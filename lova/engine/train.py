"""Step 10-2 / 10-3: one batch -> one optimizer update. AMP, gradient clipping, scheduler bookkeeping.

    batch (collate output, any device) ──to_device──> device of the model
    ┌ autocast(amp_dtype) ┐
    │ out = model(image, pointer, valid)                       (valid always passed: padding gated to 0)
    │ losses = total_loss(out.as_dict(), batch, loss_cfg)
    └─────────────────────┘
    losses["total"].backward()            exactly ONE backward per step (per-loss gradients were a 9-5 diagnostic)
    grad_norm = clip_grad_norm_(params, grad_clip_norm)        global norm; the returned value is the PRE-clip norm
    optimizer.step(); scheduler.step()    the schedule counts successful parameter updates only

grad_clip_norm = 100 is a first-training safeguard, not a tuned value: on the first 40 real steps the
pre-clip norm had median ~50 and p95 ~380, so 1.0 clipped every step (gradient normalisation rather
than a safeguard; it also erases the step-to-step magnitude that AdamW's moments see). Log
`clipped` and watch the clip fraction: < 5-10 % is a safeguard, > 50 % means the threshold defines the
dynamics. Global-norm clipping scales all losses' gradients by one scalar, so the (1, 1, 3) loss
weights keep their meaning.

AMP policy:  "bf16" (V0 default, no GradScaler: bf16 has fp32's exponent range, loss scaling is not needed;
             a non-finite loss OR gradient is an error, not something to skip: nothing may reach the optimizer)
             "fp16" (GradScaler; unscale before clipping; an overflow skips the optimizer update AND the
             scheduler step, detected as a decrease of the scale)
             "fp32" (autocast off)
Out of scope here: gradient accumulation, EMA, DDP synchronisation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..losses.total_loss import TotalLossCfg, total_loss
from ..models.backbone import valid_to_stride

AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}


@dataclass(frozen=True)
class TrainCfg:
    amp: bool = True
    amp_dtype: str = "bf16"          # "bf16" | "fp16" | "fp32"
    grad_clip_norm: float | None = 100.0


def _validate(cfg: TrainCfg) -> None:
    if cfg.amp_dtype not in AMP_DTYPES:
        raise ValueError(f"amp_dtype must be one of {sorted(AMP_DTYPES)}, got {cfg.amp_dtype!r}")
    if cfg.grad_clip_norm is not None and (not math.isfinite(cfg.grad_clip_norm) or cfg.grad_clip_norm <= 0):
        raise ValueError(f"grad_clip_norm must be a finite number > 0, or None to disable clipping, got {cfg.grad_clip_norm}")


def amp_dtype(cfg: TrainCfg) -> torch.dtype | None:
    """The autocast dtype, or None when autocast is off."""
    _validate(cfg)
    return AMP_DTYPES[cfg.amp_dtype] if cfg.amp else None


def make_scaler(cfg: TrainCfg, device: torch.device | str) -> torch.amp.GradScaler | None:
    """A GradScaler only for fp16 autocast; bf16 / fp32 run without one."""
    if amp_dtype(cfg) is not torch.float16:
        return None
    return torch.amp.GradScaler(torch.device(device).type)


def to_device(batch: dict, device: torch.device | str) -> dict:
    """Tensors move (non_blocking for pinned memory); lists of per-image tensors move element-wise."""
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, list) and v and torch.is_tensor(v[0]):
            out[k] = [t.to(device, non_blocking=True) for t in v]
        else:
            out[k] = v
    return out


def train_step(model: nn.Module, batch: dict, optimizer: torch.optim.Optimizer, scheduler, loss_cfg: TotalLossCfg = TotalLossCfg(),
               cfg: TrainCfg = TrainCfg(), scaler: torch.amp.GradScaler | None = None, r_override=None, detach_r: bool = False) -> dict:
    """One update. r_override (e.g. 1.0 for the dense A0 baseline) / detach_r (predictor learns R_GT only) are the
    training-time gate policy and belong to the run meta. Returns detached 0-d tensors (losses, grad_norm, r_budget, clipped) plus python values
    (lr used, did_step, scale). Without a scaler, a non-finite loss or gradient raises FloatingPointError
    before the optimizer is touched."""
    dtype = amp_dtype(cfg)
    if (dtype is torch.float16) != (scaler is not None):
        raise ValueError("fp16 autocast needs a GradScaler and nothing else must have one (see make_scaler)")
    device = next(model.parameters()).device
    batch = to_device(batch, device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device.type, dtype=dtype if dtype is not None else torch.float32, enabled=dtype is not None):
        out = model(batch["image"], batch["pointer"], batch["valid"], r_override, detach_r)   # plain forward; intermediates are for diagnostics
        losses = total_loss(out.as_dict(), batch, loss_cfg)
    total = losses["total"]
    lr = optimizer.param_groups[0]["lr"]
    if scaler is None:
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite loss {float(total.detach())} (r {float(losses['r_raw'])}, heat {float(losses['heat_raw'])}, mask {float(losses['mask_raw'])})")
        total.backward()
        try:
            grad_norm = _clip(model, cfg, error_if_nonfinite=True)      # a NaN / Inf gradient must never reach the optimizer
        except RuntimeError as e:
            raise FloatingPointError("non-finite gradient (loss was finite)") from e
        optimizer.step()
        did_step, scale = True, 1.0
    else:
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)                       # clip on real gradients; non-finite ones are recorded by the scaler
        grad_norm = _clip(model, cfg)
        old = scaler.get_scale()
        scaler.step(optimizer)                           # skipped internally when any grad is inf / nan
        scaler.update()
        scale = scaler.get_scale()
        did_step = scale >= old                          # a skipped step decreases the scale
    if did_step:
        scheduler.step()
    with torch.no_grad():
        r_budget = model.backbone.gate_budget(out.r_gate, valid_to_stride(batch["valid"], 4)).mean()
        clipped = grad_norm > cfg.grad_clip_norm if cfg.grad_clip_norm is not None else torch.zeros((), dtype=torch.bool, device=grad_norm.device)
    # every tensor is detached: a loop may keep these logs without keeping the step's graph alive
    return {"total": total.detach(), "r_raw": losses["r_raw"].detach(), "heat_raw": losses["heat_raw"].detach(), "mask_raw": losses["mask_raw"].detach(),
            "r": losses["r"].detach(), "heat": losses["heat"].detach(), "mask": losses["mask"].detach(),
            "grad_norm": grad_norm.detach(), "clipped": clipped.detach(), "r_budget": r_budget.detach(),
            "lr": lr, "did_step": did_step, "scale": scale}


def _clip(model: nn.Module, cfg: TrainCfg, *, error_if_nonfinite: bool = False) -> torch.Tensor:
    """Global-norm clipping; returns the pre-clip norm (0-d). Without clipping the norm is still measured."""
    max_norm = float("inf") if cfg.grad_clip_norm is None else cfg.grad_clip_norm
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, error_if_nonfinite=error_if_nonfinite)
