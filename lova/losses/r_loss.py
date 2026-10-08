"""Step 8-1: R loss. The R predictor is trained to reproduce R_GT where the cell is valid.

    L_R = sum_x V_R(x) * (R_pred(x) - R_GT(x))^2  /  sum_x V_R(x)

    r_pred   [B, H, W] (or [B, 1, H, W]) in [0, 1]      R predictor output at the supervision stride
    r_gt     [B, H, W]                                  TrainingSample.r_gt
    r_valid  [B, H, W] bool                             TrainingSample.r_valid: padding is IGNORED,
                                                        never a "0" target

MSE rather than BCE: R is a continuous computation-budget field, not a probability. Reduction is a
mean over valid cells of the whole batch (not per image), so a half-padded image does not weigh
twice as much per cell. If no cell is valid the loss is 0 with a gradient path kept.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RLossCfg:
    kind: str = "mse"     # "mse" | "l1"


def _as_bhw(r_pred: torch.Tensor, r_gt: torch.Tensor, r_valid: torch.Tensor) -> torch.Tensor:
    """Strict shape contract: r_pred [B,H,W] or [B,1,H,W]; r_gt / r_valid [B,H,W]. Anything else raises
    (a [B,2,H,W] head output must not silently lose a channel)."""
    if r_pred.dim() == 4:
        if r_pred.shape[1] != 1:
            raise ValueError(f"r_pred must have one channel, got {tuple(r_pred.shape)}")
        r_pred = r_pred[:, 0]
    elif r_pred.dim() != 3:
        raise ValueError(f"r_pred must be [B,H,W] or [B,1,H,W], got {tuple(r_pred.shape)}")
    if r_gt.dim() != 3 or r_valid.dim() != 3 or r_pred.shape != r_gt.shape or r_gt.shape != r_valid.shape:
        raise ValueError(f"shape mismatch: r_pred {tuple(r_pred.shape)} r_gt {tuple(r_gt.shape)} r_valid {tuple(r_valid.shape)}")
    return r_pred


def r_loss(r_pred: torch.Tensor, r_gt: torch.Tensor, r_valid: torch.Tensor, cfg: RLossCfg = RLossCfg()) -> torch.Tensor:
    r_pred = _as_bhw(r_pred, r_gt, r_valid)
    diff = r_pred.float() - r_gt.float()
    if cfg.kind == "mse":
        per = diff * diff
    elif cfg.kind == "l1":
        per = diff.abs()
    else:
        raise ValueError(f"kind={cfg.kind!r} (mse | l1)")
    v = r_valid.to(per.dtype)
    return (per * v).sum() / v.sum().clamp(min=1.0)


@torch.no_grad()
def r_stats(r_pred: torch.Tensor, r_gt: torch.Tensor, r_valid: torch.Tensor) -> dict:
    """Numbers worth logging next to the loss: mean |error| over valid cells, and how much of the
    valid area each field lights (R > 0.5) - the compute-budget view of the same tensors.
    Returned as 0-d tensors (no host sync); call .item() only at logging time."""
    r_pred = _as_bhw(r_pred, r_gt, r_valid)
    v = r_valid.float()
    n = v.sum().clamp(min=1.0)
    return dict(r_mae=((r_pred - r_gt).abs() * v).sum() / n,
                r_lit_pred=((r_pred > 0.5).float() * v).sum() / n,
                r_lit_gt=((r_gt > 0.5).float() * v).sum() / n)
