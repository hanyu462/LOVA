"""Step 10-1: optimizer and learning-rate schedule. Pure functions over a model and a step counter.

    model ──param_groups──> [{decay: conv / linear weights}, {no decay: biases, norm affine}]
          ──build_optimizer──> AdamW
    step  ──lr_factor(step, cfg)──> multiplier in [min_lr_ratio, 1]:
              linear warm-up over W optimizer updates (1/W, 2/W, ..., 1), then cosine from 1 at update W
              down to min_lr_ratio at the FINAL update (total_steps - 1); constant after that
          ──build_scheduler──> LambdaLR applying lr_factor every update
              (call scheduler.step() only when an optimizer update actually happened: a GradScaler
               overflow skip must not advance the schedule — handled in the train step)

This optimizer policy excludes GroupNorm affine parameters and biases from weight decay, by shape:
anything with dim <= 1 is "no decay". The schedule is indexed by optimizer updates, not epochs or
micro-batches: the caller must choose total_steps / warmup_steps consistently with the effective
batch size, gradient accumulation and world size.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class OptimCfg:
    lr: float = 2e-4
    weight_decay: float = 0.05
    betas: tuple = (0.9, 0.999)


@dataclass(frozen=True)
class SchedCfg:
    total_steps: int = 20_000      # number of optimizer updates (indices 0 .. total_steps - 1)
    warmup_steps: int = 1_000
    min_lr_ratio: float = 0.01     # lr at the final update = min_lr_ratio * lr
    kind: str = "cosine"           # "cosine" | "constant" (warm-up then flat)


def _validate_optim(cfg: OptimCfg) -> None:
    if cfg.lr <= 0 or cfg.weight_decay < 0:
        raise ValueError(f"need lr > 0 and weight_decay >= 0, got {cfg.lr} / {cfg.weight_decay}")
    if len(cfg.betas) != 2 or any(not 0.0 <= b < 1.0 for b in cfg.betas):
        raise ValueError(f"betas must be two values in [0, 1), got {cfg.betas}")


def _validate_sched(cfg: SchedCfg) -> None:
    if cfg.total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {cfg.total_steps}")
    if cfg.warmup_steps < 0 or cfg.warmup_steps > cfg.total_steps:
        raise ValueError(f"need 0 <= warmup_steps <= total_steps, got {cfg.warmup_steps} / {cfg.total_steps}")
    if not 0.0 <= cfg.min_lr_ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must lie in [0, 1], got {cfg.min_lr_ratio}")
    if cfg.kind not in ("cosine", "constant"):
        raise ValueError(f"unknown schedule {cfg.kind!r}")
    if cfg.kind == "cosine" and cfg.warmup_steps >= cfg.total_steps:
        raise ValueError("cosine schedule needs at least one post-warm-up update")


def param_groups(model: nn.Module, cfg: OptimCfg = OptimCfg()) -> list[dict]:
    _validate_optim(cfg)
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() > 1]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() <= 1]
    return [{"params": decay, "weight_decay": cfg.weight_decay, "name": "decay"},
            {"params": no_decay, "weight_decay": 0.0, "name": "no_decay"}]


def build_optimizer(model: nn.Module, cfg: OptimCfg = OptimCfg()) -> torch.optim.AdamW:
    return torch.optim.AdamW(param_groups(model, cfg), lr=cfg.lr, betas=cfg.betas)


def lr_factor(step: int, cfg: SchedCfg = SchedCfg()) -> float:
    """Multiplier of the base lr at optimizer update `step` (0-based)."""
    _validate_sched(cfg)
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    if step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    if cfg.kind == "constant":
        return 1.0
    decay_steps = cfg.total_steps - cfg.warmup_steps          # updates W .. total_steps - 1
    if decay_steps == 1:
        return cfg.min_lr_ratio                               # the only cosine update is the final one
    t = min(step - cfg.warmup_steps, decay_steps - 1)
    return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * t / (decay_steps - 1)))


def build_scheduler(opt: torch.optim.Optimizer, cfg: SchedCfg = SchedCfg()) -> torch.optim.lr_scheduler.LambdaLR:
    _validate_sched(cfg)
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, cfg))
