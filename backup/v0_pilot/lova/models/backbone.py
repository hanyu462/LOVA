"""Step 9-3: the R-gated backbone. F0 [B,c0,S/4,S/4] + R_gate [B,1,S/4,S/4] -> {s4, s8, s16, s32}.

    r_pred  (predictor output, supervised by r_loss, left untouched)
        ──resolve_r(r_pred, valid_s4, r_override)──> r_gate = (r_override or r_pred) * valid_s4
        ──r_pyramid(r_gate, valid_s4)──> R at stride 4 / 8 / 16 / 32 by VALID-NORMALISED area averaging:
                       R_s = Pool(R_4 V_4) / Pool(V_4) on coarse cells with majority valid pixels, 0 elsewhere.
                       (R is a budget field, not a mask: averaging keeps the budget, max pooling would inflate
                       it. Normalising by the valid area keeps a constant R = 1 equal to 1 on every valid coarse
                       cell, so the A0 baseline is "full residual on every coarse cell the majority-valid
                       convention counts as valid" (a cell with exactly half its pixels in the image is dropped);
                       a plain average would read 0.5 / 0.25 on coarse cells that straddle the padding boundary.)
    stage i:  x = trans(x)                       ungated base path (strided conv; identity for stage 0)
              for each block:  x = x + g(R_s) * Body(x)      Body = two pre-activation 3x3 convs

    gate g (GateCfg.kind):  "linear"  g = R                       <- default, V0 soft gating
                            "hard"    g = 1[R > tau]              inference / train-vs-exec gap checks
                            "sigmoid" g = sigmoid((R - tau) / temp)   binary-targeted relaxation (ablation)
                            "none"    g = 1 everywhere, padding included: dense algebra reference /
                                      debugging only. NOT the A0 baseline: A0 = "linear" with
                                      r_override = 1.0 (full residual inside the image, padding 0).

With g = 0 a block is an exact identity at that location (nothing is added to the stream), so a
location with R = 0 keeps its coarse base-path feature through the whole stage. Everything is
MASKING here (all blocks computed everywhere, then multiplied); real skipping comes later.

r_override: None (use r_pred), a float (constant-R forward: 0 / 0.5 / 1 sweeps, the R = 1
baseline), or a tensor [B,1,S/4,S/4] (random / matched-budget fields). Must be finite and in
[0, 1]: a negative override would subtract the residual, which is no ablation we mean. valid is
canvas geometry (where the image actually is), not supervision: padding contributes zero gated
residual-body budget (the stem, transitions, neck and head still run there in this dense V0).

TODO (sparse execution), two sources of dense-vs-sparse mismatch to revisit when real skipping is built:
  * GroupNorm statistics are taken over the whole map, so a dense masked forward and a real
    location-skipping forward do not share normalisation statistics even where both compute.
  * Convolution halos: the second 3x3 conv of a body at an active site needs the first conv's
    output on the 3x3 neighbourhood, so the computed region is the active region dilated by the
    receptive field, not the active region itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import PreActConv, ResBlock

STAGE_STRIDES = (4, 8, 16, 32)


@dataclass(frozen=True)
class GateCfg:
    kind: str = "linear"      # "linear" | "hard" | "sigmoid" | "none"
    tau: float = 0.5
    temp: float = 0.1


@dataclass(frozen=True)
class BackboneCfg:
    widths: tuple = (64, 128, 256, 384)
    depths: tuple = (2, 2, 3, 2)
    gate: GateCfg = field(default_factory=GateCfg)


def gate(r: torch.Tensor, cfg: GateCfg = GateCfg()) -> torch.Tensor:
    if cfg.kind == "linear":
        return r
    if cfg.kind == "hard":
        return (r > cfg.tau).to(r.dtype)
    if cfg.kind == "sigmoid":
        if cfg.temp <= 0:
            raise ValueError(f"sigmoid gate needs temp > 0, got {cfg.temp}")
        return torch.sigmoid((r - cfg.tau) / cfg.temp)
    if cfg.kind == "none":
        return torch.ones_like(r)
    raise ValueError(f"unknown gate kind {cfg.kind!r}")


def valid_to_stride(valid: torch.Tensor, stride: int) -> torch.Tensor:
    """[B,S,S] bool -> [B,1,S/stride,S/stride] float {0,1}: a cell is valid when more than half of its
    pixels are (the same rule build() uses for r_valid)."""
    return (F.avg_pool2d(valid[:, None].float(), stride) > 0.5).float()


def resolve_r(r_pred: torch.Tensor, valid_s4: torch.Tensor | None, r_override=None) -> torch.Tensor:
    """r_gate [B,1,h,w]: the field the backbone actually runs on. r_pred is never modified."""
    if r_override is None:
        r = r_pred
    elif torch.is_tensor(r_override):
        if r_override.shape != r_pred.shape:
            raise ValueError(f"r_override {tuple(r_override.shape)} must match r_pred {tuple(r_pred.shape)}")
        r = r_override.to(device=r_pred.device, dtype=r_pred.dtype)
    else:
        r = torch.full_like(r_pred, float(r_override))
    if r_override is not None:
        if not torch.isfinite(r).all():
            raise ValueError("r_override contains NaN / Inf")
        if r.min() < 0 or r.max() > 1:
            raise ValueError(f"r_override must lie in [0, 1], got [{float(r.min()):.3g}, {float(r.max()):.3g}]")
    if valid_s4 is not None:
        if valid_s4.shape != r.shape:
            raise ValueError(f"valid_s4 {tuple(valid_s4.shape)} must match r {tuple(r.shape)}")
        r = r * valid_s4.to(device=r.device, dtype=r.dtype)
    return r


def valid_pyramid(valid_s4: torch.Tensor, n: int = len(STAGE_STRIDES)) -> list[torch.Tensor]:
    """[V@4, V@8, V@16, V@32] float {0,1}: a coarse cell is valid when more than half of its stride-4 cells are."""
    return [valid_s4 if i == 0 else (F.avg_pool2d(valid_s4, 2 ** i) > 0.5).to(valid_s4.dtype) for i in range(n)]


def r_pyramid(r_gate: torch.Tensor, valid_s4: torch.Tensor | None = None, n: int = len(STAGE_STRIDES)) -> list[torch.Tensor]:
    """[R@4, R@8, R@16, R@32]. Without valid: plain area averages. With valid:
    R_s = Pool(R_4 V_4) / Pool(V_4) on majority-valid coarse cells, 0 on the others.
    V_4 is multiplied in here as well, so a raw (unresolved) R gives the same result as a resolved one:
    (R V) V = R V. The invariant "padding never enters the numerator" does not depend on the caller."""
    if valid_s4 is None:
        return [r_gate if i == 0 else F.avg_pool2d(r_gate, 2 ** i) for i in range(n)]
    v4 = valid_s4.to(device=r_gate.device, dtype=r_gate.dtype)
    rv = r_gate * v4
    out = [rv]
    for i, v_s in enumerate(valid_pyramid(v4, n)):
        if i == 0:
            continue
        out.append(F.avg_pool2d(rv, 2 ** i) / F.avg_pool2d(v4, 2 ** i).clamp(min=1e-6) * v_s)
    return out


class GatedResBlock(ResBlock):
    """x + g * Body(x). Body as in ResBlock (second conv zero-initialised)."""

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:   # type: ignore[override]
        return x + g * self.body(x)


class Stage(nn.Module):
    def __init__(self, cin: int, cout: int, depth: int, stride: int):
        super().__init__()
        self.trans = nn.Identity() if (stride == 1 and cin == cout) else PreActConv(cin, cout, s=stride)
        self.blocks = nn.ModuleList(GatedResBlock(cout) for _ in range(depth))

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        x = self.trans(x)
        for blk in self.blocks:
            x = blk(x, g)
        return x


class Backbone(nn.Module):
    def __init__(self, c0: int = 64, cfg: BackboneCfg = BackboneCfg()):
        super().__init__()
        if len(cfg.widths) != len(STAGE_STRIDES) or len(cfg.depths) != len(STAGE_STRIDES):
            raise ValueError(f"widths / depths must have {len(STAGE_STRIDES)} entries (strides {STAGE_STRIDES})")
        self.cfg = cfg
        stages, cin = [], c0
        for i, (w, d) in enumerate(zip(cfg.widths, cfg.depths)):
            stages.append(Stage(cin, w, d, stride=1 if i == 0 else 2))
            cin = w
        self.stages = nn.ModuleList(stages)
        self.out_channels = dict(zip((f"s{s}" for s in STAGE_STRIDES), cfg.widths))

    def forward(self, f0: torch.Tensor, r_gate: torch.Tensor, valid_s4: torch.Tensor | None = None) -> dict:
        """f0 [B,c0,S/4,S/4], r_gate [B,1,S/4,S/4] (from resolve_r, with the same valid_s4) -> {"s4","s8","s16","s32"}."""
        if r_gate.shape != (f0.shape[0], 1, *f0.shape[2:]):
            raise ValueError(f"r_gate must be [B,1,{f0.shape[2]},{f0.shape[3]}], got {tuple(r_gate.shape)}")
        feats, x = {}, f0
        for i, (stage, r_s) in enumerate(zip(self.stages, r_pyramid(r_gate, valid_s4))):
            x = stage(x, gate(r_s.to(x.dtype), self.cfg.gate))
            feats[f"s{STAGE_STRIDES[i]}"] = x
        return feats

    @torch.no_grad()
    def gate_budget(self, r_gate: torch.Tensor, valid_s4: torch.Tensor | None = None) -> torch.Tensor:
        """[B] gated residual-body budget PROXY:
            B = sum_s c_s sum_{x in V_s} g_s(x)  /  sum_s c_s |V_s|
        with c_s the dense cost of one cell of stage s (two 3x3 convs of width c per block, times
        the number of blocks) and V_s the valid cells of that stage (all cells without valid_s4).
        Logging only.

        This is NOT the FLOPs fraction of the current model: every block is computed densely and
        then multiplied, so V0 FLOPs do not depend on R at all. With a linear gate, R = 0.5 is a
        budget weight of 0.5, not half a convolution. Stem, R predictor, stage transitions, neck and
        head are outside the weighting. For hard gates it is an idealised active-site compute proxy;
        actual sparse FLOPs additionally depend on convolution halos, normalisation, indexing and
        executor overhead."""
        used = torch.zeros(r_gate.shape[0], device=r_gate.device, dtype=torch.float32)
        total = torch.zeros_like(used)
        vs = None if valid_s4 is None else valid_pyramid(valid_s4.to(r_gate.dtype))
        for i, (stage, r_s) in enumerate(zip(self.stages, r_pyramid(r_gate, valid_s4))):
            g = gate(r_s, self.cfg.gate).float()
            v = torch.ones_like(g) if vs is None else vs[i].float()
            c = 2 * 9 * self.cfg.widths[i] ** 2 * len(stage.blocks)
            used += (g * v).flatten(1).sum(1) * c
            total += v.flatten(1).sum(1) * c
        return used / total.clamp(min=1)
