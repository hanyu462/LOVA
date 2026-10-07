"""Initial block (C_phi) and the R-conditioned backbone A_psi(F0, R).

Interface contract (keep this stable so V1/V2 can swap the implementation):

    stem:      I [B,3,H,W]                    -> F0 [B,C0,H/4,W/4]
    backbone:  (F0, R [B,1,H/4,W/4])          -> {"s4","s8","s16","s32"} features
               plus backbone.last_gates: list of per-block gate maps (for logging)

V0 conditioning = soft depth gating of residual blocks:

    F_{l+1} = F_l + g_l(R) * Block_l(F_l),   g_l(R) = sigmoid((R - tau_l) / T)

Within a stage the thresholds tau_l increase, so R acts as a continuous
"how many refinement blocks run here" knob: R=0 -> (almost) identity only,
R=1 -> every block. This is MASKING in V0 (all blocks are computed
everywhere). In V1 the same thresholds become hard tile-level skip
decisions, which is where real FLOPs savings come from.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvGNAct, PreActConv


class Stem(nn.Module):
    """Initial conv block. Shallow on purpose: it is shared by R_theta and the
    backbone, and must not be strong enough to solve the task on its own."""

    def __init__(self, c0: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            ConvGNAct(3, c0 // 2, s=2),  # H/2
            ConvGNAct(c0 // 2, c0, s=2),  # H/4
        )

    def forward(self, x):
        return self.body(x)


class GatedResBlock(nn.Module):
    """Pre-activation residual block with an R-dependent gate.

    g == 0 gives an exact identity (no activation on the stream), so a low R
    location really keeps its coarse F_l.
    """

    def __init__(self, c: int, tau: float):
        super().__init__()
        self.tau = tau
        self.body = nn.Sequential(PreActConv(c, c), PreActConv(c, c))

    def gate(self, r: torch.Tensor, mode: str, temp: float) -> torch.Tensor:
        if mode == "depth":
            return torch.sigmoid((r - self.tau) / temp)
        if mode == "linear":
            return r
        if mode == "none":
            return torch.ones_like(r)
        raise ValueError(mode)

    def forward(self, x, r, mode, temp):
        g = self.gate(r, mode, temp)
        return x + g * self.body(x), g


class RAStage(nn.Module):
    def __init__(self, cin: int, cout: int, depth: int, stride: int, transition: str = "conv"):
        super().__init__()
        # Transition = part of the always-on base path (not gated). Its strength bounds how
        # good low-R perception can be: "light" (avgpool + 1x1) makes R matter more.
        if stride == 1 and cin == cout:
            self.trans = nn.Identity()
        elif transition == "conv":
            self.trans = PreActConv(cin, cout, s=stride)
        elif transition == "light":
            self.trans = nn.Sequential(nn.AvgPool2d(stride) if stride > 1 else nn.Identity(), PreActConv(cin, cout, k=1))
        else:
            raise ValueError(transition)
        taus = [(i + 0.5) / depth for i in range(depth)]
        self.blocks = nn.ModuleList(GatedResBlock(cout, t) for t in taus)

    def forward(self, x, r, mode, temp):
        x = self.trans(x)
        gates = []
        for blk in self.blocks:
            x, g = blk(x, r, mode, temp)
            gates.append(g)
        return x, gates


class RABackbone(nn.Module):
    """Stages at strides 4/8/16/32. R (stride 4) is average-pooled to each stage."""

    def __init__(self, c0=64, widths=(64, 128, 256, 384), depths=(2, 2, 3, 2),
                 gate_mode: str = "depth", gate_temp: float = 0.1, transition: str = "conv"):
        super().__init__()
        self.widths = widths
        self.gate_mode = gate_mode
        self.gate_temp = gate_temp
        stages, cin = [], c0
        for i, (w, d) in enumerate(zip(widths, depths)):
            stages.append(RAStage(cin, w, d, stride=1 if i == 0 else 2, transition=transition))
            cin = w
        self.stages = nn.ModuleList(stages)
        self.last_gates = []

    def forward(self, f0: torch.Tensor, r: torch.Tensor):
        feats, self.last_gates = {}, []
        x = f0
        for i, stage in enumerate(self.stages):
            r_s = r if i == 0 else F.avg_pool2d(r, 2**i)
            x, gates = stage(x, r_s, self.gate_mode, self.gate_temp)
            feats[f"s{4 * 2**i}"] = x
            self.last_gates.append(gates)
        return feats

    @torch.no_grad()
    def virtual_compute(self, r: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        """Fraction of gated-block FLOPs that a hard-routing V1 would execute
        (block runs where R > tau). Per image, [B]. Not a V0 speedup."""
        total, used = 0.0, 0.0
        for i, stage in enumerate(self.stages):
            r_s = r if i == 0 else F.avg_pool2d(r, 2**i)
            v = None if valid is None else F.avg_pool2d(valid, 2**i) > 0.5
            c = self.widths[i]
            flops_px = 2 * 9 * c * c  # two 3x3 convs per block
            npx = r_s.shape[2] * r_s.shape[3]
            for blk in stage.blocks:
                on = (r_s > blk.tau).float()
                if v is not None:
                    frac = (on * v).flatten(1).sum(1) / v.flatten(1).sum(1).clamp(min=1)
                else:
                    frac = on.flatten(1).mean(1)
                used = used + frac * flops_px * npx
                total += flops_px * npx
        return used / total
