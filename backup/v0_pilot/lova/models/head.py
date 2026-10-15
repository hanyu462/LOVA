"""Step 9-4b: the segmentation head (SOLOv2-style dynamic kernels, CenterNet centre assignment).

    p8 [B,c,S/8,S/8] ──heat tower (2 x ConvGNAct) ──3x3──> heat_logits [B,K,S/8,S/8]   -> heat_loss
    p8 + coords      ──kernel tower (2 x ConvGNAct)──3x3──> kernel_map  [B,D,S/8,S/8]   -> mask_loss
    p4 + coords      ──mask branch (2 x ConvGNAct, 1x1, GN, GELU)──> mask_feat [B,D,S/4,S/4] -> mask_loss

    the loss (not the head) forms   mask_logits[q] = <kernel_map[:, q], mask_feat>   at the positive cells q

coords: normalised (x, y) in (-1, 1) of each cell centre, so kernels and the mask feature can encode
position (a dynamic kernel has to separate "this instance here" from a same-looking one elsewhere).

Initialisation, set from the loss tests (8-4) and checked on a real batch (9-4 / 9-5):
    heat_out bias = logit(prior), prior 0.1 -> starts with a low foreground prior instead of p ~ 0.5,
        which removes most of the initial negative focal loss
    heat_out weights ~ N(0, 0.01)   -> preserves that prior approximately (p ~ 0.10 +- 0.02 at init)
    kernel_out weights ~ N(0, 0.01) -> initial dynamic-mask logits at a moderate, unsaturated scale
        (|z| mean ~ 1, p95 ~ 3.5 on a real batch) instead of the saturated std-1 kernels of the loss test
The head never receives R or the pointer directly; R affects it only through the backbone / neck
features, so lower R only changes the quality of what arrives.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .common import ConvGNAct, coord_grid, gn


@dataclass(frozen=True)
class HeadCfg:
    c: int = 128            # tower width (= neck width)
    d: int = 64             # dynamic kernel / mask feature dimension D
    prior: float = 0.1      # initial heat probability
    out_std: float = 0.01   # std of the heat / kernel output weights


def norm_coords(h: int, w: int, stride: int, device, dtype) -> torch.Tensor:
    """[1, 2, h, w] cell-centre coordinates mapped to (-1, 1) over the canvas h*stride x w*stride."""
    g = coord_grid(h, w, stride, device, dtype)
    size = torch.tensor([w * stride, h * stride], device=device, dtype=dtype).view(2, 1, 1)
    return ((g + 0.5) / size * 2 - 1)[None]


class Head(nn.Module):
    def __init__(self, num_classes: int, cfg: HeadCfg = HeadCfg()):
        super().__init__()
        if num_classes <= 0 or cfg.c <= 0 or cfg.d <= 0:
            raise ValueError(f"num_classes / c / d must be > 0, got {num_classes} / {cfg.c} / {cfg.d}")
        if not 0.0 < cfg.prior < 1.0 or cfg.out_std < 0:
            raise ValueError(f"prior must lie in (0, 1) and out_std be >= 0, got {cfg.prior} / {cfg.out_std}")
        c, d = cfg.c, cfg.d
        self.cfg, self.num_classes = cfg, num_classes
        self.heat_tower = nn.Sequential(ConvGNAct(c, c), ConvGNAct(c, c))
        self.heat_out = nn.Conv2d(c, num_classes, 3, padding=1)
        nn.init.normal_(self.heat_out.weight, std=cfg.out_std)
        nn.init.constant_(self.heat_out.bias, -math.log((1 - cfg.prior) / cfg.prior))
        self.kernel_tower = nn.Sequential(ConvGNAct(c + 2, c), ConvGNAct(c, c))
        self.kernel_out = nn.Conv2d(c, d, 3, padding=1)
        nn.init.normal_(self.kernel_out.weight, std=cfg.out_std)
        nn.init.zeros_(self.kernel_out.bias)
        self.mask_branch = nn.Sequential(ConvGNAct(c + 2, c), ConvGNAct(c, c), nn.Conv2d(c, d, 1, bias=False), gn(d), nn.GELU())

    def forward(self, neck: dict) -> dict:
        p8, p4 = neck["p8"], neck["p4"]
        # neck -> head contract: p8 [B,c,H,W], p4 [B,c,2H,2W]
        if p8.dim() != 4 or p4.dim() != 4 or p8.shape[0] != p4.shape[0]:
            raise ValueError(f"p8 / p4 must be 4-D with the same batch, got {tuple(p8.shape)} / {tuple(p4.shape)}")
        if p8.shape[1] != self.cfg.c or p4.shape[1] != self.cfg.c:
            raise ValueError(f"head expects {self.cfg.c} channels, got p8 {p8.shape[1]} / p4 {p4.shape[1]}")
        if p4.shape[-2:] != (2 * p8.shape[-2], 2 * p8.shape[-1]):
            raise ValueError(f"p4 must be 2x p8 spatially, got p8 {tuple(p8.shape[-2:])} / p4 {tuple(p4.shape[-2:])}")
        c8 = norm_coords(p8.shape[2], p8.shape[3], 8, p8.device, p8.dtype).expand(p8.shape[0], -1, -1, -1)
        c4 = norm_coords(p4.shape[2], p4.shape[3], 4, p4.device, p4.dtype).expand(p4.shape[0], -1, -1, -1)
        return {"heat_logits": self.heat_out(self.heat_tower(p8)),
                "kernel_map": self.kernel_out(self.kernel_tower(torch.cat([p8, c8], 1))),
                "mask_feat": self.mask_branch(torch.cat([p4, c4], 1))}
