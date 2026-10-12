"""Step 9-2b: pointer encoding + Resolution Predictor R_theta(F0, p) -> r_pred [B,1,S/4,S/4].

    pointer [B,2] (x, y) canvas px ──pointer_maps──> P [B,4,h,w]   dx, dy, distance, gaussian
    F0 [B,c0,h,w] + P ──> ConvGNAct (skip @ h) ──> s2 ──> s2 ──> + P@h/4, dilated context ──> up ──> fuse ──> 1x1 ──> sigmoid

The pointer enters the network ONLY here (the backbone, neck and head never see it): everything
pointer-specific must be expressed through R. The context path (stride 16, dilations 2 and 4) gives
R a receptive field of a few hundred pixels so it can follow an object away from the pointer; the
skip path keeps the stride-4 detail. r_pred starts as the constant `init_r` (zero output weights).

pointer_maps uses cell centres (coord_grid), so the gaussian peaks on the cell that contains the
pointer, consistent with the R_GT geometry.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvGNAct, coord_grid

POINTER_CHANNELS = 4


@dataclass(frozen=True)
class RPredictorCfg:
    c: int = 64                 # width of the predictor
    sigma_frac: float = 0.08    # gaussian pointer cue width as a fraction of the canvas side (0.08 * 640 = 51 px)
    init_r: float = 0.5         # constant output at initialisation


def pointer_maps(pointer: torch.Tensor, h: int, w: int, stride: int, canvas: int, sigma_frac: float = 0.08) -> torch.Tensor:
    """[B,2] px -> [B,4,h,w]: (x - p_x)/canvas, (y - p_y)/canvas, their norm, exp(-norm^2 / 2 sigma^2)."""
    grid = coord_grid(h, w, stride, pointer.device, pointer.dtype)[None]      # [1,2,h,w] px
    d = (grid - pointer.view(-1, 2, 1, 1)) / canvas
    dist = d.norm(dim=1, keepdim=True)
    g = torch.exp(-dist.pow(2) / (2 * sigma_frac ** 2))
    return torch.cat([d, dist, g], 1)


class RPredictor(nn.Module):
    def __init__(self, c0: int = 64, cfg: RPredictorCfg = RPredictorCfg(), stride: int = 4):
        super().__init__()
        c, cp = cfg.c, POINTER_CHANNELS
        self.cfg, self.stride = cfg, stride
        self.inp = ConvGNAct(c0 + cp, c)
        self.down1 = ConvGNAct(c, c, s=2)
        self.down2 = ConvGNAct(c, c, s=2)
        self.ctx = nn.Sequential(ConvGNAct(c + cp, c, d=2), ConvGNAct(c, c, d=4))
        self.fuse = ConvGNAct(2 * c, c)
        self.out = nn.Conv2d(c, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, float(torch.logit(torch.tensor(cfg.init_r))))

    def forward(self, f0: torch.Tensor, pointer: torch.Tensor, canvas: int) -> torch.Tensor:
        """f0 [B,c0,h,w] (stride 4), pointer [B,2] px, canvas = input side S -> [B,1,h,w] in (0,1)."""
        b, _, h, w = f0.shape
        p_hi = pointer_maps(pointer, h, w, self.stride, canvas, self.cfg.sigma_frac).to(f0.dtype)
        x = self.inp(torch.cat([f0, p_hi], 1))
        y = self.down2(self.down1(x))
        p_lo = pointer_maps(pointer, y.shape[2], y.shape[3], self.stride * 4, canvas, self.cfg.sigma_frac).to(y.dtype)
        y = self.ctx(torch.cat([y, p_lo], 1))
        y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
        y = self.fuse(torch.cat([x, y], 1))
        return torch.sigmoid(self.out(y))
