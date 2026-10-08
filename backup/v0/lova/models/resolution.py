"""Pointer encoding + Resolution Predictor R_theta(F0, p).

R is predicted at the F0 resolution (stride 4). It is deliberately small
(~0.3M params): it is the "cheap coarse perception" that decides where the
expensive backbone should refine.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvGNAct, coord_grid

POINTER_CHANNELS = 4  # P_x, P_y, distance, gaussian


def pointer_maps(pointers: torch.Tensor, h: int, w: int, canvas: int, sigma: float = 0.08) -> torch.Tensor:
    """Spatial pointer representation.

    pointers: [B, 2] (x, y) in input-pixel coordinates.
    canvas:   input side length (pixels), used for normalization.
    returns:  [B, 4, h, w] = (x - p_x, y - p_y, ||.||, exp(-||.||^2 / 2 sigma^2)),
              offsets normalized to [-1, 1]-ish by canvas size.
    """
    b = pointers.shape[0]
    grid = (coord_grid(b, h, w, pointers.device) + 1) / 2  # [0,1], pixel-center approximate
    p = (pointers / canvas).view(b, 2, 1, 1)
    d = grid - p
    dist = d.norm(dim=1, keepdim=True)
    g = torch.exp(-dist.pow(2) / (2 * sigma**2))
    return torch.cat([d, dist, g], 1)


class ResolutionPredictor(nn.Module):
    """R_theta(F0, p) -> R in [0,1]^{H/4 x W/4}.

    Two inputs, two sources of evidence:
      * pointer maps   -> pointer-driven attention
      * F0 + context   -> image-driven importance (dilated low-res path gives a
                          large receptive field so R can rise away from p)

    F0 [B,C0,H/4,W/4] + P [B,4,H/4,W/4]
      -> ConvGNAct            [B,c,H/4]     (skip)
      -> ConvGNAct s2         [B,c,H/8]
      -> ConvGNAct s2         [B,c,H/16]
      -> cat P@H/16, dilated ConvGNAct x2 (d=2,4)   (context)
      -> upsample to H/4, cat skip, ConvGNAct, 1x1 -> sigmoid  [B,1,H/4,W/4]
    """

    def __init__(self, c0: int = 64, c: int = 64, init_r: float = 0.5):
        super().__init__()
        cp = POINTER_CHANNELS
        self.inp = ConvGNAct(c0 + cp, c)
        self.down1 = ConvGNAct(c, c, s=2)
        self.down2 = ConvGNAct(c, c, s=2)
        self.ctx = nn.Sequential(ConvGNAct(c + cp, c, d=2), ConvGNAct(c, c, d=4))
        self.fuse = ConvGNAct(2 * c, c)
        self.out = nn.Conv2d(c, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, torch.logit(torch.tensor(init_r)).item())

    def forward(self, f0: torch.Tensor, pointers: torch.Tensor, canvas: int) -> torch.Tensor:
        b, _, h, w = f0.shape
        p_hi = pointer_maps(pointers, h, w, canvas)
        x = self.inp(torch.cat([f0, p_hi], 1))
        y = self.down2(self.down1(x))
        p_lo = pointer_maps(pointers, y.shape[2], y.shape[3], canvas)
        y = self.ctx(torch.cat([y, p_lo], 1))
        y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
        y = self.fuse(torch.cat([x, y], 1))
        return torch.sigmoid(self.out(y))
