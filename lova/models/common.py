"""Step 9-1: building blocks shared by every model part. No task logic here.

    gn(c)                GroupNorm with the largest group count (<= 32) dividing c; batch-size independent
    ConvGNAct            Conv -> GN -> GELU               (post-activation; stems, necks, heads)
    PreActConv           GN -> GELU -> Conv               (pre-activation; inside residual streams)
    ResBlock             x + PreActConv(PreActConv(x))    (the ungated residual unit; 9-3 wraps it with a gate)
    coord_grid           cell-centre coordinates in canvas pixels at a stride (half-pixel convention,
                         the same one lova.utils.geometry.pointer_to_stride uses)
    count_params         parameter count in millions
"""
from __future__ import annotations

import torch
import torch.nn as nn


def gn(c: int, max_groups: int = 32) -> nn.GroupNorm:
    g = max_groups
    while c % g != 0:
        g //= 2
    return nn.GroupNorm(g, c)


class ConvGNAct(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int = 3, s: int = 1, d: int = 1):
        super().__init__(nn.Conv2d(cin, cout, k, s, padding=d * (k // 2), dilation=d, bias=False), gn(cout), nn.GELU())


class PreActConv(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int = 3, s: int = 1):
        super().__init__(gn(cin), nn.GELU(), nn.Conv2d(cin, cout, k, s, padding=k // 2, bias=False))


class ResBlock(nn.Module):
    """Two pre-activation 3x3 convs on a residual stream of width c. The second conv starts at zero,
    so a fresh block is the identity and a deep stack trains from a sane starting point."""

    def __init__(self, c: int):
        super().__init__()
        self.body = nn.Sequential(PreActConv(c, c), PreActConv(c, c))
        nn.init.zeros_(self.body[1][2].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


def coord_grid(h: int, w: int, stride: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """[2, h, w]: (x, y) canvas-pixel coordinates of the centre of every cell of an h x w grid at
    `stride`. Cell j covers pixels [j*stride, (j+1)*stride), its centre is (j + 0.5) * stride - 0.5."""
    ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) * stride - 0.5
    xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) * stride - 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], 0)


def count_params(m: nn.Module) -> float:
    return sum(p.numel() for p in m.parameters()) / 1e6
