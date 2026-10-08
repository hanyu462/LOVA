"""Small grid-geometry helpers shared by pointer.py and make_rgt.py (chamfer propagation)."""
from __future__ import annotations

import math

import torch

SHIFTS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
          (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]


def shift(x: torch.Tensor, dy: int, dx: int, fill: float) -> torch.Tensor:
    """out[y, x] = x[y - dy, x - dx], `fill` outside. x [h, w]."""
    h, w = x.shape
    out = torch.full_like(x, fill)
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = x[ys, xs]
    return out


def distance_to(target: torch.Tensor, max_iter: int = 4096) -> torch.Tensor:
    """target [h, w] bool -> [h, w] float: 8-neighbour chamfer distance (cells) to the nearest TRUE cell
    (0 on TRUE cells, +inf if empty or not reached within max_iter steps; <= 9 % from Euclidean)."""
    inf = float("inf")
    d = torch.where(target, torch.zeros_like(target, dtype=torch.float32), torch.full_like(target, inf, dtype=torch.float32))
    for _ in range(max_iter):
        best = d
        for dy, dx, wgt in SHIFTS:
            best = torch.minimum(best, shift(d, dy, dx, inf) + wgt)
        if torch.equal(best, d):
            return d
        d = best
    return d
