"""Grid geometry on [h, w] bool / float tensors (CPU or GPU). No policy here: pointer.py decides
where pointers go and make_rgt.py defines R; both are built from these primitives.

    downsample_mask(mask, stride)          area average >= thr
    pointer_to_stride(xy, stride)          half-pixel convention p' = (p + 0.5) / stride - 0.5
    seed_cell(mask, p)                     nearest mask cell to a continuous point
    distance_to(target)                    chamfer distance to the nearest TRUE cell
    depth(mask)                            0 on the outermost ring, +1 per ring inward
    erode(mask, px)                        8-neighbour erosion, px times
    geodesic(mask, seed)                   chamfer distance from the seed travelling only through the mask
    propagate_value(sources, value, reach) distance to the nearest source + the source's value

All distances are 8-neighbour chamfer (steps 1 / sqrt 2), within 9 % of Euclidean.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

SHIFTS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
          (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]
INF = float("inf")


def shift(x: torch.Tensor, dy: int, dx: int, fill: float) -> torch.Tensor:
    """out[y, x] = x[y - dy, x - dx], `fill` outside. x [h, w]."""
    h, w = x.shape
    out = torch.full_like(x, fill)
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = x[ys, xs]
    return out


# ---- resolution / coordinates -------------------------------------------------------------------

def downsample_mask(mask: torch.Tensor, stride: int, thr: float = 0.5) -> torch.Tensor:
    """[S, S] bool -> [S/stride, S/stride] bool (area average >= thr)."""
    if stride == 1:
        return mask
    return F.avg_pool2d(mask[None, None].float(), stride)[0, 0] >= thr


def pointer_to_stride(xy, stride: int) -> tuple[float, float]:
    """Full-res pixel (x, y) -> continuous cell coordinates at `stride` (half-pixel convention)."""
    return (float(xy[0]) + 0.5) / stride - 0.5, (float(xy[1]) + 0.5) / stride - 0.5


def seed_cell(mask: torch.Tensor, p: tuple[float, float]) -> tuple[int, int]:
    """Nearest mask cell (y, x) to the continuous point p = (x, y); snaps to the closest mask cell
    if the rounded cell is outside the mask."""
    h, w = mask.shape
    cx, cy = int(round(p[0])), int(round(p[1]))
    cx, cy = min(max(cx, 0), w - 1), min(max(cy, 0), h - 1)
    if mask[cy, cx]:
        return cy, cx
    ys, xs = torch.nonzero(mask, as_tuple=True)
    if len(ys) == 0:
        raise ValueError("empty mask")
    j = int(((ys.float() - p[1]) ** 2 + (xs.float() - p[0]) ** 2).argmin())
    return int(ys[j]), int(xs[j])


# ---- distances ----------------------------------------------------------------------------------

def distance_to(target: torch.Tensor, max_iter: int = 4096) -> torch.Tensor:
    """target [h, w] bool -> [h, w] float: chamfer distance (cells) to the nearest TRUE cell
    (0 on TRUE cells, +inf if empty or not reached within max_iter steps)."""
    d = torch.where(target, torch.zeros_like(target, dtype=torch.float32), torch.full_like(target, INF, dtype=torch.float32))
    for _ in range(max_iter):
        best = d
        for dy, dx, wgt in SHIFTS:
            best = torch.minimum(best, shift(d, dy, dx, INF) + wgt)
        if torch.equal(best, d):
            return d
        d = best
    return d


def depth(mask: torch.Tensor) -> torch.Tensor:
    """mask [h, w] bool -> [h, w] float: 0 on the outermost ring (cells with a non-mask neighbour),
    1 one ring further in, ... ; 0 outside. The canvas border is NOT a boundary (the object may
    continue beyond it), so a mask touching the border keeps growing depth there."""
    d = (distance_to(~mask) - 1.0).clamp(min=0)
    return torch.where(mask, d, torch.zeros_like(d))


def erode(mask: torch.Tensor, px: int) -> torch.Tensor:
    """mask [S, S] bool -> cells with Chebyshev distance >= px + 1 from any non-mask cell
    (3x3 / 8-neighbour erosion applied px times)."""
    m = mask
    for _ in range(px):
        inv = (~m)[None, None].float()
        m = F.max_pool2d(inv, 3, 1, 1)[0, 0] == 0
    return m


def geodesic(mask: torch.Tensor, seed: tuple[int, int], max_iter: int = 4096) -> torch.Tensor:
    """mask [h, w] bool, seed (y, x) inside it -> [h, w] float: chamfer distance from the seed that
    may only travel through mask cells; +inf outside the mask and in parts not connected to the seed."""
    d = torch.full_like(mask, INF, dtype=torch.float32)
    d[seed] = 0.0
    for _ in range(max_iter):
        best = d
        for dy, dx, wgt in SHIFTS:
            best = torch.minimum(best, shift(d, dy, dx, INF) + wgt)
        best = torch.where(mask, best, torch.full_like(best, INF))
        if torch.equal(best, d):
            return d
        d = best
    return d


def propagate_value(sources: torch.Tensor, value: torch.Tensor, reach: int) -> tuple[torch.Tensor, torch.Tensor]:
    """sources [h, w] bool with value [h, w] float on them -> (distance to the nearest source,
    that source's value) for every cell, propagated for at most `reach` steps (+inf / 0 beyond)."""
    d = torch.where(sources, torch.zeros_like(value), torch.full_like(value, INF))
    v = torch.where(sources, value, torch.zeros_like(value))
    for _ in range(reach):
        best_d, best_v = d, v
        for dy, dx, wgt in SHIFTS:
            cand = shift(d, dy, dx, INF) + wgt
            better = cand < best_d
            best_v = torch.where(better, shift(v, dy, dx, 0.0), best_v)
            best_d = torch.where(better, cand, best_d)
        if torch.equal(best_d, d):
            break
        d, v = best_d, best_v
    return d, v
