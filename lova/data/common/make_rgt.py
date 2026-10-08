"""Step 5: (target mask, pointer) -> R_GT, the pseudo ground truth for the R predictor.

    R_GT = make_rgt(mask, pointer, cfg)        [S/stride, S/stride] in [0, 1]

5-1 INSIDE profile (geodesic from the pointer), 5-2 OUTSIDE profile (boundary value carried out).

    d_g(p, x)   geodesic distance from the pointer to x INSIDE the mask (8-neighbour chamfer
                propagation that may only pass through mask cells; <= 9 % from true Euclidean geodesic)
    R_in(x)     = exp[ -( d_g(p, x) / lambda_in )^gamma ],   lambda_in = lambda_in_frac * max_x d_g(p, x)

    R_in(p) = 1, smooth decrease along the object, following its shape (a U-shaped object has a low
    value on the far arm although it is close in Euclidean terms). The boundary value is NOT a
    constant: it depends on how far each boundary point is from the pointer, so moving the pointer
    changes the whole field:  R_GT(M, p1) != R_GT(M, p2).

Resolution: geometry is computed on the mask downsampled by `stride` (2 -> 320 x 320 for a 640
canvas); the pointer is mapped with the half-pixel convention p' = (p + 0.5) / stride - 0.5 and
snapped to the nearest mask cell if downsampling left its cell outside. Supervision at stride 4 is
a later bilinear downsample of the continuous field.

OUTSIDE profile:
    b(x)        nearest mask cell to the outside cell x (chamfer propagation outward from the mask
                that carries the R_in value of the cell it came from)
    d_out(x)    that chamfer distance (cells), minus 0.5 (cell centre -> boundary)
    R_out(x)    = R_in(b(x)) * exp(-d_out(x) / lambda_out),   lambda_out in pixels (absolute)

    The value is continuous across the boundary (R_out -> R_in(b) as d_out -> 0) and decays much
    faster outside than inside (lambda_out << lambda_in for all but tiny objects). Cells with
    d_out > 6 * lambda_out are exactly 0 (exp(-6) < 0.003); the propagation runs only as many
    iterations as that distance needs.

Pure torch on [h, w] tensors (CPU or GPU).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RgtCfg:
    stride: int = 2             # geometry resolution (canvas / stride)
    gamma: float = 2.0          # inside exponent: 2 = Gaussian-like plateau around the pointer, 1 = exponential
    lambda_in_frac: float = 1.0 # lambda_in = frac * max geodesic distance from the pointer (object-relative)
    lambda_out_px: float = 32.0 # outside decay length in INPUT pixels (absolute; 0.05 * 640)
    mask_thr: float = 0.5       # downsampled soft mask -> bool


_SHIFTS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
           (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]


def _shift(x: torch.Tensor, dy: int, dx: int, fill: float) -> torch.Tensor:
    """out[y, x] = x[y - dy, x - dx], `fill` outside. x [h, w]."""
    h, w = x.shape
    out = torch.full_like(x, fill)
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = x[ys, xs]
    return out


def downsample_mask(mask: torch.Tensor, stride: int, thr: float = 0.5) -> torch.Tensor:
    """[S, S] bool -> [S/stride, S/stride] bool (area average >= thr)."""
    if stride == 1:
        return mask
    return F.avg_pool2d(mask[None, None].float(), stride)[0, 0] >= thr


def pointer_to_stride(xy, stride: int) -> tuple[float, float]:
    """Full-res pixel (x, y) -> continuous cell coordinates at `stride` (half-pixel convention)."""
    return (float(xy[0]) + 0.5) / stride - 0.5, (float(xy[1]) + 0.5) / stride - 0.5


def seed_cell(mask_s: torch.Tensor, p_s: tuple[float, float]) -> tuple[int, int]:
    """Nearest mask cell (y, x) to the continuous pointer p_s = (x, y); snaps if the rounded cell is
    outside the (downsampled) mask."""
    h, w = mask_s.shape
    cx, cy = int(round(p_s[0])), int(round(p_s[1]))
    cx, cy = min(max(cx, 0), w - 1), min(max(cy, 0), h - 1)
    if mask_s[cy, cx]:
        return cy, cx
    ys, xs = torch.nonzero(mask_s, as_tuple=True)
    if len(ys) == 0:
        raise ValueError("empty mask")
    d2 = (ys.float() - p_s[1]) ** 2 + (xs.float() - p_s[0]) ** 2
    j = int(d2.argmin())
    return int(ys[j]), int(xs[j])


def geodesic_from_pointer(mask_s: torch.Tensor, seed: tuple[int, int], max_iter: int = 4096) -> torch.Tensor:
    """mask_s [h, w] bool, seed (y, x) inside it -> [h, w] float: geodesic (within-mask) chamfer
    distance from the seed; +inf outside the mask and in mask parts not connected to the seed."""
    inf = float("inf")
    d = torch.full_like(mask_s, inf, dtype=torch.float32)
    d[seed] = 0.0
    for _ in range(max_iter):
        best = d
        for dy, dx, wgt in _SHIFTS:
            best = torch.minimum(best, _shift(d, dy, dx, inf) + wgt)
        best = torch.where(mask_s, best, torch.full_like(best, inf))  # may only travel through the mask
        if torch.equal(best, d):
            return d
        d = best
    return d


def inside_profile(mask_s: torch.Tensor, d_g: torch.Tensor, cfg: RgtCfg = RgtCfg()) -> torch.Tensor:
    """R_in = exp(-(d_g / lambda_in)^gamma) on the mask, 0 elsewhere. lambda_in = frac * max finite d_g."""
    finite = torch.isfinite(d_g) & mask_s
    d_max = d_g[finite].max() if finite.any() else torch.tensor(1.0)
    lam = cfg.lambda_in_frac * float(d_max)
    if lam <= 0:  # single-cell object: R = 1 there
        return finite.float()
    r = torch.exp(-((d_g / lam).clamp(min=0)) ** cfg.gamma)
    return torch.where(finite, r, torch.zeros_like(r))


def outside_profile(mask_s: torch.Tensor, r_in: torch.Tensor, cfg: RgtCfg = RgtCfg()) -> torch.Tensor:
    """R_out on cells outside the mask: the R_in value of the nearest mask cell, decayed with
    exp(-d_out / lambda_out). Returns [h, w] with R_in kept on the mask itself."""
    inf = float("inf")
    lam = cfg.lambda_out_px / cfg.stride                      # cells
    cutoff = 6.0 * lam                                        # R_out definition: 0 beyond this distance (exp(-6) < 0.003)
    reach = int(math.ceil(cutoff)) + 1                        # propagation budget only (one step per iteration; a step
                                                              # is 1 or sqrt(2) cells, so this is NOT the distance cutoff)
    d = torch.where(mask_s, torch.zeros_like(r_in), torch.full_like(r_in, inf))
    v = torch.where(mask_s, r_in, torch.zeros_like(r_in))
    for _ in range(reach):
        best_d, best_v = d, v
        for dy, dx, wgt in _SHIFTS:
            cand = _shift(d, dy, dx, inf) + wgt
            better = cand < best_d
            best_v = torch.where(better, _shift(v, dy, dx, 0.0), best_v)
            best_d = torch.where(better, cand, best_d)
        if torch.equal(best_d, d):
            break
        d, v = best_d, best_v
    d_out = (d - 0.5).clamp(min=0)                            # cell centre -> boundary
    within = torch.isfinite(d_out) & (d_out <= cutoff)
    r_out = torch.where(within, v * torch.exp(-d_out / lam), torch.zeros_like(v))
    return torch.where(mask_s, r_in, r_out)


def make_r_in(mask: torch.Tensor, pointer, cfg: RgtCfg = RgtCfg()) -> torch.Tensor:
    """Full-res mask [S, S] bool + pointer (x, y) in full-res pixels -> R_in [S/stride, S/stride] (0 outside)."""
    mask_s = downsample_mask(mask, cfg.stride, cfg.mask_thr)
    seed = seed_cell(mask_s, pointer_to_stride(pointer, cfg.stride))
    return inside_profile(mask_s, geodesic_from_pointer(mask_s, seed), cfg)


def make_rgt(mask: torch.Tensor, pointer, cfg: RgtCfg = RgtCfg()) -> torch.Tensor:
    """Full-res mask [S, S] bool + pointer (x, y) in full-res pixels -> R_GT [S/stride, S/stride] in [0, 1]:
    inside profile on the mask, outside decay around it, 0 far away."""
    mask_s = downsample_mask(mask, cfg.stride, cfg.mask_thr)
    seed = seed_cell(mask_s, pointer_to_stride(pointer, cfg.stride))
    r_in = inside_profile(mask_s, geodesic_from_pointer(mask_s, seed), cfg)
    return outside_profile(mask_s, r_in, cfg)
