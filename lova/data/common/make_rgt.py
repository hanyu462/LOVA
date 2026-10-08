"""Step 5: (target mask, pointer) -> R_GT, the pseudo ground truth for the R predictor.

    R_GT = make_rgt(mask, pointer, cfg)        [S/stride, S/stride] in [0, 1]

Three candidate definitions (RgtCfg.mode), compared side by side in tests/common/test_make_rgt.py:

  "geodesic"     A. object-shape field: 5-1 inside profile along the mask (geodesic from the
                    pointer) + 5-2 outside decay carrying the boundary value. R follows the object's
                    topology (far arm of a U is low). Needs iterative propagation.
  "radial"       B. pointer-centred computational prior, closed form:
                    R(x) = exp[-(||x - p|| / sigma)^gamma],  sigma = sigma_frac * max_{x in M} ||x - p||
                    (object- and pointer-relative scale). Same distance -> same R, target or background.
  "radial_bias"  C. B times a soft target-mask factor  [eta + (1 - eta) * S(x)]:  S = 1 on the whole mask,
                    ramps linearly to 0 over a band of band_px OUTSIDE the mask, 0 beyond. On the object C
                    equals B (so R(p) = 1); at equal distance the background keeps eta of the radial value,
                    reached smoothly over the band instead of at the boundary.

  Far-point value for B/C: R(d_max) = exp(-(1 / sigma_frac)^gamma); sigma_frac >= 1 / (-ln r_min)^(1/gamma)
  keeps the whole target above r_min (gamma 2, r_min 0.5 -> 1.20).

    d_g(p, x)   geodesic distance from the pointer to x INSIDE the mask (8-neighbour chamfer
                propagation that may only pass through mask cells; <= 9 % from true Euclidean geodesic)
    R_in(x)     = exp[ -( d_g(p, x) / lambda_in )^gamma ],   lambda_in = lambda_in_frac * max_x d_g(p, x)

    R_in(p) = 1, smooth decrease along the object, following its shape (a U-shaped object has a low
    value on the far arm although it is close in Euclidean terms). The boundary value is NOT a
    constant: it depends on how far each boundary point is from the pointer, so moving the pointer
    changes the whole field:  R_GT(M, p1) != R_GT(M, p2).

Resolution: geometry is computed on the mask downsampled by `stride` (2 -> 320 x 320 for a 640
canvas); the pointer is mapped with the half-pixel convention p' = (p + 0.5) / stride - 0.5 and
snapped to the nearest mask cell if downsampling left its cell outside. The R predictor is
supervised at stride 4: to_supervision(r, 4 // stride) area-averages the continuous field.

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

from .geometry import SHIFTS as _SHIFTS
from .geometry import distance_to
from .geometry import shift as _shift


@dataclass(frozen=True)
class RgtCfg:
    mode: str = "geodesic"      # "geodesic" (A) | "radial" (B) | "radial_bias" (C)
    stride: int = 2             # geometry resolution (canvas / stride)
    gamma: float = 2.0          # exponent: 2 = Gaussian-like plateau around the pointer, 1 = exponential
    # A
    lambda_in_frac: float = 1.0 # lambda_in = frac * max geodesic distance from the pointer (object-relative)
    lambda_out_px: float = 32.0 # outside decay length in INPUT pixels (absolute; 0.05 * 640)
    # B / C
    sigma_frac: float = 1.0     # sigma = frac * max Euclidean distance from the pointer within the mask
    eta: float = 0.3            # C: background keeps eta of the radial value at equal distance
    band_px: float = 24.0       # C: width (input px) of the ramp 1 -> 0 OUTSIDE the mask
    mask_thr: float = 0.5       # downsampled soft mask -> bool


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


def to_supervision(r: torch.Tensor, factor: int) -> torch.Tensor:
    """R_GT at the geometry stride -> the R predictor's stride (area average over factor x factor
    cells, which for integer factors equals antialiased bilinear downsampling). factor 1 = identity."""
    return r if factor == 1 else F.avg_pool2d(r[None, None], factor)[0, 0]


def radial_profile(mask_s: torch.Tensor, p_s: tuple[float, float], cfg: RgtCfg = RgtCfg()) -> torch.Tensor:
    """B: exp[-(||x - p|| / sigma)^gamma] on the whole canvas, sigma = sigma_frac * max distance from the
    pointer to a mask cell. p_s in cell coordinates (half-pixel convention)."""
    h, w = mask_s.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=mask_s.device, dtype=torch.float32),
                            torch.arange(w, device=mask_s.device, dtype=torch.float32), indexing="ij")
    dist = torch.sqrt((xs - p_s[0]) ** 2 + (ys - p_s[1]) ** 2)
    d_max = dist[mask_s].max() if mask_s.any() else torch.tensor(1.0)
    sigma = max(cfg.sigma_frac * float(d_max), 1e-6)
    return torch.exp(-((dist / sigma) ** cfg.gamma))


def soft_mask(mask_s: torch.Tensor, band_cells: float) -> torch.Tensor:
    """S: exactly 1 on the mask, linear ramp 1 -> 0 over `band_cells` outside it, 0 beyond."""
    if band_cells <= 0:
        return mask_s.float()
    d = distance_to(mask_s, max_iter=int(math.ceil(band_cells)) + 1)   # cells to the mask (0 on it)
    d = torch.where(torch.isfinite(d), d, torch.full_like(d, band_cells + 1))
    return (1.0 - (d - 0.5).clamp(min=0) / band_cells).clamp(0, 1)


def make_rgt(mask: torch.Tensor, pointer, cfg: RgtCfg = RgtCfg()) -> torch.Tensor:
    """Full-res mask [S, S] bool + pointer (x, y) in full-res pixels -> R_GT [S/stride, S/stride] in [0, 1]
    according to cfg.mode (see module docstring)."""
    mask_s = downsample_mask(mask, cfg.stride, cfg.mask_thr)
    p_s = pointer_to_stride(pointer, cfg.stride)
    if cfg.mode == "geodesic":
        seed = seed_cell(mask_s, p_s)
        r_in = inside_profile(mask_s, geodesic_from_pointer(mask_s, seed), cfg)
        return outside_profile(mask_s, r_in, cfg)
    if cfg.mode == "radial":
        return radial_profile(mask_s, p_s, cfg)
    if cfg.mode == "radial_bias":
        s_soft = soft_mask(mask_s, cfg.band_px / cfg.stride)
        return radial_profile(mask_s, p_s, cfg) * (cfg.eta + (1.0 - cfg.eta) * s_soft)
    raise ValueError(f"mode={cfg.mode!r} (geodesic | radial | radial_bias)")


def field_stats(r: torch.Tensor, mask_s: torch.Tensor, p_s: tuple[float, float], valid_s: torch.Tensor | None = None,
                band_cells: int = 16) -> dict:
    """The four numbers used to compare definitions: R at the pointer, min / mean on the target mask,
    mean in the outside band (band_cells wide), fraction of the valid canvas with R > 0.5."""
    h, w = mask_s.shape
    cy, cx = min(max(int(round(p_s[1])), 0), h - 1), min(max(int(round(p_s[0])), 0), w - 1)
    valid_s = torch.ones_like(mask_s) if valid_s is None else valid_s
    k = 2 * band_cells + 1
    dil = F.max_pool2d(mask_s[None, None].float(), k, 1, band_cells)[0, 0] > 0
    band = dil & ~mask_s & valid_s
    return dict(r_pointer=float(r[cy, cx]),
                r_mask_min=float(r[mask_s].min()) if mask_s.any() else float("nan"),
                r_mask_mean=float(r[mask_s].mean()) if mask_s.any() else float("nan"),
                r_band_mean=float(r[band].mean()) if band.any() else float("nan"),
                area_gt_half=float(((r > 0.5) & valid_s).sum() / valid_s.sum()))
