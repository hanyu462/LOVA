"""Step 5: (target mask, pointer) -> R_GT, the pseudo ground truth for the R predictor.

    R_GT = make_r_gt(mask, pointer, cfg)        [S/stride, S/stride] in [0, 1]

Three candidate definitions (RgtCfg.mode), compared side by side in tests/common/test_make_r_gt.py.
Decision (2026-10-08): C "radial_bias" with gamma 2, sigma_frac 1.25, eta 0.3, band_px 96 is the V0
definition: a pointer-centred computational prior (not an object-shape field) with the target mask
as a soft bias. A stays as the comparison baseline, B as the ablation without the mask bias.

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

  Levels of statement (continuous field / geometry grid / supervision grid):
    * continuous field: R(p) = 1 exactly; on the geometry grid the pointer lies between cell centres,
      so the maximum cell is ~1 (0.99+), never snapped (snapping would move the field's centre)
    * geometry grid: every target-mask cell has R >= exp(-(1 / sigma_frac)^gamma) = 0.527 for the
      defaults (sigma_frac >= 1 / (-ln r_min)^(1/gamma) keeps it above r_min)
    * supervision grid (stride 4): area averages; boundary-straddling cells may fall below that
  Outside the band S = 0 but R = eta * R_radial is NOT 0: a low radial tail covers the canvas
  (LOVA's low R means coarse perception, not "unseen"); eta therefore affects compute directly.

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
Grid primitives (downsample, geodesic, chamfer propagation) live in lova/utils/geometry.py; this file only
defines R.

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

from ...utils.geometry import (distance_to, downsample_mask, geodesic, pointer_to_stride, propagate_value,  # noqa: F401
                       seed_cell)


@dataclass(frozen=True)
class RgtCfg:
    mode: str = "radial_bias"   # "radial_bias" (C, chosen 2026-10-08) | "radial" (B) | "geodesic" (A, comparison)
    stride: int = 2             # geometry resolution (canvas / stride)
    gamma: float = 2.0          # exponent: 2 = Gaussian-like plateau around the pointer, 1 = exponential
    # A
    lambda_in_frac: float = 1.0 # lambda_in = frac * max geodesic distance from the pointer (object-relative)
    lambda_out_px: float = 32.0 # outside decay length in INPUT pixels (absolute; 0.05 * 640)
    # B / C
    sigma_frac: float = 1.25    # sigma = frac * max Euclidean distance from the pointer within the mask
                                # (1.25 keeps the farthest target cell at R >= 0.5 for gamma 2)
    eta: float = 0.3            # C: background keeps eta of the radial value at equal distance
    band_px: float = 96.0       # C: width (input px) of the ramp 1 -> 0 OUTSIDE the mask (chosen visually:
                                # soft transition, ~10 % of the canvas lit for a 1 % object)
    mask_thr: float = 0.5       # downsampled soft mask -> bool


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
    """A: R_out on cells outside the mask = R_in value of the nearest mask cell, decayed with
    exp(-d_out / lambda_out); exactly 0 beyond 6 lambda_out. R_in is kept on the mask itself."""
    lam = cfg.lambda_out_px / cfg.stride                      # cells
    cutoff = 6.0 * lam                                        # R_out definition: 0 beyond this distance (exp(-6) < 0.003)
    reach = int(math.ceil(cutoff)) + 1                        # propagation budget only (a step is 1 or sqrt 2 cells)
    d, v = propagate_value(mask_s, torch.where(mask_s, r_in, torch.zeros_like(r_in)), reach)
    d_out = (d - 0.5).clamp(min=0)                            # cell centre -> boundary
    within = torch.isfinite(d_out) & (d_out <= cutoff)
    r_out = torch.where(within, v * torch.exp(-d_out / lam), torch.zeros_like(v))
    return torch.where(mask_s, r_in, r_out)


def make_r_in(mask: torch.Tensor, pointer, cfg: RgtCfg = RgtCfg()) -> torch.Tensor:
    """Full-res mask [S, S] bool + pointer (x, y) in full-res pixels -> R_in [S/stride, S/stride] (0 outside)."""
    mask_s = downsample_mask(mask, cfg.stride, cfg.mask_thr)
    seed = seed_cell(mask_s, pointer_to_stride(pointer, cfg.stride))
    return inside_profile(mask_s, geodesic(mask_s, seed), cfg)


def to_supervision(r: torch.Tensor, factor: int) -> torch.Tensor:
    """Area-average R_GT from the geometry grid to the predictor grid (factor x factor cells per
    output cell; factor 1 = identity). Note: a supervision cell straddling the target boundary
    averages target and background values, so the "target cells >= 0.527" guarantee below holds on
    the geometry grid, not necessarily on every supervision cell."""
    return r if factor == 1 else F.avg_pool2d(r[None, None], factor)[0, 0]


def radial_profile(mask_s: torch.Tensor, p_s: tuple[float, float], cfg: RgtCfg = RgtCfg()) -> torch.Tensor:
    """B: exp[-(||x - p|| / sigma)^gamma] on the whole canvas, sigma = sigma_frac * max distance from the
    pointer to a mask cell. p_s in cell coordinates (half-pixel convention)."""
    h, w = mask_s.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=mask_s.device, dtype=torch.float32),
                            torch.arange(w, device=mask_s.device, dtype=torch.float32), indexing="ij")
    dist = torch.sqrt((xs - p_s[0]) ** 2 + (ys - p_s[1]) ** 2)
    if not mask_s.any():
        raise ValueError("radial_profile needs a non-empty mask to set sigma")
    sigma = max(cfg.sigma_frac * float(dist[mask_s].max()), 1e-6)
    return torch.exp(-((dist / sigma) ** cfg.gamma))


def soft_mask(mask_s: torch.Tensor, band_cells: float, coarse: int = 2) -> torch.Tensor:
    """S: exactly 1 on the mask, linear ramp 1 -> 0 over `band_cells` outside it, 0 beyond.
    The distance is propagated on a `coarse`x coarser grid and interpolated back; the mask itself is
    re-imposed exactly. Measured against coarse=1 on 109 val2017 samples (band 96 px): mean |dR|
    0.0008, max 0.021, R>0.5 area 0.1437 vs 0.1442, values on the target mask identical."""
    if band_cells <= 0:
        return mask_s.float()
    h, w = mask_s.shape
    if coarse > 1 and h % coarse == 0 and w % coarse == 0:
        mc = F.avg_pool2d(mask_s[None, None].float(), coarse)[0, 0] > 0
        d = distance_to(mc, max_iter=int(math.ceil(band_cells / coarse)) + 1) * coarse
        d = torch.where(torch.isfinite(d), d, torch.full_like(d, band_cells + 1))
        ramp = (1.0 - (d - 0.5).clamp(min=0) / band_cells).clamp(0, 1)
        ramp = F.interpolate(ramp[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    else:
        d = distance_to(mask_s, max_iter=int(math.ceil(band_cells)) + 1)
        d = torch.where(torch.isfinite(d), d, torch.full_like(d, band_cells + 1))
        ramp = (1.0 - (d - 0.5).clamp(min=0) / band_cells).clamp(0, 1)
    return torch.where(mask_s, torch.ones_like(ramp), ramp)


def make_r_gt(mask: torch.Tensor, pointer, cfg: RgtCfg = RgtCfg()) -> torch.Tensor:
    """Full-res mask [S, S] bool + pointer (x, y) in full-res pixels -> R_GT [S/stride, S/stride] in [0, 1]
    according to cfg.mode (see module docstring)."""
    mask_s = downsample_mask(mask, cfg.stride, cfg.mask_thr)
    if not mask_s.any():
        # Never seen on COCO val2017 (checked over all candidates, eval + train transforms). Fail loudly
        # rather than build a field around an arbitrary sigma; decide a fallback only if it ever happens.
        raise ValueError(f"target mask vanished after downsampling by {cfg.stride} ({int(mask.sum())} px at full res)")
    p_s = pointer_to_stride(pointer, cfg.stride)
    if cfg.mode == "geodesic":
        seed = seed_cell(mask_s, p_s)
        r_in = inside_profile(mask_s, geodesic(mask_s, seed), cfg)
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
