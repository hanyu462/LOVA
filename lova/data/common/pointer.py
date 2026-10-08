"""Step 4: pick ONE pointer target among the candidates and ONE point inside it.

    owned  = pointer_region(idx, masks)                    M_i minus pixels owned by a smaller instance
    safe   = safe_region(owned, cfg)                       {x : D(x) >= alpha * max D},  D = 0 on the boundary ring
    region = sampling_region(owned, cfg)                   safe if non-empty else owned   (computed ONCE)
    p      = sample_from(region, generator)                p ~ Uniform(region)            (cheap, repeatable)
    idx, p = make_pointer(transformed, cands, cfg, generator)
             candidates in random order; the first one with a non-empty owned region is the target
             (pick_target alone is the plain "one uniform candidate" helper)

Pointer ownership rule (V0 training / evaluation convention, deterministic):
    owner(x) = argmin_{i : x in M_i} |M_i|      the SMALLEST instance containing the pixel
A click on a cat lying on a bed means the cat, so bed pointers are never placed on the cat.
Ownership is resolved against ALL instances (not only candidates): a remote too small to be a
target still owns its pixels, so a click there is never read as "bed". This says nothing about
real depth order; it only fixes what an ambiguous click means. owner_of() applies the same rule
with GT masks for evaluation / visualisation; the deployed model has no masks and must have
learned the rule from (I, p) -> R.

Policy (deliberately simple): uniform over pointer-feasible eligible instances (those with a
non-empty owned region), uniform over the safe interior.
No centroid, no bbox centre: the user may click anywhere inside an object, so training pointers
should cover the interior, just not the boundary band where a 1-2 px annotation error would put
the pointer outside the mask and where the later R field would be very one-sided.

Depth D is computed on the mask downsampled by `depth_stride` (8-neighbour chamfer propagation,
<= 9 % from exact Euclidean) and the safe region is brought back to full resolution. The excluded
band is >= alpha * max depth, i.e. several pixels, so stride-4 depth is precise enough and the
cost stays in the millisecond range for a data-loader worker.
Returns pointer coordinates (x, y) in canvas pixels (ints).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ...utils.geometry import depth_l1, erode, owner_map  # grid primitives live in lova/utils/geometry.py; this file is policy only
from .transform import Transformed


@dataclass(frozen=True)
class PointerCfg:
    alpha: float = 0.1        # safe interior: depth >= alpha * max depth of the object
    depth_stride: int = 4     # resolution at which the depth map is computed
    erode_px: int = 2         # require Chebyshev distance >= erode_px + 1 from any non-mask pixel (full res)


def safe_region(mask: torch.Tensor, cfg: PointerCfg = PointerCfg()) -> torch.Tensor:
    """mask [S, S] bool -> [S, S] bool: interior cells with depth >= alpha * max depth.
    With alpha > 0 the outermost coarse ring (depth 0) is always excluded, and erode_px removes a
    full-res margin, so every safe pixel is >= erode_px + 1 px from the true boundary. alpha = 0 and
    erode_px = 0 keep the whole mask. Empty for thin objects; the caller then falls back to the mask."""
    S = mask.shape[-1]
    st = cfg.depth_stride
    if st > 1:
        small = F.avg_pool2d(mask[None, None].float(), st)[0, 0] >= 0.5
    else:
        small = mask
    if not small.any():
        return torch.zeros_like(mask)
    d = depth_l1(small)                                   # Manhattan depth: exact, separable, no per-ring loop
    dmax = d.max()
    if not torch.isfinite(dmax):
        # no outside cell at all in the coarse mask: the canvas border is not a boundary by
        # convention (the object may continue), so there is no ring to exclude
        safe_small = small
    else:
        safe_small = small & (d >= cfg.alpha * dmax) & ((d >= 1) if (cfg.alpha > 0 and dmax >= 1) else small)
    if st > 1:
        safe = F.interpolate(safe_small[None, None].float(), size=(S, S), mode="nearest")[0, 0] > 0.5
    else:
        safe = safe_small
    safe = safe & mask
    if cfg.erode_px > 0:  # coarse cells can touch the jagged true boundary: keep a full-res margin
        safe = safe & erode(mask, cfg.erode_px)
    return safe


def pointer_region(idx: int, masks: torch.Tensor, owners: torch.Tensor | None = None) -> torch.Tensor:
    """masks [N, S, S] bool -> [S, S] bool: pixels of instance idx that it OWNS (not covered by any
    smaller instance; ties broken by lower index). Pass `owners = owner_map(masks)` when calling
    for several instances of the same sample (computed once instead of N times)."""
    if owners is None:
        owners = owner_map(masks)
    return owners == idx


def owner_of(xy, masks: torch.Tensor) -> int | None:
    """GT-side counterpart for evaluation / visualisation: the smallest instance containing pixel
    (x, y), or None (outside every mask or outside the canvas)."""
    x, y = int(xy[0]), int(xy[1])
    h, w = masks.shape[-2:]
    if not (0 <= x < w and 0 <= y < h):
        return None
    inside = torch.nonzero(masks[:, y, x])[:, 0]
    if len(inside) == 0:
        return None
    areas = masks[inside].flatten(1).sum(1)
    return int(inside[areas.argmin()])


def pick_target(cands: list[int], generator: torch.Generator | None = None) -> int:
    """One candidate index, uniformly at random."""
    if not cands:
        raise ValueError("no pointer candidates")
    return cands[int(torch.randint(len(cands), (1,), generator=generator).item())]


def sampling_region(region: torch.Tensor, cfg: PointerCfg = PointerCfg()) -> torch.Tensor:
    """Where pointers may land: the safe interior of `region`, or `region` itself if that is empty
    (thin objects). Compute once, then sample_from() as often as needed."""
    safe = safe_region(region, cfg)
    return safe if safe.any() else region


def sample_from(region: torch.Tensor, generator: torch.Generator | None = None) -> tuple[int, int] | None:
    """One pixel (x, y) uniform over the TRUE cells of region; None if empty."""
    idx = torch.nonzero(region, as_tuple=False)  # [K, 2] (y, x)
    if len(idx) == 0:
        return None
    y, x = idx[int(torch.randint(len(idx), (1,), generator=generator).item())].tolist()
    return int(x), int(y)


def sample_pointer(region: torch.Tensor, cfg: PointerCfg = PointerCfg(),
                   generator: torch.Generator | None = None) -> tuple[int, int] | None:
    """sampling_region + sample_from in one call (one pointer per region)."""
    return sample_from(sampling_region(region, cfg), generator)


def make_pointer(t: Transformed, cands: list[int], cfg: PointerCfg = PointerCfg(),
                 generator: torch.Generator | None = None, owners: torch.Tensor | None = None) -> tuple[int, tuple[int, int]] | None:
    """(pointed instance index, (x, y)) for one transformed sample. Candidates are tried in random
    order; one whose owned region is empty (fully covered by smaller instances) is skipped.
    None if no candidate has an owned pixel. `owners` = owner_map(t.masks) if already computed."""
    if owners is None:
        owners = owner_map(t.masks)
    order = [cands[i] for i in torch.randperm(len(cands), generator=generator).tolist()]
    for idx in order:
        p = sample_pointer(pointer_region(idx, t.masks, owners), cfg, generator)
        if p is not None:
            return idx, p
    return None
