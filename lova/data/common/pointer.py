"""Step 4: pick ONE pointer target among the candidates and ONE point inside it.

    idx   = pick_target(cands, generator)                 i ~ Uniform(candidates)
    safe  = safe_region(mask, cfg)                        {x in M : D(x) >= alpha * max D},  D = depth to boundary
    p     = sample_pointer(mask, cfg, generator)          p ~ Uniform(safe)   (falls back to M if safe is empty)
    idx, p = make_pointer(transformed, cands, cfg, generator)

Policy (deliberately simple): uniform over eligible instances, uniform over the safe interior.
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

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .transform import Transformed


@dataclass(frozen=True)
class PointerCfg:
    alpha: float = 0.1        # safe interior: depth >= alpha * max depth of the object
    depth_stride: int = 4     # resolution at which the depth map is computed


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


def distance_to(target: torch.Tensor, max_iter: int = 4096) -> torch.Tensor:
    """target [h, w] bool -> [h, w] float: chamfer distance (cells) from every cell to the nearest
    TRUE cell (0 on TRUE cells, +inf if target is empty). Converges in <= max distance iterations."""
    inf = float("inf")
    d = torch.where(target, torch.zeros(target.shape), torch.full(target.shape, inf))
    for _ in range(max_iter):
        best = d
        for dy, dx, wgt in _SHIFTS:
            best = torch.minimum(best, _shift(d, dy, dx, inf) + wgt)
        if torch.equal(best, d):
            return d
        d = best
    return d


def depth(mask: torch.Tensor) -> torch.Tensor:
    """mask [h, w] bool -> [h, w] float: distance from each mask cell to the nearest NON-mask cell
    (0 outside the mask). The image border counts as inside, as the object may continue beyond it."""
    d = distance_to(~mask)
    return torch.where(mask, d, torch.zeros_like(d))


def safe_region(mask: torch.Tensor, cfg: PointerCfg = PointerCfg()) -> torch.Tensor:
    """mask [S, S] bool -> [S, S] bool: interior cells with depth >= alpha * max depth.
    Empty if the mask is empty; may be empty for very thin masks (caller falls back to the mask)."""
    S = mask.shape[-1]
    st = cfg.depth_stride
    if st > 1:
        small = F.avg_pool2d(mask[None, None].float(), st)[0, 0] >= 0.5
    else:
        small = mask
    if not small.any():
        return torch.zeros_like(mask)
    d = depth(small)
    safe_small = small & (d >= cfg.alpha * d.max())
    if st > 1:
        safe = F.interpolate(safe_small[None, None].float(), size=(S, S), mode="nearest")[0, 0] > 0.5
    else:
        safe = safe_small
    return safe & mask


def pick_target(cands: list[int], generator: torch.Generator | None = None) -> int:
    """One candidate index, uniformly at random."""
    if not cands:
        raise ValueError("no pointer candidates")
    return cands[int(torch.randint(len(cands), (1,), generator=generator).item())]


def sample_pointer(mask: torch.Tensor, cfg: PointerCfg = PointerCfg(),
                   generator: torch.Generator | None = None) -> tuple[int, int]:
    """One pixel (x, y), uniform over the safe interior (fallback: uniform over the mask)."""
    region = safe_region(mask, cfg)
    if not region.any():
        region = mask
    idx = torch.nonzero(region, as_tuple=False)  # [K, 2] (y, x)
    if len(idx) == 0:
        raise ValueError("empty mask")
    y, x = idx[int(torch.randint(len(idx), (1,), generator=generator).item())].tolist()
    return int(x), int(y)


def make_pointer(t: Transformed, cands: list[int], cfg: PointerCfg = PointerCfg(),
                 generator: torch.Generator | None = None) -> tuple[int, tuple[int, int]]:
    """(pointed instance index, (x, y)) for one transformed sample."""
    idx = pick_target(cands, generator)
    return idx, sample_pointer(t.masks[idx], cfg, generator)
