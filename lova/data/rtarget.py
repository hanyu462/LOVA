"""Pseudo ground truth R* for the R predictor, built on the fly from the pointed instance mask
(after augmentation; nothing is stored). Two definitions:

mode "geo" (default)  pointer-centred, boundary value varies along the boundary
    R_in(x)  = exp[-(d_g(p, x) / sigma_in)^gamma]            x in M
    R_out(x) = R_in(b(x)) * exp(-d_out(x) / sigma_out)        x not in M
    d_g      : geodesic distance from the pointer INSIDE the mask (8-neighbour chamfer
               propagation, <= ~8% from true Euclidean geodesic)
    sigma_in : fraction of the max geodesic distance in the object (so R_in on the farthest
               boundary point = exp(-1) for sigma_in = 1), gamma = 2 -> Gaussian-like plateau
               around the pointer
    b(x)     : nearest boundary cell; its R_in value is carried outward by the same propagation
    sigma_out: pixels (context halo)

mode "dt"             object-centred, constant boundary value (first version, kept for comparison)
    R_in(x)  = r_b + (1 - r_b) * (D_in(x) / max D_in)^gamma   (depth inside the mask)
    R_out(x) = r_b * exp(-d_out(x) / sigma_out)

Resolution: compute on the mask you pass (full input resolution recommended so thin structures
keep their geometry), then `downsample(r, 4)` to the R predictor's stride. All torch, CPU or GPU.
Under binary execution G = 1[R > 0.5] the active area is {R* > 0.5}; see `viz_rtarget.py`.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_SHIFTS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
           (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]


def _shift(x: torch.Tensor, dy: int, dx: int, fill: float) -> torch.Tensor:
    """x [B, h, w] shifted so that out[y, x] = x[y - dy, x - dx] (fill outside)."""
    b, h, w = x.shape
    out = torch.full_like(x, fill)
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[:, yd, xd] = x[:, ys, xs]
    return out


def propagate(dist: torch.Tensor, value: torch.Tensor, allowed: torch.Tensor,
              max_iter: int = 4096, check_every: int = 16):
    """Chamfer (8-neighbour) distance propagation with value carry-over.
    dist [B,h,w] (0 at seeds, inf elsewhere), value [B,h,w] (value at seeds), allowed [B,h,w] bool:
    cells that may be reached. Returns (dist, value) where value[x] = value at the seed nearest to x."""
    inf = float("inf")
    for it in range(max_iter):
        best_d, best_v = dist, value
        for dy, dx, wgt in _SHIFTS:
            cand = _shift(dist, dy, dx, inf) + wgt
            better = (cand < best_d) & allowed
            best_v = torch.where(better, _shift(value, dy, dx, 0.0), best_v)
            best_d = torch.where(better, cand, best_d)
        if (it + 1) % check_every == 0 and torch.equal(best_d, dist):
            return best_d, best_v
        dist, value = best_d, best_v
    return dist, value


def distance_transform(mask: torch.Tensor, max_dist: float | None = None) -> torch.Tensor:
    """mask [B,h,w] bool -> chamfer distance (cells) to the nearest TRUE cell. max_dist caps the
    propagation (cells farther than that keep +inf), which bounds the cost for large maps."""
    b, h, w = mask.shape
    dist = torch.where(mask, torch.zeros(b, h, w, device=mask.device), torch.full((b, h, w), float("inf"), device=mask.device))
    val = torch.zeros_like(dist)
    allowed = torch.ones_like(mask)
    iters = 4096 if max_dist is None else int(math.ceil(max_dist)) + 1
    d, _ = propagate(dist, val, allowed, max_iter=iters)
    return d


def _pointer_cells(pointers: torch.Tensor, stride: int, h: int, w: int):
    px = (pointers[:, 0] / stride).long().clamp(0, w - 1)
    py = (pointers[:, 1] / stride).long().clamp(0, h - 1)
    return px, py


def r_profile(pointed_mask: torch.Tensor, pointers: torch.Tensor, mode: str = "geo",
              sigma_in: float = 1.0, gamma: float = 2.0, sigma_out: float = 25.6,
              r_b: float = 0.7, mask_stride: int = 1) -> torch.Tensor:
    """pointed_mask [B,1,H,W] (soft or binary) at input resolution / `mask_stride`,
    pointers [B,2] (x, y) in INPUT pixels, sigma_out in INPUT pixels -> R* [B,1,H,W] in [0,1]."""
    m = pointed_mask[:, 0] > 0.5
    empty = ~m.flatten(1).any(1)
    if empty.any():
        mx = pointed_mask[:, 0].flatten(1).max(1).values.view(-1, 1, 1)
        m = torch.where(empty.view(-1, 1, 1), pointed_mask[:, 0] >= mx, m)
    b, h, w = m.shape
    dev = m.device
    inf = float("inf")
    s_out = max(sigma_out / mask_stride, 1e-6)  # cells
    cap_out = 6.0 * s_out  # beyond ~6 sigma R_out < 0.0025 -> treat as 0

    if mode == "geo":
        # seed = pointer cell, snapped into the mask if it fell just outside
        px, py = _pointer_cells(pointers, mask_stride, h, w)
        seed = torch.zeros_like(m)
        seed[torch.arange(b, device=dev), py, px] = True
        outside = seed & ~m
        if outside.any():  # snap: move the seed to the mask cell closest to the pointer
            ys, xs = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32),
                                    torch.arange(w, device=dev, dtype=torch.float32), indexing="ij")
            for i in outside.nonzero()[:, 0].tolist():
                dd = (ys - py[i].float()) ** 2 + (xs - px[i].float()) ** 2
                dd = torch.where(m[i], dd, torch.full_like(dd, inf))
                j = int(dd.flatten().argmin().item())
                seed[i] = False
                seed[i, j // w, j % w] = True
        dist = torch.where(seed, torch.zeros(b, h, w, device=dev), torch.full((b, h, w), inf, device=dev))
        d_g, _ = propagate(dist, torch.zeros_like(dist), m)  # geodesic inside the mask
        d_g = torch.where(m, d_g, torch.zeros_like(d_g))
        d_max = d_g.flatten(1).max(1).values.clamp(min=1e-6).view(b, 1, 1)
        r_in = torch.exp(-((d_g / (sigma_in * d_max)).clamp(min=0)) ** gamma)
        # outward: seeds = mask cells, carried value = R_in on the (boundary) seed
        dist0 = torch.where(m, torch.zeros(b, h, w, device=dev), torch.full((b, h, w), inf, device=dev))
        d_out, r_bnd = propagate(dist0, torch.where(m, r_in, torch.zeros_like(r_in)), torch.ones_like(m),
                                 max_iter=int(math.ceil(cap_out)) + 1)
        d_out = (d_out - 0.5).clamp(min=0)  # cell centre -> boundary
        r_out = torch.where(torch.isfinite(d_out), r_bnd * torch.exp(-d_out / s_out), torch.zeros_like(d_out))
        return torch.where(m, r_in, r_out)[:, None].float()

    if mode == "dt":
        d_in = (distance_transform(~m) - 0.5).clamp(min=0)
        d_in = torch.where(m, d_in, torch.zeros_like(d_in))
        s = d_in / d_in.flatten(1).max(1).values.clamp(min=1e-6).view(b, 1, 1)
        r_in = r_b + (1.0 - r_b) * s.pow(gamma)
        d_out = (distance_transform(m, max_dist=cap_out) - 0.5).clamp(min=0)
        r_out = torch.where(torch.isfinite(d_out), r_b * torch.exp(-d_out / s_out), torch.zeros_like(d_out))
        return torch.where(m, r_in, r_out)[:, None].float()
    raise ValueError(mode)


def downsample(r: torch.Tensor, stride: int) -> torch.Tensor:
    """R* [B,1,H,W] -> [B,1,H/stride,W/stride] (area average = bilinear-with-antialias for integer strides)."""
    return r if stride == 1 else F.avg_pool2d(r, stride)


def sample_interior_pointer(mask4: torch.Tensor, frac: float = 0.25,
                            generator: torch.Generator | None = None) -> torch.Tensor:
    """mask4 [h, w] (stride 4) -> random cell with depth >= frac * max depth (never right at the
    boundary), in input-pixel coordinates like common.sample_pointer."""
    m = mask4 > 0.5
    if not m.any():
        m = mask4 >= mask4.max()
    d_in = torch.where(m, distance_transform(~m[None])[0], torch.zeros_like(mask4))
    cand = (d_in >= frac * d_in.max()).nonzero()
    if len(cand) == 0:
        cand = m.nonzero()
    j = int(torch.randint(len(cand), (1,), generator=generator).item())
    y, x = cand[j].tolist()
    off = torch.rand(2, generator=generator)
    return torch.tensor([(x + off[0].item()) * 4, (y + off[1].item()) * 4])


def sigma_out_px(img_size: int, frac: float = 0.05) -> float:
    """sigma_out = frac * min(H, W) input pixels (512 px, 0.05 -> 25.6 px)."""
    return frac * img_size


def halo_px(r_boundary: float, sigma_out: float, tau: float = 0.5) -> float:
    """Distance outside the mask (px) where R_out crosses tau, for a boundary value r_boundary."""
    return sigma_out * math.log(r_boundary / tau) if r_boundary > tau else 0.0
