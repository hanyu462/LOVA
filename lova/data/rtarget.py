"""Pseudo ground truth for R ("profile" target) built from the pointed instance mask.

    R*(x) = R_in(x) = R_b + (1 - R_b) * s(x)^gamma          x inside the mask
            R_out(x) = R_b * exp(-d_out(x) / lambda)        x outside

    s(x) in [0,1]: how deep inside the object x is
        mode "dt"     : D_in(x) / max D_in     (mask distance transform; exactly R_b on the
                        boundary for any shape, 1 at the deepest point, pointer-invariant)
        mode "euclid" : 1 - ||x - p|| / d_max  (1 at the pointer; R_b only approximately on
                        the boundary of non-convex / elongated objects)
    d_out(x): Euclidean distance (cells) from the cell centre to the mask boundary.

Asymmetric by design: a gentle 1 -> R_b ramp inside (focus), a fast R_b -> 0 decay outside
(context halo), 0 far away. Under binary execution G = 1[R > 0.5] only the halo width matters
(R_out = 0.5 at d = lambda * ln(R_b / 0.5)); the interior ramp is kept for continuous /
multi-level variants and for the R-bin analysis.

Pure torch (CPU or GPU), exact Euclidean distance transform via the boundary cells
(the nearest TRUE cell of a set is always one of its boundary cells).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def distance_transform(mask: torch.Tensor) -> torch.Tensor:
    """mask [B, h, w] bool -> [B, h, w] float: Euclidean distance (cells) to the nearest TRUE cell
    (0 on TRUE cells, +inf for an all-false mask). Batched over B with a padded boundary set."""
    b, h, w = mask.shape
    dev = mask.device
    pad = F.pad(mask, (1, 1, 1, 1), value=False)
    interior = pad[:, :-2, 1:-1] & pad[:, 2:, 1:-1] & pad[:, 1:-1, :-2] & pad[:, 1:-1, 2:]
    bnd = mask & ~interior  # [B, h, w]
    ys, xs = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32),
                            torch.arange(w, device=dev, dtype=torch.float32), indexing="ij")
    grid = torch.stack([ys.reshape(-1), xs.reshape(-1)], 1)  # [N, 2]
    out = torch.full((b, h * w), float("inf"), device=dev)
    counts = bnd.flatten(1).sum(1)
    nmax = int(counts.max().item()) if b else 0
    if nmax == 0:
        return out.view(b, h, w)
    # padded boundary coordinates [B, nmax, 2]; pad slots far away so they never win
    pts = torch.full((b, nmax, 2), 1e6, device=dev)
    for i in range(b):
        idx = bnd[i].flatten().nonzero()[:, 0]
        pts[i, :len(idx)] = grid[idx]
    d = torch.cdist(grid[None].expand(b, -1, -1), pts)  # [B, N, nmax]
    out = d.min(2).values
    out = torch.where(mask.flatten(1), torch.zeros_like(out), out)
    out[counts == 0] = float("inf")
    return out.view(b, h, w)


def r_profile(pointed_mask4: torch.Tensor, pointers: torch.Tensor | None = None, r_b: float = 0.7,
              gamma: float = 0.7, lam: float = 6.4, mode: str = "dt") -> torch.Tensor:
    """pointed_mask4 [B, 1, h, w] (soft or binary) -> R* [B, 1, h, w] in [0, 1].
    pointers [B, 2] (x, y) in input pixels, only used by mode="euclid". lam in cells."""
    m = pointed_mask4[:, 0] > 0.5
    empty = ~m.flatten(1).any(1)
    if empty.any():  # degenerate (tiny) mask: fall back to its max cells
        mx = pointed_mask4[:, 0].flatten(1).max(1).values.view(-1, 1, 1)
        m = torch.where(empty.view(-1, 1, 1), pointed_mask4[:, 0] >= mx, m)
    b, h, w = m.shape
    if mode == "dt":
        d_in = (distance_transform(~m) - 0.5).clamp(min=0)  # depth: cell centre -> boundary
        d_in = torch.where(m, d_in, torch.zeros_like(d_in))
        s = d_in / d_in.flatten(1).max(1).values.clamp(min=1e-6).view(b, 1, 1)
    elif mode == "euclid":
        ys, xs = torch.meshgrid(torch.arange(h, device=m.device, dtype=torch.float32),
                                torch.arange(w, device=m.device, dtype=torch.float32), indexing="ij")
        p = (pointers.float() / 4.0).view(b, 2, 1, 1)
        dist = torch.sqrt((xs + 0.5 - p[:, 0]) ** 2 + (ys + 0.5 - p[:, 1]) ** 2)
        d_max = torch.where(m, dist, torch.zeros_like(dist)).flatten(1).max(1).values.clamp(min=1e-6).view(b, 1, 1)
        s = (1.0 - dist / d_max).clamp(0, 1)
    else:
        raise ValueError(mode)
    r_in = r_b + (1.0 - r_b) * s.pow(gamma)
    d_out = (distance_transform(m) - 0.5).clamp(min=0)
    r_out = r_b * torch.exp(-d_out / max(lam, 1e-6))
    return torch.where(m, r_in, r_out)[:, None].float()


def sample_interior_pointer(mask4: torch.Tensor, frac: float = 0.25,
                            generator: torch.Generator | None = None) -> torch.Tensor:
    """mask4 [h, w] -> random cell with depth D_in >= frac * max D_in (never right at the boundary),
    in input-pixel coordinates like common.sample_pointer."""
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


def lam_cells(img_size: int, frac: float = 0.05) -> float:
    """lambda = frac * min(H, W) input pixels, in stride-4 cells (512 px, 0.05 -> 6.4 cells)."""
    return frac * img_size / 4.0


def halo_cells(r_b: float, lam: float, tau: float = 0.5) -> float:
    """Distance outside the mask (cells) where R_out crosses tau: the binary-execution halo."""
    return lam * math.log(r_b / tau) if r_b > tau else 0.0
