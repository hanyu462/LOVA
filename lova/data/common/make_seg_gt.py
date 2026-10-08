"""Step 6: Segmentation GT for the instance head, from ALL instances of a Transformed sample.

Independent of select / pointer / R_GT: every non-crowd instance is a segmentation target.

    gt = make_seg_gt(transformed, num_classes, cfg)

The head (SOLOv2-style dynamic kernels with CenterNet centre assignment) predicts, per stride-8 cell,
a class heatmap and a kernel vector; a kernel at cell q times the stride-4 mask feature gives the
mask of "the instance centred at q". The GT therefore has three parts and an ignore mask:

    heat        [K, S/8, S/8]  class-specific gaussian around each instance centre (peak exactly 1)
                               -> focal loss on the class heatmap
    heat_valid  [S/8, S/8]     cells that take part in the heatmap loss. False on padding (majority of
                               the cell) and on any cell touching a crowd region (there objects exist
                               without annotations, so "no centre" is NOT a valid negative; crowd
                               shapes are irregular, hence the stricter rule). Positive cells stay True
    pos_index   [P]            flat stride-8 cells that own an instance: the centre cell plus its 3x3
    pos_inst    [P]            neighbours inside the instance; the smaller instance wins overlaps
                               -> which kernels are trained, and against which instance
    masks_s4    [N, S/4, S/4]  soft (area-averaged) instance masks at the mask-feature stride
    mask_valid  [S/4, S/4]     dice ignore: False on crowd and padding
    centers     [N, 2]         (x, y) full-res pixels, for visualisation / analysis
    center_pixel_inside [N]    whether the centre PIXEL lies on the instance mask (concave shapes may
                               not for "centroid"); the stride-8 centre cell is always a positive anyway

Centre definition (cfg.center), default "deepest_owned":
    "deepest_owned"  deepest cell of the pixels the instance OWNS (minus every smaller instance that
                     covers it, the pointer ownership rule): always on the mask, never on another
                     annotated object. Unannotated things (COCO has no "plate") stay part of the mask
    "deepest"        deepest cell of the whole mask: always on the mask, may sit on a covering object
    "centroid"       mask centroid (V0): outside the mask for ~6 % of COCO instances (concave shapes)
Gaussian sigma = max(sigma_min, sqrt(area) / 8 / sigma_div) cells.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ...utils.geometry import depth
from .pointer import pointer_region
from .transform import Transformed


@dataclass(frozen=True)
class SegGtCfg:
    stride_heat: int = 8        # heatmap / kernel grid
    stride_mask: int = 4        # mask feature grid (dice targets)
    center: str = "deepest_owned"  # "deepest_owned" (default, 2026-10-08) | "deepest" | "centroid" (V0; ablations)
    sigma_min: float = 0.8      # gaussian sigma lower bound (cells)
    sigma_div: float = 6.0      # sigma = object size (cells) / sigma_div
    pos_occupancy: float = 0.25 # a 3x3 neighbour cell counts as inside if >= this fraction of it is mask
    crowd_ignore: bool = True   # crowd / padding cells leave the heatmap and dice losses
    max_pos: int = 256          # cap on positive cells per sample: one cell per instance is always kept,
                                # the rest is a random subset (P > 256 in 0.4 % of val2017 samples)


@dataclass
class SegGT:
    heat: torch.Tensor          # [K, h8, w8] float
    heat_valid: torch.Tensor    # [h8, w8] bool
    pos_index: torch.Tensor     # [P] long
    pos_inst: torch.Tensor      # [P] long
    masks_s4: torch.Tensor      # [N, h4, w4] float
    mask_valid: torch.Tensor    # [h4, w4] bool
    centers: torch.Tensor       # [N, 2] float (x, y) full-res
    center_pixel_inside: torch.Tensor  # [N] bool


def instance_centers(masks: torch.Tensor, how: str, stride: int = 4) -> torch.Tensor:
    """[N, S, S] bool -> [N, 2] (x, y) full-res pixels. "deepest*" work on the mask at `stride`."""
    n, h, w = masks.shape
    out = torch.zeros(n, 2)
    if n == 0:
        return out
    if how == "centroid":
        ys = torch.arange(h, dtype=torch.float32).view(1, -1, 1)
        xs = torch.arange(w, dtype=torch.float32).view(1, 1, -1)
        m = masks.float()
        area = m.sum((1, 2)).clamp(min=1e-6)
        out[:, 0] = (m * xs).sum((1, 2)) / area
        out[:, 1] = (m * ys).sum((1, 2)) / area
        return out
    if how in ("deepest", "deepest_owned"):
        st, half = stride, (stride - 1) / 2
        for i in range(n):
            region = pointer_region(i, masks) if how == "deepest_owned" else masks[i]
            if not region.any():                                                # fully covered by smaller instances
                region = masks[i]
            occ = F.avg_pool2d(region[None, None].float(), st)[0, 0]            # occupancy at `stride`
            small = occ >= 0.5 if bool((occ >= 0.5).any()) else occ > 0
            # for a CENTRE the canvas border counts as a boundary (unlike pointer depth): pad with False
            d = depth(F.pad(small, (1, 1, 1, 1), value=False))[1:-1, 1:-1]
            score = d + occ                                                     # deepest cell, ties -> most occupied
            j = int(score.flatten().argmax())
            cy, cx = j // small.shape[1], j % small.shape[1]
            block = region[cy * st:cy * st + st, cx * st:cx * st + st]           # a region pixel inside that cell,
            by, bx = torch.nonzero(block, as_tuple=True)                        # closest to the cell centre
            k = int(((by.float() - half) ** 2 + (bx.float() - half) ** 2).argmin())
            out[i, 0], out[i, 1] = cx * st + int(bx[k]), cy * st + int(by[k])
        return out
    raise ValueError(f"center={how!r} (centroid | deepest | deepest_owned)")


def make_seg_gt(t: Transformed, num_classes: int, cfg: SegGtCfg = SegGtCfg(),
                generator: torch.Generator | None = None) -> SegGT:
    masks, labels = t.masks, t.labels
    n, S, _ = masks.shape
    s8, s4 = cfg.stride_heat, cfg.stride_mask
    h8, w8 = S // s8, S // s8
    heat = torch.zeros(num_classes, h8, w8)
    owner = torch.full((h8, w8), -1, dtype=torch.long)

    # ignore regions at both grids: padding and (optionally) crowd
    valid8 = F.avg_pool2d(t.valid[None, None].float(), s8)[0, 0] > 0.5
    valid4 = F.avg_pool2d(t.valid[None, None].float(), s4)[0, 0] > 0.5
    heat_valid, mask_valid = valid8.clone(), valid4.clone()
    if cfg.crowd_ignore and t.crowd.any():  # any crowd pixel in the cell -> ignore (crowd shapes are irregular)
        heat_valid &= ~(F.avg_pool2d(t.crowd[None, None].float(), s8)[0, 0] > 0)
        mask_valid &= ~(F.avg_pool2d(t.crowd[None, None].float(), s4)[0, 0] > 0)

    masks_s4 = F.avg_pool2d(masks[:, None].float(), s4)[:, 0] if n else torch.zeros(0, S // s4, S // s4)
    centers = instance_centers(masks, cfg.center, s4)
    center_pixel_inside = torch.zeros(n, dtype=torch.bool)
    if n == 0:
        return SegGT(heat, heat_valid, torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long),
                     masks_s4, mask_valid, centers, center_pixel_inside)

    area = masks.flatten(1).sum(1).float()
    occ8 = F.avg_pool2d(masks[:, None].float(), s8)[:, 0]                      # [N, h8, w8] occupancy
    ys = torch.arange(h8, dtype=torch.float32).view(-1, 1)
    xs = torch.arange(w8, dtype=torch.float32).view(1, -1)
    for i in area.argsort(descending=True).tolist():                           # smaller instances assigned last -> win
        cx = int(min(max(round((float(centers[i, 0]) + 0.5) / s8 - 0.5), 0), w8 - 1))
        cy = int(min(max(round((float(centers[i, 1]) + 0.5) / s8 - 0.5), 0), h8 - 1))
        center_pixel_inside[i] = bool(masks[i, min(int(centers[i, 1]), S - 1), min(int(centers[i, 0]), S - 1)])
        size8 = math.sqrt(float(area[i])) / s8
        sigma = max(cfg.sigma_min, size8 / cfg.sigma_div)
        g = torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
        c = int(labels[i])
        heat[c] = torch.maximum(heat[c], g)
        y0, y1, x0, x1 = max(cy - 1, 0), min(cy + 2, h8), max(cx - 1, 0), min(cx + 2, w8)
        win = occ8[i, y0:y1, x0:x1] >= cfg.pos_occupancy
        win[cy - y0, cx - x0] = True                                           # centre cell always owned
        owner[y0:y1, x0:x1][win] = i

    pos_index = torch.nonzero(owner.flatten() >= 0, as_tuple=False)[:, 0]
    pos_inst = owner.flatten()[pos_index]
    heat_valid.view(-1)[pos_index] = True                                      # positives are never ignored
    if pos_index.numel() > cfg.max_pos:
        # keep one cell per instance (its centre cell if it owns it, else any), then a random subset of the rest
        first = torch.zeros(pos_index.numel(), dtype=torch.bool)
        seen = set()
        for k, inst in enumerate(pos_inst.tolist()):
            if inst not in seen:
                seen.add(inst)
                first[k] = True
        rest = torch.nonzero(~first)[:, 0]
        budget = max(cfg.max_pos - int(first.sum()), 0)
        extra = rest[torch.randperm(rest.numel(), generator=generator)[:budget]]
        keep = torch.cat([torch.nonzero(first)[:, 0], extra]).sort().values
        pos_index, pos_inst = pos_index[keep], pos_inst[keep]
    return SegGT(heat, heat_valid, pos_index, pos_inst, masks_s4, mask_valid, centers, center_pixel_inside)
