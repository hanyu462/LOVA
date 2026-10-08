"""Step 6: Segmentation GT for the instance head, from ALL instances of a Transformed sample.

Independent of select / pointer / R_GT: every non-crowd instance is a segmentation target.

    gt = make_seg_gt(transformed, num_classes, cfg)

The head (SOLOv2-style dynamic kernels with CenterNet centre assignment) predicts, per stride-8 cell,
a class heatmap and a kernel vector; a kernel at cell q times the stride-4 mask feature gives the
mask of "the instance centred at q". The GT therefore has three parts and an ignore mask:

    heat        [K, S/8, S/8]  class-specific gaussian around each instance centre (peak exactly 1)
                               -> focal loss on the class heatmap
    heat_valid  [S/8, S/8]     cells that take part in the heatmap loss. False on crowd regions and on
                               padding (there objects may exist without annotations, so "no centre"
                               is NOT a valid negative). Positive cells stay True even on crowd
    pos_index   [P]            flat stride-8 cells that own an instance: the centre cell plus its 3x3
    pos_inst    [P]            neighbours inside the instance; the smaller instance wins overlaps
                               -> which kernels are trained, and against which instance
    masks_s4    [N, S/4, S/4]  soft (area-averaged) instance masks at the mask-feature stride
    mask_valid  [S/4, S/4]     dice ignore: False on crowd and padding
    centers     [N, 2]         (x, y) full-res pixels, for visualisation / analysis
    center_inside [N]          whether the centre cell lies on the instance (concave shapes may not)

Centre definition (cfg.center): "centroid" (mask centroid, V0) or "deepest" (cell with the largest
depth, always inside the mask). Gaussian sigma = max(sigma_min, sqrt(area) / 8 / sigma_div) cells.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ...utils.geometry import depth
from .transform import Transformed


@dataclass(frozen=True)
class SegGtCfg:
    stride_heat: int = 8        # heatmap / kernel grid
    stride_mask: int = 4        # mask feature grid (dice targets)
    center: str = "centroid"    # "centroid" | "deepest"
    sigma_min: float = 0.8      # gaussian sigma lower bound (cells)
    sigma_div: float = 6.0      # sigma = object size (cells) / sigma_div
    pos_occupancy: float = 0.25 # a 3x3 neighbour cell counts as inside if >= this fraction of it is mask
    crowd_ignore: bool = True   # crowd / padding cells leave the heatmap and dice losses
    max_pos: int = 256          # cap on positive cells per sample (random subset beyond)


@dataclass
class SegGT:
    heat: torch.Tensor          # [K, h8, w8] float
    heat_valid: torch.Tensor    # [h8, w8] bool
    pos_index: torch.Tensor     # [P] long
    pos_inst: torch.Tensor      # [P] long
    masks_s4: torch.Tensor      # [N, h4, w4] float
    mask_valid: torch.Tensor    # [h4, w4] bool
    centers: torch.Tensor       # [N, 2] float (x, y) full-res
    center_inside: torch.Tensor # [N] bool


def instance_centers(masks: torch.Tensor, how: str) -> torch.Tensor:
    """[N, S, S] bool -> [N, 2] (x, y) full-res pixels."""
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
    if how == "deepest":
        for i in range(n):
            occ = F.avg_pool2d(masks[i][None, None].float(), 4)[0, 0]          # stride-4 occupancy
            small = occ >= 0.5 if bool((occ >= 0.5).any()) else occ > 0
            score = depth(small) + occ                                          # deepest cell, ties -> most occupied
            j = int(score.flatten().argmax())
            cy, cx = j // small.shape[1], j % small.shape[1]
            block = masks[i, cy * 4:cy * 4 + 4, cx * 4:cx * 4 + 4]               # a mask pixel inside that cell,
            by, bx = torch.nonzero(block, as_tuple=True)                        # closest to the cell centre
            k = int(((by.float() - 1.5) ** 2 + (bx.float() - 1.5) ** 2).argmin())
            out[i, 0], out[i, 1] = cx * 4 + int(bx[k]), cy * 4 + int(by[k])
        return out
    raise ValueError(f"center={how!r} (centroid | deepest)")


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
    if cfg.crowd_ignore and t.crowd.any():
        heat_valid &= ~(F.avg_pool2d(t.crowd[None, None].float(), s8)[0, 0] > 0.5)
        mask_valid &= ~(F.avg_pool2d(t.crowd[None, None].float(), s4)[0, 0] > 0.5)

    masks_s4 = F.avg_pool2d(masks[:, None].float(), s4)[:, 0] if n else torch.zeros(0, S // s4, S // s4)
    centers = instance_centers(masks, cfg.center)
    center_inside = torch.zeros(n, dtype=torch.bool)
    if n == 0:
        return SegGT(heat, heat_valid, torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long),
                     masks_s4, mask_valid, centers, center_inside)

    area = masks.flatten(1).sum(1).float()
    occ8 = F.avg_pool2d(masks[:, None].float(), s8)[:, 0]                      # [N, h8, w8] occupancy
    ys = torch.arange(h8, dtype=torch.float32).view(-1, 1)
    xs = torch.arange(w8, dtype=torch.float32).view(1, -1)
    for i in area.argsort(descending=True).tolist():                           # smaller instances assigned last -> win
        cx = int(min(max(round((float(centers[i, 0]) + 0.5) / s8 - 0.5), 0), w8 - 1))
        cy = int(min(max(round((float(centers[i, 1]) + 0.5) / s8 - 0.5), 0), h8 - 1))
        center_inside[i] = bool(masks[i, min(int(centers[i, 1]), S - 1), min(int(centers[i, 0]), S - 1)])
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
        keep = torch.randperm(pos_index.numel(), generator=generator)[:cfg.max_pos]
        pos_index, pos_inst = pos_index[keep], pos_inst[keep]
    return SegGT(heat, heat_valid, pos_index, pos_inst, masks_s4, mask_valid, centers, center_inside)
