"""Center-based assignment for the dynamic-kernel instance head.

Per instance (masks given at stride 4):
  * center   = mask centroid, quantized to the stride-8 grid
  * heatmap  = class-specific gaussian at the center (CenterNet style)
  * kernel positives = 3x3 cells around the center that fall inside the mask
                       (center cell always kept). Smaller instances win ties.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def build_targets(masks4: torch.Tensor, classes: torch.Tensor, num_classes: int,
                  max_pos: int = 256, generator: torch.Generator | None = None):
    """masks4: [N, H4, W4] float in [0,1]; classes: [N] long (0..K-1).

    returns heat [K, H8, W8], pos_index [P] (flat index in H8*W8), pos_inst [P]
    """
    n, h4, w4 = masks4.shape
    h8, w8 = h4 // 2, w4 // 2
    heat = torch.zeros(num_classes, h8, w8)
    owner = torch.full((h8, w8), -1, dtype=torch.long)
    if n == 0:
        return heat, torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)

    m8 = F.avg_pool2d(masks4[:, None], 2)[:, 0] > 0.25  # [N, H8, W8]
    ys = torch.arange(h8, dtype=torch.float32).view(-1, 1)
    xs = torch.arange(w8, dtype=torch.float32).view(1, -1)
    yy4 = torch.arange(h4, dtype=torch.float32).view(1, -1, 1)
    xx4 = torch.arange(w4, dtype=torch.float32).view(1, 1, -1)
    area4 = masks4.sum((1, 2)).clamp(min=1e-6)
    cy4 = (masks4 * yy4).sum((1, 2)) / area4
    cx4 = (masks4 * xx4).sum((1, 2)) / area4

    for i in torch.argsort(area4, descending=True).tolist():  # small instances assigned last -> win
        cx = int(min(max(round((cx4[i].item() + 0.5) / 2 - 0.5), 0), w8 - 1))
        cy = int(min(max(round((cy4[i].item() + 0.5) / 2 - 0.5), 0), h8 - 1))
        size8 = math.sqrt(area4[i].item()) / 2
        sigma = max(0.8, size8 / 6)
        g = torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2))
        c = int(classes[i])
        heat[c] = torch.maximum(heat[c], g)
        y0, y1, x0, x1 = max(cy - 1, 0), min(cy + 2, h8), max(cx - 1, 0), min(cx + 2, w8)
        win = m8[i, y0:y1, x0:x1].clone()
        win[cy - y0, cx - x0] = True
        owner[y0:y1, x0:x1][win] = i

    pos_index = torch.nonzero(owner.flatten() >= 0, as_tuple=False)[:, 0]
    pos_inst = owner.flatten()[pos_index]
    if pos_index.numel() > max_pos:
        keep = torch.randperm(pos_index.numel(), generator=generator)[:max_pos]
        pos_index, pos_inst = pos_index[keep], pos_inst[keep]
    return heat, pos_index, pos_inst
