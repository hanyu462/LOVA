"""One-stage dense instance head (SOLOv2-style dynamic kernels, center assignment).

  inst feature [B,C,H/8,W/8]
     ├─ cls tower  -> heat logits   [B,K,H/8,W/8]   (objectness x class, CenterNet focal)
     └─ kernel tower (+coords) -> kernels [B,E,H/8,W/8]
  mask feature [B,C,H/4,W/4] (+coords) -> mask_feat [B,E,H/4,W/4]

  instance at grid cell q:  mask_logits = <kernels[:, q], mask_feat>   (1x1 dynamic conv)

Every location of the image produces a prediction; nothing here depends on the
pointer or on R. Lower R only changes the quality of the incoming features.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvGNAct, coord_grid, gn


class InstanceHead(nn.Module):
    def __init__(self, num_classes: int, c: int = 128, e: int = 64, prior: float = 0.1):
        super().__init__()
        self.cls_tower = nn.Sequential(ConvGNAct(c, c), ConvGNAct(c, c))
        self.cls_out = nn.Conv2d(c, num_classes, 3, padding=1)
        nn.init.normal_(self.cls_out.weight, std=0.01)
        nn.init.constant_(self.cls_out.bias, -math.log((1 - prior) / prior))
        self.ker_tower = nn.Sequential(ConvGNAct(c + 2, c), ConvGNAct(c, c))
        self.ker_out = nn.Conv2d(c, e, 3, padding=1)
        nn.init.normal_(self.ker_out.weight, std=0.01)
        nn.init.zeros_(self.ker_out.bias)
        self.mask_branch = nn.Sequential(
            ConvGNAct(c + 2, c), ConvGNAct(c, c), nn.Conv2d(c, e, 1, bias=False), gn(e), nn.GELU())

    def forward(self, neck):
        x8, x4 = neck["inst"], neck["mask"]
        b = x8.shape[0]
        heat = self.cls_out(self.cls_tower(x8))
        kernels = self.ker_out(self.ker_tower(torch.cat([x8, coord_grid(b, *x8.shape[2:], x8.device, x8.dtype)], 1)))
        mask_feat = self.mask_branch(torch.cat([x4, coord_grid(b, *x4.shape[2:], x4.device, x4.dtype)], 1))
        return {"heat_logits": heat, "kernels": kernels, "mask_feat": mask_feat}


def dynamic_masks(kernels_b: torch.Tensor, mask_feat_b: torch.Tensor, pos_index: torch.Tensor) -> torch.Tensor:
    """kernels_b [E,H8,W8], mask_feat_b [E,H4,W4], pos_index [P] -> mask logits [P,H4,W4]."""
    k = kernels_b.flatten(1)[:, pos_index].t()  # [P,E]
    return torch.einsum("pe,ehw->phw", k, mask_feat_b)


def matrix_nms(masks: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    """SOLOv2 Matrix NMS (gaussian). masks [N,HW] binary float sorted by score desc."""
    n = len(scores)
    if n == 0:
        return scores
    inter = masks @ masks.t()
    areas = masks.sum(1)
    iou = (inter / (areas[:, None] + areas[None] - inter).clamp(min=1e-6)).triu(1)
    same = (labels[:, None] == labels[None]).float().triu(1)
    decay_iou = iou * same
    comp = decay_iou.max(0).values.expand(n, n).t()
    decay = (torch.exp(-sigma * decay_iou**2) / torch.exp(-sigma * comp**2)).min(0).values
    return scores * decay


@torch.no_grad()
def postprocess(out, score_thr: float = 0.05, topk: int = 100, mask_thr: float = 0.5, max_det: int = 100):
    """Returns per image: dict(scores [D], labels [D], masks [D,H4,W4] prob, centers [D,2] input px)."""
    heat = torch.sigmoid(out["heat_logits"].float())
    peaks = heat * (F.max_pool2d(heat, 3, 1, 1) == heat)
    b, k, h8, w8 = heat.shape
    results = []
    for i in range(b):
        sc, idx = peaks[i].flatten().topk(min(topk, peaks[i].numel()))
        keep = sc > score_thr
        sc, idx = sc[keep], idx[keep]
        labels, pos = idx // (h8 * w8), idx % (h8 * w8)
        prob = torch.sigmoid(dynamic_masks(out["kernels"][i].float(), out["mask_feat"][i].float(), pos))
        binm = (prob > mask_thr).float()
        area = binm.flatten(1).sum(1)
        ok = area > 0
        sc, labels, pos, prob, binm, area = sc[ok], labels[ok], pos[ok], prob[ok], binm[ok], area[ok]
        maskness = (prob * binm).flatten(1).sum(1) / area.clamp(min=1)
        sc = sc * maskness
        order = sc.argsort(descending=True)
        sc, labels, pos, prob, binm = sc[order], labels[order], pos[order], prob[order], binm[order]
        sc = matrix_nms(binm.flatten(1), labels, sc)
        keep = sc > score_thr
        order = sc[keep].argsort(descending=True)[:max_det]
        sel = keep.nonzero()[:, 0][order]
        centers = torch.stack([(pos[sel] % w8).float(), (pos[sel] // w8).float()], 1) * 8 + 4
        results.append({"scores": sc[sel], "labels": labels[sel], "masks": prob[sel], "centers": centers})
    return results
