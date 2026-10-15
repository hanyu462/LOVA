"""Step 11-1: decoding. ModelOutput -> per-image detections (scores, labels, soft masks, centres).

    heat = sigmoid(heat_logits)                [B,K,H8,W8]
    peaks: cells that are the 3x3 maximum of their class map (CenterNet NMS), top-k by score, score > score_thr
    masks: sigmoid(<kernel_map[:, cell], mask_feat>)   [D, H4, W4]   (the same dynamic kernel the loss trains)
    score = peak score * maskness (mean prob inside the binarised mask, SOLOv2)
    Matrix NMS (gaussian, per class) on the binarised masks, then score > score_thr again, at most max_det

Padding: the canvas has padding where no image is (eval transform: resize + pad). The losses ignore it
through heat_valid / mask_valid, so the head outputs there are untrained; decode must not read them
either: invalid heat cells are removed BEFORE the 3x3 peak test (so they neither become peaks nor
suppress a valid neighbour), and mask probabilities are zeroed outside mask_valid BEFORE maskness,
area and Matrix NMS. Pass the same heat_valid / mask_valid the losses use.

select_pointed(det, pointer): LOVA's actual output is ONE mask, the pointed object. Among the detections
whose mask probability at the pointer is >= min_prob, the highest score; if none covers the pointer, the
one with the highest probability there. This is post-processing over the full-scene detections (the V0
head still predicts every instance); a pointed-only head is a later design step.

Everything else here is inference-time policy; nothing is learned. The masks are at stride 4 on the
canvas; the evaluator / visualiser upsample and undo the transform. decode() reads only the head
outputs (no R, no pointer); only select_pointed uses the pointer.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..losses.mask_loss import dynamic_masks
from ..models.output import ModelOutput


@dataclass(frozen=True)
class DecodeCfg:
    score_thr: float = 0.05
    topk: int = 100
    mask_thr: float = 0.5
    max_det: int = 100
    nms_sigma: float = 2.0


def matrix_nms(masks: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    """SOLOv2 Matrix NMS (gaussian). masks [N,HW] binary float, sorted by score descending -> decayed scores [N]."""
    n = len(scores)
    if n == 0:
        return scores
    inter = masks @ masks.t()
    areas = masks.sum(1)
    iou = (inter / (areas[:, None] + areas[None] - inter).clamp(min=1e-6)).triu(1)
    same = (labels[:, None] == labels[None]).float().triu(1)
    decay_iou = iou * same
    comp = decay_iou.max(0).values.expand(n, n).t()
    decay = (torch.exp(-sigma * decay_iou ** 2) / torch.exp(-sigma * comp ** 2)).min(0).values
    return scores * decay


@torch.no_grad()
def decode(out: ModelOutput, cfg: DecodeCfg = DecodeCfg(), heat_valid: torch.Tensor | None = None,
           mask_valid: torch.Tensor | None = None) -> list[dict]:
    """Per image: {"scores" [D], "labels" [D] long, "masks" [D,H4,W4] prob, "centers" [D,2] canvas px (x, y)}.
    heat_valid [B,H8,W8] bool and mask_valid [B,H4,W4] bool (SegGT / collate): padding is excluded."""
    heat = torch.sigmoid(out.heat_logits.float())
    b, k, h8, w8 = heat.shape
    if heat_valid is not None:
        if heat_valid.shape != (b, h8, w8):
            raise ValueError(f"heat_valid must be [B,{h8},{w8}], got {tuple(heat_valid.shape)}")
        hv = heat_valid.bool().to(heat.device)[:, None]
        heat_nms = heat.masked_fill(~hv, -1.0)                 # BEFORE the 3x3 max: an untrained padding value must not
        peaks = heat * ((F.max_pool2d(heat_nms, 3, 1, 1) == heat_nms) & hv)   # hide a valid neighbour's peak, nor be one
    else:
        peaks = heat * (F.max_pool2d(heat, 3, 1, 1) == heat)
    if mask_valid is not None and mask_valid.shape != (b, *out.mask_feat.shape[2:]):
        raise ValueError(f"mask_valid must be [B,{out.mask_feat.shape[2]},{out.mask_feat.shape[3]}], got {tuple(mask_valid.shape)}")
    results = []
    for i in range(b):
        sc, idx = peaks[i].flatten().topk(min(cfg.topk, peaks[i].numel()))
        keep = sc > cfg.score_thr
        sc, idx = sc[keep], idx[keep]
        labels, cell = idx // (h8 * w8), idx % (h8 * w8)
        prob = torch.sigmoid(dynamic_masks(out.kernel_map[i].float(), out.mask_feat[i].float(), cell))
        if mask_valid is not None:
            prob = prob * mask_valid[i].to(device=prob.device, dtype=prob.dtype)[None]
        binm = (prob > cfg.mask_thr).float()
        area = binm.flatten(1).sum(1)
        ok = area > 0
        sc, labels, cell, prob, binm, area = sc[ok], labels[ok], cell[ok], prob[ok], binm[ok], area[ok]
        sc = sc * (prob * binm).flatten(1).sum(1) / area.clamp(min=1)              # maskness
        order = sc.argsort(descending=True)
        sc, labels, cell, prob, binm = sc[order], labels[order], cell[order], prob[order], binm[order]
        sc = matrix_nms(binm.flatten(1), labels, sc, cfg.nms_sigma)
        keep = sc > cfg.score_thr
        sel = keep.nonzero()[:, 0][sc[keep].argsort(descending=True)[:cfg.max_det]]
        centers = torch.stack([(cell[sel] % w8).float(), (cell[sel] // w8).float()], 1) * 8 + 3.5     # cell centre, half-pixel
        results.append({"scores": sc[sel], "labels": labels[sel], "masks": prob[sel], "centers": centers})
    return results


def select_pointed(det: dict, pointer, min_prob: float = 0.5, stride: int = 4) -> int | None:
    """Index into det["masks"] of the mask to output for `pointer` (x, y canvas px), or None without detections."""
    masks = det["masks"]
    if masks.shape[0] == 0:
        return None
    h, w = masks.shape[-2:]
    cx = min(max(int((float(pointer[0]) + 0.5) / stride), 0), w - 1)       # the stride-4 cell containing the pointer
    cy = min(max(int((float(pointer[1]) + 0.5) / stride), 0), h - 1)
    at_pointer = masks[:, cy, cx]
    covering = at_pointer >= min_prob
    if bool(covering.any()):
        scores = det["scores"].masked_fill(~covering, -1.0)
        return int(scores.argmax())
    return int(at_pointer.argmax())
