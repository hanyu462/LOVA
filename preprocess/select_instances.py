"""Step 1: which instances may be POINTER targets (every instance stays a segmentation target).

    ratio = visible mask area / visible image area
    keep  = ratio >= threshold   (and not iscrowd)

Two places use the same rule:
  1a  pre-filter on the original annotation  (`prefilter`, cheap: ann["area"] / (H*W))
      -> images with no candidate can be skipped before loading pixels
  1b  recheck AFTER augmentation on the transformed masks (`recheck`, uses the visible area:
      a random crop can leave only an arm of a person that was large in the original)

Only `keep_by_ratio` holds the rule; the other two just build the ratio array.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


def keep_by_ratio(ratios, threshold: float) -> list[int]:
    """Indices whose area ratio >= threshold, largest ratio first. The single place the rule lives."""
    r = torch.as_tensor(ratios, dtype=torch.float32)
    idx = torch.nonzero(r >= threshold)[:, 0]
    return idx[r[idx].argsort(descending=True)].tolist()


@dataclass(frozen=True)
class Candidate:
    ann_id: int
    category_id: int
    area: float   # original-image pixels
    ratio: float  # area / (H * W) of the original image
    bbox: tuple   # (x, y, w, h) original-image pixels


def annotation_ratios(coco, img_id: int, area_from: str = "annotation") -> list[Candidate]:
    """All non-crowd instances of an image (annotation order) with their original-image area ratio.
    area_from: "annotation" uses ann["area"] (polygon area); "mask" rasterises with annToMask."""
    info = coco.loadImgs(img_id)[0]
    ha = info["height"] * info["width"]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False))
    out = []
    for a in anns:
        area = float(a["area"]) if area_from == "annotation" else float(coco.annToMask(a).sum())
        out.append(Candidate(a["id"], a["category_id"], area, area / ha, tuple(a["bbox"])))
    return out


def prefilter(coco, img_id: int, threshold: float, area_from: str = "annotation") -> list[Candidate]:
    """1a: candidates on the ORIGINAL image, largest first. [] = no pointer target in this image."""
    cands = annotation_ratios(coco, img_id, area_from)
    return [cands[i] for i in keep_by_ratio([c.ratio for c in cands], threshold)]


def visible_ratios(masks: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    """masks [N, h, w] (soft or binary, any stride) -> [N] visible area / visible image area.
    valid [1, h, w] or [h, w] marks the real image (1) vs padding (0); None = whole canvas."""
    m = masks.flatten(1).float()
    if valid is None:
        denom = float(masks.shape[-1] * masks.shape[-2])
    else:
        denom = float(valid.float().sum().clamp(min=1))
    return m.sum(1) / denom


def recheck(masks: torch.Tensor, threshold: float, valid: torch.Tensor | None = None) -> list[int]:
    """1b: after augmentation. Indices (into `masks`) still large enough to be pointed at, largest first."""
    return keep_by_ratio(visible_ratios(masks, valid), threshold)


def selection_stats(coco, threshold: float, img_ids=None) -> dict:
    """How much of the dataset survives the ORIGINAL-image pre-filter (fast, annotation areas)."""
    img_ids = list(img_ids) if img_ids is not None else coco.getImgIds()
    n_inst = n_keep = n_img_keep = 0
    for i in img_ids:
        cands = annotation_ratios(coco, i)
        k = len(keep_by_ratio([c.ratio for c in cands], threshold))
        n_inst += len(cands)
        n_keep += k
        n_img_keep += k > 0
    return dict(threshold=threshold, images=len(img_ids), images_with_target=n_img_keep,
                instances=n_inst, instances_kept=n_keep)
