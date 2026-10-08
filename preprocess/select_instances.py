"""Step 1: pick the instances that are large enough to be pointer targets.

    ratio = mask area / image area   (both in original-image pixels)
    keep  = ratio >= threshold  and  not iscrowd

Used by the training dataset to decide which instances may be pointed at. Every instance
(kept or not) still remains a segmentation target; this only restricts the POINTER targets.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Instance:
    ann_id: int
    category_id: int
    area: float          # mask area in original-image pixels
    ratio: float         # area / (H * W)
    bbox: tuple          # (x, y, w, h) original-image pixels


def image_area(coco, img_id: int) -> int:
    info = coco.loadImgs(img_id)[0]
    return info["height"] * info["width"]


def instance_ratios(coco, img_id: int, area_from: str = "annotation") -> list[Instance]:
    """All non-crowd instances of an image with their area ratio, largest first.
    area_from: "annotation" uses ann["area"] (polygon area, cheap); "mask" rasterises with annToMask."""
    ha = image_area(coco, img_id)
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False))
    out = []
    for a in anns:
        area = float(a["area"]) if area_from == "annotation" else float(coco.annToMask(a).sum())
        out.append(Instance(a["id"], a["category_id"], area, area / ha, tuple(a["bbox"])))
    return sorted(out, key=lambda x: x.ratio, reverse=True)


def select_instances(coco, img_id: int, threshold: float, area_from: str = "annotation") -> list[Instance]:
    """Instances with area ratio >= threshold (largest first). Empty list = image has no pointer target."""
    return [x for x in instance_ratios(coco, img_id, area_from) if x.ratio >= threshold]


def selection_stats(coco, threshold: float, img_ids=None) -> dict:
    """How much of the dataset survives the threshold (annotation areas, so this is fast)."""
    img_ids = list(img_ids) if img_ids is not None else coco.getImgIds()
    n_inst = n_keep = n_img_keep = 0
    for i in img_ids:
        inst = instance_ratios(coco, i)
        k = sum(x.ratio >= threshold for x in inst)
        n_inst += len(inst)
        n_keep += k
        n_img_keep += k > 0
    return dict(threshold=threshold, images=len(img_ids), images_with_target=n_img_keep,
                instances=n_inst, instances_kept=n_keep)
