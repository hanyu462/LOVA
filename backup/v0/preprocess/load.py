"""Step 2: COCO image id -> original image + per-instance masks (nothing resized yet).

    sample = load_sample(coco, root, split, img_id)
    sample.image   PIL RGB (H, W)
    sample.masks   torch bool [N, H, W]   one rasterised polygon per non-crowd instance
    sample.labels  torch long [N]         contiguous class index 0..K-1 (sorted COCO category ids)
    sample.cands   list[Candidate]        same N, annotation order (ann_id, category_id, area, ratio, bbox)

This is the only place that touches image files and pycocotools. Everything after it works on
tensors, so it can be tested with synthetic masks.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from .select_instances import Candidate, annotation_ratios


@dataclass
class Loaded:
    image: Image.Image
    masks: torch.Tensor      # [N, H, W] bool
    labels: torch.Tensor     # [N] long
    cands: list[Candidate]   # [N]
    image_id: int

    @property
    def size(self) -> tuple[int, int]:
        return self.image.height, self.image.width


class LabelMap:
    """COCO category id <-> contiguous label and name."""

    def __init__(self, coco):
        self.cat_ids = sorted(coco.getCatIds())
        self.to_label = {c: i for i, c in enumerate(self.cat_ids)}
        self.names = {c["id"]: c["name"] for c in coco.loadCats(self.cat_ids)}
        self.num_classes = len(self.cat_ids)

    def name(self, category_id: int) -> str:
        return self.names[category_id]


def load_sample(coco, root: str, split: str, img_id: int, labels: LabelMap | None = None) -> Loaded:
    labels = labels or LabelMap(coco)
    info = coco.loadImgs(img_id)[0]
    image = Image.open(os.path.join(root, split, info["file_name"])).convert("RGB")
    cands = annotation_ratios(coco, img_id)
    anns = coco.loadAnns([c.ann_id for c in cands])
    if anns:
        masks = torch.from_numpy(np.stack([coco.annToMask(a) for a in anns])).bool()
    else:
        masks = torch.zeros(0, info["height"], info["width"], dtype=torch.bool)
    lab = torch.tensor([labels.to_label[c.category_id] for c in cands], dtype=torch.long)
    return Loaded(image, masks, lab, cands, img_id)
