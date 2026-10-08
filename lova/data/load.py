"""Step 1 of the data pipeline: COCO image id -> original image + one mask per instance.

    coco   = open_coco(root, split)                 once per process
    sample = load(coco, img_id)                     per sample

    sample.image    PIL RGB, original size (H, W)
    sample.masks    bool  [N, H, W]   rasterised polygon of every non-crowd instance
    sample.labels   long  [N]         contiguous class index 0..K-1 (sorted COCO category ids)
    sample.ann_ids  list  [N]         COCO annotation ids (for evaluation / bookkeeping)
    sample.areas    float [N]         mask area in pixels (from the annotation polygon)
    sample.ratios   float [N]         areas / (H * W): the quantity later steps threshold on

Nothing is resized, cropped or filtered here. Crowd regions (iscrowd=1) are skipped: they are
unlabeled blobs, not instances. This is the only module that reads files or uses pycocotools;
every later step takes tensors, so it can be tested on synthetic masks.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


@dataclass
class CocoSet:
    """Annotation index for one split + the paths and label map that go with it."""
    coco: object               # pycocotools.coco.COCO
    root: str
    split: str
    cat_ids: list[int]         # sorted COCO category ids
    cat_to_label: dict[int, int]
    names: dict[int, str]      # category id -> name

    @property
    def num_classes(self) -> int:
        return len(self.cat_ids)

    def image_ids(self) -> list[int]:
        return sorted(self.coco.getImgIds())

    def name_of_label(self, label: int) -> str:
        return self.names[self.cat_ids[label]]


def open_coco(root: str, split: str = "train2017") -> CocoSet:
    """Reads root/annotations/instances_{split}.json. Images are expected at root/{split}/."""
    from pycocotools.coco import COCO

    coco = COCO(os.path.join(root, "annotations", f"instances_{split}.json"))
    cat_ids = sorted(coco.getCatIds())
    names = {c["id"]: c["name"] for c in coco.loadCats(cat_ids)}
    return CocoSet(coco, root, split, cat_ids, {c: i for i, c in enumerate(cat_ids)}, names)


@dataclass
class Sample:
    image_id: int
    image: Image.Image
    masks: torch.Tensor      # [N, H, W] bool
    labels: torch.Tensor     # [N] long
    ann_ids: list[int]
    areas: torch.Tensor      # [N] float
    ratios: torch.Tensor     # [N] float

    @property
    def size(self) -> tuple[int, int]:
        return self.image.height, self.image.width

    def __len__(self) -> int:
        return len(self.ann_ids)


def load(cs: CocoSet, img_id: int) -> Sample:
    info = cs.coco.loadImgs(img_id)[0]
    image = Image.open(os.path.join(cs.root, cs.split, info["file_name"])).convert("RGB")
    anns = cs.coco.loadAnns(cs.coco.getAnnIds(imgIds=img_id, iscrowd=False))
    h, w = info["height"], info["width"]
    if anns:
        masks = torch.from_numpy(np.stack([cs.coco.annToMask(a) for a in anns])).bool()
    else:
        masks = torch.zeros(0, h, w, dtype=torch.bool)
    labels = torch.tensor([cs.cat_to_label[a["category_id"]] for a in anns], dtype=torch.long)
    areas = torch.tensor([float(a["area"]) for a in anns])
    return Sample(img_id, image, masks, labels, [a["id"] for a in anns], areas, areas / float(h * w))
