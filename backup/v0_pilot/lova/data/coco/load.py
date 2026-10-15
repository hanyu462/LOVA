"""COCO reader: image id -> Sample (see lova/data/common/sample.py for the contract).

    cs     = open_coco(root, split)     once per process (annotation index + label map)
    sample = load(cs, img_id)           per sample

COCO specifics handled here:
  * instances  = annotations with iscrowd=0; polygons rasterised with annToMask -> masks [N,H,W]
  * labels     = COCO category ids (1..90 with gaps) sorted and mapped to 0..79
  * areas      = annotation "area" (polygon area; the raster can differ by a few pixels). Cheap;
                 steps that need the area of a transformed mask use mask.sum() instead
  * crowd      = union of iscrowd=1 regions (RLE) -> Sample.crowd ignore mask

Nothing is resized, cropped or filtered here. This is the only module that reads files or uses
pycocotools; every later step takes tensors, so it can be tested on synthetic masks.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from ..common.sample import Sample


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

    def labels_of(self, names) -> frozenset:
        """Class names -> label indices (e.g. for SelectCfg.exclude_labels)."""
        by_name = {n: i for i, n in ((self.cat_to_label[c], self.names[c]) for c in self.cat_ids)}
        missing = [n for n in names if n not in by_name]
        if missing:
            raise KeyError(f"unknown COCO class names: {missing}")
        return frozenset(by_name[n] for n in names)


def open_coco(root: str, split: str = "train2017") -> CocoSet:
    """Reads root/annotations/instances_{split}.json. Images are expected at root/{split}/."""
    from pycocotools.coco import COCO

    coco = COCO(os.path.join(root, "annotations", f"instances_{split}.json"))
    cat_ids = sorted(coco.getCatIds())
    names = {c["id"]: c["name"] for c in coco.loadCats(cat_ids)}
    return CocoSet(coco, root, split, cat_ids, {c: i for i, c in enumerate(cat_ids)}, names)


def load(cs: CocoSet, img_id: int) -> Sample:
    info = cs.coco.loadImgs(img_id)[0]
    image = Image.open(os.path.join(cs.root, cs.split, info["file_name"])).convert("RGB")
    h, w = info["height"], info["width"]
    anns = cs.coco.loadAnns(cs.coco.getAnnIds(imgIds=img_id, iscrowd=False))
    if anns:
        masks = torch.from_numpy(np.stack([cs.coco.annToMask(a) for a in anns])).bool()
    else:
        masks = torch.zeros(0, h, w, dtype=torch.bool)
    labels = torch.tensor([cs.cat_to_label[a["category_id"]] for a in anns], dtype=torch.long)
    areas = torch.tensor([float(a["area"]) for a in anns])
    crowd = torch.zeros(h, w, dtype=torch.bool)
    for a in cs.coco.loadAnns(cs.coco.getAnnIds(imgIds=img_id, iscrowd=True)):
        crowd |= torch.from_numpy(cs.coco.annToMask(a)).bool()
    return Sample(img_id, image, masks, labels, [a["id"] for a in anns], areas, areas / float(h * w), crowd)
