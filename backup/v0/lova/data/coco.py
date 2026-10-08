"""COCO instance segmentation with a sampled pointer.

Each __getitem__ picks one non-crowd instance uniformly and samples the pointer
inside its mask. ALL instances are kept as targets (graceful degradation, not
suppression). Crowd regions are treated as background in V0.

Layout expected:
  root/annotations/instances_{split}.json
  root/{split}/*.jpg
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .common import MEAN, STD, finalize


class COCOPointerDataset(Dataset):
    def __init__(self, root: str, split: str = "train2017", img_size: int = 512, train: bool = True,
                 scale_range=(0.6, 1.25), min_area: float = 32.0, max_pos: int = 256,
                 pointer_mode: str = "uniform", pointers_per_image: int = 1, mask_stride: int | None = None):
        """mask_stride: also return the pointed instance's transformed mask at this stride
        (sample["pointed_mask_s"]) for the R* profile; None = stride-4 masks only."""
        from pycocotools.coco import COCO

        assert img_size % 32 == 0
        self.root, self.split, self.size, self.train = root, split, img_size, train
        self.scale_range, self.min_area, self.max_pos = scale_range, min_area, max_pos
        self.pointer_mode, self.pointers_per_image, self.mask_stride = pointer_mode, pointers_per_image, mask_stride
        self.coco = COCO(os.path.join(root, "annotations", f"instances_{split}.json"))
        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat_to_label = {c: i for i, c in enumerate(self.cat_ids)}
        self.num_classes = len(self.cat_ids)
        self.ids = [i for i in sorted(self.coco.getImgIds()) if self._anns(i)]

    def _anns(self, img_id):
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, iscrowd=False))
        return [a for a in anns if a["area"] >= self.min_area]

    def __len__(self):
        return len(self.ids)

    def load(self, index):
        img_id = self.ids[index]
        info = self.coco.loadImgs(img_id)[0]
        img = Image.open(os.path.join(self.root, self.split, info["file_name"])).convert("RGB")
        anns = self._anns(img_id)
        masks = torch.from_numpy(np.stack([self.coco.annToMask(a) for a in anns])).float()
        classes = torch.tensor([self.cat_to_label[a["category_id"]] for a in anns])
        meta = dict(image_id=img_id, ann_ids=[a["id"] for a in anns], orig_size=(info["height"], info["width"]))
        return img, masks, classes, meta

    def _transform(self, img, masks, train):
        w, h = img.size
        s = self.size / max(h, w)
        if train:
            s *= random.uniform(*self.scale_range)
        nh, nw = max(4, round(h * s / 4) * 4), max(4, round(w * s / 4) * 4)
        flip = train and random.random() < 0.5
        ox = random.randint(0, max(nw - self.size, 0)) // 4 * 4 if train else 0
        oy = random.randint(0, max(nh - self.size, 0)) // 4 * 4 if train else 0

        im = torch.from_numpy(np.array(img.resize((nw, nh), Image.BILINEAR))).permute(2, 0, 1).float() / 255
        im = (im - MEAN) / STD
        m4 = F.interpolate(masks[None], size=(nh // 4, nw // 4), mode="area")[0]
        if flip:
            im, m4 = im.flip(-1), m4.flip(-1)

        S, S4 = self.size, self.size // 4
        image = torch.zeros(3, S, S)
        masks4 = torch.zeros(len(masks), S4, S4)
        valid4 = torch.zeros(1, S4, S4)
        ch, cw = min(nh - oy, S), min(nw - ox, S)
        image[:, :ch, :cw] = im[:, oy:oy + ch, ox:ox + cw]
        masks4[:, :ch // 4, :cw // 4] = m4[:, oy // 4:(oy + ch) // 4, ox // 4:(ox + cw) // 4]
        valid4[:, :ch // 4, :cw // 4] = 1
        tf = dict(scale=s, new_size=(nh, nw), offset=(ox, oy), flip=flip)
        return image, valid4, masks4, tf

    def _mask_fn(self, masks, tf):
        """Closure: original instance index -> transformed mask [1, S/st, S/st] at self.mask_stride."""
        st = self.mask_stride
        nh, nw = tf["new_size"]
        ox, oy = tf["offset"]
        S = self.size

        def fn(i):
            m = F.interpolate(masks[i][None, None], size=(nh // st, nw // st), mode="area")[0]
            if tf["flip"]:
                m = m.flip(-1)
            out = torch.zeros(1, S // st, S // st)
            ch, cw = min(nh - oy, S), min(nw - ox, S)
            out[:, :ch // st, :cw // st] = m[:, oy // st:(oy + ch) // st, ox // st:(ox + cw) // st]
            return out
        return fn

    def __getitem__(self, index, pointed: int | None = None):
        img, masks, classes, meta = self.load(index)
        for attempt in range(5):
            train = self.train and attempt < 4  # last attempt: deterministic, no crop
            image, valid4, masks4, tf = self._transform(img, masks, train)
            sample = finalize(image, valid4, masks4, classes, self.num_classes, {**meta, **tf},
                              pointed=pointed, max_pos=self.max_pos, pointer_mode=self.pointer_mode,
                              pointers_per_image=self.pointers_per_image if pointed is None else 1,
                              mask_fn=self._mask_fn(masks, tf) if self.mask_stride else None)
            if sample is not None:
                return sample
        return None
