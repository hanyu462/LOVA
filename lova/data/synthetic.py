"""Tiny synthetic instance dataset for smoke tests and fast sanity runs.

Classes are shape x color combinations (circle/square/triangle x 2 tints = 6),
so the class must be read from local shape, not just color.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .common import MEAN, STD, finalize

SHAPES = ("circle", "square", "triangle")
TINTS = (torch.tensor([0.9, 0.3, 0.2]), torch.tensor([0.2, 0.5, 0.9]))


class SyntheticPointerDataset(Dataset):
    num_classes = len(SHAPES) * len(TINTS)

    def __init__(self, length: int = 512, img_size: int = 256, max_objects: int = 6, seed: int = 0,
                 max_pos: int = 256, pointer_mode: str = "uniform", pointers_per_image: int = 1):
        assert img_size % 32 == 0
        self.length, self.size, self.max_objects, self.seed, self.max_pos = length, img_size, max_objects, seed, max_pos
        self.pointer_mode, self.pointers_per_image = pointer_mode, pointers_per_image

    def __len__(self):
        return self.length

    def load(self, index):
        g = torch.Generator().manual_seed(self.seed * 100003 + index)
        S = self.size
        yy, xx = torch.meshgrid(torch.arange(S).float(), torch.arange(S).float(), indexing="ij")
        img = 0.5 + 0.1 * torch.randn(3, S, S, generator=g)
        n = torch.randint(2, self.max_objects + 1, (1,), generator=g).item()
        masks, classes = [], []
        for _ in range(n):
            shape = torch.randint(len(SHAPES), (1,), generator=g).item()
            tint = torch.randint(len(TINTS), (1,), generator=g).item()
            r = S * (0.04 + 0.12 * torch.rand(1, generator=g).item())
            cx, cy = (torch.rand(2, generator=g) * (S - 2 * r) + r).tolist()
            if SHAPES[shape] == "circle":
                m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
            elif SHAPES[shape] == "square":
                m = ((xx - cx).abs() <= r * 0.85) & ((yy - cy).abs() <= r * 0.85)
            else:
                m = (yy - cy <= r * 0.8) & (yy - cy >= -r) & ((xx - cx).abs() <= (yy - cy + r) / math.sqrt(3))
            # later objects occlude earlier ones
            for prev in masks:
                prev &= ~m
            masks.append(m)
            classes.append(shape * len(TINTS) + tint)
            img = torch.where(m, TINTS[tint].view(3, 1, 1) + 0.05 * torch.randn(3, S, S, generator=g), img)
        masks = torch.stack(masks).float()
        return img.clamp(0, 1), masks, torch.tensor(classes), dict(image_id=index, orig_size=(S, S))

    def __getitem__(self, index, pointed: int | None = None):
        img, masks, classes, meta = self.load(index)
        S4 = self.size // 4
        image = (img - MEAN) / STD
        masks4 = F.avg_pool2d(masks[None], 4)[0]
        valid4 = torch.ones(1, S4, S4)
        g = torch.Generator().manual_seed(self.seed * 7919 + index)
        return finalize(image, valid4, masks4, classes, self.num_classes,
                        {**meta, "scale": 1.0, "offset": (0, 0), "new_size": (self.size, self.size), "flip": False},
                        pointed=pointed, generator=g, max_pos=self.max_pos, pointer_mode=self.pointer_mode,
                        pointers_per_image=self.pointers_per_image if pointed is None else 1)
