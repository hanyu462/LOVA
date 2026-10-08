"""The contract between dataset readers and the rest of the pipeline.

Every reader (lova/data/coco, later synthetic / custom data) produces a Sample; every later step
(select, transform, pointer, targets, R*) consumes one. Nothing here knows about files or COCO.

    image    PIL RGB, original size (H, W)
    masks    bool  [N, H, W]   one mask per instance
    labels   long  [N]         contiguous class index 0..K-1
    ann_ids  list  [N]         reader-specific instance ids (bookkeeping / evaluation)
    areas    float [N]         instance area as given by the reader (annotation area for COCO)
    ratios   float [N]         areas / (H * W): what select thresholds on
    crowd    bool  [H, W]      ignore region: objects present but not individually annotated.
                               Never a pointer or segmentation target, never background either;
                               the loss step must ignore these pixels

Index i refers to the same instance in masks, labels, ann_ids, areas and ratios.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image


@dataclass
class Sample:
    image_id: int
    image: Image.Image
    masks: torch.Tensor      # [N, H, W] bool
    labels: torch.Tensor     # [N] long
    ann_ids: list[int]
    areas: torch.Tensor      # [N] float
    ratios: torch.Tensor     # [N] float, areas / (H * W)
    crowd: torch.Tensor      # [H, W] bool

    @property
    def size(self) -> tuple[int, int]:
        return self.image.height, self.image.width

    def __len__(self) -> int:
        return len(self.ann_ids)
