"""Step 9-2a: the stem. image [B,3,H,W] -> F0 [B,c0,H/4,W/4].

Two strided 3x3 convs. Deliberately shallow: F0 is the ONLY thing the R predictor sees besides the
pointer, and it is also the backbone's input. Keeping it shallow limits the ungated compute and
discourages F0 from becoming a strong standalone representation (a design choice, not a guarantee).

Input invariant: H and W must be positive multiples of 32 (the backbone goes to stride 32; the data
pipeline always produces a 640 x 640 canvas). Checked here because the stem is where the image enters.
The stem itself accepts rectangular input; the square-canvas requirement is Lova's (R predictor geometry).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .common import ConvGNAct

MAX_STRIDE = 32


class Stem(nn.Module):
    def __init__(self, c0: int = 64):
        super().__init__()
        if c0 < 2 or c0 % 2:
            raise ValueError(f"c0 must be an even integer >= 2 (the stem goes 3 -> c0/2 -> c0), got {c0}")
        self.c0 = c0
        self.body = nn.Sequential(ConvGNAct(3, c0 // 2, s=2),     # S/2
                                  ConvGNAct(c0 // 2, c0, s=2))    # S/4

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() != 4 or image.shape[1] != 3:
            raise ValueError(f"image must be [B,3,H,W], got {tuple(image.shape)}")
        h, w = image.shape[-2:]
        if h < MAX_STRIDE or w < MAX_STRIDE or h % MAX_STRIDE or w % MAX_STRIDE:
            raise ValueError(f"H and W must be positive multiples of {MAX_STRIDE}, got {h}x{w}")
        return self.body(image)
