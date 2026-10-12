"""Step 9-2a: the stem. image [B,3,S,S] -> F0 [B,c0,S/4,S/4].

Two strided 3x3 convs. Deliberately shallow: F0 is the ONLY thing the R predictor sees besides the
pointer, and it is also the backbone's input, so it must be cheap (it runs everywhere, ungated) and
must not be strong enough to solve the task on its own.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .common import ConvGNAct


class Stem(nn.Module):
    def __init__(self, c0: int = 64):
        super().__init__()
        self.c0 = c0
        self.body = nn.Sequential(ConvGNAct(3, c0 // 2, s=2),     # S/2
                                  ConvGNAct(c0 // 2, c0, s=2))    # S/4

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.body(image)
