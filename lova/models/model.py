"""LOVA V0 wiring.

    I [B,3,H,W] ──Stem──> F0 [B,64,H/4,W/4]
                           ├──(+ pointer maps)── ResolutionPredictor ──> R [B,1,H/4,W/4]
                           └────────────── RABackbone(F0, R) ──> {s4,s8,s16,s32}
                                                            └─ FPNLite ─ InstanceHead

Hard constraints:
  * the pointer enters ONLY through R
  * neck/head never see R or F0 directly (no bypass of the R-conditioned backbone)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import RABackbone, Stem
from .head import InstanceHead
from .neck import FPNLite
from .resolution import ResolutionPredictor


class LOVAv0(nn.Module):
    def __init__(self, num_classes: int = 80, c0: int = 64, widths=(64, 128, 256, 384), depths=(2, 2, 3, 2),
                 neck_c: int = 128, kernel_dim: int = 64, gate_mode: str = "depth", gate_temp: float = 0.1, transition: str = "conv"):
        super().__init__()
        self.config = dict(num_classes=num_classes, c0=c0, widths=tuple(widths), depths=tuple(depths),
                           neck_c=neck_c, kernel_dim=kernel_dim, gate_mode=gate_mode, gate_temp=gate_temp, transition=transition)
        self.stem = Stem(c0)
        self.rpred = ResolutionPredictor(c0)
        self.backbone = RABackbone(c0, widths, depths, gate_mode, gate_temp, transition)
        self.neck = FPNLite(widths, neck_c)
        self.head = InstanceHead(num_classes, neck_c, kernel_dim)

    def predict_r(self, images, pointers, f0=None):
        f0 = self.stem(images) if f0 is None else f0
        return self.rpred(f0, pointers, canvas=max(images.shape[-2:]))

    def forward(self, images: torch.Tensor, pointers: torch.Tensor, r_override: torch.Tensor | None = None,
                r_only: bool = False):
        """images [B,3,H,W], pointers [B,2] (x,y px), r_override [B,1,H/4,W/4] or None.
        r_only: skip backbone/head (phase B without task loss; goes through forward() so DDP syncs)."""
        f0 = self.stem(images)
        if r_only:
            return {"r": self.predict_r(images, pointers, f0)}
        r_pred = self.predict_r(images, pointers, f0) if r_override is None else None
        r = r_override if r_override is not None else r_pred
        out = self.head(self.neck(self.backbone(f0, r)))
        out["r"] = r
        out["r_pred"] = r_pred
        return out
