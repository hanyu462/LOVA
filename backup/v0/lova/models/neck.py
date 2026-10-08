"""Neck: backbone features -> (instance feature @ stride 8, mask feature @ stride 4).

Not R-conditioned and never sees R or the pointer. Everything the head knows
about "where to look" must come through the backbone features.
"""
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvGNAct, gn


class FPNLite(nn.Module):
    def __init__(self, widths=(64, 128, 256, 384), c: int = 128):
        super().__init__()
        w4, w8, w16, w32 = widths
        self.lat = nn.ModuleDict({
            k: nn.Sequential(nn.Conv2d(w, c, 1, bias=False), gn(c))
            for k, w in (("s4", w4), ("s8", w8), ("s16", w16), ("s32", w32))
        })
        self.smooth8 = ConvGNAct(c, c)
        self.smooth4 = ConvGNAct(c, c)

    def forward(self, feats):
        p32 = self.lat["s32"](feats["s32"])
        p16 = self.lat["s16"](feats["s16"]) + F.interpolate(p32, scale_factor=2, mode="nearest")
        p8 = self.lat["s8"](feats["s8"]) + F.interpolate(p16, scale_factor=2, mode="nearest")
        p8 = self.smooth8(p8)
        p4 = self.lat["s4"](feats["s4"]) + F.interpolate(p8, scale_factor=2, mode="bilinear", align_corners=False)
        p4 = self.smooth4(p4)
        return {"inst": p8, "mask": p4}
