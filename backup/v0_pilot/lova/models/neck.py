"""Step 9-4a: the neck. {s4, s8, s16, s32} -> {p8 [B,c,S/8,S/8], p4 [B,c,S/4,S/4]}.

    s32 ──1x1 lat──> p32 ──up x2 (nearest)──+── s16 lat ──> p16 ──up x2──+── s8 lat ──> smooth ──> p8   (heat / kernels)
                                                                          p8 ──up x2 (bilinear)──+── s4 lat ──> smooth ──> p4   (mask feature)

A small top-down FPN. It never receives R or the pointer directly; their effect reaches the neck only
through the R-conditioned backbone features. p8 feeds the heat and kernel towers, p4 the mask feature
branch.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvGNAct, gn


@dataclass(frozen=True)
class NeckCfg:
    c: int = 128


class FPN(nn.Module):
    def __init__(self, in_channels: dict, cfg: NeckCfg = NeckCfg()):
        """in_channels: Backbone.out_channels, {"s4": w4, "s8": w8, "s16": w16, "s32": w32}."""
        super().__init__()
        if set(in_channels) != {"s4", "s8", "s16", "s32"}:
            raise ValueError(f"neck expects s4/s8/s16/s32 features, got {sorted(in_channels)}")
        if not isinstance(cfg.c, int) or cfg.c <= 0:
            raise ValueError(f"neck width c must be a positive integer, got {cfg.c!r}")
        if any(not isinstance(w, int) or w <= 0 for w in in_channels.values()):
            raise ValueError(f"in_channels must be positive integers, got {in_channels}")
        c = cfg.c
        self.c, self.in_channels = c, dict(in_channels)
        self.lat = nn.ModuleDict({k: nn.Sequential(nn.Conv2d(w, c, 1, bias=False), gn(c)) for k, w in in_channels.items()})
        self.smooth8 = ConvGNAct(c, c)
        self.smooth4 = ConvGNAct(c, c)

    def _check(self, feats: dict) -> None:
        """backbone -> neck contract: the four levels, declared channels, one batch, exact 2x steps."""
        keys = ("s4", "s8", "s16", "s32")
        if set(feats) != set(keys):
            raise ValueError(f"neck expects s4/s8/s16/s32 features, got {sorted(feats)}")
        for k in keys:
            x = feats[k]
            if x.dim() != 4 or x.shape[1] != self.in_channels[k] or x.shape[0] != feats["s4"].shape[0]:
                raise ValueError(f"{k} must be [B,{self.in_channels[k]},h,w] with the batch of s4, got {tuple(x.shape)}")
        for fine, coarse in zip(keys[:-1], keys[1:]):
            if tuple(feats[fine].shape[-2:]) != (2 * feats[coarse].shape[-2], 2 * feats[coarse].shape[-1]):
                raise ValueError(f"{fine} must be exactly 2x {coarse} spatially, got {tuple(feats[fine].shape[-2:])} / {tuple(feats[coarse].shape[-2:])}")

    def forward(self, feats: dict) -> dict:
        self._check(feats)
        p32 = self.lat["s32"](feats["s32"])
        p16 = self.lat["s16"](feats["s16"]) + F.interpolate(p32, size=feats["s16"].shape[-2:], mode="nearest")
        p8 = self.lat["s8"](feats["s8"]) + F.interpolate(p16, size=feats["s8"].shape[-2:], mode="nearest")
        p8 = self.smooth8(p8)
        p4 = self.lat["s4"](feats["s4"]) + F.interpolate(p8, size=feats["s4"].shape[-2:], mode="bilinear", align_corners=False)
        return {"p8": p8, "p4": self.smooth4(p4)}
