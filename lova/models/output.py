"""Step 9-1: the contract between the model and the losses. Fixed before any block is written.

    out = model(image, pointer)                image [B,3,S,S] normalized, pointer [B,2] (x, y) canvas px

    r_pred       [B, 1, S/4, S/4]  in [0, 1]   -> r_loss      (sigmoid output, the Resolution Field)
    heat_logits  [B, K, S/8, S/8]  logits      -> heat_loss   (CenterNet class heatmap)
    kernel_map   [B, D, S/8, S/8]              -> mask_loss   (dynamic kernel per stride-8 cell)
    mask_feat    [B, D, S/4, S/4]              -> mask_loss   (shared mask feature)

`as_dict()` is what total_loss consumes (it reads keys, see lova/losses/total_loss.REQUIRED_OUT).
`check()` is the shape assertion used by the tests and by debug runs; the training loop does not call it.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

R_STRIDE, HEAT_STRIDE, MASK_STRIDE = 4, 8, 4


@dataclass
class ModelOutput:
    r_pred: torch.Tensor        # [B, 1, S/4, S/4]
    heat_logits: torch.Tensor   # [B, K, S/8, S/8]
    kernel_map: torch.Tensor    # [B, D, S/8, S/8]
    mask_feat: torch.Tensor     # [B, D, S/4, S/4]

    def as_dict(self) -> dict:
        return {"r_pred": self.r_pred, "heat_logits": self.heat_logits,
                "kernel_map": self.kernel_map, "mask_feat": self.mask_feat}

    def check(self, image_shape: tuple, num_classes: int) -> "ModelOutput":
        """Raises if any tensor disagrees with the contract for an input of `image_shape` [B,3,H,W]."""
        B, _, H, W = image_shape
        want = {"r_pred": (B, 1, H // R_STRIDE, W // R_STRIDE),
                "heat_logits": (B, num_classes, H // HEAT_STRIDE, W // HEAT_STRIDE)}
        for k, s in want.items():
            if tuple(getattr(self, k).shape) != s:
                raise ValueError(f"{k}: expected {s}, got {tuple(getattr(self, k).shape)}")
        D = self.kernel_map.shape[1]
        if tuple(self.kernel_map.shape) != (B, D, H // HEAT_STRIDE, W // HEAT_STRIDE):
            raise ValueError(f"kernel_map: expected {(B, D, H // HEAT_STRIDE, W // HEAT_STRIDE)}, got {tuple(self.kernel_map.shape)}")
        if tuple(self.mask_feat.shape) != (B, D, H // MASK_STRIDE, W // MASK_STRIDE):
            raise ValueError(f"mask_feat: expected {(B, D, H // MASK_STRIDE, W // MASK_STRIDE)}, got {tuple(self.mask_feat.shape)}")
        if self.r_pred.min() < 0 or self.r_pred.max() > 1:
            raise ValueError("r_pred must lie in [0, 1]")
        return self
