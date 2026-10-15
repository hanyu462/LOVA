"""Step 9-5: the whole model. (image, pointer, valid) -> ModelOutput, nothing else in between.

    image [B,3,S,S] ──Stem──> F0 [B,c0,S/4,S/4]
                               ├── RPredictor(F0, pointer) ──> r_pred ──resolve_r(·, valid@4, r_override)──> r_gate
                               └── Backbone(F0, r_gate) ──> {s4..s32} ──FPN──> {p8, p4} ──Head──> heat_logits, kernel_map, mask_feat

use_r = False is the pure No-R baseline: no R predictor (no parameters, no pointer input, nothing for
r_loss to train: pair it with w_r = 0), the backbone runs on gate = 1 inside the image, i.e. the plain
residual network F + Body(F). It differs from A0 (train_r_override = 1.0): A0 still has the predictor,
trained by r_loss, whose gradient reaches the shared stem. r_pred is reported as the constant gate so
the output contract and the diagnostics stay the same.

Hard constraints (the causal structure of LOVA):
  * the pointer enters ONLY through the R predictor; the backbone, neck and head never see it
  * the neck and head never see F0 or R directly: everything arrives through the gated backbone
  * r_pred is supervised by r_loss; r_source = r_pred unless r_override is supplied;
    r_gate = r_source * valid@4 is what controls the backbone (an override goes through valid too)

Input contract: image [B,3,S,S] on one device with H == W (V0 uses a square canvas; the R predictor
normalises pointer offsets by a single canvas side), pointer [B,2] and valid [B,S,S] bool on the same
device. Training must pass valid (padding is gated to 0); None is for debug / unpadded inference only.
r_override: see backbone.resolve_r (floats for constant-R sweeps and the R = 1 baseline, tensors for
matched-budget fields).

forward_with_intermediates additionally returns {f0, feats, neck}: the shared tensors the training
diagnostics (per-loss gradient at F0) need. The training loop uses forward().
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .backbone import Backbone, BackboneCfg, resolve_r, valid_to_stride
from .head import Head, HeadCfg
from .neck import FPN, NeckCfg
from .output import ModelOutput
from .r_predictor import RPredictor, RPredictorCfg
from .stem import Stem


@dataclass(frozen=True)
class LovaCfg:
    c0: int = 64
    r_predictor: RPredictorCfg = field(default_factory=RPredictorCfg)
    backbone: BackboneCfg = field(default_factory=BackboneCfg)
    neck: NeckCfg = field(default_factory=NeckCfg)
    head: HeadCfg = field(default_factory=HeadCfg)
    use_r: bool = True            # False: No-R baseline (see module docstring)


class Lova(nn.Module):
    def __init__(self, num_classes: int, cfg: LovaCfg = LovaCfg()):
        super().__init__()
        if cfg.head.c != cfg.neck.c:
            raise ValueError(f"head width {cfg.head.c} must equal neck width {cfg.neck.c}")
        self.cfg, self.num_classes = cfg, num_classes
        self.stem = Stem(cfg.c0)
        self.r_predictor = RPredictor(cfg.c0, cfg.r_predictor) if cfg.use_r else None
        self.backbone = Backbone(cfg.c0, cfg.backbone)
        self.neck = FPN(self.backbone.out_channels, cfg.neck)
        self.head = Head(num_classes, cfg.head)

    def forward_with_intermediates(self, image: torch.Tensor, pointer: torch.Tensor, valid: torch.Tensor | None = None,
                                   r_override=None, detach_r: bool = False) -> tuple[ModelOutput, dict]:
        """detach_r: the backbone runs on r_pred.detach(), so the segmentation losses cannot reach the predictor
        (ablation: the predictor then learns R_GT only). r_pred itself is returned with its graph for r_loss."""
        if image.dim() != 4 or image.shape[1] != 3:
            raise ValueError(f"image must be [B,3,H,W], got {tuple(image.shape)}")
        if image.shape[-2] != image.shape[-1]:
            raise ValueError(f"LOVA V0 requires a square canvas, got {image.shape[-2]}x{image.shape[-1]}")
        if self.cfg.use_r and (pointer.shape != (image.shape[0], 2) or pointer.device != image.device):
            raise ValueError(f"pointer must be [B,2] on {image.device}, got {tuple(pointer.shape)} on {pointer.device}")
        if valid is not None:
            if valid.shape != (image.shape[0], *image.shape[2:]) or valid.dtype != torch.bool or valid.device != image.device:
                raise ValueError(f"valid must be bool [B,H,W] on {image.device}, got {valid.dtype} {tuple(valid.shape)} on {valid.device}")
        f0 = self.stem(image)
        valid_s4 = None if valid is None else valid_to_stride(valid, 4)
        if self.r_predictor is not None:
            r_pred = self.r_predictor(f0, pointer, canvas=image.shape[-1])
        else:                                                    # No-R: the pointer is ignored, gate = 1 inside the image
            r_pred = torch.ones(f0.shape[0], 1, *f0.shape[2:], device=f0.device, dtype=f0.dtype) if valid_s4 is None else valid_s4.to(f0.dtype)
        r_gate = resolve_r(r_pred.detach() if detach_r else r_pred, valid_s4, r_override)
        feats = self.backbone(f0, r_gate, valid_s4)
        neck = self.neck(feats)
        head = self.head(neck)
        out = ModelOutput(r_pred=r_pred, r_gate=r_gate, **head)
        return out, {"f0": f0, "feats": feats, "neck": neck, "valid_s4": valid_s4}

    def forward(self, image: torch.Tensor, pointer: torch.Tensor, valid: torch.Tensor | None = None, r_override=None,
                detach_r: bool = False) -> ModelOutput:
        return self.forward_with_intermediates(image, pointer, valid, r_override, detach_r)[0]

    def parts(self) -> dict:
        d = {"stem": self.stem, "r_predictor": self.r_predictor, "backbone": self.backbone, "neck": self.neck, "head": self.head}
        return {k: v for k, v in d.items() if v is not None}
