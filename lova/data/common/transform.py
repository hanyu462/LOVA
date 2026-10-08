"""Step 2: the same geometric transform for image, instance masks and crowd mask.

    params = sample_params(sample.size, cfg, generator)     draw scale / flip / crop ONLY
    out    = apply(sample, params, cfg)                      apply them to image + masks + crowd

Why two functions: everything random is decided in sample_params, so apply() is deterministic.
Tests can fix the params, evaluation uses `TransformCfg(train=False)` (resize + pad only), and
the params are stored in the output so predictions can be mapped back to original coordinates.

Order inside apply():  resize (longest side -> size * scale)  ->  horizontal flip  ->  crop to
size x size if larger  ->  pad with zeros (top-left aligned) if smaller.  `valid` marks real pixels.

Masks stay at FULL output resolution [N, S, S] (bool). Downsampling to stride 4 / 8 happens in
the steps that need it (targets, R*), so thin structures are not lost here.
Normalisation: ImageNet mean / std on the image only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .sample import Sample

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@dataclass(frozen=True)
class TransformCfg:
    size: int = 512                   # output canvas S x S
    train: bool = True                # False -> no random scale / flip / crop
    scale_range: tuple = (0.6, 1.25)  # multiplier on "longest side = size"
    flip_prob: float = 0.5
    mask_thr: float = 0.5             # bilinear-resized soft mask -> bool


@dataclass(frozen=True)
class TransformParams:
    scale: float          # resize factor applied to the original image
    new_h: int            # resized size before crop / pad
    new_w: int
    flip: bool
    crop_x: int           # top-left of the S x S window in the resized image (0 if not cropped)
    crop_y: int


def sample_params(orig_size: tuple[int, int], cfg: TransformCfg,
                  generator: torch.Generator | None = None) -> TransformParams:
    h, w = orig_size
    s = cfg.size / max(h, w)
    if cfg.train:
        lo, hi = cfg.scale_range
        s *= lo + (hi - lo) * torch.rand((), generator=generator).item()
    new_h, new_w = max(1, round(h * s)), max(1, round(w * s))
    flip = cfg.train and torch.rand((), generator=generator).item() < cfg.flip_prob
    cx = int(torch.randint(0, max(new_w - cfg.size, 0) + 1, (), generator=generator).item()) if cfg.train else 0
    cy = int(torch.randint(0, max(new_h - cfg.size, 0) + 1, (), generator=generator).item()) if cfg.train else 0
    return TransformParams(s, new_h, new_w, flip, cx, cy)


@dataclass
class Transformed:
    image: torch.Tensor      # [3, S, S] float, normalised
    masks: torch.Tensor      # [N, S, S] bool
    crowd: torch.Tensor      # [S, S] bool
    valid: torch.Tensor      # [S, S] bool  (True = real image, False = padding)
    labels: torch.Tensor     # [N] long   (unchanged)
    ann_ids: list[int]       # unchanged
    image_id: int
    params: TransformParams
    orig_size: tuple[int, int]

    def __len__(self) -> int:
        return len(self.ann_ids)


def _place(x: torch.Tensor, p: TransformParams, size: int, fill=0) -> torch.Tensor:
    """x [C, new_h, new_w] (already resized + flipped) -> [C, S, S]: crop window then pad top-left."""
    ch, cw = min(p.new_h - p.crop_y, size), min(p.new_w - p.crop_x, size)
    out = torch.full((x.shape[0], size, size), fill, dtype=x.dtype)
    out[:, :ch, :cw] = x[:, p.crop_y:p.crop_y + ch, p.crop_x:p.crop_x + cw]
    return out


def apply(sample: Sample, p: TransformParams, cfg: TransformCfg) -> Transformed:
    S = cfg.size
    # image: PIL bilinear resize -> tensor -> flip -> crop/pad -> normalise (padding stays 0 AFTER normalisation)
    im = sample.image.resize((p.new_w, p.new_h), Image.BILINEAR)
    im = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255
    if p.flip:
        im = im.flip(-1)
    im = (im - MEAN) / STD
    image = _place(im, p, S, fill=0.0)

    # masks + crowd: one bilinear resize on the stacked float masks -> threshold -> flip -> crop/pad
    stack = torch.cat([sample.masks, sample.crowd[None]], 0).float()[None]        # [1, N+1, H, W]
    stack = F.interpolate(stack, size=(p.new_h, p.new_w), mode="bilinear", align_corners=False)[0] >= cfg.mask_thr
    if p.flip:
        stack = stack.flip(-1)
    stack = _place(stack, p, S, fill=False)
    masks, crowd = stack[:-1], stack[-1]

    valid = torch.zeros(S, S, dtype=torch.bool)
    valid[:min(p.new_h - p.crop_y, S), :min(p.new_w - p.crop_x, S)] = True
    return Transformed(image, masks, crowd, valid, sample.labels, list(sample.ann_ids), sample.image_id, p, sample.size)


def transform(sample: Sample, cfg: TransformCfg, generator: torch.Generator | None = None) -> Transformed:
    """Convenience: sample_params + apply."""
    return apply(sample, sample_params(sample.size, cfg, generator), cfg)


def denormalize(image: torch.Tensor) -> torch.Tensor:
    """[3, S, S] normalised -> [3, S, S] in [0, 1] (for visualisation)."""
    return (image * STD + MEAN).clamp(0, 1)


def to_original(xy, p: TransformParams):
    """Point (x, y) in output-canvas pixels -> original-image pixels (undo pad/crop, flip, scale)."""
    x, y = float(xy[0]) + p.crop_x, float(xy[1]) + p.crop_y
    if p.flip:
        x = p.new_w - 1 - x
    return x / p.scale, y / p.scale
