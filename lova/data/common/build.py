"""Step 7: one Sample -> one TrainingSample (the whole per-sample pipeline), plus collate.

    ts = build(sample, num_classes, cfg, generator)     None if the sample cannot carry a pointer

    Sample ──transform──> Transformed ──select──> cands ──pointer──> (idx, p) ──make_r_gt──> R_GT @4
                              │                                                            (padding = 0)
                              └──────────────── make_seg_gt ────────────────────────────> SegGT

Two independent branches share the Transformed sample: the R branch (select / pointer / R_GT)
and the segmentation branch (SegGT from every visible instance). A sample without a pointer
candidate, or whose candidates have no owned pixel, returns None. Handling None is the Dataset's
job (it retries another image and always returns a TrainingSample); collate() takes only
TrainingSamples, so a batch is never silently smaller than requested.

TrainingSample fields and where they go:
    image        [3, S, S] float      model input
    valid        [S, S] bool          padding mask (losses, R stats)
    pointer      (x, y) px            R predictor input
    pointed_idx  int                  which instance was pointed at (evaluation: pointer gain)
    r_gt         [S/4, S/4] float     R predictor target (also zeroed on padding, for display / safety)
    r_valid      [S/4, S/4] bool      cells that take part in the R loss: padding is IGNORED, not a 0 target
                                      L_R = sum(r_valid * l(R_pred, R_GT)) / sum(r_valid)
    seg_gt       SegGT                heat / heat_valid / pos_index / pos_inst / masks_s4 / mask_valid
    labels, ann_ids, image_id, params, orig_size      bookkeeping (evaluation maps back to COCO)

collate() stacks the fixed-size tensors and keeps per-sample lists for the variable-length ones.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .make_r_gt import RgtCfg, make_r_gt, to_supervision
from .make_seg_gt import SegGT, SegGtCfg, make_seg_gt
from .pointer import PointerCfg, make_pointer
from .sample import Sample
from .select import SelectCfg, select
from .transform import TransformCfg, TransformParams, transform
from ...utils.geometry import owner_map


@dataclass(frozen=True)
class PipelineCfg:
    transform: TransformCfg = field(default_factory=TransformCfg)
    select: SelectCfg = field(default_factory=SelectCfg)
    pointer: PointerCfg = field(default_factory=PointerCfg)
    r_gt: RgtCfg = field(default_factory=RgtCfg)
    seg_gt: SegGtCfg = field(default_factory=SegGtCfg)
    r_stride: int = 4            # R predictor / supervision stride


@dataclass
class TrainingSample:
    image: torch.Tensor          # [3, S, S]
    valid: torch.Tensor          # [S, S] bool
    pointer: torch.Tensor        # [2] float (x, y) canvas px
    pointed_idx: int
    r_gt: torch.Tensor           # [S/r_stride, S/r_stride]
    r_valid: torch.Tensor        # [S/r_stride, S/r_stride] bool
    seg_gt: SegGT
    labels: torch.Tensor         # [N]
    ann_ids: list[int]
    image_id: int
    params: TransformParams
    orig_size: tuple[int, int]


def build(sample: Sample, num_classes: int, cfg: PipelineCfg = PipelineCfg(),
          generator: torch.Generator | None = None) -> TrainingSample | None:
    if cfg.r_stride < cfg.r_gt.stride or cfg.r_stride % cfg.r_gt.stride:
        raise ValueError(f"r_stride {cfg.r_stride} must be a multiple of the R_GT geometry stride {cfg.r_gt.stride}")
    t = transform(sample, cfg.transform, generator)
    # full-resolution passes shared by the branches, each done once and only when needed
    masks_s4 = torch.nn.functional.avg_pool2d(t.masks[:, None].float(), cfg.seg_gt.stride_mask)[:, 0]
    cands = select(t, cfg.select, masks_s4)
    if not cands:
        return None
    owners = owner_map(t.masks)                                                  # after select: skipped for samples without candidates
    picked = make_pointer(t, cands, cfg.pointer, generator, owners)
    if picked is None:
        return None
    idx, ptr = picked
    r = make_r_gt(t.masks[idx], ptr, cfg.r_gt)                                   # geometry stride
    r = to_supervision(r, cfg.r_stride // cfg.r_gt.stride)                       # -> r_stride
    r_valid = torch.nn.functional.avg_pool2d(t.valid[None, None].float(), cfg.r_stride)[0, 0] > 0.5
    r = r * r_valid                                                              # display / safety; the loss uses r_valid
    seg = make_seg_gt(t, num_classes, cfg.seg_gt, generator, masks_s4, owners)
    return TrainingSample(t.image, t.valid, torch.tensor(ptr, dtype=torch.float32), idx, r, r_valid, seg,
                          t.labels, t.ann_ids, t.image_id, t.params, t.orig_size)


def collate(samples: list[TrainingSample]) -> dict:
    """List of TrainingSample -> batch dict. Fixed-size tensors are stacked, variable-length ones
    (per-image instance lists) stay as lists indexed by batch position. No None filtering here:
    the Dataset guarantees every element is a TrainingSample."""
    assert len(samples) > 0 and all(s is not None for s in samples), "collate takes TrainingSamples only"
    return {
        "image": torch.stack([s.image for s in samples]),
        "valid": torch.stack([s.valid for s in samples]),
        "pointer": torch.stack([s.pointer for s in samples]),
        "r_gt": torch.stack([s.r_gt for s in samples]),
        "r_valid": torch.stack([s.r_valid for s in samples]),
        "heat": torch.stack([s.seg_gt.heat for s in samples]),
        "heat_valid": torch.stack([s.seg_gt.heat_valid for s in samples]),
        "mask_valid": torch.stack([s.seg_gt.mask_valid for s in samples]),
        "masks_s4": [s.seg_gt.masks_s4 for s in samples],
        "pos_index": [s.seg_gt.pos_index for s in samples],
        "pos_inst": [s.seg_gt.pos_inst for s in samples],
        "labels": [s.labels for s in samples],
        "pointed_idx": [s.pointed_idx for s in samples],
        "ann_ids": [s.ann_ids for s in samples],
        "image_id": [s.image_id for s in samples],
        "params": [s.params for s in samples],
        "orig_size": [s.orig_size for s in samples],
    }
