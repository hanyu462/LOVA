"""Step 3: which instances of a TRANSFORMED sample may be pointer targets.

    ratio_i = |mask_i ∧ valid| / |valid|       visible instance area over visible image area
    keep    = { i : ratio_i >= threshold  and  label_i not in exclude_labels }     largest first

exclude_labels: classes that are never pointer targets (still segmentation targets). DEFAULT EMPTY:
V0 keeps pointer conditioning class-independent and trains on every eligible instance (large
objects teach wide R). Kept as an ablation / deployment option only. Label indices are
dataset-specific, so the reader converts names (CocoSet.labels_of). Reference numbers on COCO
val2017 (600 images) for "surface" classes: dining table 4.2 % of candidates / owned p10 68 %,
bed 1.2 % / 67 %, bench 1.1 % / 68 %, couch 1.8 % / 98 %; chair 6.7 % / 97 %.
A class-agnostic alternative, if heavily covered targets ever matter: skip instances whose owned
fraction (pointer.pointer_region) is below a threshold.

    cands = select(transformed, SelectCfg(threshold=0.01))   -> list[int] indices into transformed.masks

Decided after the transform on purpose: a random crop can shrink a large object to a sliver or
enlarge a small one, so the original-image ratio is not the right quantity. Everything else is
untouched: masks / labels stay complete (all instances remain segmentation targets); this only
restricts where a pointer may be placed. Instances cropped away entirely have ratio 0 and drop out.

Only keep_by_ratio holds the rule, so the same policy can be reused on other ratio sources.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .transform import Transformed


@dataclass(frozen=True)
class SelectCfg:
    threshold: float = 0.01   # visible-area ratio a pointer target must reach. Working value, not
                              # final: val2017 at 0.01 -> 93 % of images have a candidate, 42 % of
                              # instances qualify (person-sized objects stay, bats / clocks drop)
    exclude_labels: frozenset = frozenset()   # label indices never used as pointer targets


def visible_ratios(masks: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """masks [N, S, S] bool, valid [S, S] bool -> [N] float. Padding counts in neither numerator nor
    denominator. Pure tensor ops (no host sync) so it can also run on GPU batches later."""
    v = valid.to(masks.device)
    area = (masks & v).flatten(1).sum(1).float()
    return area / v.sum().float().clamp(min=1)


def keep_by_ratio(ratios: torch.Tensor, threshold: float) -> list[int]:
    """Indices with ratio >= threshold, sorted by ratio descending. The single place the rule lives."""
    r = torch.as_tensor(ratios, dtype=torch.float32)
    idx = torch.nonzero(r >= threshold)[:, 0]
    return idx[r[idx].argsort(descending=True)].tolist()


def select(t: Transformed, cfg: SelectCfg = SelectCfg()) -> list[int]:
    """Pointer candidates of a transformed sample (indices into t.masks), largest first."""
    ratios = visible_ratios(t.masks, t.valid)
    if cfg.exclude_labels:
        excluded = torch.tensor([int(l) in cfg.exclude_labels for l in t.labels], dtype=torch.bool)
        ratios = torch.where(excluded, torch.zeros_like(ratios) - 1.0, ratios)   # -1 never passes
    return keep_by_ratio(ratios, cfg.threshold)
