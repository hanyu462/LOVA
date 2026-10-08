"""Step 3: which instances of a TRANSFORMED sample may be pointer targets.

    ratio_i = |mask_i ∧ valid| / |valid|       visible instance area over visible image area
    keep    = { i : ratio_i >= threshold }     largest first

    cands = select(transformed, threshold)     -> list[int] of indices into transformed.masks

Decided after the transform on purpose: a random crop can shrink a large object to a sliver or
enlarge a small one, so the original-image ratio is not the right quantity. Everything else is
untouched: masks / labels stay complete (all instances remain segmentation targets); this only
restricts where a pointer may be placed. Instances cropped away entirely have ratio 0 and drop out.

Only keep_by_ratio holds the rule, so the same policy can be reused on other ratio sources.
"""
from __future__ import annotations

import torch

from .transform import Transformed


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


def select(t: Transformed, threshold: float) -> list[int]:
    """Pointer candidates of a transformed sample (indices into t.masks), largest first."""
    return keep_by_ratio(visible_ratios(t.masks, t.valid), threshold)
