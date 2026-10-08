"""Shared sample format for COCO and synthetic data.

A sample (dict) contains:
  image        [3, S, S]  normalized float
  valid4       [1, S/4, S/4]  1 = real image, 0 = padding
  masks4       [N, S/4, S/4]  soft instance masks at stride 4
  classes      [N]           contiguous class ids
  pointer      [2]           (x, y) in input pixels, inside instance `pointed`
  pointed      int           index of the pointed instance
  heat         [K, S/8, S/8]
  pos_index    [P], pos_inst [P]
  meta         dict (image_id, ann_ids, orig size, scale, ...)
"""
from __future__ import annotations

import torch

from .rtarget import sample_interior_pointer
from .targets import build_targets

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def sample_pointer(mask4: torch.Tensor, generator: torch.Generator | None = None,
                   mode: str = "uniform") -> torch.Tensor:
    """Random pixel inside a stride-4 mask, in input-pixel coords.
    mode "uniform": any mask cell; "interior": cells with depth >= 25% of the max depth
    (never right at the boundary; pointer jitter stays inside the object)."""
    if mode == "interior":
        return sample_interior_pointer(mask4, generator=generator)
    idx = torch.nonzero(mask4 > 0.5, as_tuple=False)
    if idx.numel() == 0:
        idx = torch.nonzero(mask4 == mask4.max(), as_tuple=False)
    j = torch.randint(len(idx), (1,), generator=generator).item()
    y, x = idx[j].tolist()
    off = torch.rand(2, generator=generator)
    return torch.tensor([(x + off[0].item()) * 4, (y + off[1].item()) * 4])


def with_pointer(sample: dict, pointed: int, generator=None, mode: str = "uniform") -> dict:
    """Same image / targets, different pointed instance (targets do not depend on the pointer)."""
    out = dict(sample)
    out["pointed"] = pointed
    out["pointer"] = sample_pointer(sample["masks4"][pointed], generator, mode)
    return out


def finalize(image, valid4, masks4, classes, num_classes, meta, pointed=None,
             generator=None, max_pos=256, pointer_mode="uniform", pointers_per_image=1):
    """Returns one sample dict, or a list of `pointers_per_image` samples sharing the image and
    targets but pointing at different instances (None if no instance survives)."""
    keep = masks4.flatten(1).sum(1) >= 1.0  # drop instances that vanished after resize/crop
    if pointed is not None:
        if not keep[pointed]:
            return None
        pointed = int(keep[:pointed].sum())
    masks4, classes = masks4[keep], classes[keep]
    meta = dict(meta)
    if "ann_ids" in meta:
        meta["ann_ids"] = [a for a, k in zip(meta["ann_ids"], keep.tolist()) if k]
    n = len(classes)
    if n == 0:
        return None
    if pointed is None:
        pointed = torch.randint(n, (1,), generator=generator).item()
    pointer = sample_pointer(masks4[pointed], generator, pointer_mode)
    heat, pos_index, pos_inst = build_targets(masks4, classes, num_classes, max_pos, generator)
    sample = dict(image=image, valid4=valid4, masks4=masks4, classes=classes, pointer=pointer,
                  pointed=pointed, heat=heat, pos_index=pos_index, pos_inst=pos_inst, meta=meta)
    if pointers_per_image <= 1:
        return sample
    # K pointers on K distinct instances when possible (same instance, new pointer otherwise)
    order = torch.randperm(n, generator=generator).tolist()
    others = [i for i in order if i != pointed] + [pointed] * pointers_per_image
    return [sample] + [with_pointer(sample, j, generator, pointer_mode) for j in others[:pointers_per_image - 1]]


def collate(batch):
    flat = []
    for b in batch:  # a dataset item may be a list of samples (pointers_per_image > 1)
        flat.extend(b if isinstance(b, list) else [b])
    batch = [b for b in flat if b is not None]
    out = {
        "image": torch.stack([b["image"] for b in batch]),
        "valid4": torch.stack([b["valid4"] for b in batch]),
        "pointer": torch.stack([b["pointer"] for b in batch]).float(),
        "heat": torch.stack([b["heat"] for b in batch]),
        "pointed_mask4": torch.stack([b["masks4"][b["pointed"]][None] for b in batch]),
    }
    for k in ("masks4", "classes", "pos_index", "pos_inst", "pointed", "meta"):
        out[k] = [b[k] for b in batch]
    return out
