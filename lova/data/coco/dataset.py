"""torch Dataset over COCO: index -> TrainingSample (load + build), for DataLoader with collate.

    ds = CocoDataset(root, "train2017", PipelineCfg(), train=True)
    dl = DataLoader(ds, batch_size=16, num_workers=8, collate_fn=collate, drop_last=True,
                    worker_init_fn=worker_init)

worker_init pins each worker to one intra-op thread: the pipeline is many small tensor ops, and
letting every worker spawn a full thread pool makes them fight for the cores. Measured on a
16-core desktop (val2017, train transform): build ~150 ms per sample single-threaded (20
instances), 58 samples/s with 8 workers; 12 workers were slower (contention).

Images without a non-crowd instance are dropped at construction. If build() returns None for an
index (no pointer candidate after the transform), the next index is tried; a sample is never
silently empty. TODO before evaluation: in eval mode the retry must not substitute another image
(index 10 -> image 11 would evaluate image 11 twice and drop image 10); precompute the set of
pointer-able indices deterministically instead. In eval mode (train=False) the transform is deterministic and the pointer draw is
seeded by the index, so every evaluation sees the same (image, pointer) pairs.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from ..common.build import PipelineCfg, TrainingSample, build
from ..common.transform import TransformCfg
from .load import CocoSet, load, open_coco


def worker_init(_worker_id: int) -> None:
    torch.set_num_threads(1)


class CocoDataset(Dataset):
    def __init__(self, root: str, split: str, cfg: PipelineCfg = PipelineCfg(), train: bool = True,
                 max_retries: int = 8):
        self.cs: CocoSet = open_coco(root, split)
        tcfg = cfg.transform if train else TransformCfg(**{**cfg.transform.__dict__, "train": False})
        self.cfg = PipelineCfg(**{**cfg.__dict__, "transform": tcfg})
        self.train = train
        self.max_retries = max_retries
        coco = self.cs.coco
        self.ids = [i for i in self.cs.image_ids() if coco.getAnnIds(imgIds=i, iscrowd=False)]

    @property
    def num_classes(self) -> int:
        return self.cs.num_classes

    def __len__(self) -> int:
        return len(self.ids)

    def build_one(self, index: int) -> TrainingSample | None:
        g = None if self.train else torch.Generator().manual_seed(index)
        return build(load(self.cs, self.ids[index]), self.num_classes, self.cfg, g)

    def __getitem__(self, index: int) -> TrainingSample:
        for k in range(self.max_retries):
            ts = self.build_one((index + k) % len(self.ids))
            if ts is not None:
                return ts
        raise RuntimeError(f"no pointer-able sample within {self.max_retries} images from index {index}")
