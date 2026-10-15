"""lova.data.common.build: the whole per-sample pipeline on a synthetic Sample, plus collate.

    python tests/common/test_build.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lova.data.common.build import PipelineCfg, TrainingSample, build, collate  # noqa: E402
from lova.data.common.select import SelectCfg  # noqa: E402
from lova.data.common.transform import TransformCfg  # noqa: E402
from tests.common.test_transform import synthetic_sample  # noqa: E402


def unit_test():
    s = synthetic_sample()                      # 400x300, 3 instances (square, disc, thin bar) + crowd
    cfg = PipelineCfg(transform=TransformCfg(size=256, train=False))
    ts = build(s, num_classes=3, cfg=cfg, generator=torch.Generator().manual_seed(0))
    assert isinstance(ts, TrainingSample)
    S = 256
    assert ts.image.shape == (3, S, S) and ts.valid.shape == (S, S)
    assert ts.r_gt.shape == (S // 4, S // 4) and 0 <= ts.r_gt.min() and ts.r_gt.max() <= 1
    assert ts.seg_gt.heat.shape == (3, S // 8, S // 8) and ts.seg_gt.masks_s4.shape == (3, S // 4, S // 4)
    # pointer lies on the pointed instance; R_GT peaks near it and is 0 on padding
    x, y = ts.pointer.tolist()
    assert bool(s.masks[ts.pointed_idx].any()) and ts.pointed_idx in (0, 1, 2)
    px, py = int((x + 0.5) / 4 - 0.5), int((y + 0.5) / 4 - 0.5)
    assert float(ts.r_gt[py, px]) > 0.9
    valid_r = torch.nn.functional.avg_pool2d(ts.valid[None, None].float(), 4)[0, 0] > 0.5
    assert torch.equal(ts.r_valid, valid_r) and (~valid_r).any(), "r_valid marks the padding cells to ignore"
    assert float(ts.r_gt[~ts.r_valid].max()) == 0.0, "padding rows carry no R target"
    # reproducible with the same generator
    ts2 = build(s, 3, cfg, torch.Generator().manual_seed(0))
    assert torch.equal(ts.pointer, ts2.pointer) and torch.equal(ts.r_gt, ts2.r_gt) and torch.equal(ts.seg_gt.pos_index, ts2.seg_gt.pos_index)
    # no candidate -> None (threshold above every instance)
    assert build(s, 3, PipelineCfg(transform=TransformCfg(size=256, train=False), select=SelectCfg(threshold=0.9))) is None
    # train mode runs (random transform) and collate stacks
    batch = collate([build(s, 3, PipelineCfg(transform=TransformCfg(size=256)), torch.Generator().manual_seed(k)) for k in range(3)])
    assert batch["image"].shape == (3, 3, S, S) and batch["r_gt"].shape == (3, S // 4, S // 4)
    assert batch["heat"].shape == (3, 3, S // 8, S // 8) and batch["pointer"].shape == (3, 2)
    assert len(batch["masks_s4"]) == 3 and len(batch["pos_index"]) == 3 and len(batch["image_id"]) == 3
    assert batch["r_valid"].shape == (3, S // 4, S // 4)
    for bad in ([], [None, ts]):
        try:
            collate(bad)
            raise AssertionError("collate must reject empty / None")
        except AssertionError as e:
            assert "TrainingSamples only" in str(e)
    # stride validation
    from lova.data.common.make_r_gt import RgtCfg
    try:
        build(s, 3, PipelineCfg(transform=TransformCfg(size=256, train=False), r_gt=RgtCfg(stride=2), r_stride=3))
        raise AssertionError("bad stride must raise")
    except ValueError:
        pass
    print("build unit test OK")


if __name__ == "__main__":
    unit_test()
