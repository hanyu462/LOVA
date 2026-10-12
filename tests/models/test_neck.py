"""lova.models.neck: FPN shapes, top-down path, gradient flow. Synthetic only.

    python tests/models/test_neck.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lova.models.backbone import Backbone  # noqa: E402
from lova.models.common import count_params  # noqa: E402
from lova.models.neck import FPN, NeckCfg  # noqa: E402


def unit_test():
    torch.manual_seed(0)
    B, S, c = 2, 64, 32
    widths = {"s4": 16, "s8": 24, "s16": 32, "s32": 48}
    feats = {k: torch.randn(B, w, S // int(k[1:]), S // int(k[1:]), requires_grad=True) for k, w in widths.items()}
    neck = FPN(widths, NeckCfg(c=c))
    out = neck(feats)
    assert set(out) == {"p8", "p4"} and out["p8"].shape == (B, c, S // 8, S // 8) and out["p4"].shape == (B, c, S // 4, S // 4)
    # top-down: a change in s32 alone reaches p8 and p4
    feats2 = dict(feats); feats2["s32"] = feats["s32"].detach() + 1.0
    out2 = neck(feats2)
    assert not torch.allclose(out2["p8"], out["p8"]) and not torch.allclose(out2["p4"], out["p4"])
    # a change in s4 alone reaches p4 but not p8
    feats3 = dict(feats); feats3["s4"] = feats["s4"].detach() + 1.0
    out3 = neck(feats3)
    assert torch.allclose(out3["p8"], out["p8"]) and not torch.allclose(out3["p4"], out["p4"])
    (out["p8"].sum() + out["p4"].sum()).backward()
    assert all(p.grad is not None for p in neck.parameters()) and all(f.grad is not None for f in feats.values())
    for bad_ctor in (({"s4": 16, "s8": 24}, NeckCfg()), (widths, NeckCfg(c=0)), (widths, NeckCfg(c=128.5)), ({**widths, "s8": 0}, NeckCfg())):
        try:
            FPN(*bad_ctor); raise AssertionError("must raise")
        except ValueError:
            pass
    # input contract: missing level, wrong channels, batch mismatch, broken 2x pyramid, 3-D
    d = {k: v.detach() for k, v in feats.items()}
    for bad in ({k: v for k, v in d.items() if k != "s16"},
                {**d, "s8": d["s8"][:, :-1]},
                {**d, "s32": d["s32"][:1]},
                {**d, "s16": d["s16"][:, :, :-1]},
                {**d, "s4": d["s4"][0]}):
        try:
            neck(bad); raise AssertionError("must raise")
        except ValueError:
            pass
    # plugs onto the backbone's declared channels
    bb = Backbone(64)
    full = FPN(bb.out_channels)
    o = full(bb(torch.randn(1, 64, 32, 32), torch.ones(1, 1, 32, 32)))
    assert o["p8"].shape == (1, 128, 16, 16) and o["p4"].shape == (1, 128, 32, 32)
    print(f"neck unit test OK   default FPN c=128 on (64,128,256,384): {count_params(full):.2f}M")


if __name__ == "__main__":
    unit_test()
