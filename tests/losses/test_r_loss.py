"""lova.losses.r_loss: values and gradients on synthetic tensors.

    python tests/losses/test_r_loss.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lova.losses.r_loss import RLossCfg, r_loss, r_stats  # noqa: E402


def unit_test():
    B, H, W = 2, 8, 8
    r_gt = torch.rand(B, H, W)
    valid = torch.ones(B, H, W, dtype=torch.bool)
    valid[:, 6:, :] = False                       # padding rows

    # exact prediction -> 0; constant offset -> offset^2 on valid cells only
    assert r_loss(r_gt, r_gt, valid) == 0
    assert abs(float(r_loss(r_gt + 0.1, r_gt, valid)) - 0.01) < 1e-6
    # padding cells do not count: wrong values there change nothing
    bad = r_gt.clone(); bad[:, 6:, :] = 1.0
    assert r_loss(bad, r_gt, valid) == 0
    # [B,1,H,W] input accepted; l1 variant; batch-level mean (not per-image)
    assert torch.allclose(r_loss(r_gt[:, None] + 0.1, r_gt, valid), torch.tensor(0.01), atol=1e-6)
    assert torch.allclose(r_loss(r_gt + 0.1, r_gt, valid, RLossCfg("l1")), torch.tensor(0.1), atol=1e-6)
    v2 = valid.clone(); v2[1] = False; v2[1, 0, 0] = True          # image 1 has a single valid cell
    p = r_gt.clone(); p[0] += 0.1; p[1, 0, 0] += 0.3
    expect = (48 * 0.01 + 1 * 0.09) / 49
    assert abs(float(r_loss(p, r_gt, v2)) - expect) < 1e-6
    # gradient: flows to valid cells only, zero elsewhere; no valid cell -> loss 0, finite grad
    pred = r_gt.clone().requires_grad_(True)
    r_loss(pred + 0.2, r_gt, valid).backward()
    assert bool((pred.grad[:, 6:, :] == 0).all()) and bool((pred.grad[:, :6, :] != 0).all())
    pred2 = torch.rand(B, H, W, requires_grad=True)
    l0 = r_loss(pred2, r_gt, torch.zeros_like(valid))
    assert l0 == 0
    l0.backward()
    assert torch.isfinite(pred2.grad).all()
    # shape mismatch is an error
    try:
        r_loss(r_gt[:, :4], r_gt, valid)
        raise AssertionError("must raise")
    except ValueError:
        pass
    st = r_stats(r_gt + 0.1, r_gt, valid)
    assert abs(st["r_mae"] - 0.1) < 1e-6 and 0 <= st["r_lit_pred"] <= 1 and 0 <= st["r_lit_gt"] <= 1
    print("r_loss unit test OK")


if __name__ == "__main__":
    unit_test()
