"""lova.losses.heat_loss: positive / shoulder / negative / ignored cells, normalisation, monotonicity, gradients.

    python tests/losses/test_heat_loss.py
"""
from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lova.losses.heat_loss import HeatLossCfg, heat_loss, heat_stats  # noqa: E402


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def unit_test():
    B, K, H, W = 1, 2, 6, 6
    heat = torch.zeros(B, K, H, W)
    heat[0, 0, 2, 2] = 1.0                       # positive, class 0
    heat[0, 0, 2, 3] = 0.6                       # gaussian shoulder of that centre
    heat[0, 1, 4, 4] = 1.0                       # positive, class 1
    valid = torch.ones(B, H, W, dtype=torch.bool)
    valid[0, 0, :] = False                       # an ignored row (crowd / padding)

    # one hand-computed case: all logits = logit(0.5) -> p = 0.5 everywhere
    logits = torch.full((B, K, H, W), logit(0.5))
    n_pos = 2
    pos_term = n_pos * (0.5 ** 2) * (-math.log(0.5))
    n_valid_cells = (H - 1) * W                                   # 30 valid cells per class
    neg_cells = K * n_valid_cells - n_pos - 1                     # all valid non-positive cells except the shoulder
    neg_term = neg_cells * (0.5 ** 2) * (-math.log(0.5)) + (1 - 0.6) ** 4 * (0.5 ** 2) * (-math.log(0.5))
    expect = (pos_term + neg_term) / n_pos
    assert abs(float(heat_loss(logits, heat, valid)) - expect) < 1e-5, (float(heat_loss(logits, heat, valid)), expect)

    # ignored cells contribute nothing: put extreme wrong logits there
    bad = logits.clone(); bad[0, :, 0, :] = 20.0
    assert abs(float(heat_loss(bad, heat, valid)) - expect) < 1e-5
    # a positive on an ignored cell does not count as positive (N_pos follows valid cells)
    h2 = heat.clone(); h2[0, 1, 0, 0] = 1.0
    assert abs(float(heat_loss(logits, h2, valid)) - expect) < 1e-5

    # monotonicity: at a positive, loss falls as p -> 1; at a negative, loss falls as p -> 0
    def loss_with(cell_logit, c, y, x):
        lg = torch.full((B, K, H, W), -6.0)                      # everything else confidently negative
        lg[0, c, y, x] = cell_logit
        return float(heat_loss(lg, heat, valid))
    pos_curve = [loss_with(v, 0, 2, 2) for v in (-2.0, 0.0, 2.0, 4.0)]
    assert all(a > b for a, b in zip(pos_curve, pos_curve[1:])), pos_curve
    neg_curve = [loss_with(v, 0, 3, 3) for v in (4.0, 2.0, 0.0, -2.0)]
    assert all(a > b for a, b in zip(neg_curve, neg_curve[1:])), neg_curve
    # the shoulder is penalised less than a plain negative at the same prediction
    assert loss_with(2.0, 0, 2, 3) < loss_with(2.0, 0, 3, 3)
    # near-perfect prediction -> near-zero loss
    perfect = torch.full((B, K, H, W), -12.0); perfect[0, 0, 2, 2] = 12.0; perfect[0, 1, 4, 4] = 12.0
    assert float(heat_loss(perfect, heat, valid)) < 1e-3

    # gradients: present at valid cells, zero at ignored cells, finite; no positives -> finite (N_pos clamp)
    lg = logits.clone().requires_grad_(True)
    heat_loss(lg, heat, valid).backward()
    assert bool((lg.grad[0, :, 0, :] == 0).all()) and bool((lg.grad[0, :, 1:, :] != 0).any()) and torch.isfinite(lg.grad).all()
    lg2 = torch.randn(B, K, H, W, requires_grad=True)
    l_nopos = heat_loss(lg2, torch.zeros_like(heat), valid)
    assert torch.isfinite(l_nopos); l_nopos.backward(); assert torch.isfinite(lg2.grad).all()
    # bf16 logits are handled in float32
    assert torch.isfinite(heat_loss(logits.to(torch.bfloat16), heat, valid))

    # shape contract
    for bad_args in ((logits[:, :1], heat, valid), (logits, heat, valid[:, None]), (logits[0], heat[0], valid[0])):
        try:
            heat_loss(*bad_args)
            raise AssertionError("must raise")
        except ValueError:
            pass
    st = heat_stats(logits, heat, valid)
    assert st["n_pos"] == 2 and abs(float(st["p_pos"]) - 0.5) < 1e-6 and all(torch.is_tensor(v) for v in st.values())
    print("heat_loss unit test OK")


if __name__ == "__main__":
    unit_test()
