"""lova.losses.mask_loss: dynamic-kernel mask logits + soft dice with ignore, on synthetic tensors.

    python tests/losses/test_mask_loss.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lova.losses.mask_loss import MaskLossCfg, dynamic_masks, mask_loss, mask_stats, soft_dice_loss  # noqa: E402


def unit_test():
    torch.manual_seed(0)
    B, D, H8, W8, H4, W4 = 2, 4, 4, 4, 8, 8

    # ---- soft dice ----
    t = torch.zeros(1, H4, W4); t[0, 2:6, 2:6] = 1.0
    valid = torch.ones(H4, W4, dtype=torch.bool)
    big = torch.full((1, H4, W4), -20.0); big[0, 2:6, 2:6] = 20.0        # perfect prediction
    assert float(soft_dice_loss(big, t, valid)) < 1e-3
    wrong = -big                                                            # inverted
    assert float(soft_dice_loss(wrong, t, valid)) > 0.99
    # ignore: pixels with valid=False influence neither numerator nor denominator
    v2 = valid.clone(); v2[:, 6:] = False
    leak = big.clone(); leak[0, :, 6:] = 20.0                               # confidently wrong only in ignored columns
    assert abs(float(soft_dice_loss(leak, t, v2)) - float(soft_dice_loss(big, t, v2))) < 1e-6
    # soft targets are used as-is (0.5 target: p = 0.5 is the dice optimum, not p = 1)
    soft_t = torch.full((1, H4, W4), 0.5)
    assert float(soft_dice_loss(torch.zeros(1, H4, W4), soft_t, valid)) < float(soft_dice_loss(torch.full((1, H4, W4), 20.0), soft_t, valid))

    # ---- dynamic masks: kernel at the chosen cell dotted with the feature ----
    km = torch.randn(D, H8, W8); mf = torch.randn(D, H4, W4)
    out = dynamic_masks(km, mf, torch.tensor([5, 10]))
    assert out.shape == (2, H4, W4)
    assert torch.allclose(out[0], (km.flatten(1)[:, 5][:, None, None] * mf).sum(0), atol=1e-5)

    # ---- batch loss with per-image local indices ----
    kernel_map = torch.randn(B, D, H8, W8, requires_grad=True)
    mask_feat = torch.randn(B, D, H4, W4, requires_grad=True)
    masks = [torch.zeros(2, H4, W4), torch.zeros(1, H4, W4)]
    masks[0][0, 0:4, 0:4] = 1.0; masks[0][1, 4:8, 4:8] = 1.0; masks[1][0, 2:6, 2:6] = 1.0
    pos_index = [torch.tensor([0, 1, 15]), torch.tensor([10])]            # local flat indices on the 4x4 grid
    pos_inst = [torch.tensor([0, 0, 1]), torch.tensor([0])]
    mask_valid = torch.ones(B, H4, W4, dtype=torch.bool)
    l = mask_loss(kernel_map, mask_feat, pos_index, pos_inst, masks, mask_valid)
    assert 0 <= float(l.detach()) <= 1
    l.backward()
    assert torch.isfinite(kernel_map.grad).all() and torch.isfinite(mask_feat.grad).all()
    # only the positive cells' kernels receive gradient
    g = kernel_map.grad.flatten(2)
    assert bool((g[0, :, [0, 1, 15]] != 0).any()) and bool((g[0, :, 2] == 0).all()) and bool((g[1, :, 10] != 0).any()) and bool((g[1, :, 0] == 0).all())
    # equals the mean over all positives of the per-instance dice losses computed by hand
    per = []
    for b in range(B):
        lg = dynamic_masks(kernel_map[b].detach(), mask_feat[b].detach(), pos_index[b])
        per.append(soft_dice_loss(lg, masks[b][pos_inst[b]], mask_valid[b]))
    assert abs(float(l) - float(torch.cat(per).mean())) < 1e-6
    # an image without positives is skipped; a batch without any positive gives 0 with a gradient path
    l2 = mask_loss(kernel_map, mask_feat, [pos_index[0], torch.zeros(0, dtype=torch.long)], [pos_inst[0], torch.zeros(0, dtype=torch.long)], masks, mask_valid)
    assert abs(float(l2) - float(per[0].mean())) < 1e-6
    km2 = torch.randn(B, D, H8, W8, requires_grad=True)
    l0 = mask_loss(km2, mask_feat.detach(), [torch.zeros(0, dtype=torch.long)] * B, [torch.zeros(0, dtype=torch.long)] * B, masks, mask_valid)
    assert l0 == 0; l0.backward(); assert km2.grad is not None
    # a perfectly fitted kernel drives the loss to ~0: construct mask_feat with the target as one channel
    mf_fit = torch.zeros(1, D, H4, W4); mf_fit[0, 0] = masks[1][0] * 40 - 20
    km_fit = torch.zeros(1, D, H8, W8); km_fit[0, 0, 2, 2] = 1.0           # cell 10 = (2, 2)
    assert float(mask_loss(km_fit, mf_fit, [torch.tensor([10])], [torch.tensor([0])], [masks[1]], mask_valid[:1])) < 1e-3
    # bf16 inputs are computed in float32
    assert torch.isfinite(mask_loss(kernel_map.detach().to(torch.bfloat16), mask_feat.detach().to(torch.bfloat16), pos_index, pos_inst, masks, mask_valid))
    # shape / index contract: D mismatch, missing image, out-of-range cell, NEGATIVE cell, length mismatch, bad instance
    for args in ((kernel_map, mask_feat[:, :2], pos_index, pos_inst, masks, mask_valid),
                 (kernel_map, mask_feat, pos_index[:1], pos_inst, masks, mask_valid),
                 (kernel_map, mask_feat, [torch.tensor([99]), pos_index[1]], pos_inst, masks, mask_valid),
                 (kernel_map, mask_feat, [torch.tensor([-1]), pos_index[1]], [torch.tensor([0]), pos_inst[1]], masks, mask_valid),
                 (kernel_map, mask_feat, [torch.tensor([0, 1]), pos_index[1]], [torch.tensor([0]), pos_inst[1]], masks, mask_valid),
                 (kernel_map, mask_feat, pos_index, [torch.tensor([0, 0, 5]), pos_inst[1]], masks, mask_valid)):
        try:
            mask_loss(*args)
            raise AssertionError("must raise")
        except ValueError:
            pass
    st = mask_stats(kernel_map.detach(), mask_feat.detach(), pos_index, pos_inst, masks, mask_valid, loss=l.detach())
    assert st["n_pos_cells"] == 4 and abs(float(st["dice"]) - (1 - float(l.detach()))) < 1e-6
    st0 = mask_stats(kernel_map.detach(), mask_feat.detach(), [torch.zeros(0, dtype=torch.long)] * B, [torch.zeros(0, dtype=torch.long)] * B, masks, mask_valid)
    assert st0["n_pos_cells"] == 0 and torch.isnan(st0["dice"]), "no positives -> dice undefined (NaN), not 1"
    # soft_dice_loss contract
    try:
        soft_dice_loss(torch.zeros(1, H4, W4), torch.zeros(1, H4, W4 - 1), valid)
        raise AssertionError("must raise")
    except ValueError:
        pass
    print("mask_loss unit test OK")


if __name__ == "__main__":
    unit_test()
