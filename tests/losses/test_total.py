"""lova.losses.total: weighted sum of the three losses on a synthetic model output + a real collated batch.

    python tests/losses/test_total.py                              # synthetic
    python tests/losses/test_total.py --root datasets/coco         # + one real batch through random outputs
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lova.losses.heat_loss import heat_loss  # noqa: E402
from lova.losses.mask_loss import mask_loss  # noqa: E402
from lova.losses.r_loss import r_loss  # noqa: E402
from lova.losses.total import REQUIRED_OUT, TotalLossCfg, total_loss, total_stats  # noqa: E402


def fake_out(B, K, D, S, device="cpu"):
    return {"r_pred": torch.rand(B, 1, S // 4, S // 4, device=device, requires_grad=True),
            "heat_logits": torch.randn(B, K, S // 8, S // 8, device=device, requires_grad=True),
            "kernel_map": torch.randn(B, D, S // 8, S // 8, device=device, requires_grad=True),
            "mask_feat": torch.randn(B, D, S // 4, S // 4, device=device, requires_grad=True)}


def fake_batch(B, K, S):
    H4, H8 = S // 4, S // 8
    heat = torch.zeros(B, K, H8, W8 := H8)
    heat[:, 0, 2, 2] = 1.0
    masks = [torch.zeros(1, H4, H4) for _ in range(B)]
    for m in masks:
        m[0, 4:12, 4:12] = 1.0
    return {"r_gt": torch.rand(B, H4, H4), "r_valid": torch.ones(B, H4, H4, dtype=torch.bool),
            "heat": heat, "heat_valid": torch.ones(B, H8, W8, dtype=torch.bool),
            "pos_index": [torch.tensor([2 * H8 + 2]) for _ in range(B)], "pos_inst": [torch.tensor([0]) for _ in range(B)],
            "masks_s4": masks, "mask_valid": torch.ones(B, H4, H4, dtype=torch.bool)}


def unit_test():
    torch.manual_seed(0)
    B, K, D, S = 2, 3, 4, 64
    out, batch = fake_out(B, K, D, S), fake_batch(B, K, S)
    cfg = TotalLossCfg(w_r=2.0, w_heat=0.5, w_mask=3.0)
    L = total_loss(out, batch, cfg)
    lr = r_loss(out["r_pred"], batch["r_gt"], batch["r_valid"])
    lh = heat_loss(out["heat_logits"], batch["heat"], batch["heat_valid"])
    lm = mask_loss(out["kernel_map"], out["mask_feat"], batch["pos_index"], batch["pos_inst"], batch["masks_s4"], batch["mask_valid"])
    assert torch.allclose(L["total"], 2.0 * lr + 0.5 * lh + 3.0 * lm)
    assert torch.allclose(L["r_raw"], lr.detach()) and torch.allclose(L["heat_raw"], lh.detach()) and torch.allclose(L["mask_raw"], lm.detach())
    assert L["total"].requires_grad and not L["r_raw"].requires_grad
    L["total"].backward()
    assert all(torch.isfinite(out[k].grad).all() for k in REQUIRED_OUT)
    # defaults are 1, 1, 1 placeholders
    L1 = total_loss({k: v.detach() for k, v in out.items()}, batch)
    assert torch.allclose(L1["total"], lr.detach() + lh.detach() + lm.detach())
    # stats reuse the computed mask loss, all 0-d tensors
    st = total_stats(out, batch, L1)
    for k in ("r_mae", "r_lit_pred", "r_lit_gt", "n_pos", "p_pos", "p_neg", "n_pos_cells", "dice"):
        assert k in st and torch.is_tensor(st[k]) and st[k].dim() == 0, k
    assert abs(float(st["dice"]) - (1 - float(L1["mask_raw"]))) < 1e-6
    # missing keys are reported
    for bad_out in ({k: v for k, v in out.items() if k != "mask_feat"},):
        try:
            total_loss(bad_out, batch)
            raise AssertionError("must raise")
        except KeyError:
            pass
    print("total loss unit test OK")


def real_batch(root):
    from torch.utils.data import DataLoader
    from lova.data.coco.dataset import CocoDataset
    from lova.data.common.build import PipelineCfg, collate
    ds = CocoDataset(root, "val2017", PipelineCfg(), train=False)
    batch = next(iter(DataLoader(ds, batch_size=4, collate_fn=collate)))
    B, K, S = batch["heat"].shape[0], ds.num_classes, batch["image"].shape[-1]
    out = fake_out(B, K, 64, S)
    L = total_loss(out, batch)
    st = total_stats(out, batch, L)
    print(f"real batch (B={B}, {sum(len(p) for p in batch['pos_index'])} positive cells), random outputs:")
    print("  raw losses:  " + "  ".join(f"{k}={float(L[k + '_raw']):.4f}" for k in ("r", "heat", "mask")))
    print("  stats:       " + "  ".join(f"{k}={float(v):.3f}" for k, v in st.items()))
    L["total"].backward()
    print("  grad norms:  " + "  ".join(f"{k}={float(out[k].grad.norm()):.3f}" for k in REQUIRED_OUT))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=None)
    a = p.parse_args()
    unit_test()
    if a.root:
        real_batch(a.root)
