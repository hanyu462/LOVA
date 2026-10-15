"""lova.models.head: output shapes against the ModelOutput contract, initialisation (heat prior, small
kernels -> unsaturated mask logits), coordinate channels, gradient flow from heat_loss and mask_loss.

    python tests/models/test_head.py
    python tests/models/test_head.py --root datasets/coco     # real batch through stem -> R -> backbone -> neck -> head at init
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lova.losses.heat_loss import heat_loss, heat_stats  # noqa: E402
from lova.losses.mask_loss import dynamic_masks, mask_loss  # noqa: E402
from lova.models.common import count_params  # noqa: E402
from lova.models.head import Head, HeadCfg, norm_coords  # noqa: E402
from lova.models.output import ModelOutput  # noqa: E402


def unit_test():
    torch.manual_seed(0)
    B, K, S, c, d = 2, 5, 64, 32, 8
    neck = {"p8": torch.randn(B, c, S // 8, S // 8, requires_grad=True), "p4": torch.randn(B, c, S // 4, S // 4, requires_grad=True)}
    head = Head(K, HeadCfg(c=c, d=d))
    out = head(neck)
    # shapes satisfy the contract (with a dummy r_pred)
    ModelOutput(torch.rand(B, 1, S // 4, S // 4), **out).check((B, 3, S, S), K)
    # coords: (-1, 1) range, x varies along width, y along height, consistent across strides
    c8, c4 = norm_coords(S // 8, S // 8, 8, "cpu", torch.float32), norm_coords(S // 4, S // 4, 4, "cpu", torch.float32)
    assert float(c8.min()) > -1 and float(c8.max()) < 1 and torch.allclose(c8[0, 0, 0], c8[0, 0, -1]) and torch.allclose(c8[0, 1, :, 0], c8[0, 1, :, -1])
    assert float(c8[0, 0, 0, 1] - c8[0, 0, 0, 0]) > 0 and float(c8[0, 1, 1, 0] - c8[0, 1, 0, 0]) > 0
    assert abs(float(c4[0, 0, 0, 2 * 3] + c4[0, 0, 0, 2 * 3 + 1]) / 2 - float(c8[0, 0, 0, 3])) < 1e-6   # stride-4 pair centre = stride-8 cell
    # init: heat probability ~ prior everywhere, kernels small, mask logits unsaturated
    p = torch.sigmoid(out["heat_logits"])
    p = p.detach()
    assert abs(float(p.mean()) - 0.1) < 0.02 and float(p.std()) < 0.02
    assert float(out["kernel_map"].abs().mean()) < 0.1
    logits = dynamic_masks(out["kernel_map"][0], out["mask_feat"][0], torch.arange(out["kernel_map"].shape[2] ** 2))
    assert float(logits.abs().mean()) < 1.0, float(logits.abs().mean())
    # gradients from both task losses reach every head parameter and both neck inputs
    H8 = S // 8
    heat = torch.zeros(B, K, H8, H8); heat[:, 0, 2, 2] = 1.0
    hv = torch.ones(B, H8, H8, dtype=torch.bool)
    masks = [torch.zeros(1, S // 4, S // 4) for _ in range(B)]
    for m in masks: m[0, 4:12, 4:12] = 1.0
    lh = heat_loss(out["heat_logits"], heat, hv)
    lm = mask_loss(out["kernel_map"], out["mask_feat"], [torch.tensor([2 * H8 + 2])] * B, [torch.tensor([0])] * B, masks, torch.ones(B, S // 4, S // 4, dtype=torch.bool))
    (lh + lm).backward()
    assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in head.parameters())
    assert neck["p8"].grad is not None and neck["p4"].grad is not None
    assert float(head.kernel_out.weight.grad.abs().sum()) > 0 and float(head.mask_branch[0][0].weight.grad.abs().sum()) > 0
    # the prior drives the initial heat loss down (vs prior 0.5)
    head5 = Head(K, HeadCfg(c=c, d=d, prior=0.5))
    assert float(heat_loss(head5(neck)["heat_logits"], heat, hv)) > 3 * float(lh.detach())
    for bad in ({"p8": neck["p8"][:, :c - 1], "p4": neck["p4"]},                 # channels
                {"p8": neck["p8"], "p4": neck["p4"][:, :, :-1]},                 # p4 not 2x p8
                {"p8": neck["p8"][:1], "p4": neck["p4"]},                        # batch mismatch
                {"p8": neck["p8"][0], "p4": neck["p4"]}):                        # 3-D
        try:
            head(bad); raise AssertionError("must raise")
        except ValueError:
            pass
    for bad_args in ((0, HeadCfg()), (K, HeadCfg(prior=1.0)), (K, HeadCfg(prior=0.0)), (K, HeadCfg(d=0)), (K, HeadCfg(out_std=-1))):
        try:
            Head(*bad_args); raise AssertionError("must raise")
        except ValueError:
            pass
    print(f"head unit test OK   default head (c=128, d=64, K=80): {count_params(Head(80)):.2f}M   init heat p {float(p.mean()):.3f}, mask logit |mean| {float(logits.abs().mean()):.3f}")


def real(a):
    from torch.utils.data import DataLoader
    from lova.data.coco.dataset import CocoDataset
    from lova.data.common.build import PipelineCfg, collate
    from lova.losses.mask_loss import mask_stats
    from lova.models.backbone import Backbone, resolve_r, valid_to_stride
    from lova.models.neck import FPN
    from lova.models.r_predictor import RPredictor
    from lova.models.stem import Stem
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = CocoDataset(a.root, a.split, PipelineCfg(), train=False)
    batch = next(iter(DataLoader(ds, batch_size=4, collate_fn=collate)))
    K = ds.num_classes
    img, ptr, valid = batch["image"].to(dev), batch["pointer"].to(dev), batch["valid"].to(dev)
    stem, rp, bb = Stem(64).to(dev), RPredictor(64).to(dev), Backbone(64).to(dev)
    neck, head = FPN(bb.out_channels).to(dev), Head(K).to(dev)
    with torch.no_grad():
        f0 = stem(img)
        r_pred = rp(f0, ptr, img.shape[-1])
        out = head(neck(bb(f0, resolve_r(r_pred, valid_to_stride(valid, 4)))))
        ModelOutput(r_pred, **out).check(tuple(img.shape), K)
        hl = heat_loss(out["heat_logits"], batch["heat"].to(dev), batch["heat_valid"].to(dev))
        hs = heat_stats(out["heat_logits"], batch["heat"].to(dev), batch["heat_valid"].to(dev))
        ml = mask_loss(out["kernel_map"], out["mask_feat"], batch["pos_index"], batch["pos_inst"], batch["masks_s4"], batch["mask_valid"].to(dev))
        ms = mask_stats(out["kernel_map"], out["mask_feat"], batch["pos_index"], batch["pos_inst"], batch["masks_s4"], batch["mask_valid"].to(dev), loss=ml)
        logits = torch.cat([dynamic_masks(out["kernel_map"][b], out["mask_feat"][b], batch["pos_index"][b].to(dev)) for b in range(img.shape[0])])
    n = count_params(stem) + count_params(rp) + count_params(bb) + count_params(neck) + count_params(head)
    print(f"real batch at init ({dev}, total {n:.2f}M params): heat loss {float(hl):.1f} (p_pos {float(hs['p_pos']):.3f}, p_neg {float(hs['p_neg']):.3f})   "
          f"mask loss {float(ml):.3f} (dice {float(ms['dice']):.3f}, {int(ms['n_pos_cells'])} cells, mask logit |mean| {float(logits.abs().mean()):.3f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=None); p.add_argument("--split", default="val2017")
    a = p.parse_args()
    unit_test()
    if a.root:
        real(a)
