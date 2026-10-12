"""lova.models.backbone: constant-R identity / full / half, spatial gating, valid masking, r_override,
R pyramid, stage shapes, gradient flow. Synthetic; --root runs one real batch through stem -> R predictor
-> backbone and prints the gate budget proxy for predicted / constant R.

    python tests/models/test_backbone.py
    python tests/models/test_backbone.py --root datasets/coco
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lova.models.backbone import (Backbone, BackboneCfg, GateCfg, GatedResBlock, STAGE_STRIDES, gate,  # noqa: E402
                                  r_pyramid, resolve_r, valid_pyramid, valid_to_stride)
from lova.models.common import count_params  # noqa: E402


def randomize(m: torch.nn.Module):
    """ResBlock bodies start at zero (identity); give them weights so the tests are not vacuous."""
    for p in m.parameters():
        if p.dim() == 4 and p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.05)
    return m


def unit_test():
    torch.manual_seed(0)
    B, c0, S = 2, 16, 64
    h = S // 4
    cfg = BackboneCfg(widths=(16, 24, 32, 48), depths=(2, 2, 2, 1))
    bb = randomize(Backbone(c0, cfg))
    f0 = torch.randn(B, c0, h, h)
    ones, zeros = torch.ones(B, 1, h, h), torch.zeros(B, 1, h, h)

    # ---- stage shapes ----
    out = bb(f0, ones)
    for s, w in zip(STAGE_STRIDES, cfg.widths):
        assert out[f"s{s}"].shape == (B, w, S // s, S // s), s
    # ---- gate function ----
    r = torch.tensor([0.0, 0.3, 0.5, 0.7, 1.0])
    assert torch.equal(gate(r, GateCfg("linear")), r)
    assert torch.equal(gate(r, GateCfg("hard", tau=0.5)), torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]))
    assert torch.equal(gate(r, GateCfg("none")), torch.ones(5))
    assert float(gate(torch.tensor(0.5), GateCfg("sigmoid"))) == 0.5
    for bad_cfg in (GateCfg("sigmoid", temp=0.0), GateCfg("tanh")):
        try:
            gate(r, bad_cfg); raise AssertionError("must raise")
        except ValueError:
            pass
    # ---- block level: identity at g=0, full residual at g=1, exactly halfway at g=0.5 ----
    blk = randomize(GatedResBlock(16)); x = torch.randn(B, 16, h, h)
    assert torch.equal(blk(x, zeros), x)
    full = blk(x, ones)
    assert not torch.allclose(full, x)
    assert torch.allclose(blk(x, 0.5 * ones), x + 0.5 * (full - x), atol=1e-6)
    # spatial: g = 1 on the left half, 0 on the right -> right half untouched, left half = full
    half = ones.clone(); half[..., h // 2:] = 0
    y = blk(x, half)
    assert torch.equal(y[..., h // 2:], x[..., h // 2:]) and torch.allclose(y[..., :h // 2], full[..., :h // 2])
    # ---- backbone level: R = 0 -> every stage is just its base path (blocks are identities) ----
    out0 = bb(f0, zeros)
    x = f0
    for i, stage in enumerate(bb.stages):
        x = stage.trans(x)
        assert torch.equal(out0[f"s{STAGE_STRIDES[i]}"], x), f"stage {i} not identity at R=0"
    # R = 1 == gate kind "none" on a fully valid canvas (dense reference; it ignores R AND padding)
    bb_none = Backbone(c0, BackboneCfg(cfg.widths, cfg.depths, GateCfg("none"))); bb_none.load_state_dict(bb.state_dict())
    out1 = bb(f0, ones); outn = bb_none(f0, 0.37 * ones)
    for k in out1:
        assert torch.allclose(out1[k], outn[k], atol=1e-6), k
    assert not torch.allclose(out1["s32"], out0["s32"])
    # R = 0.5 lies between: stage-0 output is exactly halfway for the FIRST block, and overall differs from both ends
    outh = bb(f0, 0.5 * ones)
    assert not torch.allclose(outh["s4"], out0["s4"]) and not torch.allclose(outh["s4"], out1["s4"])
    d0, d1 = (outh["s4"] - out0["s4"]).norm().detach(), (outh["s4"] - out1["s4"]).norm().detach()
    assert 0.2 < float(d0 / (d0 + d1)) < 0.8
    # hard gate: 0.3 -> identical to R=0, 0.7 -> identical to R=1
    bb_hard = Backbone(c0, BackboneCfg(cfg.widths, cfg.depths, GateCfg("hard"))); bb_hard.load_state_dict(bb.state_dict())
    assert torch.equal(bb_hard(f0, 0.3 * ones)["s32"], out0["s32"]) and torch.allclose(bb_hard(f0, 0.7 * ones)["s32"], out1["s32"])
    # ---- R pyramid: area average; a 2x2 lit block at stride 4 -> 1.0 at stride 8, 0.25 at stride 16 ----
    rg = torch.zeros(1, 1, h, h); rg[0, 0, 4:6, 8:10] = 1.0
    pyr = r_pyramid(rg)
    assert [p.shape[-1] for p in pyr] == [h, h // 2, h // 4, h // 8]
    assert float(pyr[1][0, 0, 2, 4]) == 1.0 and float(pyr[2][0, 0, 1, 2]) == 0.25 and float(pyr[1].sum()) == 1.0
    # ---- valid-normalised pyramid: a constant R = 1 stays 1 on every majority-valid coarse cell, even where the
    # coarse cell straddles the padding boundary (a plain average would read 0.5 / 0.25 there) ----
    S2 = 64; h2 = S2 // 4
    vb = torch.ones(1, S2, S2, dtype=torch.bool); vb[:, :, 40:] = False          # right 24 px padded: stride-4 cols 10.. are padding
    v4b = valid_to_stride(vb, 4)
    pyr_b = r_pyramid(resolve_r(torch.ones(1, 1, h2, h2), v4b, 1.0), v4b)
    vp = valid_pyramid(v4b)
    assert [p.shape[-1] for p in vp] == [h2, h2 // 2, h2 // 4, h2 // 8] and vp[2][0, 0, 0].tolist() == [1, 1, 0, 0] and vp[3][0, 0, 0].tolist() == [1, 0]   # exactly half valid -> not majority -> excluded
    for p_, v_ in zip(pyr_b, vp):
        assert torch.equal(p_, v_), "R = 1 must be exactly 1 on valid coarse cells and 0 elsewhere"
    raw = r_pyramid(torch.ones(1, 1, h2, h2), v4b)                     # unresolved R: same result, (RV)V = RV
    assert all(torch.equal(a_, b_) for a_, b_ in zip(raw, pyr_b))
    plain = r_pyramid(resolve_r(torch.ones(1, 1, h2, h2), v4b, 1.0))
    assert float(plain[2][0, 0, 0, 2]) == 0.5 and float(plain[3][0, 0, 0, 1]) == 0.25        # what the plain average gives
    # a half-lit coarse cell is still 0.5 when the whole cell is valid (normalisation only divides by the valid area)
    rg2 = torch.zeros(1, 1, h2, h2); rg2[:, :, :, :8] = 1.0
    assert float(r_pyramid(rg2, torch.ones(1, 1, h2, h2))[3][0, 0, 0, 0]) == 1.0 and float(r_pyramid(rg2, torch.ones(1, 1, h2, h2))[2][0, 0, 0, 1]) == 1.0
    rg3 = torch.zeros(1, 1, h2, h2); rg3[:, :, :, :2] = 1.0
    assert float(r_pyramid(rg3, torch.ones(1, 1, h2, h2))[2][0, 0, 0, 0]) == 0.5
    # ---- valid masking + r_override ----
    valid = torch.ones(B, S, S, dtype=torch.bool); valid[:, S // 2:] = False       # bottom half is padding
    v4 = valid_to_stride(valid, 4)
    assert v4.shape == (B, 1, h, h) and float(v4[:, :, :h // 2].min()) == 1 and float(v4[:, :, h // 2:].max()) == 0
    r_pred = torch.rand(B, 1, h, h)
    rg = resolve_r(r_pred, v4)
    assert torch.equal(rg[:, :, :h // 2], r_pred[:, :, :h // 2]) and float(rg[:, :, h // 2:].abs().max()) == 0
    assert torch.equal(resolve_r(r_pred, v4, 1.0), v4) and torch.equal(resolve_r(r_pred, None, 0.25), 0.25 * ones)
    assert torch.equal(resolve_r(r_pred, None, r_pred * 0.5), r_pred * 0.5)
    assert torch.equal(resolve_r(r_pred, None), r_pred)                                   # r_pred untouched
    # padded bottom half with R_pred = 1 everywhere: EXACTLY the R = 0 stream there (stage 0, no stride mixing),
    # while the lit half is computed. (It is not bit-identical to the all-R=1 run: GroupNorm statistics inside
    # the second block see the whole map, so gating is local on the stream but not on the normalisation.)
    out_pad = bb(f0, resolve_r(ones, v4))
    assert torch.equal(out_pad["s4"][:, :, h // 2:], out0["s4"][:, :, h // 2:])
    assert not torch.allclose(out_pad["s4"][:, :, :h // 2], out0["s4"][:, :, :h // 2])
    for bad in ((r_pred, v4[:1]), (r_pred, None, r_pred[:, :, :-1]), (r_pred, None, -0.5), (r_pred, None, 1.5),
                (r_pred, None, r_pred * float("nan"))):
        try:
            resolve_r(*bad); raise AssertionError("must raise")
        except ValueError:
            pass
    # ---- gate fraction: constant R gives that constant; valid restricts the average ----
    assert abs(float(bb.gate_budget(0.3 * ones)[0]) - 0.3) < 1e-6
    assert abs(float(bb.gate_budget(resolve_r(ones, v4), v4)[0]) - 1.0) < 1e-6 and abs(float(bb.gate_budget(resolve_r(ones, v4))[0]) - 0.5) < 1e-6
    # ---- gradient: reaches every parameter, and flows back into R (the predictor's path) ----
    rq = torch.rand(B, 1, h, h, requires_grad=True)
    f0q = f0.clone().requires_grad_(True)
    sum(o.sum() for o in bb(f0q, rq).values()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in bb.parameters())
    assert rq.grad is not None and float(rq.grad.abs().sum()) > 0 and f0q.grad is not None
    # default config: parameter count and a timing on 640^2
    full_bb = Backbone(64)
    n = count_params(full_bb)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    full_bb, x = full_bb.to(dev), torch.randn(2, 64, 160, 160, device=dev)
    with torch.no_grad():
        for _ in range(2): full_bb(x, torch.ones(2, 1, 160, 160, device=dev))
        if dev == "cuda": torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(5): full_bb(x, torch.ones(2, 1, 160, 160, device=dev))
        if dev == "cuda": torch.cuda.synchronize()
        ms = (time.perf_counter() - t) / 5 * 1e3
    print(f"backbone unit test OK   default (64,128,256,384)x(2,2,3,2): {n:.2f}M params, forward B=2 640^2 on {dev}: {ms:.1f} ms")


def real(a):
    from torch.utils.data import DataLoader
    from lova.data.coco.dataset import CocoDataset
    from lova.data.common.build import PipelineCfg, collate
    from lova.models.r_predictor import RPredictor
    from lova.models.stem import Stem
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = CocoDataset(a.root, a.split, PipelineCfg(), train=False)
    batch = next(iter(DataLoader(ds, batch_size=4, collate_fn=collate)))
    img, ptr, valid = batch["image"].to(dev), batch["pointer"].to(dev), batch["valid"].to(dev)
    stem, rp, bb = Stem(64).to(dev), RPredictor(64).to(dev), randomize(Backbone(64)).to(dev)
    torch.nn.init.normal_(rp.out.weight, std=0.5)                       # untrained but not constant
    with torch.no_grad():
        f0 = stem(img); r_pred = rp(f0, ptr, img.shape[-1]); v4 = valid_to_stride(valid, 4)
        print(f"real batch B={img.shape[0]}: valid fraction {float(v4.mean()):.3f}, r_pred mean {float(r_pred.mean()):.3f}, "
              f"mean over valid {float((r_pred * v4).sum() / v4.sum()):.3f}")
        for name, ov in (("predicted", None), ("R=0", 0.0), ("R=0.5", 0.5), ("R=1", 1.0), ("R_GT", batch["r_gt"][:, None].to(dev))):
            rg = resolve_r(r_pred, v4, ov)
            feats = bb(f0, rg, v4)
            gf = bb.gate_budget(rg, v4)
            print(f"  {name:9s} gate budget (valid) {[f'{float(g):.3f}' for g in gf]}   s32 |mean| {float(feats['s32'].abs().mean()):.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=None); p.add_argument("--split", default="val2017")
    a = p.parse_args()
    unit_test()
    if a.root:
        real(a)
