"""lova.models.lova: the whole model against total_loss.

    python tests/models/test_lova.py                        # synthetic: forward / shapes / backward / override / valid gating
    python tests/models/test_lova.py --root datasets/coco   # 9-5a one real batch at init (+ saturation stats, R sweeps)
    python tests/models/test_lova.py --root datasets/coco --warmup 30 --batches 20
        9-5b: after a short warm-up, per-loss gradient norms at F0 and on the stem over several batches
        (the quantity the loss weights should be set from; L_R never reaches the backbone)
    python tests/models/test_lova.py --root datasets/coco --overfit 8 --steps 600 [--out dir]
        what the model learns, visibly: overfit the whole model on n images and draw, per image,
        image + pointer | R_GT | r_pred | heat GT | heat pred | GT masks (at positive cells) | predicted masks
        (a learning-signal check on seen images, not an accuracy result)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lova.losses.mask_loss import dynamic_masks  # noqa: E402
from lova.losses.total_loss import TotalLossCfg, total_loss, total_stats  # noqa: E402
from lova.models.backbone import BackboneCfg, GateCfg  # noqa: E402
from lova.models.common import count_params  # noqa: E402
from lova.models.head import HeadCfg  # noqa: E402
from lova.models.lova import Lova, LovaCfg  # noqa: E402
from lova.models.neck import NeckCfg  # noqa: E402
from lova.models.r_predictor import RPredictorCfg  # noqa: E402

SMALL = LovaCfg(c0=16, r_predictor=RPredictorCfg(c=16), backbone=BackboneCfg((16, 24, 32, 48), (1, 1, 1, 1)),
                neck=NeckCfg(c=32), head=HeadCfg(c=32, d=8))


def fake_batch(B, K, S):
    H4, H8 = S // 4, S // 8
    heat = torch.zeros(B, K, H8, H8); heat[:, 0, 2, 2] = 1.0
    masks = [torch.zeros(1, H4, H4) for _ in range(B)]
    for m in masks: m[0, 4:12, 4:12] = 1.0
    valid = torch.ones(B, S, S, dtype=torch.bool); valid[:, S // 2:] = False          # bottom half padded
    return {"image": torch.randn(B, 3, S, S), "pointer": torch.tensor([[10.0, 12.0]] * B), "valid": valid,
            "r_gt": torch.rand(B, H4, H4), "r_valid": torch.ones(B, H4, H4, dtype=torch.bool),
            "heat": heat, "heat_valid": torch.ones(B, H8, H8, dtype=torch.bool),
            "pos_index": [torch.tensor([2 * H8 + 2])] * B, "pos_inst": [torch.tensor([0])] * B,
            "masks_s4": masks, "mask_valid": torch.ones(B, H4, H4, dtype=torch.bool)}


def unit_test():
    torch.manual_seed(0)
    B, K, S = 2, 5, 64
    model = Lova(K, SMALL)
    batch = fake_batch(B, K, S)
    out = model(batch["image"], batch["pointer"], batch["valid"])
    out.check((B, 3, S, S), K)
    assert out.r_gate is not None and out.r_gate.shape == out.r_pred.shape
    # valid gating: padded half of r_gate is 0, r_pred itself untouched (constant init_r there)
    assert float(out.r_gate.detach()[:, :, S // 8:].abs().max()) == 0 and float(out.r_pred.detach()[:, :, S // 8:].min()) > 0.4
    # total loss: finite, backward reaches every part. The R predictor's upstream layers and the stem get
    # nothing from L_R on the first step (out.weight = 0) but the stem gets gradient from the task losses.
    L = total_loss(out.as_dict(), batch)
    assert torch.isfinite(L["total"])
    L["total"].backward()
    for name, part in model.parts().items():
        g = sum(float(p.grad.abs().sum()) for p in part.parameters() if p.grad is not None)
        assert g > 0, f"no gradient in {name}"
    # r_override: constant 1 / 0 and a tensor; output differs from the predicted-R run once blocks are non-zero
    for p in model.backbone.parameters():
        if p.dim() == 4 and p.abs().sum() == 0: torch.nn.init.normal_(p, std=0.05)
    with torch.no_grad():
        o_pred = model(batch["image"], batch["pointer"], batch["valid"])
        o1 = model(batch["image"], batch["pointer"], batch["valid"], r_override=1.0)
        o0 = model(batch["image"], batch["pointer"], batch["valid"], r_override=0.0)
        ot = model(batch["image"], batch["pointer"], batch["valid"], r_override=torch.rand_like(o_pred.r_pred))
    assert torch.equal(o1.r_pred, o_pred.r_pred) and torch.equal(o0.r_pred, o_pred.r_pred)       # r_pred is never overridden
    assert float(o1.r_gate[:, :, :S // 8].min()) == 1 and float(o1.r_gate[:, :, S // 8:].max()) == 0
    assert float(o0.r_gate.max()) == 0
    assert not torch.allclose(o1.heat_logits, o0.heat_logits) and not torch.allclose(ot.heat_logits, o0.heat_logits)
    # no valid (debug): padding is NOT gated
    with torch.no_grad():
        on = model(batch["image"], batch["pointer"], None, r_override=1.0)
    assert float(on.r_gate.min()) == 1
    # intermediates expose F0 for the gradient diagnostics
    out2, inter = model.forward_with_intermediates(batch["image"], batch["pointer"], batch["valid"])
    assert inter["f0"].shape == (B, SMALL.c0, S // 4, S // 4) and set(inter["feats"]) == {"s4", "s8", "s16", "s32"}
    for bad in ((batch["image"], batch["pointer"][:1], batch["valid"]),                       # pointer batch
                (batch["image"], batch["pointer"], batch["valid"][:, :-1]),                   # valid shape
                (batch["image"], batch["pointer"], batch["valid"].float()),                   # valid not bool
                (batch["image"][:, :, :, :S - 32], batch["pointer"], batch["valid"][:, :, :S - 32]),   # not square
                (batch["image"][:, :2], batch["pointer"], batch["valid"])):                   # not RGB
        try:
            model(*bad); raise AssertionError("must raise")
        except ValueError:
            pass
    if torch.cuda.is_available():                                                              # device mismatch
        try:
            model.cuda()(batch["image"].cuda(), batch["pointer"], batch["valid"].cuda()); raise AssertionError("must raise")
        except ValueError:
            pass
        model.cpu()
    try:
        Lova(K, LovaCfg(neck=NeckCfg(c=64), head=HeadCfg(c=128))); raise AssertionError("must raise")
    except ValueError:
        pass
    full = Lova(80)
    print(f"lova unit test OK   default model (K=80): {count_params(full):.2f}M = " +
          " + ".join(f"{k} {count_params(m):.2f}" for k, m in full.parts().items()))


def to_dev(batch, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}


def saturation(out, batch, dev):
    with torch.no_grad():
        z = torch.cat([dynamic_masks(out.kernel_map[b], out.mask_feat[b], batch["pos_index"][b].to(dev)) for b in range(out.kernel_map.shape[0])]).abs().flatten()
        sub = z[torch.randperm(z.numel(), device=z.device)[:1_000_000]] if z.numel() > 1_000_000 else z   # quantile() caps at 16M
        return float(z.mean()), float(sub.quantile(0.95)), float((z > 5).float().mean())


@torch.no_grad()
def sweep(model, batch, title):
    print(f"  r_override sweep {title} (budget = gated residual-body proxy over valid cells):")
    for name, ov in (("predicted", None), ("R=0", 0.0), ("R=0.5", 0.5), ("R=1", 1.0)):
        o, inter = model.forward_with_intermediates(batch["image"], batch["pointer"], batch["valid"], r_override=ov)
        Lo = total_loss(o.as_dict(), batch)
        print(f"    {name:9s} budget {float(model.backbone.gate_budget(o.r_gate, inter['valid_s4']).mean()):.3f}  heat {float(Lo['heat_raw']):.3f}  mask {float(Lo['mask_raw']):.4f}")


def real(a):
    from torch.utils.data import DataLoader
    from lova.data.coco.dataset import CocoDataset, worker_init
    from lova.data.common.build import PipelineCfg, collate
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    ds = CocoDataset(a.root, a.split, PipelineCfg(), train=True)
    loader = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, collate_fn=collate, worker_init_fn=worker_init, drop_last=True)
    it = iter(loader)
    model = Lova(ds.num_classes).to(dev)
    K = ds.num_classes
    # ---- 9-5a: one batch at init ----
    batch = to_dev(next(it), dev)
    out = model(batch["image"], batch["pointer"], batch["valid"])
    out.check(tuple(batch["image"].shape), K)
    L = total_loss(out.as_dict(), batch)
    st = total_stats(out.as_dict(), batch, L)
    L["total"].backward()
    print(f"9-5a  {dev}, B={a.batch}, {count_params(model):.2f}M params, init:")
    print("  raw losses   " + "  ".join(f"{k}={float(L[k + '_raw']):.4f}" for k in ("r", "heat", "mask")))
    print("  stats        " + "  ".join(f"{k}={float(v):.3f}" for k, v in st.items()))
    print("  grad |sum|   " + "  ".join(f"{k}={sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None):.3g}" for k, m in model.parts().items()))
    m_, p95, frac = saturation(out, batch, dev)
    print(f"  mask logits  mean|z| {m_:.3f}  p95|z| {p95:.3f}  frac(|z|>5) {frac:.4f}")
    sweep(model, batch, "at init (blocks are zero-initialised: R cannot matter yet)")
    if a.warmup == 0:
        return
    # ---- 9-5b: warm-up, then per-loss gradient norms at F0 and on the stem over several batches ----
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    t0 = time.perf_counter()
    for k in range(a.warmup):
        batch = to_dev(next(it), dev)
        out = model(batch["image"], batch["pointer"], batch["valid"])
        L = total_loss(out.as_dict(), batch)
        opt.zero_grad(); L["total"].backward(); opt.step()
        if k % max(a.warmup // 5, 1) == 0 or k == a.warmup - 1:
            print(f"  warm-up {k:3d}  r {float(L['r_raw']):.4f}  heat {float(L['heat_raw']):.3f}  mask {float(L['mask_raw']):.4f}")
    print(f"9-5b  warm-up {a.warmup} steps, {(time.perf_counter() - t0) / a.warmup * 1e3:.0f} ms/step; measuring {a.batches} batches")
    sweep(model, batch, f"after {a.warmup} warm-up steps")
    stem_params = list(model.stem.parameters())
    cfg = TotalLossCfg()
    weights = {"r": cfg.w_r, "heat": cfg.w_heat, "mask": cfg.w_mask}
    rows = {k: {"raw_f0": [], "raw_stem": [], "w_f0": [], "loss": []} for k in ("r", "heat", "mask")}
    sat = []
    for _ in range(a.batches):
        batch = to_dev(next(it), dev)
        out, inter = model.forward_with_intermediates(batch["image"], batch["pointer"], batch["valid"])
        L = total_loss(out.as_dict(), batch, cfg)
        for k in ("r", "heat", "mask"):
            raw_term = L[k] / weights[k]                       # L[k] is the WEIGHTED term; divide to get the raw loss with its graph
            g = torch.autograd.grad(raw_term, [inter["f0"]] + stem_params, retain_graph=True, allow_unused=True)
            gf0 = float(g[0].norm()) if g[0] is not None else 0.0
            rows[k]["raw_f0"].append(gf0)
            rows[k]["w_f0"].append(gf0 * weights[k])            # gradient of the weighted term = w * raw gradient
            rows[k]["raw_stem"].append(float(torch.sqrt(sum((x.norm() ** 2 for x in g[1:] if x is not None), torch.zeros((), device=dev)))))
            rows[k]["loss"].append(float(L[k + "_raw"]))
        sat.append(saturation(out, batch, dev))
    med = lambda xs: float(torch.tensor(xs).median())
    base = med(rows["heat"]["raw_f0"])
    print(f"  RAW loss gradient (medians over {a.batches} batches)")
    print(f"  {'loss':6s} {'raw loss':>10s} {'|dL/dF0|':>12s} {'|grad stem|':>12s} {'ratio vs heat':>14s}")
    for k in ("r", "heat", "mask"):
        print(f"  {k:6s} {med(rows[k]['loss']):10.4f} {med(rows[k]['raw_f0']):12.3e} {med(rows[k]['raw_stem']):12.3e} {med(rows[k]['raw_f0']) / base:14.3f}")
    print("  naive equalisation of raw |dL/dF0| (heat = 1): " + "  ".join(f"w_{k}={base / max(med(rows[k]['raw_f0']), 1e-12):.2f}" for k in ("r", "heat", "mask")))
    print(f"  WEIGHTED loss gradient with the chosen V0 weights (w_r, w_heat, w_mask) = ({cfg.w_r:g}, {cfg.w_heat:g}, {cfg.w_mask:g})")
    wbase = med(rows["heat"]["w_f0"])
    print("  ratio vs heat: " + "  ".join(f"{k} {med(rows[k]['w_f0']) / wbase:.3f}" for k in ("r", "heat", "mask")))
    s = torch.tensor(sat)
    print(f"  mask logits after warm-up: mean|z| {float(s[:, 0].mean()):.3f}  p95|z| {float(s[:, 1].mean()):.3f}  frac(|z|>5) {float(s[:, 2].mean()):.4f}")


def overfit(a):
    import numpy as np
    from PIL import Image
    from tests.util.viz import draw_pointer, field_overlay, hstack, overlay_masks, vstack
    from lova.data.coco.dataset import CocoDataset
    from lova.data.common.build import PipelineCfg, collate
    from lova.data.common.transform import denormalize
    from lova.losses.r_loss import r_stats
    from lova.losses.heat_loss import heat_stats
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    ds = CocoDataset(a.root, a.split, PipelineCfg(), train=False)
    samples = []
    i = a.index
    while len(samples) < a.overfit:
        ts = ds.build_one(i); i += 1
        if ts is not None:
            samples.append(ts)
    batch = collate(samples); b_dev = to_dev(batch, dev)
    model = Lova(ds.num_classes).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    print(f"overfit {a.overfit} images, {a.steps} steps on {dev}")
    t0 = time.perf_counter()
    for it in range(a.steps):
        out = model(b_dev["image"], b_dev["pointer"], b_dev["valid"])
        L = total_loss(out.as_dict(), b_dev)
        opt.zero_grad(); L["total"].backward(); opt.step()
        if it % max(a.steps // 6, 1) == 0 or it == a.steps - 1:
            st = total_stats(out.as_dict(), b_dev, L)
            print(f"  step {it:4d}  r {float(L['r_raw']):.4f}  heat {float(L['heat_raw']):.3f}  mask {float(L['mask_raw']):.4f}   "
                  f"r_mae {float(st['r_mae']):.3f}  p_pos {float(st['p_pos']):.2f}  p_neg {float(st['p_neg']):.3f}  dice {float(st['dice']):.3f}")
    print(f"  {(time.perf_counter() - t0) / a.steps * 1e3:.0f} ms/step")
    model.eval()
    with torch.no_grad():
        out = model(b_dev["image"], b_dev["pointer"], b_dev["valid"])
    rows = []
    for b in range(a.overfit):
        base = Image.fromarray((denormalize(batch["image"][b]).permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        ptr = batch["pointer"][b]
        idx, inst = batch["pos_index"][b], batch["pos_inst"][b]
        heat_gt = batch["heat"][b].amax(0)                                               # [H8,W8] max over classes
        heat_pred = torch.sigmoid(out.heat_logits[b]).amax(0).cpu()
        gt_masks = batch["masks_s4"][b][inst] > 0.5                                      # one per positive cell
        with torch.no_grad():
            pred = torch.sigmoid(dynamic_masks(out.kernel_map[b], out.mask_feat[b], idx.to(dev))).cpu() > 0.5
        up = lambda m: torch.nn.functional.interpolate(m[None].float(), size=base.size[::-1], mode="nearest")[0] > 0.5
        panels = [draw_pointer(base.copy(), ptr),
                  draw_pointer(field_overlay(base, batch["r_gt"][b]), ptr),
                  draw_pointer(field_overlay(base, out.r_pred[b, 0].cpu()), ptr),
                  field_overlay(base, heat_gt), field_overlay(base, heat_pred),
                  overlay_masks(base, up(gt_masks)), overlay_masks(base, up(pred))]
        panels = [im.resize((a.panel, a.panel)) for im in panels]
        rows.append(hstack(panels, [f"img {batch['image_id'][b]} + pointer", "R_GT", "r_pred", "heat GT (max over classes)",
                                    "heat pred", f"GT masks at {len(idx)} positive cells", "predicted masks (same cells)"]))
    canvas = vstack(rows)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        where = os.path.join(a.out, f"lova_overfit_{a.index}_{a.overfit}_{a.steps}.png"); canvas.save(where); print("saved", where)
    else:
        canvas.show(title="LOVA overfit")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=None); p.add_argument("--split", default="val2017")
    p.add_argument("--batch", type=int, default=8); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--warmup", type=int, default=0, help="warm-up steps before the gradient measurement (0 = 9-5a only)")
    p.add_argument("--batches", type=int, default=20); p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--overfit", type=int, default=0, help="overfit the whole model on this many images and draw the predictions")
    p.add_argument("--steps", type=int, default=600); p.add_argument("--index", type=int, default=0)
    p.add_argument("--panel", type=int, default=300); p.add_argument("--out", default=None)
    a = p.parse_args()
    unit_test()
    if a.root and a.overfit:
        overfit(a)
    elif a.root:
        real(a)
