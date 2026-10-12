"""lova.models.r_predictor: pointer encoding + R predictor, alone, against r_loss.

    python tests/models/test_r_predictor.py                                   # synthetic
    python tests/models/test_r_predictor.py --root datasets/coco --n 8 --pointers 4 --steps 300 [--out dir]
        overfits stem + R predictor on n val images x k pointers each and shows, per image:
        image + a trained pointer | its R_GT | r_pred | r_pred for a HELD-OUT pointer | its R_GT
        (the last two columns are the pointer-conditioning check: that pair was never trained on)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lova.losses.r_loss import r_loss, r_stats  # noqa: E402
from lova.models.common import count_params  # noqa: E402
from lova.models.r_predictor import POINTER_CHANNELS, RPredictor, RPredictorCfg, pointer_maps  # noqa: E402
from lova.models.stem import Stem  # noqa: E402


def unit_test():
    torch.manual_seed(0)
    # ---- pointer maps ----
    S, stride = 64, 4
    h = w = S // stride
    p = torch.tensor([[17.0, 33.0], [2.0, 60.0]])                           # cells (4, 8) and (0, 15)
    P = pointer_maps(p, h, w, stride, S)
    assert P.shape == (2, POINTER_CHANNELS, h, w)
    assert P[0, 3].argmax() == 8 * w + 4 and P[1, 3].argmax() == 15 * w + 0   # gaussian peaks on the pointer cell
    assert P[0, 2].argmin() == 8 * w + 4                                      # distance minimal there
    assert float(P[0, 3].max()) > 0.99 and float(P[0, 2].min()) < 1.0 / S
    assert float(P[0, 0, 8, 10]) > 0 and float(P[0, 0, 8, 2]) < 0             # dx sign: right of the pointer positive
    assert float(P[0, 1, 12, 4]) > 0 and float(P[0, 1, 3, 4]) < 0             # dy sign
    assert abs(float(P[0, 2, 8, 4 + 8]) - 32 / S) < 0.01                      # 8 cells = 32 px right -> 0.5 canvas
    # ---- predictor ----
    c0 = 16
    rp = RPredictor(c0, RPredictorCfg(c=16, init_r=0.3))
    f0 = torch.randn(2, c0, h, w)
    r = rp(f0, p, S)
    assert r.shape == (2, 1, h, w) and torch.allclose(r, torch.full_like(r, 0.3), atol=1e-6)   # constant at init
    assert rp(torch.randn(1, c0, 24, 40), p[:1], 160).shape == (1, 1, 24, 40)                # rectangular
    # once the output layer is non-zero the pointer matters: same F0, different pointer -> different R
    torch.nn.init.normal_(rp.out.weight, std=0.5)
    r1, r2 = rp(f0, p, S), rp(f0, p.flip(0), S)
    assert not torch.allclose(r1, r2)
    # and the same pointer gives the same R regardless of batch position
    assert torch.allclose(rp(f0[:1], p[:1], S), rp(f0, p, S)[:1], atol=1e-5)
    # ---- through r_loss: gradient reaches every parameter of predictor and stem ----
    stem = Stem(c0)
    img = torch.randn(2, 3, S, S)
    r_gt = torch.rand(2, h, w); r_valid = torch.ones(2, h, w, dtype=torch.bool); r_valid[:, :, -3:] = False
    loss = r_loss(rp(stem(img), p, S), r_gt, r_valid)
    loss.backward()
    assert all(p_.grad is not None and torch.isfinite(p_.grad).all() for p_ in list(rp.parameters()) + list(stem.parameters()))
    # ---- learnability: fit one image with its R_GT = the pointer gaussian ----
    rp2, st2 = RPredictor(c0, RPredictorCfg(c=16)), Stem(c0)
    opt = torch.optim.Adam(list(rp2.parameters()) + list(st2.parameters()), lr=2e-3)
    target = pointer_maps(p, h, w, stride, S, sigma_frac=0.15)[:, 3]
    first = None
    for it in range(150):
        l = r_loss(rp2(st2(img), p, S), target, r_valid)
        first = float(l.detach()) if first is None else first
        opt.zero_grad(); l.backward(); opt.step()
    assert float(l.detach()) < 0.1 * first, (first, float(l.detach()))
    print(f"r_predictor unit test OK   fit loss {first:.4f} -> {float(l.detach()):.4f}   params c0=64/c=64: {count_params(RPredictor(64)):.3f}M")


def real(a):
    """Overfit stem + R predictor on n images x k pointers each (k seeds per image); a (k+1)-th pointer per
    image is held out. If R really depends on the pointer, the held-out field must match its R_GT even
    though that (image, pointer) pair was never trained on."""
    import numpy as np
    from PIL import Image
    from tests.util.viz import draw_pointer, field_overlay, hstack, vstack
    from lova.data.coco.dataset import CocoDataset
    from lova.data.coco.load import load
    from lova.data.common.build import PipelineCfg, build, collate
    from lova.data.common.transform import denormalize
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = CocoDataset(a.root, a.split, PipelineCfg(), train=False)
    train, held = [], []
    i = a.index
    while len(held) < a.n:                      # eval transform is deterministic: only the pointer differs between seeds
        sample = load(ds.cs, ds.ids[i])
        ts = [build(sample, ds.num_classes, ds.cfg, torch.Generator().manual_seed(1000 * i + s)) for s in range(a.pointers + 1)]
        i += 1
        if any(t is None for t in ts):
            continue
        train += ts[:a.pointers]; held.append(ts[a.pointers])
    batch, batch_h = collate(train), collate(held)
    S = batch["image"].shape[-1]
    img, ptr, r_gt, r_valid = (batch[k].to(dev) for k in ("image", "pointer", "r_gt", "r_valid"))
    stem, rp = Stem(64).to(dev), RPredictor(64, RPredictorCfg()).to(dev)
    params = list(stem.parameters()) + list(rp.parameters())
    print(f"{dev}: {a.n} images x {a.pointers} pointers = {len(train)} training pairs, {a.n} held-out pointers; "
          f"stem {count_params(stem):.3f}M + R predictor {count_params(rp):.3f}M, {a.steps} steps")
    img_h, ptr_h, r_gt_h, r_valid_h = (batch_h[k].to(dev) for k in ("image", "pointer", "r_gt", "r_valid"))
    opt = torch.optim.Adam(params, lr=a.lr)
    t0 = time.perf_counter()
    for it in range(a.steps):
        r = rp(stem(img), ptr, S)
        l = r_loss(r, r_gt, r_valid)
        opt.zero_grad(); l.backward(); opt.step()
        if it % max(a.steps // 6, 1) == 0 or it == a.steps - 1:
            with torch.no_grad():
                lh = r_loss(rp(stem(img_h), ptr_h, S), r_gt_h, r_valid_h)
            print(f"  step {it:4d}  train r_loss {float(l.detach()):.4f}   held-out pointers r_loss {float(lh):.4f}")
    print(f"  {(time.perf_counter() - t0) / a.steps * 1e3:.0f} ms/step")
    with torch.no_grad():
        r_fit = rp(stem(img), ptr, S).cpu()
        r_hold = rp(stem(img_h), ptr_h, S).cpu()
        st = r_stats(r_hold.to(dev), r_gt_h, r_valid_h)
    print(f"  held-out: mae {float(st['r_mae']):.3f}  lit pred/gt {float(st['r_lit_pred']):.3f}/{float(st['r_lit_gt']):.3f}")
    rows = []
    for b in range(a.n):
        j = b * a.pointers                         # first training pointer of image b
        base = Image.fromarray((denormalize(batch_h["image"][b]).permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        p1, ph = batch["pointer"][j], batch_h["pointer"][b]
        panels = [draw_pointer(base.copy(), p1),
                  draw_pointer(field_overlay(base, batch["r_gt"][j]), p1),
                  draw_pointer(field_overlay(base, r_fit[j, 0]), p1),
                  draw_pointer(field_overlay(base, r_hold[b, 0]), ph),
                  draw_pointer(field_overlay(base, batch_h["r_gt"][b]), ph)]
        panels = [im.resize((a.panel, a.panel)) for im in panels]
        rows.append(hstack(panels, [f"img {batch_h['image_id'][b]}: a trained pointer", "its R_GT", f"r_pred ({a.steps} steps)",
                                    "r_pred for the HELD-OUT pointer", "its R_GT (never trained)"]))
    canvas = vstack(rows)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        where = os.path.join(a.out, f"r_predictor_{a.index}_{a.n}x{a.pointers}.png"); canvas.save(where); print("saved", where)
    else:
        canvas.show(title="R predictor overfit")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=None); p.add_argument("--split", default="val2017")
    p.add_argument("--index", type=int, default=0, help="first dataset index"); p.add_argument("--n", type=int, default=8)
    p.add_argument("--pointers", type=int, default=4, help="training pointers per image (one more is held out)")
    p.add_argument("--steps", type=int, default=300); p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--panel", type=int, default=320); p.add_argument("--out", default=None)
    a = p.parse_args()
    unit_test()
    if a.root:
        real(a)
