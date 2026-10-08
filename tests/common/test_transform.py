"""lova.data.common.transform: image / mask / crowd stay aligned under resize, flip, crop, pad.

    python tests/common/test_transform.py                              # unit tests (synthetic, no data)
    python tests/common/test_transform.py --root datasets/coco --image-id 139 --seed 0     # window: original | transformed
    python tests/common/test_transform.py --root datasets/coco --image-id 139 --seed 0 --eval          # deterministic eval transform
    python tests/common/test_transform.py --root datasets/coco --image-id 139 --seed 3 --out viz/transform

Alignment test idea: paint each instance region of a synthetic image in a unique colour; after
the transform, pixels inside mask i must still have colour i (and padding must be exactly 0).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lova.data.common.sample import Sample  # noqa: E402
from lova.data.common.transform import (TransformCfg, TransformParams, apply, denormalize,  # noqa: E402
                                        sample_params, to_original, transform)
from tests.util.args import add_image_args, resolve_image_id  # noqa: E402
from tests.util.viz import hstack, overlay_masks  # noqa: E402

COLORS = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], np.uint8)


def synthetic_sample(h=300, w=400):
    """Grey image with three painted shapes (one per instance) and a crowd rectangle."""
    img = np.full((h, w, 3), 128, np.uint8)
    masks = torch.zeros(3, h, w, dtype=torch.bool)
    masks[0, 40:140, 30:130] = True                        # square, left
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    masks[1] = (yy - 200) ** 2 + (xx - 300) ** 2 < 60 ** 2  # circle, right
    masks[2, 250:290, 150:350] = True                      # thin bar, bottom
    for i in range(3):
        img[masks[i].numpy()] = COLORS[i]
    crowd = torch.zeros(h, w, dtype=torch.bool)
    crowd[10:40, 300:390] = True
    img[crowd.numpy()] = [255, 255, 0]
    areas = masks.flatten(1).sum(1).float()
    return Sample(1, Image.fromarray(img), masks, torch.tensor([0, 1, 2]), [10, 11, 12], areas, areas / (h * w), crowd)


def colour_inside_masks(t, tol=60):
    """Fraction of pixels inside each transformed mask whose colour is that instance's colour."""
    img = (denormalize(t.image) * 255).permute(1, 2, 0).numpy()
    out = []
    for i in range(len(t)):
        m = t.masks[i].numpy()
        if m.sum() == 0:
            out.append(float("nan"))
            continue
        ok = (np.abs(img[m] - COLORS[i]).max(1) < tol).mean()
        out.append(float(ok))
    return out


def unit_test():
    s = synthetic_sample()
    cfg = TransformCfg(size=256)

    # 1. eval transform: resize only, padding at the bottom (400x300 -> 256x192), nothing else
    t = apply(s, sample_params(s.size, TransformCfg(size=256, train=False)), cfg)
    assert t.image.shape == (3, 256, 256) and t.masks.shape == (3, 256, 256) and t.masks.dtype == torch.bool
    assert t.params.new_h == 192 and t.params.new_w == 256 and not t.params.flip and t.params.crop_x == t.params.crop_y == 0
    assert bool(t.valid[:192].all()) and not bool(t.valid[192:].any())
    assert torch.equal(t.image[:, 192:, :], torch.zeros(3, 64, 256)), "padding must be exactly 0 after normalisation"
    assert not bool(t.masks[:, 192:, :].any()) and not bool(t.crowd[192:].any())
    frac = colour_inside_masks(t)
    assert min(frac) > 0.9, frac                          # image and masks moved together
    assert bool(t.crowd[8:24, 194:246].all()) and not bool(t.crowd[30:, :].any())  # crowd moved too (y 6..26, x 192..250 at scale 0.64)

    # 2. flip: masks mirror, image mirrors, colours still match
    pf = TransformParams(scale=0.64, new_h=192, new_w=256, flip=True, crop_x=0, crop_y=0)
    tf = apply(s, pf, cfg)
    assert torch.equal(tf.masks[:, :192], t.masks[:, :192].flip(-1))
    assert min(colour_inside_masks(tf)) > 0.9
    # the square was on the left (x 30..130 of 400) -> after flip it is on the right
    ys, xs = torch.nonzero(tf.masks[0], as_tuple=True)
    assert xs.float().mean() > 128

    # 3. crop: scale up so the resized image (640x480) exceeds 256, crop a window; thin bar partly survives
    pc = TransformParams(scale=1.6, new_h=480, new_w=640, flip=False, crop_x=200, crop_y=224)  # window y 224..480, x 200..456
    tc = apply(s, pc, cfg)
    assert bool(tc.valid.all()), "crop window fully inside the image -> no padding"
    assert min(f for f in colour_inside_masks(tc) if f == f) > 0.9
    assert not bool(tc.masks[0].any()), "square (y 40..140 -> 64..224) ends where the crop window starts"
    assert bool(tc.masks[2].any()), "bar (y 250..290 -> 400..464) is inside the crop window"
    # to_original maps a canvas point back: bar centre
    ys, xs = torch.nonzero(tc.masks[2], as_tuple=True)
    ox, oy = to_original((xs.float().mean(), ys.float().mean()), pc)
    assert 150 < ox < 350 and 250 < oy < 290, (ox, oy)

    # 4. random params are reproducible with a generator and stay in range
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    p1, p2 = sample_params(s.size, cfg, g1), sample_params(s.size, cfg, g2)
    assert p1 == p2
    for _ in range(50):
        p = sample_params(s.size, cfg, g1)
        assert 0.6 * 256 / 400 - 1e-6 <= p.scale <= 1.25 * 256 / 400 + 1e-6
        assert 0 <= p.crop_x <= max(p.new_w - 256, 0) and 0 <= p.crop_y <= max(p.new_h - 256, 0)
        tt = apply(s, p, cfg)
        assert tt.image.shape == (3, 256, 256) and tt.masks.shape[0] == 3
        assert torch.equal(tt.image[:, ~tt.valid], torch.zeros(3, int((~tt.valid).sum())))
    print("transform unit test OK")


def main():
    p = argparse.ArgumentParser()
    add_image_args(p)
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval", action="store_true", help="deterministic transform (resize + pad only)")
    p.add_argument("--out", default=None, help="save PNG here instead of opening a window")
    p.add_argument("--compare-interp", type=float, default=None, metavar="SCALE",
                   help="render bilinear+thr vs nearest mask resizing at this fixed scale (e.g. 0.4), zoomed on each mask")
    a = p.parse_args()

    unit_test()
    if a.root is None:
        return
    from lova.data.coco.load import load, open_coco

    cs = open_coco(a.root, a.split)
    img_id = resolve_image_id(cs, a)
    s = load(cs, img_id)
    if a.compare_interp is not None:
        return compare_interp(cs, s, a)
    cfg = TransformCfg(size=a.size, train=not a.eval)
    t = transform(s, cfg, torch.Generator().manual_seed(a.seed))
    pr = t.params
    print(f"image {img_id} {s.size[1]}x{s.size[0]} -> scale {pr.scale:.3f} resized {pr.new_w}x{pr.new_h} "
          f"flip {pr.flip} crop ({pr.crop_x},{pr.crop_y}) valid {float(t.valid.float().mean()):.2f} of canvas")
    kept = [i for i in range(len(t)) if t.masks[i].any()]
    print(f"  instances visible after transform: {len(kept)}/{len(t)}")

    names = [cs.name_of_label(int(l)) for l in s.labels]
    before = overlay_masks(s.image, s.masks, labels=names)
    after_img = Image.fromarray((denormalize(t.image) * 255).permute(1, 2, 0).numpy().astype(np.uint8))
    after = overlay_masks(after_img, t.masks, filled=kept, labels=[names[i] if i in kept else "" for i in range(len(t))])
    if t.crowd.any():
        arr = np.asarray(after).copy()
        hatch = t.crowd.numpy() & (((np.arange(a.size)[:, None] + np.arange(a.size)[None]) % 6) < 2)
        arr[hatch] = (arr[hatch] * 0.3 + 255 * 0.7).astype(np.uint8)
        after = Image.fromarray(arr)
    panel = hstack([before, after], [f"original {s.size[1]}x{s.size[0]}",
                                     f"transformed {a.size}x{a.size}  scale {pr.scale:.2f} flip {pr.flip} crop ({pr.crop_x},{pr.crop_y})"])
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        where = os.path.join(a.out, f"transform_{img_id}_s{a.seed}{'_eval' if a.eval else ''}.png")
        panel.save(where)
    else:
        where = "(window)"
        panel.show(title=f"transform {img_id}")
    print(f"  -> {where}")


def compare_interp(cs, s, a):
    """Same fixed scale, two mask resizers. For each instance: zoomed crop [original | bilinear+0.5 | nearest]
    and the area each keeps relative to the ideal (original area * scale^2)."""
    h, w = s.size
    sc = a.compare_interp
    pr = TransformParams(scale=sc, new_h=round(h * sc), new_w=round(w * sc), flip=False, crop_x=0, crop_y=0)
    outs = {m: apply(s, pr, TransformCfg(size=a.size, mask_interp=m)) for m in ("bilinear", "nearest")}
    names = [cs.name_of_label(int(l)) for l in s.labels]
    rows = []
    print(f"image {s.image_id} scale {sc}: mask area kept vs ideal (orig * scale^2)")
    print(f"  {'instance':<16}{'ideal px':>9}{'bilinear':>10}{'nearest':>9}")
    for i in range(len(s)):
        ideal = float(s.masks[i].sum()) * sc * sc
        ab, an = float(outs["bilinear"].masks[i].sum()), float(outs["nearest"].masks[i].sum())
        print(f"  {names[i][:15]:<16}{ideal:>9.0f}{ab / max(ideal, 1):>10.2f}{an / max(ideal, 1):>9.2f}")
        ys, xs = torch.nonzero(s.masks[i], as_tuple=True)
        if len(ys) == 0:
            continue
        pad = 8
        y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad, h)
        x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad, w)
        zoom = 2 if (y1 - y0) * (x1 - x0) < 40000 else 1
        orig = overlay_masks(s.image, s.masks[i:i + 1], labels=[names[i]]).crop((x0, y0, x1, y1))
        orig = orig.resize((orig.width * zoom, orig.height * zoom), Image.NEAREST)
        panels = [orig]
        for m in ("bilinear", "nearest"):
            t = outs[m]
            img = Image.fromarray((denormalize(t.image) * 255).permute(1, 2, 0).numpy().astype(np.uint8))
            ov = overlay_masks(img, t.masks[i:i + 1]).crop((int(x0 * sc), int(y0 * sc), int(x1 * sc) + 1, int(y1 * sc) + 1))
            panels.append(ov.resize((orig.width, orig.height), Image.NEAREST))
        rows.append(hstack(panels, [f"{names[i]} original", f"bilinear+0.5 @ {sc}", f"nearest @ {sc}"]))
    W = max(r.width for r in rows)
    canvas = Image.new("RGB", (W, sum(r.height for r in rows) + 6 * len(rows)), (30, 30, 30))
    y = 0
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.height + 6
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        where = os.path.join(a.out, f"interp_{s.image_id}_{sc}.png")
        canvas.save(where)
    else:
        where = "(window)"
        canvas.show(title=f"mask interp {s.image_id}")
    print(f"  -> {where}")


if __name__ == "__main__":
    main()
