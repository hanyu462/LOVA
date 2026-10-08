"""lova.data.common.select: pointer candidates from the TRANSFORMED masks.

    python tests/common/test_select.py                                              # unit test (synthetic)
    python tests/common/test_select.py --root datasets/coco --image-id 139 --seed 0 # window: one panel per threshold
    python tests/common/test_select.py --root datasets/coco --image-id 139 --eval --thresholds 0.005,0.01,0.02

Panel per threshold on the transformed image: candidate = filled + "class ratio", other = outline.
The title shows how many candidates survive. Compare with the original-image ratio printed in the
terminal to see what the crop did.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lova.data.common.select import SelectCfg, keep_by_ratio, select, visible_ratios  # noqa: E402
from lova.data.common.transform import TransformCfg, TransformParams, apply, denormalize, transform  # noqa: E402
from tests.common.test_transform import synthetic_sample  # noqa: E402
from tests.util.args import add_image_args, resolve_image_id  # noqa: E402
from tests.util.viz import hstack, overlay_masks  # noqa: E402


def unit_test():
    # the rule
    assert keep_by_ratio(torch.tensor([0.01, 0.5, 0.02, 0.0]), 0.02) == [1, 2]
    assert keep_by_ratio(torch.tensor([0.001]), 0.02) == []

    # visible ratios with padding: canvas 8x8, valid = left 8x4
    masks = torch.zeros(3, 8, 8, dtype=torch.bool)
    masks[0, :, :4] = True          # fills the visible image          -> 1.0
    masks[1, 0:2, 0:2] = True       # 4 of 32 visible px                -> 0.125
    masks[2, 0, 0] = True           # 1 px                              -> 0.03125
    valid = torch.zeros(8, 8, dtype=torch.bool)
    valid[:, :4] = True
    r = visible_ratios(masks, valid)
    assert torch.allclose(r, torch.tensor([1.0, 0.125, 0.03125]))
    leak = masks.clone()
    leak[1, :, 4:] = True           # mask in the padding must not count
    assert torch.allclose(visible_ratios(leak, valid), r)
    assert keep_by_ratio(r, 0.1) == [0, 1]

    # end to end on the synthetic sample: crop keeps the bar, drops the square
    s = synthetic_sample()
    cfg = TransformCfg(size=256)
    full = apply(s, TransformParams(scale=0.64, new_h=192, new_w=256, flip=False, crop_x=0, crop_y=0), cfg)
    r_full = visible_ratios(full.masks, full.valid)
    assert torch.allclose(r_full, s.ratios, atol=0.01), (r_full, s.ratios)   # no crop -> same ratios as original
    crop = apply(s, TransformParams(scale=1.6, new_h=480, new_w=640, flip=False, crop_x=200, crop_y=224), cfg)
    r_crop = visible_ratios(crop.masks, crop.valid)
    assert r_crop[0] == 0 and r_crop[2] > 0                                   # square gone, bar visible
    assert 0 not in select(crop, SelectCfg(0.001)) and select(crop, SelectCfg(0.5)) == []
    assert select(full) == select(full, SelectCfg(0.01)), "default cfg"
    print("select unit test OK")


def main():
    p = argparse.ArgumentParser()
    add_image_args(p)
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--thresholds", default="0.005,0.01,0.02")
    p.add_argument("--out", default=None, help="save PNG here instead of opening a window")
    a = p.parse_args()

    unit_test()
    if a.root is None:
        return
    from lova.data.coco.load import load, open_coco

    cs = open_coco(a.root, a.split)
    img_id = resolve_image_id(cs, a)
    s = load(cs, img_id)
    t = transform(s, TransformCfg(size=a.size, train=not a.eval), torch.Generator().manual_seed(a.seed))
    pr = t.params
    ratios = visible_ratios(t.masks, t.valid)
    names = [cs.name_of_label(int(l)) for l in t.labels]
    ths = [float(x) for x in a.thresholds.split(",")]

    print(f"image {img_id}: scale {pr.scale:.2f} flip {pr.flip} crop ({pr.crop_x},{pr.crop_y})")
    print(f"  {'instance':<16}{'orig ratio':>11}{'visible':>9}  " + "".join(f"{'t=' + str(x):>9}" for x in ths))
    order = ratios.argsort(descending=True).tolist()
    for i in order:
        marks = "".join(f"{'*' if float(ratios[i]) >= x else '':>9}" for x in ths)
        print(f"  {names[i][:15]:<16}{float(s.ratios[i]):>11.4f}{float(ratios[i]):>9.4f}  {marks}")

    img = Image.fromarray((denormalize(t.image) * 255).permute(1, 2, 0).numpy().astype(np.uint8))
    panels, titles = [], []
    for x in ths:
        cands = select(t, SelectCfg(threshold=x))
        labels = [f"{names[i]} {float(ratios[i]):.3f}" if i in cands else "" for i in range(len(t))]
        panels.append(overlay_masks(img, t.masks, filled=cands, labels=labels))
        titles.append(f"threshold {x}: {len(cands)}/{len(t)} candidates")
    panel = hstack(panels, titles)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        where = os.path.join(a.out, f"select_{img_id}_s{a.seed}{'_eval' if a.eval else ''}.png")
        panel.save(where)
    else:
        where = "(window)"
        panel.show(title=f"select {img_id}")
    print(f"  -> {where}")


if __name__ == "__main__":
    main()
