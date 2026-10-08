"""Step 2 check: do the rasterised COCO masks match the objects, and what does each threshold keep?

    python tests/test_load_viz.py --coco-root datasets/coco --n 6 --out viz/load
    python tests/test_load_viz.py --coco-root datasets/coco --image-id 139 --thresholds 0.005,0.01,0.02 --out viz/load

Each PNG = one image rendered once per threshold, side by side:
  filled colour + label  = instance kept by that threshold (pointer candidate)
  outline only           = instance below the threshold (still a segmentation target)
Also prints, per image, the kept counts and the class/ratio list.
Uses preprocess.load / preprocess.select_instances; draws with preprocess.viz. No logic here.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess.load import LabelMap, load_sample  # noqa: E402
from preprocess.select_instances import keep_by_ratio  # noqa: E402
from preprocess.viz import hstack, overlay_masks  # noqa: E402


def unit_test_load(coco, root, split):
    """Mask raster is consistent with the annotation: area within 5 % (polygon vs raster), bbox contains the mask."""
    img_id = sorted(coco.getImgIds())[0]
    s = load_sample(coco, root, split, img_id)
    assert s.masks.shape[0] == len(s.cands) == len(s.labels)
    assert s.masks.shape[1:] == s.size
    for i, c in enumerate(s.cands):
        area = int(s.masks[i].sum())
        assert abs(area - c.area) <= max(0.05 * c.area, 20), (c.ann_id, area, c.area)  # polygon vs raster
        ys, xs = np.nonzero(s.masks[i].numpy())
        x0, y0, w, h = c.bbox
        assert xs.min() >= int(x0) - 1 and xs.max() <= int(x0 + w) + 1
        assert ys.min() >= int(y0) - 1 and ys.max() <= int(y0 + h) + 1
    print(f"load unit test OK (image {img_id}, {len(s.cands)} instances)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coco-root", required=True)
    p.add_argument("--split", default="val2017")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--image-id", type=int, default=None)
    p.add_argument("--thresholds", default="0.005,0.01,0.02")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="viz/load")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from pycocotools.coco import COCO

    coco = COCO(os.path.join(a.coco_root, "annotations", f"instances_{a.split}.json"))
    labels = LabelMap(coco)
    unit_test_load(coco, a.coco_root, a.split)

    ths = [float(t) for t in a.thresholds.split(",")]
    if a.image_id is not None:
        ids = [a.image_id]
    else:
        ids = [int(i) for i in np.random.RandomState(a.seed).choice(sorted(coco.getImgIds()), a.n, replace=False)]
    for img_id in ids:
        s = load_sample(coco, a.coco_root, a.split, img_id, labels)
        ratios = [c.ratio for c in s.cands]
        names = [f"{labels.name(c.category_id)} {c.ratio:.3f}" for c in s.cands]
        panels, titles = [], []
        for t in ths:
            keep = keep_by_ratio(ratios, t)
            panels.append(overlay_masks(s.image, s.masks, keep, names))
            titles.append(f"id {img_id}  threshold {t}: keep {len(keep)}/{len(s.cands)}")
        path = os.path.join(a.out, f"load_{img_id}.png")
        hstack(panels, titles).save(path)
        kept_str = "  ".join(f"t={t}: {len(keep_by_ratio(ratios, t))}" for t in ths)
        print(f"image {img_id} ({s.size[1]}x{s.size[0]}) {len(s.cands)} inst  {kept_str}  -> {path}")


if __name__ == "__main__":
    main()
