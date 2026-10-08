"""lova.data.coco.load: unit test on an in-memory COCO + visual check on real COCO.

    python tests/test_load.py                                   # unit test only (no data)
    python tests/test_load.py --root datasets/coco --n 4   # + opens a window per image
    python tests/test_load.py --root datasets/coco --image-id 139   # COCO id (sparse: 139, 285, 632, ...)
    python tests/test_load.py --root datasets/coco --index 0        # k-th image in sorted id order
    python tests/test_load.py --root datasets/coco --n 4 --out viz/load   # save PNGs instead (headless server)

PNG = original image with every instance mask filled + outlined, label "class ratio";
crowd (ignore) regions, if any, are hatched in grey.
What to look at: do the masks follow the objects? overlapping masks (bed under a cat)?
fragmented masks (occluded objects)? Those shape the later pointer / R* steps.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lova.data.coco.load import CocoSet, load, open_coco  # noqa: E402
from tests.common import add_image_args, resolve_image_id  # noqa: E402
from tests.viz import overlay_masks  # noqa: E402


def fake_cocoset(tmpdir: str) -> CocoSet:
    """One 100x200 image file + 3 instances (50 %, 2 %, 0.1 %) + one crowd region."""
    from pycocotools.coco import COCO

    os.makedirs(os.path.join(tmpdir, "val2017"), exist_ok=True)
    Image.fromarray(np.zeros((100, 200, 3), np.uint8)).save(os.path.join(tmpdir, "val2017", "img1.jpg"))
    c = COCO()
    c.dataset = {
        "images": [{"id": 1, "file_name": "img1.jpg", "height": 100, "width": 200}],
        "categories": [{"id": 3, "name": "car"}, {"id": 1, "name": "person"}],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 3, "iscrowd": 0, "area": 10000, "bbox": [0, 0, 100, 100],
             "segmentation": [[0, 0, 100, 0, 100, 100, 0, 100]]},
            {"id": 11, "image_id": 1, "category_id": 1, "iscrowd": 0, "area": 400, "bbox": [120, 10, 20, 20],
             "segmentation": [[120, 10, 140, 10, 140, 30, 120, 30]]},
            {"id": 12, "image_id": 1, "category_id": 1, "iscrowd": 0, "area": 20, "bbox": [150, 50, 5, 4],
             "segmentation": [[150, 50, 155, 50, 155, 54, 150, 54]]},
            {"id": 13, "image_id": 1, "category_id": 3, "iscrowd": 1, "area": 5000, "bbox": [0, 0, 50, 100],
             "segmentation": {"counts": [0, 5000, 15000], "size": [100, 200]}},
        ],
    }
    c.createIndex()
    cat_ids = sorted(c.getCatIds())
    names = {k["id"]: k["name"] for k in c.loadCats(cat_ids)}
    return CocoSet(c, tmpdir, "val2017", cat_ids, {k: i for i, k in enumerate(cat_ids)}, names)


def unit_test():
    with tempfile.TemporaryDirectory() as d:
        cs = fake_cocoset(d)
        assert cs.num_classes == 2 and cs.cat_ids == [1, 3]          # sorted category ids
        s = load(cs, 1)
        assert s.size == (100, 200) and s.image.size == (200, 100)   # PIL is (W, H)
        assert s.masks.shape == (3, 100, 200) and s.masks.dtype == torch.bool, "crowd excluded"
        assert s.ann_ids == [10, 11, 12]
        assert s.labels.tolist() == [1, 0, 0] and cs.name_of_label(1) == "car"
        assert torch.allclose(s.ratios, torch.tensor([0.5, 0.02, 0.001]))
        assert int(s.masks[0].sum()) == 10000, "polygon rasterised exactly for an axis-aligned square"
        assert bool(s.masks[1, 20, 130]) and not bool(s.masks[1, 20, 100])
        # crowd region (RLE, column-major: first 5000 px = columns 0..49) is carried separately
        assert s.crowd.shape == (100, 200) and s.crowd.dtype == torch.bool
        assert int(s.crowd.sum()) == 5000 and bool(s.crowd[:, :50].all()) and not bool(s.crowd[:, 50:].any())
        check_shapes(s)
    print("load unit test OK")


def check_shapes(s):
    """Invariants every loaded sample must satisfy (used on the fake and on real images)."""
    h, w = s.size
    n = len(s)
    assert s.image.mode == "RGB"
    assert s.masks.shape == (n, h, w) and s.masks.dtype == torch.bool
    assert s.labels.shape == (n,) and s.labels.dtype == torch.long
    assert len(s.ann_ids) == n and s.areas.shape == (n,) and s.ratios.shape == (n,)
    assert bool(((s.ratios >= 0) & (s.ratios <= 1)).all())
    assert s.crowd.shape == (h, w) and s.crowd.dtype == torch.bool


def check_real(cs: CocoSet, img_id: int):
    """Shapes/dtypes + raster area within 5 % of the annotation area."""
    s = load(cs, img_id)
    check_shapes(s)
    for i in range(len(s)):
        area = int(s.masks[i].sum())
        assert abs(area - float(s.areas[i])) <= max(0.05 * float(s.areas[i]), 20), (s.ann_ids[i], area, float(s.areas[i]))
    return s


def main():
    p = argparse.ArgumentParser()
    add_image_args(p)
    p.add_argument("--n", type=int, default=4, help="random images when neither --image-id nor --index is given")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="save PNGs here instead of opening a window")
    a = p.parse_args()

    unit_test()
    if a.root is None:
        return
    cs = open_coco(a.root, a.split)
    if a.image_id is not None or a.index:
        ids = [resolve_image_id(cs, a)]
    else:
        ids = [int(i) for i in np.random.RandomState(a.seed).choice(cs.image_ids(), a.n, replace=False)]
    if a.out:
        os.makedirs(a.out, exist_ok=True)
    for img_id in ids:
        s = check_real(cs, img_id)
        labels = [f"{cs.name_of_label(int(l))} {float(r):.3f}" for l, r in zip(s.labels, s.ratios)]
        img = overlay_masks(s.image, s.masks, labels=labels)
        if s.crowd.any():
            arr = np.asarray(img).copy()
            hatch = s.crowd.numpy() & (((np.arange(s.size[0])[:, None] + np.arange(s.size[1])[None]) % 6) < 2)
            arr[hatch] = (arr[hatch] * 0.3 + 255 * 0.7).astype(np.uint8)
            img = Image.fromarray(arr)
        if a.out:
            where = os.path.join(a.out, f"load_{img_id}.png")
            img.save(where)
        else:
            where = "(window)"
            img.show(title=f"load {img_id}")
        big = sum(float(r) >= 0.01 for r in s.ratios)
        crowd = f", crowd {float(s.crowd.float().mean()):.3f} of image" if s.crowd.any() else ""
        print(f"image {img_id} ({s.size[1]}x{s.size[0]}): {len(s)} instances, {big} with ratio >= 0.01{crowd} -> {where}")
        for i in range(min(len(s), 3)):
            print(f"    ann {s.ann_ids[i]} {cs.name_of_label(int(s.labels[i])):<12} raster {int(s.masks[i].sum()):>7} px  annotation {float(s.areas[i]):>9.1f}")


if __name__ == "__main__":
    main()
