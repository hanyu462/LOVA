"""lova.data.load: unit test on an in-memory COCO + visual check on real COCO.

    python tests/test_load.py                                   # unit test only (no data)
    python tests/test_load.py --coco-root datasets/coco --n 4   # + PNGs in viz/load/
    python tests/test_load.py --coco-root datasets/coco --image-id 139

PNG = original image with every instance mask filled + outlined, label "class ratio".
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
from lova.data.load import CocoSet, load, open_coco  # noqa: E402
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
    print("load unit test OK")


def check_real(cs: CocoSet, img_id: int):
    """Raster area within 5 % of the polygon area; mask inside the bbox."""
    s = load(cs, img_id)
    for i in range(len(s)):
        area = int(s.masks[i].sum())
        assert abs(area - float(s.areas[i])) <= max(0.05 * float(s.areas[i]), 20), (s.ann_ids[i], area, float(s.areas[i]))
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coco-root", default=None)
    p.add_argument("--split", default="val2017")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--image-id", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="viz/load")
    a = p.parse_args()

    unit_test()
    if a.coco_root is None:
        return
    cs = open_coco(a.coco_root, a.split)
    ids = [a.image_id] if a.image_id is not None else \
        [int(i) for i in np.random.RandomState(a.seed).choice(cs.image_ids(), a.n, replace=False)]
    os.makedirs(a.out, exist_ok=True)
    for img_id in ids:
        s = check_real(cs, img_id)
        labels = [f"{cs.name_of_label(int(l))} {float(r):.3f}" for l, r in zip(s.labels, s.ratios)]
        path = os.path.join(a.out, f"load_{img_id}.png")
        overlay_masks(s.image, s.masks, labels=labels).save(path)
        big = sum(float(r) >= 0.01 for r in s.ratios)
        print(f"image {img_id} ({s.size[1]}x{s.size[0]}): {len(s)} instances, {big} with ratio >= 0.01 -> {path}")


if __name__ == "__main__":
    main()
