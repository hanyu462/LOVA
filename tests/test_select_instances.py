"""Test / inspection for preprocess.select_instances (calls the module, re-implements nothing).

Unit test (no data needed):
    python tests/test_select_instances.py
List candidates on real COCO (original-image pre-filter):
    python tests/test_select_instances.py --coco-root datasets/coco --threshold 0.02 --n 5
    python tests/test_select_instances.py --coco-root datasets/coco --threshold 0.02 --image-id 139
    python tests/test_select_instances.py --coco-root datasets/coco --threshold 0.02 --stats
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess.select_instances import (annotation_ratios, keep_by_ratio, prefilter, recheck,  # noqa: E402
                                         selection_stats, visible_ratios)


def fake_coco():
    """Tiny in-memory COCO: one 100x200 image with 3 instances (50%, 2%, 0.1%) + one crowd."""
    from pycocotools.coco import COCO

    c = COCO()
    c.dataset = {
        "images": [{"id": 1, "height": 100, "width": 200}],
        "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 1, "iscrowd": 0, "area": 10000, "bbox": [0, 0, 100, 100],
             "segmentation": [[0, 0, 100, 0, 100, 100, 0, 100]]},
            {"id": 11, "image_id": 1, "category_id": 2, "iscrowd": 0, "area": 400, "bbox": [120, 10, 20, 20],
             "segmentation": [[120, 10, 140, 10, 140, 30, 120, 30]]},
            {"id": 12, "image_id": 1, "category_id": 2, "iscrowd": 0, "area": 20, "bbox": [150, 50, 5, 4],
             "segmentation": [[150, 50, 155, 50, 155, 54, 150, 54]]},
            {"id": 13, "image_id": 1, "category_id": 1, "iscrowd": 1, "area": 5000, "bbox": [0, 0, 50, 100],
             "segmentation": {"counts": [0, 5000, 15000], "size": [100, 200]}},
        ],
    }
    c.createIndex()
    return c


def unit_test():
    # the rule itself
    assert keep_by_ratio([0.01, 0.5, 0.02, 0.0], 0.02) == [1, 2]
    assert keep_by_ratio([0.001], 0.02) == []

    # 1a: pre-filter on annotations
    c = fake_coco()
    cands = annotation_ratios(c, 1)
    assert [x.ann_id for x in cands] == [10, 11, 12], "crowd excluded, annotation order"
    assert abs(cands[0].ratio - 0.5) < 1e-9 and abs(cands[1].ratio - 0.02) < 1e-9
    assert [x.ann_id for x in prefilter(c, 1, 0.01)] == [10, 11]
    assert [x.ann_id for x in prefilter(c, 1, 0.1)] == [10]
    assert prefilter(c, 1, 0.9) == []
    assert abs(annotation_ratios(c, 1, area_from="mask")[0].ratio - 0.5) < 0.01
    assert selection_stats(c, 0.01) == dict(threshold=0.01, images=1, images_with_target=1, instances=3, instances_kept=2)

    # 1b: recheck on transformed masks with padding. canvas 8x8, real image = left 8x4 (valid), rest padding
    masks = torch.zeros(3, 8, 8)
    masks[0, :, :4] = 1          # fills the whole visible image      -> ratio 1.0
    masks[1, 0:2, 0:2] = 1       # 4 px of 32 visible                 -> 0.125
    masks[2, 0, 0] = 1           # 1 px                               -> 0.03125
    valid = torch.zeros(1, 8, 8)
    valid[:, :, :4] = 1
    r = visible_ratios(masks, valid)
    assert torch.allclose(r, torch.tensor([1.0, 0.125, 0.03125]))
    assert recheck(masks, 0.1, valid) == [0, 1]
    assert recheck(masks, 0.1, None) == [0], "without valid the padding inflates the denominator"
    # a large original instance that got cropped down to a sliver is dropped by the recheck
    assert recheck(masks[2:3], 0.05, valid) == []
    print("select_instances unit test OK")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coco-root", default=None)
    p.add_argument("--split", default="val2017")
    p.add_argument("--threshold", type=float, default=0.02)
    p.add_argument("--n", type=int, default=5, help="first N images to list")
    p.add_argument("--image-id", type=int, default=None)
    p.add_argument("--area-from", choices=["annotation", "mask"], default="annotation")
    p.add_argument("--stats", action="store_true", help="dataset-wide survival counts of the pre-filter")
    a = p.parse_args()

    unit_test()
    if a.coco_root is None:
        return
    from pycocotools.coco import COCO

    coco = COCO(os.path.join(a.coco_root, "annotations", f"instances_{a.split}.json"))
    names = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    ids = [a.image_id] if a.image_id is not None else sorted(coco.getImgIds())[:a.n]
    for img_id in ids:
        info = coco.loadImgs(img_id)[0]
        cands = sorted(annotation_ratios(coco, img_id, a.area_from), key=lambda x: x.ratio, reverse=True)
        kept = {x.ann_id for x in prefilter(coco, img_id, a.threshold, a.area_from)}
        print(f"\nimage {img_id} ({info['width']}x{info['height']}): {len(kept)}/{len(cands)} instances >= {a.threshold}")
        print(f"  {'keep':<5}{'ann_id':>8}  {'class':<14}{'ratio':>8}{'area_px':>10}  bbox")
        for x in cands:
            print(f"  {'*' if x.ann_id in kept else '':<5}{x.ann_id:>8}  {names[x.category_id]:<14}{x.ratio:>8.4f}{x.area:>10.0f}  "
                  f"({x.bbox[0]:.0f},{x.bbox[1]:.0f},{x.bbox[2]:.0f},{x.bbox[3]:.0f})")
    if a.stats:
        print("\n" + str(selection_stats(coco, a.threshold)))


if __name__ == "__main__":
    main()
