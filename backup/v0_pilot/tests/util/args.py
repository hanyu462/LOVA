"""Shared argument handling for the test scripts (no pipeline logic)."""
from __future__ import annotations

import sys


def add_image_args(p):
    """--root / --split / --image-id / --index on an argparse parser."""
    p.add_argument("--root", default=None, help="COCO root (datasets/coco); without it only the unit test runs")
    p.add_argument("--split", default="val2017")
    p.add_argument("--image-id", type=int, default=None, help="COCO image id (ids are sparse: 139, 285, 632, ...)")
    p.add_argument("--index", type=int, default=0, help="k-th image of the split in sorted id order (if no --image-id)")


def resolve_image_id(cs, a) -> int:
    """--image-id if given (with a clear error for unknown ids), else the --index-th image."""
    all_ids = cs.image_ids()
    if a.image_id is None:
        return all_ids[a.index]
    if a.image_id not in cs.coco.imgs:
        near = [i for i in all_ids if abs(i - a.image_id) < 500][:6]
        sys.exit(f"image id {a.image_id} is not in {a.split} (COCO ids are sparse). Nearby valid ids: {near}. "
                 f"Or use --index k for the k-th image.")
    return a.image_id
