Tests mirror the package: `tests/coco/` for `lova/data/coco`, `tests/common/` for `lova/data/common`, `tests/utils/` for `lova/utils`, `tests/losses/` for `lova/losses`.
One script per module. Each imports the module and (1) asserts on a small synthetic case,
(2) optionally renders on real data (`--root datasets/coco`) for visual checking: a window by
default, PNGs with `--out`. Never re-implements module logic. `tests/util/` holds shared helpers
(argument handling, drawing) only.

    python tests/utils/test_geometry.py                 # grid primitives, synthetic only
    python tests/coco/test_load.py --root datasets/coco --image-id 139
    python tests/common/test_transform.py --root datasets/coco --image-id 2153 --seed 3
    python tests/common/test_select.py --root datasets/coco --image-id 39769 --seed 5
    python tests/common/test_pointer.py --root datasets/coco --image-id 39769 --seed 5 --k 30 --all
    python tests/common/test_make_r_gt.py --root datasets/coco --image-id 2153 --seed 0 --target 1 --pointers 2
