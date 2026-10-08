"""lova.data.common.make_rgt (step 5: geodesic inside profile R_in + outside decay R_out).

    python tests/common/test_make_rgt.py                                                  # unit test
    python tests/common/test_make_rgt.py --root datasets/coco --image-id 39769 --seed 5 --pointers 4
    python tests/common/test_make_rgt.py --root datasets/coco --image-id 2153 --seed 0 --pointers 4 --gamma 1.0

Window, one row per pointer on the SAME target instance:
    [RGB + pointer | target mask | R_GT heatmap (red high, blue low) | R_GT contours 0.9 / 0.7 / 0.5 / 0.3]
What to check: R = 1 at the pointer, smooth decrease that follows the object's shape, continuous
across the boundary, faster decay outside than inside (thin halo), 0 far away, and clearly
different fields for different pointers. --inside-only shows step 5-1 alone.
Terminal: R along the row through the pointer (mask cells marked with |), boundary R min/mean/max.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lova.data.common.pointer import PointerCfg, depth, make_pointer, pointer_region, sample_from, sampling_region  # noqa: E402
from lova.data.common.make_rgt import (RgtCfg, downsample_mask, geodesic_from_pointer, inside_profile,  # noqa: E402
                                       make_r_in, make_rgt, outside_profile, pointer_to_stride, seed_cell)
from lova.data.common.select import SelectCfg, select  # noqa: E402
from lova.data.common.transform import TransformCfg, denormalize, transform  # noqa: E402
from tests.util.args import add_image_args, resolve_image_id  # noqa: E402
from tests.util.viz import draw_pointer, hstack, overlay_masks  # noqa: E402


def unit_test():
    # half-pixel convention
    assert pointer_to_stride((0, 0), 2) == (-0.25, -0.25) and pointer_to_stride((3, 5), 2) == (1.25, 2.25)
    assert pointer_to_stride((7, 7), 1) == (7.0, 7.0)

    # U shape: pointer in the left arm. The right arm is close in Euclidean terms but far geodesically.
    S = 128
    m = torch.zeros(S, S, dtype=torch.bool)
    m[20:110, 20:35] = True          # left arm
    m[95:110, 20:110] = True         # bottom
    m[20:110, 95:110] = True         # right arm
    seed = seed_cell(m, (27.0, 30.0))
    assert seed == (30, 27)
    dg = geodesic_from_pointer(m, seed)
    assert dg[30, 27] == 0 and torch.isinf(dg[60, 65]), "outside the mask -> inf"
    assert dg[25, 102] > dg[105, 27] > 0, "far arm is farther than the bottom of the own arm"
    assert dg[25, 102] > 150, "geodesic goes around (~75 + 75 + 75 cells), not straight across (75)"
    cfg = RgtCfg(stride=1, gamma=2.0, lambda_in_frac=1.0)
    r = inside_profile(m, dg, cfg)
    assert r[30, 27] == 1.0 and r[60, 65] == 0.0
    assert r[105, 27] > r[25, 102] > 0 and abs(float(r[dg == dg[m].max()].min()) - math_exp(-1)) < 1e-4
    # monotone along the own arm
    col = r[20:110, 27]
    assert bool((col[11:] <= col[10:-1]).all()), "non-increasing below the pointer (row 30 = index 10) along the arm"
    # gamma 1 decays faster near the pointer than gamma 2
    r1 = inside_profile(m, dg, RgtCfg(stride=1, gamma=1.0))
    assert float(r1[40, 27]) < float(r[40, 27])

    # different pointers -> different fields; full pipeline helper at stride 2 with snapping
    ra = make_r_in(m, (27, 30), RgtCfg(stride=2))
    rb = make_r_in(m, (102, 25), RgtCfg(stride=2))
    assert ra.shape == (64, 64) and not torch.allclose(ra, rb)
    ms = downsample_mask(m, 2)
    assert float(ra[ms].min()) > 0 and float(ra[~ms].max()) == 0
    # pointer just outside the downsampled mask snaps to the nearest cell instead of failing
    r_snap = make_r_in(m, (19, 30), RgtCfg(stride=2))
    assert r_snap.max() == 1.0

    # disconnected component (not reachable from the pointer) gets 0
    m2 = m.clone()
    m2[5:10, 60:70] = True
    r2 = make_r_in(m2, (27, 30), RgtCfg(stride=1))
    assert float(r2[5:10, 60:70].max()) == 0.0
    # ---- outside profile ----
    # disc: R_in = 1 everywhere (pointer-centred small object) -> R_out depends on distance only
    yy, xx = torch.meshgrid(torch.arange(S), torch.arange(S), indexing="ij")
    disc = (yy - 64) ** 2 + (xx - 64) ** 2 < 20 ** 2
    cfg_o = RgtCfg(stride=1, lambda_out_px=4.0)
    r_full = make_rgt(disc, (64, 64), cfg_o)
    assert r_full[64, 64] == 1.0 and r_full.shape == (S, S)
    assert float(r_full[disc].min()) > 0.3                           # inside keeps R_in
    bval, ring1 = float(r_full[64, 64 + 19]), float(r_full[64, 64 + 20])   # last cell inside, first outside
    assert 0.8 * bval < ring1 <= bval, (bval, ring1)                   # continuous across the boundary (exp(-0.5/lambda))
    assert r_full[64, 64 + 28] < r_full[64, 64 + 24] < r_full[64, 64 + 20], "monotone decay outside"
    assert abs(float(r_full[64, 64 + 28]) / float(r_full[64, 64 + 20]) - math_exp(-8 / 4)) < 0.1, "exp(-d/lambda)"
    assert float(r_full[64, 64 + 20 + 30]) == 0.0, "zero beyond ~6 lambda"
    assert float(r_full[0, 0]) == 0.0
    # boundary value is carried outward: the outside near the pointer side is higher than far side
    rb = make_rgt(m, (27, 30), RgtCfg(stride=1, lambda_out_px=6.0))
    assert rb[30, 17] > rb[25, 112] > 0, (float(rb[30, 17]), float(rb[25, 112]))
    assert torch.equal(outside_profile(m, inside_profile(m, dg, cfg), RgtCfg(stride=1))[m], r[m]), "inside untouched"
    # stride-2 end to end: shape and range
    r2 = make_rgt(m, (27, 30), RgtCfg(stride=2))
    assert r2.shape == (64, 64) and r2.min() >= 0 and r2.max() <= 1
    print("make_rgt unit test OK")


def math_exp(x):
    import math
    return math.exp(x)


def main():
    p = argparse.ArgumentParser()
    add_image_args(p)
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--target", type=int, default=None, help="instance index to use (default: random candidate)")
    p.add_argument("--pointers", type=int, default=4, help="pointers on the same target (one row each)")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--lambda-in-frac", type=float, default=1.0)
    p.add_argument("--lambda-out-px", type=float, default=32.0)
    p.add_argument("--inside-only", action="store_true", help="step 5-1 view: no outside decay")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    unit_test()
    if a.root is None:
        return
    from lova.data.coco.load import load, open_coco

    cs = open_coco(a.root, a.split)
    img_id = resolve_image_id(cs, a)
    s = load(cs, img_id)
    g = torch.Generator().manual_seed(a.seed)
    t = transform(s, TransformCfg(size=a.size, train=not a.eval), g)
    cands = select(t, SelectCfg(a.threshold))
    names = [cs.name_of_label(int(l)) for l in t.labels]
    if not cands:
        print(f"image {img_id}: no candidates")
        return
    if a.target is None:
        idx, _ = make_pointer(t, cands, PointerCfg(), g)
    else:
        idx = a.target
    mask = t.masks[idx]
    region = sampling_region(pointer_region(idx, t.masks), PointerCfg())
    cfg = RgtCfg(stride=a.stride, gamma=a.gamma, lambda_in_frac=a.lambda_in_frac, lambda_out_px=a.lambda_out_px)
    S = a.size
    print(f"image {img_id}: target {idx} {names[idx]} ({int(mask.sum())} px), stride {a.stride} gamma {a.gamma} "
          f"lambda_in_frac {a.lambda_in_frac} lambda_out_px {a.lambda_out_px}{' inside only' if a.inside_only else ''}")

    base = Image.fromarray((denormalize(t.image) * 255).permute(1, 2, 0).numpy().astype(np.uint8))
    mask_s = downsample_mask(mask, a.stride)
    bnd = mask_s & (depth(mask_s) == 0)
    rows = []
    for k in range(a.pointers):
        ptr = sample_from(region, g)
        r = (make_r_in if a.inside_only else make_rgt)(mask, ptr, cfg)  # [S/st, S/st]
        rr = torch.nn.functional.interpolate(r[None, None], size=(S, S), mode="bilinear", align_corners=False)[0, 0]
        img_rgb = draw_pointer(base.copy(), ptr)
        img_mask = draw_pointer(overlay_masks(base, mask[None], labels=[names[idx]]), ptr)
        heat = torch.stack([rr, torch.zeros_like(rr), 1 - rr], 0) * 0.7 + torch.from_numpy(np.array(base)).permute(2, 0, 1).float() / 255 * 0.3
        img_heat = draw_pointer(Image.fromarray((heat.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)), ptr)
        cont = np.asarray(base).copy()
        for lvl, col in ((0.9, (255, 255, 255)), (0.7, (255, 200, 0)), (0.5, (255, 100, 0)), (0.3, (200, 0, 200))):
            above = rr >= lvl   # iso-contour = cells above the level that touch a cell below it
            touches_below = torch.nn.functional.max_pool2d((~above).float()[None, None], 3, 1, 1)[0, 0] > 0
            cont[(above & touches_below).numpy()] = col
        img_cont = draw_pointer(Image.fromarray(cont), ptr)
        rows.append(hstack([img_rgb, img_mask, img_heat, img_cont],
                           [f"pointer {k}: ({ptr[0]},{ptr[1]})", f"target: {names[idx]}",
                            f"{'R_in' if a.inside_only else 'R_GT'} (red 1 -> blue 0)", "contours 0.9 0.7 0.5 0.3"]))
        # numbers: profile along the pointer row (stride cells) and boundary range
        px, py = pointer_to_stride(ptr, a.stride)
        row = r[int(round(py))]
        cx = int(round(px))
        xs = list(range(max(cx - 30, 0), min(cx + 31, r.shape[1]), 5))
        rb = r[bnd]
        print(f"  pointer {k} ({ptr[0]},{ptr[1]})  R(p)={float(r[int(round(py)), cx]):.2f}  boundary R min/mean/max "
              f"{float(rb.min()):.2f}/{float(rb.mean()):.2f}/{float(rb.max()):.2f}")
        print("     x(cells): " + " ".join(f"{x:4d}" for x in xs))
        print("     R       : " + " ".join(f"{float(row[x]):4.2f}" for x in xs))
        print("     in mask : " + " ".join(f"{'   |' if bool(mask_s[int(round(py)), x]) else '    '}" for x in xs))
    W = max(rw.width for rw in rows)
    canvas = Image.new("RGB", (W, sum(rw.height for rw in rows) + 6 * len(rows)), (30, 30, 30))
    y = 0
    for rw in rows:
        canvas.paste(rw, (0, y))
        y += rw.height + 6
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        where = os.path.join(a.out, f"rgt_{img_id}_t{idx}_s{a.seed}{'_in' if a.inside_only else ''}.png")
        canvas.save(where)
    else:
        where = "(window)"
        canvas.show(title=f"R_GT {img_id}")
    print(f"  -> {where}")


if __name__ == "__main__":
    main()
