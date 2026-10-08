"""lova.data.common.make_r_gt (step 5: geodesic inside profile R_in + outside decay R_out).

    python tests/common/test_make_r_gt.py                                                  # unit test
    python tests/common/test_make_r_gt.py --root datasets/coco --image-id 39769 --seed 5 --pointers 4
    python tests/common/test_make_r_gt.py --root datasets/coco --image-id 2153 --seed 0 --pointers 4 --gamma 1.0

Window, one row per pointer on the SAME target instance:
    [RGB + pointer + mask outline | R_in @ stride 2 | R_GT @ stride 2 | R_GT @ stride 4 (supervision) | contours of R_GT @ 4]
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
from lova.utils.geometry import depth, downsample_mask, geodesic, pointer_to_stride, seed_cell  # noqa: E402
from lova.data.common.make_r_gt import (RgtCfg, field_stats, inside_profile, make_r_in, make_r_gt,  # noqa: E402
                                       outside_profile, radial_profile, soft_mask, to_supervision)
from lova.data.common.pointer import PointerCfg, make_pointer, pointer_region, sample_from, sampling_region  # noqa: E402
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
    dg = geodesic(m, seed)
    assert dg[30, 27] == 0 and torch.isinf(dg[60, 65]), "outside the mask -> inf"
    assert dg[25, 102] > dg[105, 27] > 0, "far arm is farther than the bottom of the own arm"
    assert dg[25, 102] > 150, "geodesic goes around (~75 + 75 + 75 cells), not straight across (75)"
    cfg = RgtCfg(mode="geodesic", stride=1, gamma=2.0, lambda_in_frac=1.0)
    r = inside_profile(m, dg, cfg)
    assert r[30, 27] == 1.0 and r[60, 65] == 0.0
    assert r[105, 27] > r[25, 102] > 0 and abs(float(r[dg == dg[m].max()].min()) - math_exp(-1)) < 1e-4
    # monotone along the own arm
    col = r[20:110, 27]
    assert bool((col[11:] <= col[10:-1]).all()), "non-increasing below the pointer (row 30 = index 10) along the arm"
    # gamma 1 decays faster near the pointer than gamma 2
    r1 = inside_profile(m, dg, RgtCfg(mode="geodesic", stride=1, gamma=1.0))
    assert float(r1[40, 27]) < float(r[40, 27])

    # different pointers -> different fields; full pipeline helper at stride 2 with snapping
    ra = make_r_in(m, (27, 30), RgtCfg(mode="geodesic", stride=2))
    rb = make_r_in(m, (102, 25), RgtCfg(mode="geodesic", stride=2))
    assert ra.shape == (64, 64) and not torch.allclose(ra, rb)
    ms = downsample_mask(m, 2)
    assert float(ra[ms].min()) > 0 and float(ra[~ms].max()) == 0
    # pointer just outside the downsampled mask snaps to the nearest cell instead of failing
    r_snap = make_r_in(m, (19, 30), RgtCfg(mode="geodesic", stride=2))
    assert r_snap.max() == 1.0

    # disconnected component (not reachable from the pointer) gets 0
    m2 = m.clone()
    m2[5:10, 60:70] = True
    r2 = make_r_in(m2, (27, 30), RgtCfg(mode="geodesic", stride=1))
    assert float(r2[5:10, 60:70].max()) == 0.0
    # ---- outside profile ----
    # disc: R_in = 1 everywhere (pointer-centred small object) -> R_out depends on distance only
    yy, xx = torch.meshgrid(torch.arange(S), torch.arange(S), indexing="ij")
    disc = (yy - 64) ** 2 + (xx - 64) ** 2 < 20 ** 2
    cfg_o = RgtCfg(mode="geodesic", stride=1, lambda_out_px=4.0)
    r_full = make_r_gt(disc, (64, 64), cfg_o)
    assert r_full[64, 64] == 1.0 and r_full.shape == (S, S)
    assert float(r_full[disc].min()) > 0.3                           # inside keeps R_in
    bval, ring1 = float(r_full[64, 64 + 19]), float(r_full[64, 64 + 20])   # last cell inside, first outside
    assert 0.8 * bval < ring1 <= bval, (bval, ring1)                   # continuous across the boundary (exp(-0.5/lambda))
    assert r_full[64, 64 + 28] < r_full[64, 64 + 24] < r_full[64, 64 + 20], "monotone decay outside"
    assert abs(float(r_full[64, 64 + 28]) / float(r_full[64, 64 + 20]) - math_exp(-8 / 4)) < 0.1, "exp(-d/lambda)"
    assert float(r_full[64, 64 + 20 + 30]) == 0.0, "zero beyond 6 lambda"
    # exact distance cutoff, also on the diagonal (where one propagation step covers sqrt(2) cells)
    yy2, xx2 = torch.meshgrid(torch.arange(S), torch.arange(S), indexing="ij")
    euclid = torch.sqrt((yy2 - 64.0) ** 2 + (xx2 - 64.0) ** 2) - 20.0      # ~distance to the disc boundary
    assert float(r_full[euclid > 6 * 4 + 2].max()) == 0.0, "nothing beyond the cutoff in any direction"
    assert float(r_full[(euclid < 6 * 4 - 2) & ~disc].min()) > 0.0, "everything inside the cutoff is reached"
    assert float(r_full[0, 0]) == 0.0
    # boundary value is carried outward: the outside near the pointer side is higher than far side
    rb = make_r_gt(m, (27, 30), RgtCfg(mode="geodesic", stride=1, lambda_out_px=6.0))
    assert rb[30, 17] > rb[25, 112] > 0, (float(rb[30, 17]), float(rb[25, 112]))
    assert torch.equal(outside_profile(m, inside_profile(m, dg, cfg), RgtCfg(mode="geodesic", stride=1))[m], r[m]), "inside untouched"
    # stride-2 end to end (geodesic): shape and range; supervision at stride 4 = 2x2 area average
    r2 = make_r_gt(m, (27, 30), RgtCfg(mode="geodesic", stride=2))
    assert r2.shape == (64, 64) and r2.min() >= 0 and r2.max() <= 1
    r4 = to_supervision(r2, 2)
    assert r4.shape == (32, 32) and abs(float(r4[0, 0]) - float(r2[0:2, 0:2].mean())) < 1e-6
    assert torch.equal(to_supervision(r2, 1), r2)

    # ---- B radial / C radial_bias ----
    cb = RgtCfg(mode="radial", stride=1, gamma=2.0, sigma_frac=1.0)
    rB = make_r_gt(m, (27, 30), cb)
    assert rB.shape == (S, S) and rB[30, 27] == 1.0
    # same Euclidean distance -> same value, inside or outside the mask
    assert abs(float(rB[30, 27 + 20]) - float(rB[30 + 20, 27])) < 1e-5
    # far arm is close in Euclidean terms -> HIGH under B (the opposite of A)
    assert float(rB[25, 102]) > float(rB[105, 27]) * 0.5 and float(rB[25, 102]) > 0.3
    # farthest mask cell = exp(-1) at sigma_frac 1; sigma_frac 1.25 lifts it above 0.5
    ys, xs = torch.nonzero(m, as_tuple=True)
    far = ((ys - 30.0) ** 2 + (xs - 27.0) ** 2).argmax()
    assert abs(float(rB[ys[far], xs[far]]) - math_exp(-1)) < 1e-3
    rB2 = make_r_gt(m, (27, 30), RgtCfg(mode="radial", stride=1, sigma_frac=1.25))
    assert float(rB2[ys[far], xs[far]]) > 0.5
    # C: soft mask is exactly 1 on the mask (boundary included), ramps 1 -> 0 outside over the band, 0 beyond
    Sm = soft_mask(m, 6)
    assert bool(Sm[m].min() == 1.0) and Sm[60, 60] == 0.0
    assert 0 < float(Sm[30, 18]) < 1 and float(Sm[30, 17]) < float(Sm[30, 18]), "ramp just outside the left arm"
    cc = RgtCfg(mode="radial_bias", stride=1, sigma_frac=1.0, eta=0.3, band_px=6)
    rC = make_r_gt(m, (27, 30), cc)
    assert rC[30, 27] == 1.0, "R(p) = 1"
    assert torch.allclose(rC[m], rB[m]), "on the mask C == B"
    assert abs(float(rC[30, 27 + 40]) - 0.3 * float(rB[30, 27 + 40])) < 1e-5, "beyond the band: eta * B"
    assert float(rC[30, 17]) > 0.3 * float(rB[30, 17]), "inside the band: between B and eta * B"
    st = field_stats(rC, m, (27.0, 30.0))
    assert st["r_pointer"] == 1.0 and 0 < st["r_mask_min"] <= st["r_mask_mean"] <= 1 and 0 <= st["area_gt_half"] <= 1
    # default cfg is the chosen definition
    d = RgtCfg()
    assert d.mode == "radial_bias" and d.sigma_frac == 1.25 and d.eta == 0.3 and d.band_px == 96.0 and d.gamma == 2.0
    rd = make_r_gt(m, (27, 30), RgtCfg(stride=1))            # default mode / parameters, full-res for exact indexing
    assert rd[30, 27] == 1.0 and float(rd[m].min()) > 0.5, "whole target above 0.5 with the default sigma_frac"
    assert make_r_gt(m, (27, 30)).shape == (64, 64), "default stride 2"
    # a mask that vanishes at the geometry stride is an error, not a silently wrong field
    thin = torch.zeros(S, S, dtype=torch.bool)
    thin[10, 20] = True                     # single pixel -> area average 0.25 < 0.5 at stride 2 (a 1 px line gives 0.5 and survives)
    try:
        make_r_gt(thin, (20, 10), RgtCfg(stride=2))
        raise AssertionError("vanished mask must raise")
    except ValueError:
        pass
    print("make_r_gt unit test OK")


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
    p.add_argument("--mode", default="radial_bias", choices=["radial_bias", "radial", "geodesic"])
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--sigma-frac", type=float, default=1.25)
    p.add_argument("--eta", type=float, default=0.3)
    p.add_argument("--band-px", type=float, default=96.0)
    p.add_argument("--lambda-in-frac", type=float, default=1.0, help="geodesic mode")
    p.add_argument("--lambda-out-px", type=float, default=32.0, help="geodesic mode")
    p.add_argument("--inside-only", action="store_true", help="step 5-1 view: no outside decay")
    p.add_argument("--compare", action="store_true",
                   help="A/B/C side by side per pointer: geodesic | radial s1.0 | radial s1.25 | radial_bias s1.25, with stats")
    p.add_argument("--compare-c", action="store_true",
                   help="C variants at eta 0.3 (sigma_frac 1.25): bands from --bands (default 48,64,96,128)")
    p.add_argument("--bands", default="48,64,96,128", help="band widths (px) for --compare-c")
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
    cfg = RgtCfg(mode=a.mode, stride=a.stride, gamma=a.gamma, sigma_frac=a.sigma_frac, eta=a.eta, band_px=a.band_px,
                 lambda_in_frac=a.lambda_in_frac, lambda_out_px=a.lambda_out_px)
    S = a.size
    print(f"image {img_id}: target {idx} {names[idx]} ({int(mask.sum())} px), {cfg}{' inside only' if a.inside_only else ''}")

    base = Image.fromarray((denormalize(t.image) * 255).permute(1, 2, 0).numpy().astype(np.uint8))
    mask_s = downsample_mask(mask, a.stride)
    if a.compare or a.compare_c:
        return compare_modes(a, t, mask, mask_s, region, base, names[idx], img_id, idx, g)
    bnd = mask_s & (depth(mask_s) == 0)
    rows = []
    for k in range(a.pointers):
        ptr = sample_from(region, g)
        r_in = make_r_in(mask, ptr, cfg) if a.mode == "geodesic" else make_r_gt(mask, ptr, RgtCfg(**{**cfg.__dict__, "mode": "radial"}))
        r = r_in if a.inside_only else make_r_gt(mask, ptr, cfg)
        r4 = to_supervision(r, max(4 // a.stride, 1))                    # supervision resolution (stride 4)
        up = lambda f: torch.nn.functional.interpolate(f[None, None], size=(S, S), mode="bilinear", align_corners=False)[0, 0]
        rr = up(r4)                                                      # contours / numbers below refer to the stride-4 field
        base_t = torch.from_numpy(np.array(base)).permute(2, 0, 1).float() / 255

        def heatmap(f):
            h = torch.stack([f, torch.zeros_like(f), 1 - f], 0) * 0.7 + base_t * 0.3
            return draw_pointer(Image.fromarray((h.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)), ptr)
        img_rgb = draw_pointer(overlay_masks(base, mask[None], filled=[], labels=[names[idx]]), ptr)
        r4_nearest = torch.nn.functional.interpolate(r4[None, None], size=(S, S), mode="nearest")[0, 0]  # show the real cells
        img_in, img_s2, img_s4 = heatmap(up(r_in)), heatmap(up(r)), heatmap(r4_nearest)
        cont = np.asarray(base).copy()
        for lvl, col in ((0.9, (255, 255, 255)), (0.7, (255, 200, 0)), (0.5, (255, 100, 0)), (0.3, (200, 0, 200))):
            above = rr >= lvl   # iso-contour = cells above the level that touch a cell below it
            touches_below = torch.nn.functional.max_pool2d((~above).float()[None, None], 3, 1, 1)[0, 0] > 0
            cont[(above & touches_below).numpy()] = col
        img_cont = draw_pointer(Image.fromarray(cont), ptr)
        rows.append(hstack([img_rgb, img_in, img_s2, img_s4, img_cont],
                           [f"pointer {k}: ({ptr[0]},{ptr[1]})  target {names[idx]}",
                            f"{'R_in' if a.mode == 'geodesic' else 'B radial (no mask bias)'} @ stride {a.stride}",
                            f"R_GT @ stride {a.stride}", "R_GT @ stride 4 (supervision, nearest)", "contours of R_GT@4: 0.9 0.7 0.5 0.3"]))
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
        where = os.path.join(a.out, f"r_gt_{img_id}_t{idx}_s{a.seed}{'_in' if a.inside_only else ''}.png")
        canvas.save(where)
    else:
        where = "(window)"
        canvas.show(title=f"R_GT {img_id}")
    print(f"  -> {where}")


def compare_modes(a, t, mask, mask_s, region, base, name, img_id, idx, g):
    S = a.size
    if a.compare_c:
        variants = [(f"C eta0.3 band{b}", RgtCfg(mode="radial_bias", stride=a.stride, gamma=a.gamma, sigma_frac=1.25, eta=0.3, band_px=float(b)))
                    for b in a.bands.split(",")]
    else:
        variants = [("A geodesic", RgtCfg(mode="geodesic", stride=a.stride, gamma=a.gamma)),
                    ("B radial s=1.0", RgtCfg(mode="radial", stride=a.stride, gamma=a.gamma, sigma_frac=1.0)),
                    ("B radial s=1.25", RgtCfg(mode="radial", stride=a.stride, gamma=a.gamma, sigma_frac=1.25)),
                    ("C radial_bias s=1.25", RgtCfg(mode="radial_bias", stride=a.stride, gamma=a.gamma, sigma_frac=1.25))]
    valid_s = downsample_mask(t.valid, a.stride)
    base_t = torch.from_numpy(np.array(base)).permute(2, 0, 1).float() / 255
    up = lambda f: torch.nn.functional.interpolate(f[None, None], size=(S, S), mode="bilinear", align_corners=False)[0, 0]
    rows = []
    print(f"  {'':<10}{'variant':<24}{'R(p)':>6}{'mask min':>10}{'mask mean':>10}{'band32 mean':>12}{'area>0.5':>10}")
    for k in range(a.pointers):
        ptr = sample_from(region, g)
        p_s = pointer_to_stride(ptr, a.stride)
        panels = [draw_pointer(overlay_masks(base, mask[None], filled=[], labels=[name]), ptr)]
        titles = [f"pointer {k} ({ptr[0]},{ptr[1]})  {name}"]
        for label, cfg in variants:
            r = make_r_gt(mask, ptr, cfg) * valid_s
            st = field_stats(r, mask_s, p_s, valid_s, band_cells=int(32 / a.stride))
            print(f"  {'ptr ' + str(k):<10}{label:<24}{st['r_pointer']:>6.2f}{st['r_mask_min']:>10.2f}{st['r_mask_mean']:>10.2f}"
                  f"{st['r_band_mean']:>12.2f}{st['area_gt_half']:>10.3f}")
            rr = up(r)
            h = torch.stack([rr, torch.zeros_like(rr), 1 - rr], 0) * 0.7 + base_t * 0.3
            img = Image.fromarray((h.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            above = rr >= 0.5
            ring = (above & (torch.nn.functional.max_pool2d((~above).float()[None, None], 3, 1, 1)[0, 0] > 0)).numpy()
            arr = np.asarray(img).copy()
            arr[ring] = (255, 255, 255)
            panels.append(draw_pointer(Image.fromarray(arr), ptr))
            titles.append(label + "  (white = 0.5)")
        rows.append(hstack(panels, titles))
    W = max(rw.width for rw in rows)
    canvas = Image.new("RGB", (W, sum(rw.height for rw in rows) + 6 * len(rows)), (30, 30, 30))
    y = 0
    for rw in rows:
        canvas.paste(rw, (0, y))
        y += rw.height + 6
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        where = os.path.join(a.out, f"r_gt_compare{'C' if a.compare_c else ''}_{img_id}_t{idx}_s{a.seed}.png")
        canvas.save(where)
    else:
        where = "(window)"
        canvas.show(title=f"R_GT A/B/C {img_id}")
    print(f"  -> {where}")


if __name__ == "__main__":
    main()
