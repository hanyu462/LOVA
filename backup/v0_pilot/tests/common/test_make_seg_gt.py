"""lova.data.common.make_seg_gt: segmentation GT (heatmap, positives, ignore) from all instances.

    python tests/common/test_make_seg_gt.py                                           # unit test
    python tests/common/test_make_seg_gt.py --root datasets/coco --image-id 2153 --seed 0
    python tests/common/test_make_seg_gt.py --root datasets/coco --index 3 --center deepest
    python tests/common/test_make_seg_gt.py --root datasets/coco --stats 300             # centre-outside rate etc.

Window: [image + centres (x) + positive cells (squares) | heatmap, max over classes | ignore: crowd / padding]
Terminal: per instance class, area, centre inside?, #positive cells, sigma.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lova.data.common.make_seg_gt import SegGT, SegGtCfg, instance_centers, make_seg_gt  # noqa: E402
from lova.data.common.transform import TransformCfg, Transformed, TransformParams, denormalize, transform  # noqa: E402
from tests.common.test_transform import synthetic_sample  # noqa: E402
from tests.util.args import add_image_args, resolve_image_id  # noqa: E402
from tests.util.viz import hstack, overlay_masks  # noqa: E402


def fake_transformed(S=128):
    """square (class 0), small disc INSIDE the square (class 1), U shape (class 2), crowd patch, padding."""
    masks = torch.zeros(3, S, S, dtype=torch.bool)
    masks[0, 16:80, 16:80] = True
    yy, xx = torch.meshgrid(torch.arange(S), torch.arange(S), indexing="ij")
    masks[1] = (yy - 48) ** 2 + (xx - 48) ** 2 < 10 ** 2
    masks[2, 90:120, 10:20] = True; masks[2, 110:120, 10:60] = True; masks[2, 90:120, 50:60] = True
    crowd = torch.zeros(S, S, dtype=torch.bool); crowd[20:50, 90:120] = True
    valid = torch.ones(S, S, dtype=torch.bool); valid[:, 100:] = False
    return Transformed(torch.zeros(3, S, S), masks, crowd, valid, torch.tensor([0, 1, 2]), [1, 2, 3], 0,
                       TransformParams(1.0, S, S, False, 0, 0), (S, S))


def unit_test():
    t = fake_transformed()
    gt = make_seg_gt(t, num_classes=3, cfg=SegGtCfg(center="centroid"))   # V0 centre for the geometry checks below
    h8 = 16
    assert gt.heat.shape == (3, h8, h8) and gt.heat_valid.shape == (h8, h8) and gt.masks_s4.shape == (3, 32, 32)
    # peaks exactly 1 at each instance centre cell, in its own class channel
    for i in range(3):
        cx, cy = (gt.centers[i] + 0.5) / 8 - 0.5
        cx, cy = int(round(float(cx))), int(round(float(cy)))
        assert gt.heat[i, cy, cx] == 1.0, (i, gt.heat[i, cy, cx])
    assert gt.heat.max() == 1.0 and gt.heat.min() >= 0
    # centroid of the U is outside the mask; the square's and disc's are inside
    assert bool(gt.center_pixel_inside[0]) and bool(gt.center_pixel_inside[1]) and not bool(gt.center_pixel_inside[2])
    gt_d = make_seg_gt(t, 3, SegGtCfg(center="deepest"))
    assert bool(gt_d.center_pixel_inside.all()), "deepest centre is always inside"
    # deepest_owned: the square's centre moves off the disc that sits inside it; plain deepest stays on the disc
    cx, cy = gt_d.centers[0].tolist()
    assert bool(t.masks[1, int(cy), int(cx)]), "deepest of the square lands on the (covering) disc"
    gt_o = make_seg_gt(t, 3, SegGtCfg(center="deepest_owned"))
    cx, cy = gt_o.centers[0].tolist()
    assert bool(t.masks[0, int(cy), int(cx)]) and not bool(t.masks[1, int(cy), int(cx)]), "owned: on the square, off the disc"
    assert bool(gt_o.center_pixel_inside.all())
    assert SegGtCfg().center == "deepest_owned" and bool(make_seg_gt(t, 3).center_pixel_inside.all()), "default"
    # ownership: the disc (smaller) owns its centre cell although it lies inside the square
    pos = dict(zip(gt.pos_index.tolist(), gt.pos_inst.tolist()))
    cx, cy = [int(round((float(v) + 0.5) / 8 - 0.5)) for v in gt.centers[1]]
    assert pos[cy * h8 + cx] == 1
    # positives lie on their instance (occupancy >= 0.25) except the forced centre cell
    occ8 = torch.nn.functional.avg_pool2d(t.masks[:, None].float(), 8)[:, 0]
    for q, i in pos.items():
        y, x = q // h8, q % h8
        cxi, cyi = [int(round((float(v) + 0.5) / 8 - 0.5)) for v in gt.centers[i]]
        assert occ8[i, y, x] >= 0.25 or (y, x) == (cyi, cxi)
    # ignore: crowd cells and padding cells are not valid, positives are
    # crowd x 90..120 / y 20..50; padding x >= 100. Cell 11 (x 88..96) touches crowd, cells >= 13 are padding
    assert not bool(gt.heat_valid[3:6, 11].any()), "crowd -> ignored"
    assert not bool(gt.heat_valid[2, 11]), "cell with only a few crowd pixels (rows 20..24 of 16..24) is ignored too (any-pixel rule)"
    assert not bool(gt.heat_valid[:, 13:].any()), "padding -> ignored"
    assert bool(gt.heat_valid[3:6, 0:10].all()), "ordinary background stays valid"
    assert bool(gt.heat_valid.view(-1)[gt.pos_index].all())
    assert not bool(gt.mask_valid[:, 25:].any()) and not bool(gt.mask_valid[5:13, 22:25].any())
    assert bool(make_seg_gt(t, 3, SegGtCfg(crowd_ignore=False)).heat_valid[3:6, 11].all())
    # soft masks at stride 4: area preserved
    assert abs(float(gt.masks_s4[0].sum()) * 16 - float(t.masks[0].sum())) < 1
    # an instance whose whole 3x3 centre window is taken by smaller instances reclaims one of its
    # other (unowned) cells; cells owned by a smaller instance are never stolen
    S2 = 64
    big = torch.zeros(S2, S2, dtype=torch.bool); big[8:40, 8:40] = True          # cells 1..4 x 1..4, centroid cell (2, 2)
    smalls = []
    for dy in range(3):
        for dx in range(3):
            m_ = torch.zeros(S2, S2, dtype=torch.bool)
            m_[8 + dy * 8 + 2:8 + dy * 8 + 7, 8 + dx * 8 + 2:8 + dx * 8 + 7] = True   # 5x5 blob in cells 1..3 x 1..3
            smalls.append(m_)
    masks_c = torch.stack([big] + smalls)
    t_c = Transformed(torch.zeros(3, S2, S2), masks_c, torch.zeros(S2, S2, dtype=torch.bool), torch.ones(S2, S2, dtype=torch.bool),
                      torch.zeros(len(masks_c), dtype=torch.long), list(range(len(masks_c))), 0, TransformParams(1, S2, S2, False, 0, 0), (S2, S2))
    g_c = make_seg_gt(t_c, 3, SegGtCfg(center="centroid"))
    assert 0 in g_c.pos_inst.tolist(), "big instance reclaims an unowned occupied cell"
    assert set(g_c.pos_inst.tolist()) == set(range(len(masks_c)))
    # equal-area collision: lower index wins the shared cells (same tie-break as owner_map)
    a_ = torch.zeros(64, 64, dtype=torch.bool); a_[8:24, 8:24] = True
    t_eq = Transformed(torch.zeros(3, 64, 64), torch.stack([a_, a_.clone()]), torch.zeros(64, 64, dtype=torch.bool),
                       torch.ones(64, 64, dtype=torch.bool), torch.tensor([0, 1]), [1, 2], 0, TransformParams(1, 64, 64, False, 0, 0), (64, 64))
    g_eq = make_seg_gt(t_eq, 3, SegGtCfg(center="centroid"))
    assert set(g_eq.pos_inst.tolist()) == {0}, "identical masks: lower index owns every cell; nothing is stolen for the other"
    # an instance with no visible pixel (cropped away): no peak, no positives, centre (-1, -1)
    t_inv = fake_transformed()
    t_inv.masks[1] = False
    g_inv = make_seg_gt(t_inv, 3)
    assert 1 not in g_inv.pos_inst.tolist() and g_inv.heat[1].max() == 0 and tuple(g_inv.centers[1].tolist()) == (-1.0, -1.0)
    assert not bool(g_inv.center_pixel_inside[1]) and bool(g_inv.center_pixel_inside[0])
    g_inv_c = make_seg_gt(t_inv, 3, SegGtCfg(center="centroid"))
    assert 1 not in g_inv_c.pos_inst.tolist() and tuple(g_inv_c.centers[1].tolist()) == (-1.0, -1.0)
    # empty sample
    t0 = Transformed(torch.zeros(3, 64, 64), torch.zeros(0, 64, 64, dtype=torch.bool), torch.zeros(64, 64, dtype=torch.bool),
                     torch.ones(64, 64, dtype=torch.bool), torch.zeros(0, dtype=torch.long), [], 0, TransformParams(1, 64, 64, False, 0, 0), (64, 64))
    g0 = make_seg_gt(t0, 3)
    assert g0.heat.sum() == 0 and len(g0.pos_index) == 0 and g0.masks_s4.shape == (0, 16, 16)
    # max_pos cap: every instance keeps >= 1 positive, total == cap, reproducible
    g1 = make_seg_gt(t, 3, SegGtCfg(max_pos=4), torch.Generator().manual_seed(0))
    g2 = make_seg_gt(t, 3, SegGtCfg(max_pos=4), torch.Generator().manual_seed(0))
    assert len(g1.pos_index) == 4 and torch.equal(g1.pos_index, g2.pos_index)
    assert set(g1.pos_inst.tolist()) == {0, 1, 2}, "no instance loses all its positives"
    g3 = make_seg_gt(t, 3, SegGtCfg(max_pos=2), torch.Generator().manual_seed(0))
    assert len(g3.pos_index) == 3 and set(g3.pos_inst.tolist()) == {0, 1, 2}, "cap below #instances still keeps one each"
    print("make_seg_gt unit test OK")


def stats(cs, a):
    """Centre-outside rate and positives per instance over N images (eval transform)."""
    from lova.data.coco.load import load
    ids = cs.image_ids()
    rng = np.random.RandomState(a.seed)
    n_inst = n_out = n_pos = 0
    for k in rng.choice(len(ids), a.stats, replace=False):
        s = load(cs, ids[int(k)])
        t = transform(s, TransformCfg(size=a.size, train=False), torch.Generator().manual_seed(int(k)))
        gt = make_seg_gt(t, cs.num_classes, SegGtCfg(center=a.center))
        n_inst += len(t); n_out += int((~gt.center_pixel_inside).sum()); n_pos += len(gt.pos_index)
    print(f"{a.stats} images, {n_inst} instances: centre outside the mask {n_out} ({n_out / max(n_inst, 1):.1%}), "
          f"positives per instance {n_pos / max(n_inst, 1):.2f}  [center={a.center}]")


def main():
    p = argparse.ArgumentParser()
    add_image_args(p)
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--center", default="deepest_owned", choices=["centroid", "deepest", "deepest_owned"])
    p.add_argument("--stats", type=int, default=0, help="N random images: centre-outside rate etc. (no window)")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    unit_test()
    if a.root is None:
        return
    from lova.data.coco.load import load, open_coco

    cs = open_coco(a.root, a.split)
    if a.stats:
        return stats(cs, a)
    img_id = resolve_image_id(cs, a)
    s = load(cs, img_id)
    t = transform(s, TransformCfg(size=a.size, train=not a.eval), torch.Generator().manual_seed(a.seed))
    gt = make_seg_gt(t, cs.num_classes, SegGtCfg(center=a.center))
    names = [cs.name_of_label(int(l)) for l in t.labels]
    S = a.size
    h8 = S // 8
    print(f"image {img_id}: {len(t)} instances, {len(gt.pos_index)} positive cells, center={a.center}")
    print(f"  {'instance':<14}{'area px':>9}{'centre in':>11}{'#pos':>6}{'sigma':>7}")
    for i in range(len(t)):
        npos = int((gt.pos_inst == i).sum())
        sigma = max(0.8, (float(t.masks[i].sum()) ** 0.5) / 8 / 6)
        print(f"  {names[i][:13]:<14}{int(t.masks[i].sum()):>9}{str(bool(gt.center_pixel_inside[i])):>11}{npos:>6}{sigma:>7.2f}")

    base = Image.fromarray((denormalize(t.image) * 255).permute(1, 2, 0).numpy().astype(np.uint8))
    img = overlay_masks(base, t.masks, filled=[], labels=names)
    d = ImageDraw.Draw(img)
    for q, i in zip(gt.pos_index.tolist(), gt.pos_inst.tolist()):
        y, x = q // h8, q % h8
        d.rectangle([x * 8, y * 8, x * 8 + 7, y * 8 + 7], outline=(255, 255, 0), width=1)
    for i in range(len(t)):
        cx, cy = gt.centers[i].tolist()
        col = (0, 255, 0) if gt.center_pixel_inside[i] else (255, 0, 0)
        d.line([cx - 5, cy - 5, cx + 5, cy + 5], fill=col, width=2); d.line([cx - 5, cy + 5, cx + 5, cy - 5], fill=col, width=2)
    hm = gt.heat.max(0).values
    hm = torch.nn.functional.interpolate(hm[None, None], size=(S, S), mode="nearest")[0, 0]
    base_t = torch.from_numpy(np.array(base)).permute(2, 0, 1).float() / 255
    heat_img = Image.fromarray(((torch.stack([hm, torch.zeros_like(hm), 1 - hm]) * 0.7 + base_t * 0.3).clamp(0, 1)
                                .permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    ign = ~torch.nn.functional.interpolate(gt.heat_valid[None, None].float(), size=(S, S), mode="nearest")[0, 0].bool()
    arr = np.array(base).astype(np.float32)
    arr[ign.numpy()] = arr[ign.numpy()] * 0.3 + np.array([255, 0, 255]) * 0.7
    ign_img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    panel = hstack([img, heat_img, ign_img],
                   [f"id {img_id}: centres (green in / red out) + positive cells", "heat (max over classes)",
                    f"ignored cells (magenta): crowd + padding, {int((~gt.heat_valid).sum())} of {h8 * h8}"])
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        where = os.path.join(a.out, f"seggt_{img_id}_s{a.seed}_{a.center}.png")
        panel.save(where)
    else:
        where = "(window)"
        panel.show(title=f"seg GT {img_id}")
    print(f"  -> {where}")


if __name__ == "__main__":
    main()
