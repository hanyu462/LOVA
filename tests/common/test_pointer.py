"""lova.data.common.pointer: safe interior + uniform pointer sampling.

    python tests/common/test_pointer.py                                               # unit test (synthetic)
    python tests/common/test_pointer.py --root datasets/coco --image-id 2153 --seed 0 --k 30
    python tests/common/test_pointer.py --root datasets/coco --image-id 39769 --seed 5 --k 30 --alpha 0.2

Window: transformed image; the chosen instance's mask filled, pixels owned by a smaller instance
very dark, the excluded boundary band dark, the safe interior bright, k sampled pointers (yellow). Repeated with different generators so
the spread can be judged: pointers should cover the interior, never sit on the boundary, and not
cluster at the centre. Terminal: depth statistics of the sampled points.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lova.data.common.pointer import (PointerCfg, depth, distance_to, make_pointer, owner_of,  # noqa: E402
                                      pick_target, pointer_region, safe_region, sample_from,
                                      sample_pointer, sampling_region)
from lova.data.common.select import SelectCfg, select  # noqa: E402
from lova.data.common.transform import TransformCfg, denormalize, transform  # noqa: E402
from tests.util.args import add_image_args, resolve_image_id  # noqa: E402
from tests.util.viz import draw_pointer, hstack, overlay_masks  # noqa: E402


def unit_test():
    # chamfer distance vs exact on random seeds (<= 9 %)
    rng = np.random.default_rng(0)
    for _ in range(3):
        h, w = int(rng.integers(10, 50)), int(rng.integers(10, 50))
        m = torch.from_numpy(rng.random((h, w)) < 0.05)
        if not m.any():
            m[1, 1] = True
        d = distance_to(m).numpy()
        ys, xs = np.nonzero(m.numpy())
        yy, xx = np.mgrid[0:h, 0:w]
        exact = np.sqrt(((yy[..., None] - ys) ** 2 + (xx[..., None] - xs) ** 2).min(-1))
        assert (np.abs(d - exact) / np.maximum(exact, 1)).max() < 0.09

    # circle: depth is max at the centre, 0 outside; safe region is a smaller disc
    S = 128
    yy, xx = torch.meshgrid(torch.arange(S), torch.arange(S), indexing="ij")
    circ = (yy - 64) ** 2 + (xx - 64) ** 2 < 40 ** 2
    dep = depth(circ)
    assert dep[64, 64] == dep.max() and dep[0, 0] == 0
    assert dep[64, 64 + 39] == 0, "outermost ring has depth 0"
    assert dep[64, 64 + 38] >= 1
    cfg = PointerCfg(alpha=0.2, depth_stride=1, erode_px=0)
    safe = safe_region(circ, cfg)
    assert safe.sum() < circ.sum() and bool(safe[64, 64]) and not bool(safe[64, 64 + 38])
    assert bool((safe & ~circ).sum() == 0)
    # alpha = 0 keeps everything, larger alpha shrinks; any alpha > 0 drops the boundary ring
    assert torch.equal(safe_region(circ, PointerCfg(alpha=0.0, depth_stride=1, erode_px=0)), circ)
    assert safe_region(circ, PointerCfg(alpha=0.5, depth_stride=1, erode_px=0)).sum() < safe.sum()
    tiny = safe_region(circ, PointerCfg(alpha=0.01, depth_stride=1, erode_px=0))
    assert not bool((tiny & (depth(circ) == 0) & circ).any())
    # small object at stride 4 (max depth < 10 cells): the boundary ring is still excluded
    small_disc = (yy - 64) ** 2 + (xx - 64) ** 2 < 14 ** 2       # radius 14 px = 3.5 cells
    s_small = safe_region(small_disc, PointerCfg(alpha=0.1, depth_stride=4))
    full_depth = depth(small_disc)                                  # full-res depth, px
    assert s_small.any() and float(full_depth[s_small].min()) >= 2, float(full_depth[s_small].min())
    # stride-4 depth gives nearly the same safe region as full-res (differs by a <= 4 px ring)
    yy2, xx2 = torch.meshgrid(torch.arange(256), torch.arange(256), indexing="ij")
    big = (yy2 - 128) ** 2 + (xx2 - 128) ** 2 < 100 ** 2
    s1 = safe_region(big, PointerCfg(alpha=0.2, depth_stride=1, erode_px=0))
    s4 = safe_region(big, PointerCfg(alpha=0.2, depth_stride=4, erode_px=0))
    assert (s4 ^ s1).float().sum() / s1.float().sum() < 0.2

    # sampling: always inside, never in the excluded band, spread over the interior
    g = torch.Generator().manual_seed(0)
    region = sampling_region(circ, cfg)
    assert torch.equal(region, safe)
    pts = [sample_from(region, g) for _ in range(500)]
    for x, y in pts:
        assert bool(circ[y, x]) and bool(safe[y, x])
    r = torch.tensor([((x - 64) ** 2 + (y - 64) ** 2) ** 0.5 for x, y in pts])
    inner = float((r < 20).float().mean())            # area fraction of r<20 inside r<~31 (safe disc) ~ 0.4
    assert 0.25 < inner < 0.55, inner                  # uniform over area, not clustered at the centre
    assert r.max() <= 33
    # default cfg (stride 4, erode 2) on the 40 px disc: true boundary distance of every pointer >= 3 px
    reg4 = sampling_region(circ, PointerCfg())
    fd = depth(circ)
    pts4 = [sample_from(reg4, g) for _ in range(300)]
    assert min(float(fd[y, x]) for x, y in pts4) >= 3
    # jagged mask: erosion guarantees the margin even where coarse cells touch the boundary
    jag = circ.clone()
    jag[::3, :] &= (xx[::3, :] < 90)   # notches
    regj = sampling_region(jag, PointerCfg())
    fdj = depth(jag)
    assert regj.any() and float(fdj[regj].min()) >= 2

    # thin bar: safe region may be empty at stride 4 -> fallback to the region itself
    bar = torch.zeros(S, S, dtype=torch.bool)
    bar[60:63, 10:118] = True
    x, y = sample_pointer(bar, PointerCfg(alpha=0.1, depth_stride=4), g)
    assert bool(bar[y, x])
    assert sample_pointer(torch.zeros(S, S, dtype=torch.bool), cfg, g) is None
    assert sample_from(torch.zeros(S, S, dtype=torch.bool), g) is None

    # ownership: big "bed" square containing a small "cat" disc and a tiny "remote"
    bed = torch.zeros(S, S, dtype=torch.bool); bed[8:120, 8:120] = True
    cat = (yy - 50) ** 2 + (xx - 50) ** 2 < 20 ** 2
    remote = torch.zeros(S, S, dtype=torch.bool); remote[100:106, 90:110] = True
    masks = torch.stack([bed, cat, remote])
    own_bed = pointer_region(0, masks)
    assert not bool((own_bed & cat).any()) and not bool((own_bed & remote).any())
    assert torch.equal(pointer_region(1, masks), cat) and torch.equal(pointer_region(2, masks), remote)
    assert owner_of((50, 50), masks) == 1 and owner_of((100, 103), masks) == 2 and owner_of((20, 20), masks) == 0
    assert owner_of((0, 0), masks) is None
    assert owner_of((-1, 5), masks) is None and owner_of((5, S), masks) is None   # out of canvas, no wrap-around
    bed_region = sampling_region(own_bed, cfg)
    safe_bed = safe_region(own_bed, cfg)
    for _ in range(300):  # pointer in the target mask, in its owned region, in safe (non-empty), never on cat/remote
        x, y = sample_from(bed_region, g)
        assert bool(bed[y, x]) and bool(own_bed[y, x]) and bool(safe_bed[y, x])
        assert not bool(cat[y, x]) and not bool(remote[y, x])
    # same seed -> same pointer; different seeds -> spread
    a1 = make_pointer(type("T", (), {"masks": masks}), [0, 1], cfg, torch.Generator().manual_seed(3))
    a2 = make_pointer(type("T", (), {"masks": masks}), [0, 1], cfg, torch.Generator().manual_seed(3))
    assert a1 == a2
    spread = {make_pointer(type("T", (), {"masks": masks}), [0, 1], cfg, torch.Generator().manual_seed(k))[1] for k in range(20)}
    assert len(spread) > 10
    # fully covered target is skipped by make_pointer, not pointed at via fallback
    cover = torch.stack([cat, cat.clone()])  # instance 1 identical to 0 -> index 0 owns everything, 1 owns nothing
    class T2:
        masks = cover
    for _ in range(20):
        idx, (x, y) = make_pointer(T2, [0, 1], cfg, g)
        assert idx == 0

    # pick_target is uniform over candidates
    g = torch.Generator().manual_seed(1)
    picks = torch.tensor([pick_target([3, 7, 1], g) for _ in range(900)])
    for c in (3, 7, 1):
        assert 0.25 < float((picks == c).float().mean()) < 0.42
    # make_pointer end to end on a fake Transformed-like object
    class T:  # minimal stand-in: only .masks is used
        masks = torch.stack([circ, bar])
    idx, p = make_pointer(T, [0, 1], cfg, g)
    assert idx in (0, 1) and bool(T.masks[idx][p[1], p[0]])
    assert make_pointer(T, [], cfg, g) is None
    print("pointer unit test OK")


def main():
    p = argparse.ArgumentParser()
    add_image_args(p)
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--k", type=int, default=20, help="pointers to sample per chosen instance")
    p.add_argument("--all", action="store_true", help="one panel per candidate instead of one random target")
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
    cfg = PointerCfg(alpha=a.alpha)
    names = [cs.name_of_label(int(l)) for l in t.labels]
    print(f"image {img_id}: {len(cands)} candidates {[names[i] for i in cands]}")
    if not cands:
        return
    targets = cands if a.all else [pick_target(cands, g)]

    base = Image.fromarray((denormalize(t.image) * 255).permute(1, 2, 0).numpy().astype(np.uint8))
    panels, titles = [], []
    for idx in targets:
        m = t.masks[idx]
        owned = pointer_region(idx, t.masks)
        safe = safe_region(owned, cfg)
        region = sampling_region(owned, cfg)
        pts = [q for q in (sample_from(region, g) for _ in range(a.k)) if q is not None]
        fd = depth(owned) if owned.any() else torch.zeros(1, 1)      # full-res depth (px) for reporting
        dmax = float(fd.max())
        dpts = [float(fd[y, x]) + 1 for x, y in pts] or [0.0]   # depth 0 = boundary ring -> distance to outside = depth + 1
        img = overlay_masks(base, m[None], labels=[names[idx]], alpha=0.35)
        arr = np.asarray(img).astype(np.float32)
        arr[(m & ~owned).numpy()] *= 0.25                          # owned by a smaller instance: very dark
        arr[(owned & ~safe).numpy()] *= 0.45                       # excluded boundary band: dark
        arr[safe.numpy()] = arr[safe.numpy()] * 0.6 + 255 * 0.4    # safe interior: bright
        img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
        for q in pts:
            draw_pointer(img, q, radius=4)
        panels.append(img)
        titles.append(f"{names[idx]}  owned {float(owned.sum()) / float(m.sum()):.0%}  safe {float(safe.sum()) / float(m.sum()):.0%} of mask  alpha {a.alpha}  k={len(pts)}")
        print(f"  {names[idx]:<14} mask {int(m.sum()):>7} px  owned {float(owned.sum()) / float(m.sum()):.0%}  safe {float(safe.sum()) / float(m.sum()):.0%}  "
              f"max depth {dmax:.0f} px  pointer distance to outside (full-res) min/mean {min(dpts):.0f}/{np.mean(dpts):.0f} px")
    panel = hstack(panels, titles)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        where = os.path.join(a.out, f"pointer_{img_id}_s{a.seed}.png")
        panel.save(where)
    else:
        where = "(window)"
        panel.show(title=f"pointer {img_id}")
    print(f"  -> {where}")


if __name__ == "__main__":
    main()
