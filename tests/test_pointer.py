"""lova.data.common.pointer: safe interior + uniform pointer sampling.

    python tests/test_pointer.py                                               # unit test (synthetic)
    python tests/test_pointer.py --root datasets/coco --image-id 2153 --seed 0 --k 30
    python tests/test_pointer.py --root datasets/coco --image-id 39769 --seed 5 --k 30 --alpha 0.2

Window: transformed image; the chosen instance's mask filled, its excluded boundary band darkened,
the safe interior bright, and k sampled pointers (yellow). Repeated with different generators so
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lova.data.common.pointer import (PointerCfg, depth, distance_to, make_pointer, pick_target,  # noqa: E402
                                      safe_region, sample_pointer)
from lova.data.common.select import SelectCfg, select  # noqa: E402
from lova.data.common.transform import TransformCfg, denormalize, transform  # noqa: E402
from tests.viz import draw_pointer, hstack, overlay_masks  # noqa: E402


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
    cfg = PointerCfg(alpha=0.2, depth_stride=1)
    safe = safe_region(circ, cfg)
    assert safe.sum() < circ.sum() and bool(safe[64, 64]) and not bool(safe[64, 64 + 38])
    assert bool((safe & ~circ).sum() == 0)
    # alpha = 0 keeps everything, larger alpha shrinks
    assert torch.equal(safe_region(circ, PointerCfg(alpha=0.0, depth_stride=1)), circ)
    assert safe_region(circ, PointerCfg(alpha=0.5, depth_stride=1)).sum() < safe.sum()
    # stride-4 depth gives nearly the same safe region as full-res (differs by a <= 4 px ring)
    yy2, xx2 = torch.meshgrid(torch.arange(256), torch.arange(256), indexing="ij")
    big = (yy2 - 128) ** 2 + (xx2 - 128) ** 2 < 100 ** 2
    s1 = safe_region(big, PointerCfg(alpha=0.2, depth_stride=1))
    s4 = safe_region(big, PointerCfg(alpha=0.2, depth_stride=4))
    assert (s4 ^ s1).float().sum() / s1.float().sum() < 0.2

    # sampling: always inside, never in the excluded band, spread over the interior
    g = torch.Generator().manual_seed(0)
    pts = [sample_pointer(circ, cfg, g) for _ in range(500)]
    for x, y in pts:
        assert bool(circ[y, x]) and bool(safe[y, x])
    r = torch.tensor([((x - 64) ** 2 + (y - 64) ** 2) ** 0.5 for x, y in pts])
    inner = float((r < 20).float().mean())            # area fraction of r<20 inside r<32 (safe disc) ~ 0.39
    assert 0.25 < inner < 0.55, inner                  # uniform over area, not clustered at the centre
    assert r.max() <= 33

    # thin bar: safe region may be empty at stride 4 -> fallback to the mask itself
    bar = torch.zeros(S, S, dtype=torch.bool)
    bar[60:63, 10:118] = True
    x, y = sample_pointer(bar, PointerCfg(alpha=0.1, depth_stride=4), g)
    assert bool(bar[y, x])

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
    print("pointer unit test OK")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=None)
    p.add_argument("--split", default="val2017")
    p.add_argument("--image-id", type=int, default=None)
    p.add_argument("--index", type=int, default=0)
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
    img_id = a.image_id if a.image_id is not None else cs.image_ids()[a.index]
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
        safe = safe_region(m, cfg)
        pts = [sample_pointer(m, cfg, g) for _ in range(a.k)]
        dep = depth(F_pool(m, cfg.depth_stride))
        dmax = float(dep.max())
        dpts = [float(dep[y // cfg.depth_stride, x // cfg.depth_stride]) for x, y in pts]
        img = overlay_masks(base, m[None], labels=[names[idx]], alpha=0.35)
        arr = np.asarray(img).astype(np.float32)
        band = (m & ~safe).numpy()
        arr[band] *= 0.45                                      # excluded boundary band: dark
        arr[safe.numpy()] = arr[safe.numpy()] * 0.6 + 255 * 0.4  # safe interior: bright
        img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
        for q in pts:
            draw_pointer(img, q, radius=4)
        panels.append(img)
        titles.append(f"{names[idx]}  safe {float(safe.sum()) / float(m.sum()):.0%} of mask  alpha {a.alpha}  k={a.k}")
        print(f"  {names[idx]:<14} mask {int(m.sum()):>7} px  safe {float(safe.sum()) / float(m.sum()):.0%}  "
              f"max depth {dmax * cfg.depth_stride:.0f} px  sampled depth min/mean {min(dpts) * cfg.depth_stride:.0f}/"
              f"{np.mean(dpts) * cfg.depth_stride:.0f} px")
    panel = hstack(panels, titles)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        where = os.path.join(a.out, f"pointer_{img_id}_s{a.seed}.png")
        panel.save(where)
    else:
        where = "(window)"
        panel.show(title=f"pointer {img_id}")
    print(f"  -> {where}")


def F_pool(m, st):
    import torch.nn.functional as F
    return F.avg_pool2d(m[None, None].float(), st)[0, 0] >= 0.5 if st > 1 else m


if __name__ == "__main__":
    main()
