"""lova.utils.geometry: pure grid-geometry checks on synthetic masks (no data needed).

    python tests/utils/test_geometry.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lova.utils.geometry import (depth, depth_chebyshev, depth_l1, distance_l1, distance_to,  # noqa: E402
                                       downsample_mask, erode, geodesic, owner_map, pointer_to_stride,
                                       propagate_value, seed_cell)


def unit_test():
    rng = np.random.default_rng(0)
    # chamfer distance vs exact Euclidean: <= 9 % relative error
    for _ in range(5):
        h, w = int(rng.integers(10, 60)), int(rng.integers(10, 60))
        m = torch.from_numpy(rng.random((h, w)) < 0.05)
        if not m.any():
            m[1, 1] = True
        d = distance_to(m).numpy()
        ys, xs = np.nonzero(m.numpy())
        yy, xx = np.mgrid[0:h, 0:w]
        exact = np.sqrt(((yy[..., None] - ys) ** 2 + (xx[..., None] - xs) ** 2).min(-1))
        assert (np.abs(d - exact) / np.maximum(exact, 1)).max() < 0.09
    assert torch.isinf(distance_to(torch.zeros(5, 5, dtype=torch.bool))).all(), "empty target -> inf"

    # depth: 0 on the outermost ring, growing inward, 0 outside; border is not a boundary
    S = 64
    yy, xx = torch.meshgrid(torch.arange(S), torch.arange(S), indexing="ij")
    disc = (yy - 32) ** 2 + (xx - 32) ** 2 < 20 ** 2
    dp = depth(disc)
    assert dp[32, 32] == dp.max() and dp[0, 0] == 0 and dp[32, 32 + 19] == 0 and dp[32, 32 + 18] >= 1
    full = torch.ones(S, S, dtype=torch.bool)
    assert torch.isinf(depth(full)).all() or bool((depth(full) >= 0).all())  # no outside at all: no ring
    # chebyshev depth: 0 on the ring, max at the centre, consistent with erode(), batched
    dc = depth_chebyshev(disc)
    assert dc[32, 32] == dc.max() and dc[32, 32 + 19] == 0 and dc[0, 0] == 0
    assert torch.equal(dc >= 2, erode(disc, 2)), "depth >= k  <=>  survives k erosions"
    assert torch.equal(depth_chebyshev(torch.stack([disc, disc]))[1], dc)
    # L1 distance: exact Manhattan distance vs brute force, batched, inf when empty
    for _ in range(3):
        h, w = int(rng.integers(8, 40)), int(rng.integers(8, 40))
        tgt = torch.from_numpy(rng.random((h, w)) < 0.06)
        if not tgt.any():
            tgt[2, 3] = True
        ys, xs = np.nonzero(tgt.numpy()); yy, xx = np.mgrid[0:h, 0:w]
        bf = (np.abs(yy[..., None] - ys) + np.abs(xx[..., None] - xs)).min(-1)
        assert np.array_equal(distance_l1(tgt).numpy(), bf)
        assert torch.equal(distance_l1(torch.stack([tgt, tgt]))[1], distance_l1(tgt))
    assert torch.isinf(distance_l1(torch.zeros(4, 4, dtype=torch.bool))).all()
    dl = depth_l1(disc)
    assert dl[32, 32] == dl.max() and dl[32, 32 + 19] == 0 and dl[0, 0] == 0 and dl[32, 32 + 18] == 1
    # erode: Chebyshev margin
    e = erode(disc, 2)
    assert e.sum() < disc.sum() and bool((e & ~disc).sum() == 0) and bool(e[32, 32])
    assert float(depth(disc)[e].min()) >= 2
    assert torch.equal(erode(disc, 0), disc)

    # geodesic: U shape, the far arm is far although Euclidean-close; outside -> inf
    m = torch.zeros(128, 128, dtype=torch.bool)
    m[20:110, 20:35] = True
    m[95:110, 20:110] = True
    m[20:110, 95:110] = True
    g = geodesic(m, (30, 27))
    assert g[30, 27] == 0 and torch.isinf(g[60, 65]) and g[25, 102] > 150 > g[105, 27] > 0
    m2 = m.clone()
    m2[5:10, 60:70] = True
    assert torch.isinf(geodesic(m2, (30, 27))[5:10, 60:70]).all(), "disconnected part unreachable"

    # propagate_value: nearest-source value is carried, distance grows, reach limits the extent
    src = torch.zeros(32, 32, dtype=torch.bool)
    src[5, 5] = True
    src[25, 25] = True
    val = torch.zeros(32, 32)
    val[5, 5], val[25, 25] = 0.2, 0.9
    d, v = propagate_value(src, val, reach=10)
    assert v[6, 6] == 0.2 and v[24, 24] == 0.9 and abs(float(d[5, 10]) - 5) < 1e-5
    assert torch.isinf(d[5, 25]) and v[5, 25] == 0, "beyond reach"
    assert abs(float(d[6, 6]) - 2 ** 0.5) < 1e-5

    # batched [N, h, w] distance / depth / erode == per-item
    stack = torch.stack([disc, m[:64, :64], torch.zeros(64, 64, dtype=torch.bool)])
    db = distance_to(stack)
    for i in range(3):
        assert torch.equal(db[i], distance_to(stack[i]))
    assert torch.equal(depth(stack)[0], depth(disc)) and torch.equal(erode(stack, 2)[0], erode(disc, 2))
    # owner map: smallest covering instance wins, ties -> lower index, -1 elsewhere
    big = torch.zeros(32, 32, dtype=torch.bool); big[4:28, 4:28] = True
    mid = torch.zeros(32, 32, dtype=torch.bool); mid[8:20, 8:20] = True
    tiny = torch.zeros(32, 32, dtype=torch.bool); tiny[10:13, 10:13] = True
    om = owner_map(torch.stack([big, mid, tiny]))
    assert om[5, 5] == 0 and om[9, 9] == 1 and om[11, 11] == 2 and om[0, 0] == -1
    dup = owner_map(torch.stack([mid, mid.clone()]))
    assert bool((dup[mid] == 0).all()), "equal area -> lower index owns"
    # resolution helpers
    assert pointer_to_stride((0, 0), 2) == (-0.25, -0.25) and pointer_to_stride((3, 5), 2) == (1.25, 2.25)
    ds = downsample_mask(disc, 2)
    assert ds.shape == (32, 32) and abs(float(ds.sum()) * 4 - float(disc.sum())) / float(disc.sum()) < 0.05
    assert seed_cell(m, (27.0, 30.0)) == (30, 27)
    assert seed_cell(m, (10.0, 30.0)) == (30, 20), "snaps to the nearest mask cell"
    print("geometry unit test OK")


if __name__ == "__main__":
    unit_test()
