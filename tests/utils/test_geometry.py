"""lova.utils.geometry: pure grid-geometry checks on synthetic masks (no data needed).

    python tests/utils/test_geometry.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lova.utils.geometry import (depth, distance_to, downsample_mask, erode, geodesic,  # noqa: E402
                                       pointer_to_stride, propagate_value, seed_cell)


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

    # resolution helpers
    assert pointer_to_stride((0, 0), 2) == (-0.25, -0.25) and pointer_to_stride((3, 5), 2) == (1.25, 2.25)
    ds = downsample_mask(disc, 2)
    assert ds.shape == (32, 32) and abs(float(ds.sum()) * 4 - float(disc.sum())) / float(disc.sum()) < 0.05
    assert seed_cell(m, (27.0, 30.0)) == (30, 27)
    assert seed_cell(m, (10.0, 30.0)) == (30, 20), "snaps to the nearest mask cell"
    print("geometry unit test OK")


if __name__ == "__main__":
    unit_test()
