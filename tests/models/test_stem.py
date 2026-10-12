"""lova.models.stem: image [B,3,S,S] -> F0 [B,c0,S/4,S/4]; size, cost, odd sizes. Synthetic only.

    python tests/models/test_stem.py
"""
from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lova.models.common import count_params  # noqa: E402
from lova.models.stem import Stem  # noqa: E402


def unit_test():
    torch.manual_seed(0)
    stem = Stem(64)
    x = torch.randn(2, 3, 640, 640)
    f0 = stem(x)
    assert f0.shape == (2, 64, 160, 160)
    assert stem(torch.randn(1, 3, 96, 128)).shape == (1, 64, 24, 32)      # rectangular input is fine
    assert Stem(32)(x).shape == (2, 32, 160, 160)
    for bad in (torch.randn(1, 3, 100, 128), torch.randn(1, 3, 128, 96 + 8), torch.randn(3, 128, 128),
                torch.randn(1, 3, 0, 64), torch.randn(1, 1, 64, 64)):
        try:
            stem(bad); raise AssertionError("must raise")
        except ValueError:
            pass
    for bad_c0 in (0, 1, 65, -2):
        try:
            Stem(bad_c0); raise AssertionError("must raise")
        except ValueError:
            pass
    f0.sum().backward()
    assert all(p.grad is not None for p in stem.parameters())
    n = count_params(stem)
    assert n < 0.1, n                                                       # cheap by design
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    stem, x = stem.to(dev), x.to(dev)
    with torch.no_grad():
        for _ in range(2): stem(x)
        if dev == "cuda": torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(5): stem(x)
        if dev == "cuda": torch.cuda.synchronize()
        ms = (time.perf_counter() - t) / 5 * 1e3
    print(f"stem unit test OK   params {n:.3f}M   forward B=2 640^2 on {dev}: {ms:.1f} ms")


if __name__ == "__main__":
    unit_test()
