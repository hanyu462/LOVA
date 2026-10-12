"""lova.models.common / output: block shapes, GN groups, ResBlock identity at init, coord_grid convention,
the ModelOutput contract. Synthetic only.

    python tests/models/test_common.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lova.models.common import ConvGNAct, PreActConv, ResBlock, coord_grid, count_params, gn  # noqa: E402
from lova.models.output import ModelOutput  # noqa: E402
from lova.utils.geometry import pointer_to_stride  # noqa: E402


def unit_test():
    torch.manual_seed(0)
    # gn: largest group count <= 32 dividing c
    assert gn(64).num_groups == 32 and gn(48).num_groups == 16 and gn(6).num_groups == 2 and gn(7).num_groups == 1
    x = torch.randn(2, 16, 32, 32)
    assert ConvGNAct(16, 24)(x).shape == (2, 24, 32, 32)
    assert ConvGNAct(16, 24, s=2)(x).shape == (2, 24, 16, 16)
    assert ConvGNAct(16, 24, d=2)(x).shape == (2, 24, 32, 32)          # dilation keeps the size
    assert PreActConv(16, 8)(x).shape == (2, 8, 32, 32)
    # ResBlock is the identity when fresh, and not after a weight change
    blk = ResBlock(16)
    assert torch.equal(blk(x), x)
    blk.body[1][2].weight.data.normal_()
    assert not torch.allclose(blk(x), x)
    blk(x).sum().backward()
    assert all(p.grad is not None for p in blk.parameters())
    # coord_grid matches geometry.pointer_to_stride: the pixel at a cell centre maps to that integer cell
    g = coord_grid(4, 6, 8)
    assert g.shape == (2, 4, 6) and float(g[0, 0, 0]) == 3.5 and float(g[1, 0, 0]) == 3.5 and float(g[0, 0, 1]) == 11.5
    cx, cy = pointer_to_stride((float(g[0, 2, 3]), float(g[1, 2, 3])), 8)
    assert abs(cx - 3) < 1e-6 and abs(cy - 2) < 1e-6
    # ModelOutput contract
    B, K, D, S = 2, 5, 8, 64
    out = ModelOutput(torch.rand(B, 1, S // 4, S // 4), torch.randn(B, K, S // 8, S // 8),
                      torch.randn(B, D, S // 8, S // 8), torch.randn(B, D, S // 4, S // 4))
    out.check((B, 3, S, S), K)
    assert set(out.as_dict()) == {"r_pred", "heat_logits", "kernel_map", "mask_feat"}
    for bad in (ModelOutput(out.r_pred[:, :, :-1], out.heat_logits, out.kernel_map, out.mask_feat),
                ModelOutput(out.r_pred, out.heat_logits[:, :-1], out.kernel_map, out.mask_feat),
                ModelOutput(out.r_pred, out.heat_logits, out.kernel_map, out.mask_feat[:, :-1]),
                ModelOutput(out.r_pred * 2, out.heat_logits, out.kernel_map, out.mask_feat)):
        try:
            bad.check((B, 3, S, S), K)
            raise AssertionError("must raise")
        except ValueError:
            pass
    assert abs(count_params(ConvGNAct(16, 16)) - (16 * 16 * 9 + 32) / 1e6) < 1e-9
    print("models.common / output unit test OK")


if __name__ == "__main__":
    unit_test()
