import torch
import torch.nn as nn


def gn(c: int, max_groups: int = 32) -> nn.GroupNorm:
    g = max_groups
    while c % g != 0:
        g //= 2
    return nn.GroupNorm(g, c)


class ConvGNAct(nn.Sequential):
    """Conv -> GN -> GELU (post-activation)."""

    def __init__(self, cin, cout, k=3, s=1, d=1):
        super().__init__(
            nn.Conv2d(cin, cout, k, s, padding=d * (k // 2), dilation=d, bias=False),
            gn(cout),
            nn.GELU(),
        )


class PreActConv(nn.Sequential):
    """GN -> GELU -> Conv (pre-activation). Used inside the residual stream."""

    def __init__(self, cin, cout, k=3, s=1):
        super().__init__(
            gn(cin),
            nn.GELU(),
            nn.Conv2d(cin, cout, k, s, padding=k // 2, bias=False),
        )


def coord_grid(b: int, h: int, w: int, device, dtype=torch.float32) -> torch.Tensor:
    """Normalized (x, y) coordinates in [-1, 1], shape [B, 2, H, W]."""
    ys = torch.linspace(-1, 1, h, device=device, dtype=dtype)
    xs = torch.linspace(-1, 1, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], 0).unsqueeze(0).expand(b, -1, -1, -1)
