"""Synthetic R fields for Phase A (training the backbone to be R-controllable
before any R predictor exists). Mixture per image:

  const  : R = c everywhere, c ~ U(0,1), with explicit 0 / 1 extremes ("sandwich")
  oracle : R = b + (1-b) * blur(pointed mask), b ~ U(0, 0.5)
  noise  : smooth random field (bilinear-upsampled low-res uniform noise)

Every GT instance is supervised regardless of its R, so the network learns
"quality as a function of R" instead of "suppress where R is low".
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def blur(x, k=5, n=2):
    for _ in range(n):
        x = F.avg_pool2d(x, k, 1, k // 2, count_include_pad=False)
    return x


@torch.no_grad()
def sample_r(pointed_mask4: torch.Tensor, probs=(0.3, 0.5, 0.2), p_extreme: float = 0.3) -> torch.Tensor:
    b, _, h, w = pointed_mask4.shape
    dev = pointed_mask4.device
    out = torch.empty(b, 1, h, w, device=dev)
    modes = torch.multinomial(torch.tensor(probs), b, replacement=True).tolist()
    soft = blur(pointed_mask4).clamp(0, 1)
    for i, mode in enumerate(modes):
        if mode == 0:
            u = torch.rand(()).item()
            c = float(torch.randint(2, ()).item()) if u < p_extreme else torch.rand(()).item()
            out[i] = c
        elif mode == 1:
            base = 0.5 * torch.rand(()).item()
            out[i] = base + (1 - base) * soft[i]
        else:
            g = int(torch.randint(2, 9, ()).item())
            noise = torch.rand(1, 1, g, g, device=dev)
            out[i] = F.interpolate(noise, size=(h, w), mode="bilinear", align_corners=False)[0]
    return out.clamp(0, 1)
