"""Step 8-2: heat loss. CenterNet penalty-reduced focal loss on the class heatmap, with ignore.

    heat_logits [B, K, H8, W8]    head output (pre-sigmoid)
    heat        [B, K, H8, W8]    SegGT.heat: 1 at instance centres, gaussian shoulders, 0 elsewhere
    heat_valid  [B, H8, W8] bool  SegGT.heat_valid: False on crowd / padding -> cell excluded for all classes

    p = sigmoid(logit)
    positive (heat == 1):   -(1 - p)^alpha * log p
    negative (heat <  1):   -(1 - heat)^beta * p^alpha * log(1 - p)      shoulders are down-weighted
    L = (sum_pos + sum_neg) over valid cells / max(N_pos, 1),  N_pos = number of valid cells with heat == 1
        (cells, not instances: two same-class instances sharing a centre cell count once, as in CenterNet)

Computed in float32 with logsigmoid for numerical safety under autocast.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class HeatLossCfg:
    alpha: float = 2.0    # focal exponent on the prediction
    beta: float = 4.0     # penalty reduction exponent on the gaussian shoulders


def heat_loss(heat_logits: torch.Tensor, heat: torch.Tensor, heat_valid: torch.Tensor,
              cfg: HeatLossCfg = HeatLossCfg()) -> torch.Tensor:
    if heat_logits.dim() != 4 or heat_logits.shape != heat.shape:
        raise ValueError(f"heat_logits {tuple(heat_logits.shape)} must match heat {tuple(heat.shape)} as [B,K,H,W]")
    if heat_valid.shape != (heat.shape[0], heat.shape[2], heat.shape[3]):
        raise ValueError(f"heat_valid must be [B,H,W], got {tuple(heat_valid.shape)}")
    logits = heat_logits.float()
    target = heat.float()
    valid = heat_valid.to(logits.dtype)[:, None]                      # [B,1,H,W] broadcast over classes
    pos = target.eq(1).to(logits.dtype) * valid
    neg = (1.0 - target.eq(1).to(logits.dtype)) * valid
    log_p, log_1mp = F.logsigmoid(logits), F.logsigmoid(-logits)
    p = log_p.exp()
    pos_loss = -(pos * (1.0 - p) ** cfg.alpha * log_p).sum()
    neg_loss = -(neg * (1.0 - target) ** cfg.beta * p ** cfg.alpha * log_1mp).sum()
    return (pos_loss + neg_loss) / pos.sum().clamp(min=1.0)


@torch.no_grad()
def heat_stats(heat_logits: torch.Tensor, heat: torch.Tensor, heat_valid: torch.Tensor) -> dict:
    """0-d tensors: number of positive cells, mean p at positives, mean p at valid negatives."""
    p = torch.sigmoid(heat_logits.float())
    valid = heat_valid.float()[:, None]
    pos = heat.eq(1).float() * valid
    neg = (1.0 - heat.eq(1).float()) * valid
    return dict(n_pos=pos.sum(),
                p_pos=(p * pos).sum() / pos.sum().clamp(min=1.0),
                p_neg=(p * neg).sum() / neg.sum().clamp(min=1.0))
