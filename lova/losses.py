"""Losses.

L = w_heat * L_focal + w_mask * L_dice                       (instance seg, ALL GT instances)
  + w_r_in * L_R_in + w_r_out * L_R_out + w_budget * L_budget (R supervision / budget)

R supervision is intentionally asymmetric:
  * L_R_in    : R must be high on the pointed instance (BCE toward 1)
  * L_R_out   : weak BCE toward 0 away from it (warm-up only; decays to a floor)
  * L_budget  : mean R over the image must not exceed area(pointed)+extra.
                Inside that budget the task loss is free to raise R elsewhere
                (image-driven importance) instead of R becoming a pure mask copy.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .models.head import dynamic_masks


def focal_heatmap(logits: torch.Tensor, target: torch.Tensor, alpha: float = 2.0, beta: float = 4.0):
    """CenterNet penalty-reduced focal loss, normalized by number of peaks."""
    logits = logits.float()
    pos = target.eq(1).float()
    logp, log1mp = F.logsigmoid(logits), F.logsigmoid(-logits)
    p = logp.exp()
    pos_loss = -(pos * (1 - p) ** alpha * logp).sum()
    neg_loss = -((1 - pos) * (1 - target) ** beta * p**alpha * log1mp).sum()
    return (pos_loss + neg_loss) / pos.sum().clamp(min=1)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-3):
    p = torch.sigmoid(logits.float()).flatten(1)
    t = target.flatten(1)
    return 1 - (2 * (p * t).sum(1) + eps) / ((p * p).sum(1) + (t * t).sum(1) + eps)


def instance_loss(out, batch, w_heat=1.0, w_mask=3.0):
    l_heat = focal_heatmap(out["heat_logits"], batch["heat"])
    dices = []
    for i in range(len(batch["masks4"])):
        pos = batch["pos_index"][i]
        if pos.numel() == 0:
            continue
        logits = dynamic_masks(out["kernels"][i].float(), out["mask_feat"][i].float(), pos)
        dices.append(dice_loss(logits, batch["masks4"][i][batch["pos_inst"][i]]))
    l_mask = torch.cat(dices).mean() if dices else out["kernels"].sum() * 0
    return {"heat": w_heat * l_heat, "mask": w_mask * l_mask}


def r_losses(r, pointed_mask4, valid4, w_in=1.0, w_out=0.1, w_budget=1.0, budget_extra=0.15, margin=2,
             exec_map=None):
    """r, pointed_mask4, valid4: [B,1,h,w].
    exec_map: optional soft execution map g(R) [B,1,h,w]. If given, the budget is applied to
    mean(g) (what actually decides compute under binary routing) instead of mean(R); this also
    stops the degenerate "R = tau - eps everywhere" solution that a mean(R) budget would allow."""
    r = r.float().clamp(1e-4, 1 - 1e-4)
    m = pointed_mask4
    l_in = -(m * torch.log(r)).sum() / m.sum().clamp(min=1)
    near = F.max_pool2d(m, 2 * margin + 1, 1, margin)  # don't push R down right at the instance border
    w = (1 - near) * valid4
    l_out = -(w * torch.log(1 - r)).sum() / w.sum().clamp(min=1)
    vsum = valid4.flatten(1).sum(1).clamp(min=1)
    usage = r if exec_map is None else exec_map.float()
    frac = (usage * valid4).flatten(1).sum(1) / vsum
    target = ((m * valid4).flatten(1).sum(1) / vsum + budget_extra).clamp(max=1)
    l_budget = F.relu(frac - target).pow(2).mean()
    return {"r_in": w_in * l_in, "r_out": w_out * l_out, "r_budget": w_budget * l_budget}


@torch.no_grad()
def r_stats(r, pointed_mask4, valid4):
    m = pointed_mask4 * valid4
    o = (1 - pointed_mask4) * valid4
    return {
        "R_mean": ((r * valid4).sum() / valid4.sum()).item(),
        "R_in": ((r * m).sum() / m.sum().clamp(min=1)).item(),
        "R_out": ((r * o).sum() / o.sum().clamp(min=1)).item(),
    }
