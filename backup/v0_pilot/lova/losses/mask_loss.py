"""Step 8-3: mask loss. Dynamic kernels at the positive cells reproduce the instance masks.

    kernel_map  [B, D, H8, W8]   head output: one kernel vector per stride-8 cell
    mask_feat   [B, D, H4, W4]   head output: shared mask feature at stride 4
    pos_index   list of [P_b] long  SegGT.pos_index: flat stride-8 cell indices, LOCAL to image b
    pos_inst    list of [P_b] long  SegGT.pos_inst: which instance each positive cell represents
    masks_s4    list of [N_b, H4, W4] float  SegGT.masks_s4: soft (area-averaged) instance masks
    mask_valid  [B, H4, W4] bool   SegGT.mask_valid: crowd / padding pixels leave the loss entirely

    per image b:  K_b = kernel_map[b][:, pos_index_b]              [P_b, D]
                  logits_b = K_b @ mask_feat[b]                     [P_b, H4, W4]   (1x1 dynamic conv)
                  target_b = masks_s4[b][pos_inst_b]                [P_b, H4, W4]   soft, NOT thresholded
    soft dice (squared denominator, as SOLOv2), valid applied to numerator AND denominator:
                  D_j = (2 sum V p y + eps) / (sum V p^2 + sum V y^2 + eps),   loss_j = 1 - D_j
    L_mask = mean over all P positives of the batch (an image with more positives weighs more; a
             per-image mean is the alternative, kept for an ablation). No positives -> 0 with a
             gradient path.

The GT indexing (pos_index / pos_inst) is done here, not in the model: the model only emits
kernel_map and mask_feat for every cell.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MaskLossCfg:
    eps: float = 1e-3


def dynamic_masks(kernel_map_b: torch.Tensor, mask_feat_b: torch.Tensor, pos_index_b: torch.Tensor) -> torch.Tensor:
    """[D, H8, W8], [D, H4, W4], [P] -> mask logits [P, H4, W4]."""
    k = kernel_map_b.flatten(1)[:, pos_index_b].t()                 # [P, D]
    return torch.einsum("pd,dhw->phw", k, mask_feat_b)


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """logits / target [P, H, W], valid [H, W] bool -> [P] (1 - dice). Ignored pixels contribute to
    neither numerator nor denominator."""
    if logits.dim() != 3 or logits.shape != target.shape or valid.shape != logits.shape[1:]:
        raise ValueError(f"logits {tuple(logits.shape)} / target {tuple(target.shape)} must be [P,H,W] and valid [H,W], got {tuple(valid.shape)}")
    p = torch.sigmoid(logits.float())
    t = target.to(device=p.device, dtype=p.dtype)
    v = valid.to(device=p.device, dtype=p.dtype)[None]
    inter = (v * p * t).flatten(1).sum(1)
    denom = (v * p * p).flatten(1).sum(1) + (v * t * t).flatten(1).sum(1)
    return 1.0 - (2.0 * inter + eps) / (denom + eps)


def mask_loss(kernel_map: torch.Tensor, mask_feat: torch.Tensor, pos_index: list, pos_inst: list,
              masks_s4: list, mask_valid: torch.Tensor, cfg: MaskLossCfg = MaskLossCfg()) -> torch.Tensor:
    B = kernel_map.shape[0]
    if kernel_map.dim() != 4 or mask_feat.dim() != 4 or mask_feat.shape[:2] != kernel_map.shape[:2]:
        raise ValueError(f"kernel_map {tuple(kernel_map.shape)} and mask_feat {tuple(mask_feat.shape)} must be [B,D,*,*] with the same B, D")
    if not (len(pos_index) == len(pos_inst) == len(masks_s4) == B) or mask_valid.shape != (B, *mask_feat.shape[2:]):
        raise ValueError("pos_index / pos_inst / masks_s4 must have one entry per image; mask_valid must be [B,H4,W4]")
    n_cells = kernel_map.shape[2] * kernel_map.shape[3]
    losses = []
    for b in range(B):
        idx, inst = pos_index[b].to(kernel_map.device), pos_inst[b].to(kernel_map.device)
        if idx.numel() != inst.numel():
            raise ValueError(f"image {b}: pos_index has {idx.numel()} entries but pos_inst {inst.numel()}")
        if idx.numel() == 0:
            continue
        # a negative index would silently pick the last cell / instance: refuse
        if int(idx.min()) < 0 or int(idx.max()) >= n_cells:
            raise ValueError(f"image {b}: pos_index out of range for a {tuple(kernel_map.shape[2:])} kernel grid")
        if int(inst.min()) < 0 or int(inst.max()) >= masks_s4[b].shape[0]:
            raise ValueError(f"image {b}: pos_inst out of range for {masks_s4[b].shape[0]} instances")
        logits = dynamic_masks(kernel_map[b].float(), mask_feat[b].float(), idx)
        target = masks_s4[b].to(mask_feat.device)[inst]
        losses.append(soft_dice_loss(logits, target, mask_valid[b], cfg.eps))
    if not losses:
        return kernel_map.float().sum() * 0.0 + mask_feat.float().sum() * 0.0
    return torch.cat(losses).mean()


@torch.no_grad()
def mask_stats(kernel_map: torch.Tensor, mask_feat: torch.Tensor, pos_index: list, pos_inst: list,
               masks_s4: list, mask_valid: torch.Tensor, loss: torch.Tensor | None = None) -> dict:
    """0-d tensors: number of positive cells and their mean soft dice (1 - loss). Pass the loss you
    already computed to avoid redoing the dynamic masks; without positives the dice is NaN (undefined),
    not 1. Call at logging time only."""
    n = sum(int(p.numel()) for p in pos_index)
    dev = kernel_map.device
    if n == 0:
        return dict(n_pos_cells=torch.zeros((), device=dev), dice=torch.full((), float("nan"), device=dev))
    if loss is None:
        loss = mask_loss(kernel_map, mask_feat, pos_index, pos_inst, masks_s4, mask_valid)
    return dict(n_pos_cells=torch.tensor(float(n), device=dev), dice=1.0 - loss.detach())
