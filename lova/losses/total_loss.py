"""Step 8-4: the thin wrapper. Three loss calls and a weighted sum; nothing else.

    out   = model(image, pointer)   must provide
        r_pred       [B, 1, H4, W4]   (or [B, H4, W4])
        heat_logits  [B, K, H8, W8]
        kernel_map   [B, D, H8, W8]
        mask_feat    [B, D, H4, W4]
    batch = collate(...)            provides r_gt, r_valid, heat, heat_valid, pos_index, pos_inst, masks_s4, mask_valid

    losses = total_loss(out, batch, cfg)   -> {"total", "r", "heat", "mask"}  (weighted terms; raw values too)
    stats  = total_stats(out, batch, losses)   logging only; reuses the mask loss, never recomputed per step

    L = w_r * L_R + w_heat * L_heat + w_mask * L_mask

The default weights (1, 1, 1) are integration placeholders, not a decision: once a model produces a
batch, look at the raw magnitudes and the gradient scale on the shared backbone, then set them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .heat_loss import HeatLossCfg, heat_loss, heat_stats
from .mask_loss import MaskLossCfg, mask_loss, mask_stats
from .r_loss import RLossCfg, r_loss, r_stats

REQUIRED_OUT = ("r_pred", "heat_logits", "kernel_map", "mask_feat")
REQUIRED_BATCH = ("r_gt", "r_valid", "heat", "heat_valid", "pos_index", "pos_inst", "masks_s4", "mask_valid")


@dataclass(frozen=True)
class TotalLossCfg:
    w_r: float = 1.0        # placeholders until a real batch shows the raw magnitudes
    w_heat: float = 1.0
    w_mask: float = 1.0
    r: RLossCfg = field(default_factory=RLossCfg)
    heat: HeatLossCfg = field(default_factory=HeatLossCfg)
    mask: MaskLossCfg = field(default_factory=MaskLossCfg)


def _check(out: dict, batch: dict) -> None:
    missing = [k for k in REQUIRED_OUT if k not in out]
    if missing:
        raise KeyError(f"model output lacks {missing}")
    missing = [k for k in REQUIRED_BATCH if k not in batch]
    if missing:
        raise KeyError(f"batch lacks {missing}")


def total_loss(out: dict, batch: dict, cfg: TotalLossCfg = TotalLossCfg()) -> dict:
    """Returns raw terms (r_raw, heat_raw, mask_raw), weighted terms (r, heat, mask) and their sum (total)."""
    _check(out, batch)
    lr = r_loss(out["r_pred"], batch["r_gt"], batch["r_valid"], cfg.r)
    lh = heat_loss(out["heat_logits"], batch["heat"], batch["heat_valid"], cfg.heat)
    lm = mask_loss(out["kernel_map"], out["mask_feat"], batch["pos_index"], batch["pos_inst"],
                   batch["masks_s4"], batch["mask_valid"], cfg.mask)
    terms = {"r": cfg.w_r * lr, "heat": cfg.w_heat * lh, "mask": cfg.w_mask * lm}
    return {"total": terms["r"] + terms["heat"] + terms["mask"], **terms,
            "r_raw": lr.detach(), "heat_raw": lh.detach(), "mask_raw": lm.detach()}


@torch.no_grad()
def total_stats(out: dict, batch: dict, losses: dict | None = None) -> dict:
    """Diagnostics as 0-d tensors (call .item() when logging). The mask dice is derived from the
    already computed raw mask loss when `losses` is given."""
    _check(out, batch)
    st = {}
    st.update(r_stats(out["r_pred"], batch["r_gt"], batch["r_valid"]))
    st.update(heat_stats(out["heat_logits"], batch["heat"], batch["heat_valid"]))
    st.update(mask_stats(out["kernel_map"], out["mask_feat"], batch["pos_index"], batch["pos_inst"],
                         batch["masks_s4"], batch["mask_valid"],
                         loss=None if losses is None else losses["mask_raw"]))
    return st
