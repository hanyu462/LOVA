"""Step 10-6b: fast deterministic validation. No AP here (that is the COCO evaluator, later); this answers
"is the model learning, and does R causally change the segmentation branch?" on a FIXED set of
(image, pointer) pairs so that checkpoint-to-checkpoint differences are not pointer-sampling noise.

    ValSet(root, split, data_cfg, n, seed)      a seeded permutation of the split, keeping the first n
                                                pointer-able images; eval transform (deterministic) and
                                                the pointer drawn from a generator seeded by the dataset
                                                index -> the same (image, pointer) every time
    validate(model, loader, loss_cfg, overrides, amp_dtype) -> {
        "predicted": r_loss, r_mae, heat_loss, mask_loss, dice, budget, total     (the model as trained)
        "sweep":     {"R=0", "R=0.5", "R=1", "predicted"}: heat_loss, mask_loss, dice, budget
                     (r_loss / r_mae are NOT in the sweep: an override changes what the backbone runs on,
                      not the predictor's output; they stay a predicted-R quality metric)
        "n_images" }
    Means are per-batch normalised metrics weighted by batch size: deterministic for a fixed loader (fixed
    subset, pointers, order and batch size) but not invariant to how the set is partitioned into batches.
    The COCO evaluator will aggregate exact numerators / denominators; this is a diagnostic curve.

The validation loader has its OWN torch.Generator, so creating / iterating it never consumes the global
RNG that training uses: switching validation on, off or changing its cadence leaves the training
trajectory untouched (tested).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset

from ..data.coco.dataset import CocoDataset, worker_init
from ..data.common.build import PipelineCfg, TrainingSample, collate
from ..losses.heat_loss import heat_loss
from ..losses.mask_loss import mask_loss, mask_stats
from ..losses.r_loss import r_loss, r_stats
from ..losses.total_loss import TotalLossCfg
from ..models.backbone import valid_to_stride
from .train import to_device

OVERRIDES = ((0.0, "R=0"), (0.5, "R=0.5"), (1.0, "R=1"))


@dataclass(frozen=True)
class ValCfg:
    n_images: int = 500
    seed: int = 0
    batch_size: int = 8
    workers: int = 4


class ValSet(Dataset):
    """A fixed subset of a split with fixed pointers. Built once (the pointer-able check needs a build
    per candidate image); the chosen dataset indices are then stable for the life of the object."""

    def __init__(self, root: str, split: str, data_cfg: PipelineCfg, n: int, seed: int = 0):
        self.ds = CocoDataset(root, split, data_cfg, train=False)
        order = torch.randperm(len(self.ds), generator=torch.Generator().manual_seed(seed)).tolist()
        self.indices, self.image_ids = [], []
        for i in order:
            ts = self.ds.build_one(i)
            if ts is not None:
                self.indices.append(i); self.image_ids.append(ts.image_id)
            if len(self.indices) == n:
                break
        if len(self.indices) < n:
            raise ValueError(f"only {len(self.indices)} pointer-able images in {split}, asked for {n}")

    @property
    def num_classes(self) -> int:
        return self.ds.num_classes

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int) -> TrainingSample:
        ts = self.ds.build_one(self.indices[k])
        assert ts is not None, "a pointer-able index became un-pointer-able: the data config changed"
        return ts


def val_loader(vs: ValSet, cfg: ValCfg) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    return DataLoader(vs, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers, collate_fn=collate,
                      worker_init_fn=worker_init, pin_memory=torch.cuda.is_available(), generator=g)


@torch.no_grad()
def validate(model, loader, loss_cfg: TotalLossCfg = TotalLossCfg(), overrides=OVERRIDES, amp_dtype: torch.dtype | None = torch.bfloat16) -> dict:
    was_training = model.training
    model.eval()
    try:
        return _validate(model, loader, loss_cfg, overrides, amp_dtype)
    finally:
        model.train(was_training)


def _validate(model, loader, loss_cfg, overrides, amp_dtype) -> dict:
    device = next(model.parameters()).device
    conds = [(None, "predicted")] + list(overrides)
    acc = {name: {k: 0.0 for k in ("heat_loss", "mask_loss", "dice", "budget")} for _, name in conds}
    pred = {k: 0.0 for k in ("r_loss", "r_mae", "total")}
    n = 0
    for batch in loader:
        batch = to_device(batch, device)
        b = batch["image"].shape[0]
        valid_s4 = valid_to_stride(batch["valid"], 4)
        with torch.autocast(device.type, dtype=amp_dtype if amp_dtype is not None else torch.float32, enabled=amp_dtype is not None):
            for ov, name in conds:
                out = model(batch["image"], batch["pointer"], batch["valid"], r_override=ov)
                hl = heat_loss(out.heat_logits, batch["heat"], batch["heat_valid"], loss_cfg.heat)
                ml = mask_loss(out.kernel_map, out.mask_feat, batch["pos_index"], batch["pos_inst"], batch["masks_s4"], batch["mask_valid"], loss_cfg.mask)
                ms = mask_stats(out.kernel_map, out.mask_feat, batch["pos_index"], batch["pos_inst"], batch["masks_s4"], batch["mask_valid"], loss=ml)
                bud = model.backbone.gate_budget(out.r_gate, valid_s4).mean()
                a = acc[name]
                a["heat_loss"] += float(hl) * b; a["mask_loss"] += float(ml) * b; a["dice"] += float(ms["dice"]) * b; a["budget"] += float(bud) * b
                if ov is None:
                    rl = r_loss(out.r_pred, batch["r_gt"], batch["r_valid"], loss_cfg.r)
                    rs = r_stats(out.r_pred, batch["r_gt"], batch["r_valid"])
                    pred["r_loss"] += float(rl) * b; pred["r_mae"] += float(rs["r_mae"]) * b
                    pred["total"] += float(loss_cfg.w_r * rl + loss_cfg.w_heat * hl + loss_cfg.w_mask * ml) * b
        n += b
    sweep = {name: {k: v / n for k, v in a.items()} for name, a in acc.items()}
    predicted = {**{k: v / n for k, v in pred.items()}, **sweep["predicted"]}
    return {"predicted": predicted, "sweep": sweep, "n_images": n}


def format_val(res: dict, update: int | None = None) -> str:
    p = res["predicted"]
    head = f"[val{'' if update is None else f' @ {update:06d}'}] {res['n_images']} images"
    lines = [f"{head}: total {p['total']:.3f} | r {p['r_loss']:.4f} (mae {p['r_mae']:.3f}) heat {p['heat_loss']:.3f} mask {p['mask_loss']:.4f} | dice {p['dice']:.3f} | budget {p['budget']:.3f}",
             f"  {'condition':10s} {'budget':>7s} {'heat':>8s} {'mask':>8s} {'dice':>7s}"]
    for name, s in res["sweep"].items():
        lines.append(f"  {name:10s} {s['budget']:7.3f} {s['heat_loss']:8.3f} {s['mask_loss']:8.4f} {s['dice']:7.3f}")
    return "\n".join(lines)
