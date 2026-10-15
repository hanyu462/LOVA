"""Draw what a checkpoint predicts for one image and pointer, under several R conditions.

    python visualize.py --config configs/v0.py --root datasets/coco --ckpt runs/v0/last.pt --index 0 [--seed 0] [--out dir]
    python visualize.py ... --image-id 2153 --pointer 320 240          # your own pointer (canvas px)

LOVA's output is ONE mask: the pointed object. Rows: predicted R | R=1 (dense regime) | R_GT (prior).
Columns: image + pointer | gate field (red 1, blue 0) | the SELECTED pointed mask (label, score, IoU with the GT) |
the pointed GT mask. --all-dets adds a column with every detection (score > thr). Window by default, PNG with --out.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lova.data.coco.dataset import CocoDataset  # noqa: E402
from lova.data.common.build import collate  # noqa: E402
from lova.data.common.transform import denormalize  # noqa: E402
from lova.engine.checkpoint import load_checkpoint  # noqa: E402
from lova.engine.decode import DecodeCfg, decode, select_pointed  # noqa: E402
from lova.engine.evaluate import mask_iou  # noqa: E402
from lova.engine.train import amp_dtype  # noqa: E402
from lova.models.lova import Lova  # noqa: E402
from tests.util.viz import draw_pointer, field_overlay, hstack, overlay_masks, vstack  # noqa: E402
from train import load_config  # noqa: E402


def detections_image(base: Image.Image, det: dict, names, thr: float) -> Image.Image:
    keep = det["scores"] > thr
    masks = det["masks"][keep] > 0.5
    if masks.shape[0] == 0:
        return base.copy()
    up = torch.nn.functional.interpolate(masks[None].float(), size=base.size[::-1], mode="nearest")[0] > 0.5
    labels = [f"{names(int(l))} {float(s):.2f}" for l, s in zip(det["labels"][keep], det["scores"][keep])]
    return overlay_masks(base, up.cpu(), labels=labels)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True); p.add_argument("--root", required=True); p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="val2017"); p.add_argument("--index", type=int, default=0); p.add_argument("--image-id", type=int, default=None)
    p.add_argument("--seed", type=int, default=None, help="pointer draw seed (default: the dataset index, as validation does)")
    p.add_argument("--pointer", type=float, nargs=2, default=None, help="x y in canvas pixels instead of a sampled pointer")
    p.add_argument("--thr", type=float, default=0.3); p.add_argument("--panel", type=int, default=320); p.add_argument("--out", default=None)
    p.add_argument("--all-dets", action="store_true", help="also draw every detection above --thr")
    a = p.parse_args()
    cfg = load_config(a.config)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = CocoDataset(a.root, a.split, cfg.data, train=False)
    index = a.index if a.image_id is None else ds.ids.index(a.image_id)
    from lova.data.coco.load import load
    from lova.data.common.build import build
    ts = build(load(ds.cs, ds.ids[index]), ds.num_classes, ds.cfg, torch.Generator().manual_seed(index if a.seed is None else a.seed))
    if ts is None:
        sys.exit(f"image {ds.ids[index]} has no pointer-able instance")
    if a.pointer is not None:
        ts.pointer = torch.tensor(a.pointer, dtype=torch.float32)
    batch = collate([ts])
    model = Lova(ds.num_classes, cfg.model).to(dev)
    update = load_checkpoint(a.ckpt, model, device=dev)
    model.eval()
    img, ptr, valid = batch["image"].to(dev), batch["pointer"].to(dev), batch["valid"].to(dev)
    base = Image.fromarray((denormalize(batch["image"][0]).permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    names = ds.cs.name_of_label
    pointed = ts.pointed_idx
    rows = []
    conds = [(None, "predicted R"), (1.0, "R = 1 (dense)"), (batch["r_gt"][:, None].to(dev), "R_GT (prior)")]
    gt_full = torch.nn.functional.interpolate(ts.seg_gt.masks_s4[None], size=base.size[::-1], mode="nearest")[0] > 0.5
    gt_pointed = gt_full[pointed]
    with torch.no_grad(), torch.autocast(dev, dtype=amp_dtype(cfg.train) or torch.float32, enabled=amp_dtype(cfg.train) is not None):
        for ov, name in conds:
            out = model(img, ptr, valid, r_override=ov)
            det = decode(out, DecodeCfg(), heat_valid=batch["heat_valid"].to(dev), mask_valid=batch["mask_valid"].to(dev))[0]
            budget = float(model.backbone.gate_budget(out.r_gate, (torch.nn.functional.avg_pool2d(valid[:, None].float(), 4) > 0.5).float()).mean())
            j = select_pointed(det, ts.pointer)
            if j is None:
                sel_img, sel_title = base.copy(), "no detection"
            else:
                m = torch.nn.functional.interpolate(det["masks"][j][None, None].float(), size=base.size[::-1], mode="nearest")[0] > 0.5
                iou = float(mask_iou(m.cpu(), gt_pointed))
                sel_img = overlay_masks(base, m.cpu(), labels=[f"{names(int(det['labels'][j]))} {float(det['scores'][j]):.2f}"])
                sel_title = f"selected: {names(int(det['labels'][j]))} {float(det['scores'][j]):.2f}  IoU {iou:.2f}"
            panels = [draw_pointer(base.copy(), ts.pointer),
                      draw_pointer(field_overlay(base, out.r_gate[0, 0].float().cpu()), ts.pointer),
                      draw_pointer(sel_img, ts.pointer),
                      draw_pointer(overlay_masks(base, gt_pointed[None], labels=[names(int(ts.labels[pointed]))]), ts.pointer)]
            titles = [f"img {ts.image_id}  {name}", f"gate field  budget {budget:.3f}", sel_title, f"GT pointed: {names(int(ts.labels[pointed]))}"]
            if a.all_dets:
                panels.append(draw_pointer(detections_image(base, det, names, a.thr), ts.pointer))
                titles.append(f"{int((det['scores'] > a.thr).sum())} detections > {a.thr}")
            panels = [im.resize((a.panel, a.panel)) for im in panels]
            rows.append(hstack(panels, titles))
    canvas = vstack(rows)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        where = os.path.join(a.out, f"viz_{ts.image_id}_u{update}.png"); canvas.save(where); print("saved", where)
    else:
        canvas.show(title=f"LOVA {ts.image_id} @ {update}")
