"""Step 11-2: end-to-end evaluation. ModelOutput -> decode -> ORIGINAL image coordinates -> COCO segm AP,
plus pointer diagnostics on the pointed instance.

    out = model(image, pointer, valid[, r_override])     one batch of a fixed ValSet (eval transform: resize + pad, no flip / crop)
    dets = decode(out, heat_valid, mask_valid)            per image: scores, labels, masks [D,S/4,S/4] prob, centers
                                                          (the losses' valid masks: padding never yields peaks or mask area)
    masks_to_original(prob, params, orig_size, S)         stride-4 canvas -> S x S canvas -> undo pad / crop -> undo flip
                                                          -> resize to (H, W) -> threshold: the exact inverse of transform.apply
    coco_results(...)                                     {"image_id", "category_id" (COCO id, not label), "segmentation" (RLE), "score"}
    COCOeval(segm) restricted to the evaluated image ids  -> AP, AP50, AP75, APS, APM, APL
    pointed diagnostics                                   for the pointed GT instance (its COCO annotation mask at original
                                                          resolution): best IoU over ALL detections (class-agnostic, an oracle
                                                          choice), best IoU over same-class detections, and SELECTED IoU: the one
                                                          mask select_pointed would actually output (LOVA's real output), with its
                                                          class correctness; success rates at IoU 0.5 / 0.75 for best and selected

AP uses the official COCOeval implementation, but the SET is the fixed pointer-able ValSet subset (every
image needs a pointer), so call it "COCOeval segm AP on the fixed N-image pointer-able subset", not
"COCO val2017 AP": fair for V0 vs A0 (same images, pointers, decoder, evaluator), not comparable to
published numbers. A full-split AP needs a fallback pointer for images without a pointer-able instance
(TODO). The pointed numbers are LOVA's "quality where the pointer is", deliberately not a pseudo-AP.
Determinism: fixed set, fixed pointers, no sampling anywhere -> the same checkpoint gives identical numbers.
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_util
from pycocotools.cocoeval import COCOeval

from ..data.common.transform import TransformParams
from .decode import DecodeCfg, decode, select_pointed
from .train import to_device

STAT_NAMES = ("AP", "AP50", "AP75", "APS", "APM", "APL")


@dataclass(frozen=True)
class EvalCfg:
    decode: DecodeCfg = DecodeCfg()
    iou_thrs: tuple = (0.5, 0.75)        # pointed success thresholds


def masks_to_original(prob: torch.Tensor, params: TransformParams, orig_size: tuple[int, int], canvas: int,
                      thr: float = 0.5) -> torch.Tensor:
    """[D, S/4, S/4] probabilities on the canvas -> [D, H, W] bool in original image pixels.
    Steps mirror transform.apply in reverse: upsample to the canvas, cut the real region, paste it at the crop
    offset of the resized image, undo the flip, resize to the original size, threshold."""
    if prob.dim() != 3:
        raise ValueError(f"prob must be [D,h,w], got {tuple(prob.shape)}")
    H, W = orig_size
    p = params
    d = prob.shape[0]
    if d == 0:
        return torch.zeros(0, H, W, dtype=torch.bool, device=prob.device)
    full = F.interpolate(prob[None].float(), size=(canvas, canvas), mode="bilinear", align_corners=False)[0]
    ch, cw = min(p.new_h - p.crop_y, canvas), min(p.new_w - p.crop_x, canvas)          # the real (non-padding) region
    resized = torch.zeros(d, p.new_h, p.new_w, dtype=full.dtype, device=full.device)
    resized[:, p.crop_y:p.crop_y + ch, p.crop_x:p.crop_x + cw] = full[:, :ch, :cw]
    if p.flip:
        resized = resized.flip(-1)
    orig = F.interpolate(resized[None], size=(H, W), mode="bilinear", align_corners=False)[0]
    return orig > thr


def coco_results(image_id: int, masks: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor, cat_ids: list[int]) -> list[dict]:
    """Binary masks [D,H,W] (original pixels) -> COCO result dicts with RLE segmentation and COCO category ids."""
    if masks.shape[0] == 0:
        return []
    arr = np.asfortranarray(masks.cpu().numpy().astype(np.uint8).transpose(1, 2, 0))     # [H, W, D], Fortran order
    rles = mask_util.encode(arr)
    out = []
    for rle, lab, sc in zip(rles, labels.tolist(), scores.tolist()):
        rle["counts"] = rle["counts"].decode("ascii")
        out.append({"image_id": int(image_id), "category_id": int(cat_ids[lab]), "segmentation": rle, "score": float(sc)})
    return out


def mask_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a [D,H,W] bool, b [H,W] bool -> [D] IoU."""
    inter = (a & b[None]).flatten(1).sum(1).float()
    union = (a | b[None]).flatten(1).sum(1).float()
    return inter / union.clamp(min=1)


@torch.no_grad()
def evaluate(model, vs, loader, cfg: EvalCfg = EvalCfg(), r_override=None, amp_dtype: torch.dtype | None = torch.bfloat16) -> dict:
    """vs: ValSet (for the COCO object, category ids and image ids). loader: val_loader(vs, ...).
    Returns {"coco": {AP, AP50, AP75, APS, APM, APL}, "pointed": {best_iou, best_iou_same_class, succ@0.5, succ@0.75,
    n_detected (images with >= 1 detection), n_overlapping (images with a detection touching the pointed GT)},
    "n_images", "n_dets", "budget"}."""
    was_training = model.training
    model.eval()
    try:
        return _evaluate(model, vs, loader, cfg, r_override, amp_dtype)
    finally:
        model.train(was_training)


def _evaluate(model, vs, loader, cfg, r_override, amp_dtype) -> dict:
    device = next(model.parameters()).device
    coco, cat_ids = vs.ds.cs.coco, vs.ds.cs.cat_ids
    canvas = vs.ds.cfg.transform.size
    results, image_ids = [], []
    best, best_cls, sel, sel_cls_ok, has_det, budget_sum, n = [], [], [], [], [], 0.0, 0
    from ..models.backbone import valid_to_stride
    for batch in loader:
        b_dev = to_device(batch, device)
        with torch.autocast(device.type, dtype=amp_dtype if amp_dtype is not None else torch.float32, enabled=amp_dtype is not None):
            out = model(b_dev["image"], b_dev["pointer"], b_dev["valid"], r_override=r_override)
        budget_sum += float(model.backbone.gate_budget(out.r_gate, valid_to_stride(b_dev["valid"], 4)).sum())
        dets = decode(out, cfg.decode, heat_valid=b_dev["heat_valid"], mask_valid=b_dev["mask_valid"])
        for i, det in enumerate(dets):
            img_id, params, orig = batch["image_id"][i], batch["params"][i], batch["orig_size"][i]
            masks = masks_to_original(det["masks"], params, orig, canvas, cfg.decode.mask_thr)
            results += coco_results(img_id, masks, det["labels"], det["scores"], cat_ids)
            image_ids.append(int(img_id))
            # pointed instance: its COCO annotation at original resolution
            ann = coco.loadAnns(int(batch["ann_ids"][i][int(batch["pointed_idx"][i])]))[0]
            gt = torch.from_numpy(coco.annToMask(ann).astype(bool)).to(masks.device)
            gt_label = int(batch["labels"][i][int(batch["pointed_idx"][i])])
            has_det.append(masks.shape[0] > 0)
            if masks.shape[0] == 0:
                best.append(0.0); best_cls.append(0.0); sel.append(0.0); sel_cls_ok.append(False)
            else:
                iou = mask_iou(masks, gt)
                best.append(float(iou.max()))
                same = det["labels"].to(iou.device) == gt_label
                best_cls.append(float(iou[same].max()) if bool(same.any()) else 0.0)
                j = select_pointed(det, batch["pointer"][i])
                sel.append(float(iou[j])); sel_cls_ok.append(int(det["labels"][j]) == gt_label)
            n += 1
    stats = _coco_stats(coco, results, image_ids)
    bi, bc, si = torch.tensor(best), torch.tensor(best_cls), torch.tensor(sel)
    pointed = {"best_iou": float(bi.mean()), "best_iou_same_class": float(bc.mean()),
               "selected_iou": float(si.mean()), "selected_class_acc": float(sum(sel_cls_ok)) / max(n, 1)}
    for t in cfg.iou_thrs:
        pointed[f"succ@{t:g}"] = float((bi >= t).float().mean())
        pointed[f"succ_same_class@{t:g}"] = float((bc >= t).float().mean())
        pointed[f"selected_succ@{t:g}"] = float((si >= t).float().mean())
    pointed["n_detected"] = int(sum(has_det))
    pointed["n_overlapping"] = int((bi > 0).sum())
    return {"coco": stats, "pointed": pointed, "n_images": n, "n_dets": len(results), "budget": budget_sum / max(n, 1)}


def _coco_stats(coco, results: list[dict], image_ids: list[int]) -> dict:
    if not results:
        return {k: 0.0 for k in STAT_NAMES}
    with contextlib.redirect_stdout(io.StringIO()):           # pycocotools prints its own tables
        dt = coco.loadRes(results)
        ev = COCOeval(coco, dt, "segm")
        ev.params.imgIds = sorted(set(image_ids))
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return {k: float(ev.stats[i]) for i, k in enumerate(STAT_NAMES)}


def format_eval(res: dict, name: str = "") -> str:
    c, p = res["coco"], res["pointed"]
    return (f"[eval{(' ' + name) if name else ''}] {res['n_images']} images, {res['n_dets']} detections, budget {res['budget']:.3f}\n"
            f"  COCO segm  AP {c['AP'] * 100:5.1f}  AP50 {c['AP50'] * 100:5.1f}  AP75 {c['AP75'] * 100:5.1f}  "
            f"APS {c['APS'] * 100:5.1f}  APM {c['APM'] * 100:5.1f}  APL {c['APL'] * 100:5.1f}\n"
            f"  pointed    SELECTED mask IoU {p['selected_iou']:.3f}  succ@0.5 {p['selected_succ@0.5']:.3f}  succ@0.75 {p['selected_succ@0.75']:.3f}  "
            f"class acc {p['selected_class_acc']:.3f}\n"
            f"             best-of-all IoU {p['best_iou']:.3f} (same class {p['best_iou_same_class']:.3f})  "
            f"succ@0.5 {p['succ@0.5']:.3f}  succ@0.75 {p['succ@0.75']:.3f}  "
            f"(same class {p['succ_same_class@0.5']:.3f} / {p['succ_same_class@0.75']:.3f})  "
            f"images with detections {p['n_detected']}/{res['n_images']}, overlapping the pointed GT {p['n_overlapping']}")
