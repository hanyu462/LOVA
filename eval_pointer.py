"""V0 hypothesis test: does R control perception quality?

For every evaluated (image, pointer) pair and every GT instance we record
  mean_R   : mean R inside the instance
  dist     : distance pointer -> instance (0 if pointer inside), / canvas size
  best_iou : best mask IoU with a same-class prediction
  score    : best score among same-class predictions with IoU >= 0.5 (0 = missed)
and report detection rate / confidence / mask IoU per R-bin and distance-bin,
plus (COCO, single-pointer mode) mask AP overall and per bin.

--sweep K   : up to K pointers per image, each on a different instance.
              Reports the "pointer gain": same instance, pointed vs not pointed.
--r-mode    : pred | ones | zeros | oracle | const:<v>   (ablations / controls)

Example:
  python eval_pointer.py --ckpt runs/C/last.pth --coco-root /data/coco --out eval/C_pred
  python eval_pointer.py --ckpt runs/C/last.pth --coco-root /data/coco --r-mode ones --out eval/C_ones
  python eval_pointer.py --ckpt runs/C/last.pth --coco-root /data/coco --sweep 4 --max-images 1000 --out eval/C_sweep
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from lova.data.common import MEAN, STD, collate
from lova.models.head import postprocess
from lova.models.model import LOVAv0
from lova.rsampler import blur

R_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
D_BINS = [0.0, 1e-6, 0.1, 0.25, 0.5, 10.0]  # first bin = pointed/touching (dist == 0)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", choices=["coco", "synthetic"], default="coco")
    p.add_argument("--coco-root", default="data/coco")
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--r-mode", default="pred")
    p.add_argument("--sweep", type=int, default=0)
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--det-thr", type=float, default=0.3, help="score threshold for 'detected'")
    p.add_argument("--viz", type=int, default=0, help="save N visualization PNGs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="eval/exp")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def make_r(mode, pm):
    if mode == "pred":
        return None
    if mode == "ones":
        return torch.ones_like(pm)
    if mode == "zeros":
        return torch.zeros_like(pm)
    if mode == "oracle":
        return blur(pm).clamp(0, 1)
    if mode.startswith("const:"):
        return torch.full_like(pm, float(mode.split(":")[1]))
    raise ValueError(mode)


def pointer_dist(mask_bin: torch.Tensor, pointer: torch.Tensor, canvas: int) -> torch.Tensor:
    """mask_bin [G,h4,w4] bool, pointer (x,y) px -> [G] min distance / canvas (0 if inside)."""
    g, h, w = mask_bin.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=mask_bin.device), torch.arange(w, device=mask_bin.device),
                            indexing="ij")
    d = torch.sqrt(((xs + 0.5) * 4 - pointer[0]) ** 2 + ((ys + 0.5) * 4 - pointer[1]) ** 2) / canvas
    d = torch.where(mask_bin, d[None].expand(g, -1, -1), torch.full_like(d[None].expand(g, -1, -1), 1e9))
    dmin = d.flatten(1).min(1).values
    px, py = int(pointer[0] // 4), int(pointer[1] // 4)
    inside = mask_bin[:, min(py, h - 1), min(px, w - 1)]
    return torch.where(inside, torch.zeros_like(dmin), dmin)


def analyze_sample(sample_masks4, sample_classes, r, pred, pointer, pointed, canvas, det_thr):
    gt = sample_masks4
    gt_bin = gt > 0.5
    empty = gt_bin.flatten(1).sum(1) == 0
    gt_bin[empty] = gt[empty] > 0
    mean_r = (r[0][None] * gt).flatten(1).sum(1) / gt.flatten(1).sum(1).clamp(min=1e-6)
    dist = pointer_dist(gt_bin, pointer, canvas)
    pb = pred["masks"] > 0.5
    if len(pb):
        inter = gt_bin.flatten(1).float() @ pb.flatten(1).float().t()
        union = gt_bin.flatten(1).sum(1)[:, None] + pb.flatten(1).sum(1)[None] - inter
        iou = inter / union.clamp(min=1)
        same = sample_classes[:, None] == pred["labels"][None]
        iou = iou * same
        best_iou = iou.max(1).values
        sc = torch.where(iou >= 0.5, pred["scores"][None].expand_as(iou), torch.zeros_like(iou)).max(1).values
    else:
        best_iou = torch.zeros(len(gt))
        sc = torch.zeros(len(gt))
    rows = []
    for g in range(len(gt)):
        rows.append(dict(cls=int(sample_classes[g]), area=float(gt[g].sum() * 16), mean_r=float(mean_r[g]),
                         dist=float(dist[g]), pointed=int(g == pointed), best_iou=float(best_iou[g]),
                         score=float(sc[g]), detected=int(float(sc[g]) >= det_thr)))
    return rows


def bin_summary(rows, key, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in rows if lo <= r[key] < hi]
        if not sel:
            continue
        label = "inside(=0)" if hi <= 1e-6 else f"[{lo:.2f},{hi:.2f})"
        out.append(dict(bin=label, n=len(sel),
                        det_rate=np.mean([r["detected"] for r in sel]),
                        recall50=np.mean([r["score"] > 0 for r in sel]),
                        score=np.mean([r["score"] for r in sel]),
                        iou=np.mean([r["best_iou"] for r in sel])))
    return out


def print_table(title, table):
    print(f"\n== {title}")
    print(f"{'bin':>14} {'n':>7} {'det_rate':>9} {'recall50':>9} {'score':>7} {'mIoU':>7}")
    for t in table:
        print(f"{t['bin']:>14} {t['n']:>7d} {t['det_rate']:>9.3f} {t['recall50']:>9.3f} {t['score']:>7.3f} {t['iou']:>7.3f}")


def to_coco_results(pred, meta, cat_ids):
    from pycocotools import mask as mask_util

    if len(pred["scores"]) == 0:
        return []
    nh, nw = meta["new_size"]
    h, w = meta["orig_size"]
    m = F.interpolate(pred["masks"][None], scale_factor=4, mode="bilinear", align_corners=False)[0]
    m = m[:, :nh, :nw]
    m = F.interpolate(m[None], size=(h, w), mode="bilinear", align_corners=False)[0] > 0.5
    m = np.asfortranarray(m.cpu().numpy().transpose(1, 2, 0).astype(np.uint8))
    rles = mask_util.encode(m)
    res = []
    for k, rle in enumerate(rles):
        rle["counts"] = rle["counts"].decode("ascii")
        res.append(dict(image_id=meta["image_id"], category_id=cat_ids[int(pred["labels"][k])],
                        segmentation=rle, score=float(pred["scores"][k])))
    return res


def coco_ap(coco_gt, results, img_ids, keep_ann_ids=None):
    """Mask AP. If keep_ann_ids is given, all other GT become iscrowd (ignored)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not results:
        return {"AP": 0.0, "AP50": 0.0}
    gt = coco_gt
    if keep_ann_ids is not None:
        ds = copy.deepcopy(coco_gt.dataset)
        ds["annotations"] = [dict(a, iscrowd=a["iscrowd"] if a["id"] in keep_ann_ids else 1)
                             for a in ds["annotations"] if a["image_id"] in set(img_ids)]
        gt = COCO()
        gt.dataset = ds
        gt.createIndex()
    dt = gt.loadRes(copy.deepcopy(results))
    ev = COCOeval(gt, dt, "segm")
    ev.params.imgIds = sorted(img_ids)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return {"AP": float(ev.stats[0]), "AP50": float(ev.stats[1])}


def save_viz(path, image, r, pointer, pred, det_thr):
    from PIL import Image, ImageDraw

    img = (image.cpu() * STD + MEAN).clamp(0, 1)
    S = img.shape[-1]
    rr = F.interpolate(r[None].cpu(), size=img.shape[-2:], mode="bilinear", align_corners=False)[0]
    panel_r = torch.cat([rr, torch.zeros_like(rr), 1 - rr], 0) * 0.6 + img * 0.4
    over = img.clone()
    keep = pred["scores"] >= det_thr
    gen = torch.Generator().manual_seed(0)
    for m in pred["masks"][keep].cpu():
        mm = F.interpolate(m[None, None], size=img.shape[-2:], mode="bilinear", align_corners=False)[0, 0] > 0.5
        over[:, mm] = over[:, mm] * 0.4 + torch.rand(3, 1, generator=gen) * 0.6
    canvas = torch.cat([img, panel_r, over], 2)
    pil = Image.fromarray((canvas.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    d = ImageDraw.Draw(pil)
    for k in range(3):
        x, y = float(pointer[0]) + k * S, float(pointer[1])
        d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=(255, 255, 0), width=2)
    pil.save(path)


@torch.no_grad()
def main():
    a = get_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)
    ckpt = torch.load(a.ckpt, map_location="cpu")
    train_args = ckpt.get("args", {})
    img_size = a.img_size or train_args.get("img_size", 512)
    model = LOVAv0(**ckpt["config"]).to(dev).eval()
    model.load_state_dict(ckpt["model"])

    if a.data == "synthetic":
        from lova.data.synthetic import SyntheticPointerDataset
        ds = SyntheticPointerDataset(200, img_size, seed=1)
    else:
        from lova.data.coco import COCOPointerDataset
        ds = COCOPointerDataset(a.coco_root, "val2017", img_size, train=False)
    n_img = min(len(ds), a.max_images) if a.max_images else len(ds)

    # Build (index, pointed) jobs.
    rng = np.random.RandomState(a.seed)
    jobs = []
    for i in range(n_img):
        if a.sweep:
            n_inst = len(ds.load(i)[2]) if a.data == "synthetic" else len(ds._anns(ds.ids[i]))
            for j in rng.permutation(n_inst)[:a.sweep]:
                jobs.append((i, int(j)))
        else:
            jobs.append((i, None))

    rows, results, img_ids, n_viz = [], [], [], 0
    for s in range(0, len(jobs), a.bs):
        samples = []
        for i, j in jobs[s:s + a.bs]:
            torch.manual_seed(a.seed * 1000003 + i)
            smp = ds.__getitem__(i, pointed=j)
            if smp is not None:
                smp["job"] = (i, j)
                samples.append(smp)
        if not samples:
            continue
        batch = collate(samples)
        img = batch["image"].to(dev)
        ptr = batch["pointer"].to(dev)
        r_over = make_r(a.r_mode, batch["pointed_mask4"].to(dev))
        out = model(img, ptr, r_override=r_over)
        preds = postprocess(out, score_thr=0.05)
        for b, smp in enumerate(samples):
            r = out["r"][b]
            pred = preds[b]
            sr = analyze_sample(smp["masks4"].to(dev), smp["classes"].to(dev), r, pred, ptr[b], smp["pointed"],
                                img_size, a.det_thr)
            for g, row in enumerate(sr):
                row.update(image=smp["meta"]["image_id"], job=s + b, inst=g,
                           ann_id=smp["meta"].get("ann_ids", [None] * len(sr))[g])
            rows += sr
            if a.data == "coco" and not a.sweep:
                results += to_coco_results(pred, smp["meta"], ds.cat_ids)
                img_ids.append(smp["meta"]["image_id"])
            if n_viz < a.viz:
                save_viz(os.path.join(a.out, f"viz_{n_viz:04d}.png"), smp["image"], r, ptr[b], pred, a.det_thr)
                n_viz += 1
        print(f"\r{min(s + a.bs, len(jobs))}/{len(jobs)}", end="", flush=True)
    print()

    with open(os.path.join(a.out, "instances.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    summary = {"r_mode": a.r_mode, "n_instances": len(rows),
               "by_R": bin_summary(rows, "mean_r", R_BINS), "by_dist": bin_summary(rows, "dist", D_BINS),
               "pointed": bin_summary([r for r in rows if r["pointed"]], "mean_r", [0, 1.0001]),
               "not_pointed": bin_summary([r for r in rows if not r["pointed"]], "mean_r", [0, 1.0001])}
    print_table("by mean R inside instance", summary["by_R"])
    print_table("by pointer distance (/canvas); first bin = pointed/touching", summary["by_dist"])
    print_table("pointed instances", summary["pointed"])
    print_table("non-pointed instances", summary["not_pointed"])

    if a.sweep:
        per = defaultdict(lambda: {0: [], 1: []})
        for r in rows:
            per[(r["image"], r["inst"] if r["ann_id"] is None else r["ann_id"])][r["pointed"]].append(r)
        ds_, di_ = [], []
        for v in per.values():
            if v[0] and v[1]:
                ds_.append(np.mean([x["score"] for x in v[1]]) - np.mean([x["score"] for x in v[0]]))
                di_.append(np.mean([x["best_iou"] for x in v[1]]) - np.mean([x["best_iou"] for x in v[0]]))
        summary["pointer_gain"] = {"n": len(ds_), "d_score": float(np.mean(ds_)) if ds_ else None,
                                   "d_iou": float(np.mean(di_)) if di_ else None}
        print("\n== pointer gain (same instance, pointed - not pointed):", summary["pointer_gain"])

    if a.data == "coco" and not a.sweep:
        summary["AP_all"] = coco_ap(ds.coco, results, img_ids)
        summary["AP_by_R"] = {}
        for lo, hi in zip(R_BINS[:-1], R_BINS[1:]):
            keep = {r["ann_id"] for r in rows if lo <= r["mean_r"] < hi}
            if keep:
                summary["AP_by_R"][f"[{lo:.1f},{hi:.1f})"] = coco_ap(ds.coco, results, img_ids, keep)
        summary["AP_by_dist"] = {}
        for lo, hi in zip(D_BINS[:-1], D_BINS[1:]):
            keep = {r["ann_id"] for r in rows if lo <= r["dist"] < hi}
            if keep:
                label = "inside(=0)" if hi <= 1e-6 else f"[{lo:.2f},{hi:.2f})"
                summary["AP_by_dist"][label] = coco_ap(ds.coco, results, img_ids, keep)
        print("\n== mask AP:", json.dumps({k: summary[k] for k in ("AP_all", "AP_by_R", "AP_by_dist")}, indent=1))

    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
