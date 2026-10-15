"""lova.engine.evaluate: canvas -> original coordinate restore (the inverse of transform.apply, incl. flip /
crop / pad), COCO result encoding, pointed IoU, and a determinism + sanity run on real data.

    python tests/engine/test_evaluate.py
    python tests/engine/test_evaluate.py --root datasets/coco --ckpt runs/v0/last.pt --config configs/v0.py --n 16
        (evaluates twice: numbers must be identical; also a GT-as-prediction run must give AP ~ 1)
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lova.data.common.transform import TransformCfg, TransformParams, apply, to_original  # noqa: E402
from lova.engine.evaluate import coco_results, mask_iou, masks_to_original  # noqa: E402
from tests.common.test_transform import synthetic_sample  # noqa: E402


def unit_test():
    torch.manual_seed(0)
    S = 128
    cfg = TransformCfg(size=S, train=True)
    sample = synthetic_sample(h=96, w=160)
    H, W = sample.size
    # three geometries: eval (resize + pad), flip + pad, flip + crop (scale > 1 so the resized image exceeds the canvas)
    for params in (TransformParams(S / max(H, W), round(H * S / max(H, W)), round(W * S / max(H, W)), False, 0, 0),
                   TransformParams(S / max(H, W), round(H * S / max(H, W)), round(W * S / max(H, W)), True, 0, 0),
                   TransformParams(1.1, round(H * 1.1), round(W * 1.1), True, 12, 0)):
        t = apply(sample, params, cfg)
        # the canvas masks at stride 4 (area average, as masks_s4) are the "prediction"; restoring them must give
        # back the original masks: IoU high everywhere, exactly where the instance survived the crop
        prob = torch.nn.functional.avg_pool2d(t.masks[:, None].float(), 4)[:, 0]
        back = masks_to_original(prob, params, (H, W), S)
        assert back.shape == (3, H, W) and back.dtype == torch.bool
        for k in range(3):
            if t.masks[k].sum() == 0:
                continue
            inter = (back[k] & sample.masks[k]).sum().float(); union = (back[k] | sample.masks[k]).sum().float()
            visible = t.masks[k].sum().float() / max(float(sample.masks[k].sum() * params.scale ** 2), 1)
            iou = float(inter / union)
            assert iou > 0.85 * min(visible, 1.0), (params, k, iou, float(visible))
        # a point on the canvas and its restored mask agree: the restored mask contains to_original(centre)
        for k in range(3):
            ys, xs = torch.nonzero(t.masks[k], as_tuple=True)
            if len(ys) == 0:
                continue
            cx, cy = float(xs.float().mean()), float(ys.float().mean())
            if not t.masks[k][int(cy), int(cx)]:
                continue                                                       # concave: centre outside, skip
            ox, oy = to_original((cx, cy), params)
            assert back[k][int(round(oy)), int(round(ox))], (params, k)
    # empty input
    assert masks_to_original(torch.zeros(0, S // 4, S // 4), params, (H, W), S).shape == (0, H, W)
    # coco_results: RLE round trip and category mapping
    from pycocotools import mask as mu
    import numpy as np
    m = torch.zeros(2, H, W, dtype=torch.bool); m[0, 10:30, 20:50] = True; m[1, 40:60, 60:90] = True
    res = coco_results(7, m, torch.tensor([0, 2]), torch.tensor([0.9, 0.4]), cat_ids=[1, 5, 17])
    assert [r["category_id"] for r in res] == [1, 17] and [r["image_id"] for r in res] == [7, 7] and abs(res[1]["score"] - 0.4) < 1e-6
    dec = mu.decode([r["segmentation"] for r in res])                      # [H, W, 2]
    assert np.array_equal(dec[:, :, 0].astype(bool), m[0].numpy()) and np.array_equal(dec[:, :, 1].astype(bool), m[1].numpy())
    assert coco_results(7, torch.zeros(0, H, W, dtype=torch.bool), torch.zeros(0, dtype=torch.long), torch.zeros(0), [1]) == []
    # mask_iou
    iou = mask_iou(m, m[0])
    assert abs(float(iou[0]) - 1.0) < 1e-6 and float(iou[1]) == 0.0
    print("evaluate unit test OK")


def real(a):
    from lova.engine.checkpoint import load_checkpoint
    from lova.engine.evaluate import EvalCfg, evaluate, format_eval, _coco_stats, coco_results
    from lova.engine.train import amp_dtype
    from lova.engine.validate import ValCfg, ValSet, val_loader
    from lova.models.lova import Lova
    from train import load_config
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config(a.config)
    vcfg = ValCfg(n_images=a.n, seed=cfg.val.seed, batch_size=cfg.val.batch_size, workers=cfg.val.workers)
    vs = ValSet(a.root, a.split, cfg.data, vcfg.n_images, vcfg.seed)
    # 1. GT masks fed as "predictions" through the same encoding path must score AP ~ 1 (sanity of ids / RLE / imgIds)
    coco, cat_ids = vs.ds.cs.coco, vs.ds.cs.cat_ids
    results, ids = [], []
    for k in range(len(vs)):
        img_id = vs.image_ids[k]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False))
        masks = torch.from_numpy(torch.stack([torch.from_numpy(coco.annToMask(an)) for an in anns]).numpy().astype(bool)) if anns else torch.zeros(0, 1, 1, dtype=torch.bool)
        labels = torch.tensor([vs.ds.cs.cat_to_label[an["category_id"]] for an in anns], dtype=torch.long)
        results += coco_results(img_id, masks, labels, torch.ones(len(anns)), cat_ids); ids.append(img_id)
    st = _coco_stats(coco, results, ids)
    print(f"GT as predictions: AP {st['AP']:.3f} AP50 {st['AP50']:.3f} (must be ~1)")
    assert st["AP"] > 0.97, st
    # 2. the checkpoint, twice: identical
    model = Lova(vs.num_classes, cfg.model).to(dev)
    load_checkpoint(a.ckpt, model, device=dev)
    r1 = evaluate(model, vs, val_loader(vs, vcfg), EvalCfg(), a.r_override, amp_dtype(cfg.train))
    r2 = evaluate(model, vs, val_loader(vs, vcfg), EvalCfg(), a.r_override, amp_dtype(cfg.train))
    assert r1 == r2, "evaluation must be deterministic"
    print(format_eval(r1, f"{os.path.basename(os.path.dirname(a.ckpt))} n={a.n}"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=None); p.add_argument("--split", default="val2017"); p.add_argument("--config", default="configs/v0.py")
    p.add_argument("--ckpt", default="runs/v0/last.pt"); p.add_argument("--n", type=int, default=16); p.add_argument("--r-override", type=float, default=None)
    a = p.parse_args()
    unit_test()
    if a.root:
        real(a)
