"""lova.data.coco.dataset: DataLoader throughput and a batch visualisation on real COCO.

    python tests/coco/test_dataset.py --root datasets/coco                      # window: first 4 samples
    python tests/coco/test_dataset.py --root datasets/coco --throughput 64 --workers 0,4,8
    python tests/coco/test_dataset.py --root datasets/coco --eval --index 10 --out viz/dataset

Panel per sample: [image + pointer + pointed mask outline | R_GT @4 (red high) | heat max over classes | ignored].
Throughput: samples / second through DataLoader + collate for each worker count (the number that
matters for training: one GPU process at bs 32 and 0.3 s/it needs ~100 samples/s).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lova.data.coco.dataset import CocoDataset, worker_init  # noqa: E402
from lova.data.common.build import PipelineCfg, collate  # noqa: E402
from lova.data.common.transform import denormalize  # noqa: E402
from tests.util.viz import draw_pointer, hstack, outline  # noqa: E402


def throughput(ds, n: int, workers: int, bs: int = 8) -> float:
    dl = DataLoader(ds, batch_size=bs, num_workers=workers, collate_fn=collate, shuffle=True,
                    persistent_workers=workers > 0, worker_init_fn=worker_init)
    it = iter(dl)
    next(it)                                   # warm-up (worker start, annotation index)
    t0 = time.perf_counter()
    seen = 0
    while seen < n:
        seen += next(it)["image"].shape[0]
    return seen / (time.perf_counter() - t0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--split", default="val2017")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--index", type=int, default=0, help="first sample index to show")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--throughput", type=int, default=0, help="measure samples/s over this many samples")
    p.add_argument("--workers", default="0,4,8")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    ds = CocoDataset(a.root, a.split, PipelineCfg(), train=not a.eval)
    print(f"{a.split}: {len(ds)} images with instances, {ds.num_classes} classes, train={not a.eval}")
    if a.throughput:
        for w in [int(x) for x in a.workers.split(",")]:
            print(f"  workers {w:>2}: {throughput(ds, a.throughput, w):6.1f} samples/s")
        return

    t0 = time.perf_counter()
    samples = [ds[i] for i in range(a.index, a.index + a.n)]
    print(f"  {a.n} samples built in {(time.perf_counter() - t0) / a.n * 1000:.0f} ms each")
    batch = collate(samples)
    print(f"  batch: image {tuple(batch['image'].shape)}  r_gt {tuple(batch['r_gt'].shape)}  heat {tuple(batch['heat'].shape)}  "
          f"positives {[len(p) for p in batch['pos_index']]}")
    rows = []
    for ts in samples:
        S = ts.image.shape[-1]
        base = Image.fromarray((denormalize(ts.image) * 255).permute(1, 2, 0).numpy().astype(np.uint8))
        base_t = torch.from_numpy(np.array(base)).permute(2, 0, 1).float() / 255
        m4 = ts.seg_gt.masks_s4[ts.pointed_idx]
        mfull = torch.nn.functional.interpolate(m4[None, None], size=(S, S), mode="nearest")[0, 0] > 0.5
        arr = np.array(base).copy()
        arr[outline(mfull.numpy(), 2)] = (255, 255, 0)
        img = draw_pointer(Image.fromarray(arr), ts.pointer.tolist())
        up = lambda f: torch.nn.functional.interpolate(f[None, None], size=(S, S), mode="nearest")[0, 0]

        def heatmap(f):
            h = torch.stack([f, torch.zeros_like(f), 1 - f], 0) * 0.7 + base_t * 0.3
            return Image.fromarray((h.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        r_img = draw_pointer(heatmap(up(ts.r_gt)), ts.pointer.tolist())
        h_img = heatmap(up(ts.seg_gt.heat.max(0).values))
        ign = ~up(ts.seg_gt.heat_valid.float()).bool()
        arr2 = np.array(base).astype(np.float32)
        arr2[ign.numpy()] = arr2[ign.numpy()] * 0.3 + np.array([255, 0, 255]) * 0.7
        name = ds.cs.name_of_label(int(ts.labels[ts.pointed_idx]))
        rows.append(hstack([img, r_img, h_img, Image.fromarray(arr2.clip(0, 255).astype(np.uint8))],
                           [f"id {ts.image_id}: pointer on {name} ({len(ts.labels)} inst)", "R_GT @ stride 4",
                            "heat (max over classes)", "ignored cells"]))
    W = max(r.width for r in rows)
    canvas = Image.new("RGB", (W, sum(r.height for r in rows) + 6 * len(rows)), (30, 30, 30))
    y = 0
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.height + 6
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        where = os.path.join(a.out, f"dataset_{a.split}_{a.index}{'_eval' if a.eval else ''}.png")
        canvas.save(where)
    else:
        where = "(window)"
        canvas.show(title="dataset batch")
    print(f"  -> {where}")


if __name__ == "__main__":
    main()
