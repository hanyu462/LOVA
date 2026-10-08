"""Visualise the R pseudo-GT profile (data/rtarget.py) on real samples before training with it.

    python scripts/viz_rtarget.py --coco-root datasets/coco --n 6 --out viz_rtarget
    python scripts/viz_rtarget.py --data synthetic --n 4 --out viz_rtarget_syn

Each PNG: [image + pointer | R* heatmap (red high, blue low) | G = 1[R* > 0.5] overlay].
Also prints the profile along a horizontal line through the pointer so the 1 -> r_b -> 0 shape
can be read as numbers, and the active-area fraction mean(G) per sample.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lova.data.common import MEAN, STD  # noqa: E402
from lova.data.rtarget import halo_cells, lam_cells, r_profile  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", choices=["coco", "synthetic"], default="coco")
    p.add_argument("--coco-root", default="datasets/coco")
    p.add_argument("--split", default="val2017")
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--r-b", type=float, default=0.7)
    p.add_argument("--r-gamma", type=float, default=0.7)
    p.add_argument("--r-lam-frac", type=float, default=0.05)
    p.add_argument("--mode", choices=["dt", "euclid"], default="dt")
    p.add_argument("--pointer-mode", choices=["uniform", "interior"], default="interior")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="viz_rtarget")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.data == "synthetic":
        from lova.data.synthetic import SyntheticPointerDataset
        ds = SyntheticPointerDataset(a.n, a.img_size if a.img_size <= 256 else 256, seed=a.seed, pointer_mode=a.pointer_mode)
    else:
        from lova.data.coco import COCOPointerDataset
        ds = COCOPointerDataset(a.coco_root, a.split, a.img_size, train=False, pointer_mode=a.pointer_mode)
    lam = lam_cells(ds.size, a.r_lam_frac)
    print(f"r_b={a.r_b} gamma={a.r_gamma} lambda={lam:.2f} cells ({lam * 4:.0f} px)  "
          f"binary halo (R*>0.5) = {halo_cells(a.r_b, lam):.2f} cells = {halo_cells(a.r_b, lam) * 4:.0f} px")

    rng = np.random.RandomState(a.seed)
    idxs = rng.choice(len(ds), size=min(a.n, len(ds)), replace=False)
    for k, i in enumerate(idxs):
        torch.manual_seed(a.seed * 7 + int(i))
        smp = ds[int(i)]
        if smp is None:
            continue
        pm = smp["masks4"][smp["pointed"]][None, None]
        rs = r_profile(pm, smp["pointer"][None], a.r_b, a.r_gamma, lam, a.mode)[0] * smp["valid4"]
        img = (smp["image"] * STD + MEAN).clamp(0, 1)
        S = img.shape[-1]
        rr = F.interpolate(rs[None], size=(S, S), mode="bilinear", align_corners=False)[0]
        panel_r = torch.cat([rr, torch.zeros_like(rr), 1 - rr], 0) * 0.6 + img * 0.4
        g = (rr > 0.5).float()
        panel_g = img * (0.35 + 0.65 * g) + torch.tensor([0.0, 0.3, 0.0]).view(3, 1, 1) * g * 0.3
        canvas = torch.cat([img, panel_r, panel_g.clamp(0, 1)], 2)
        pil = Image.fromarray((canvas.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        d = ImageDraw.Draw(pil)
        px, py = float(smp["pointer"][0]), float(smp["pointer"][1])
        for j in range(3):
            d.ellipse([px + j * S - 6, py - 6, px + j * S + 6, py + 6], outline=(255, 255, 0), width=3)
        path = os.path.join(a.out, f"rtarget_{k:02d}.png")
        pil.save(path)
        # numeric profile along the row through the pointer (stride-4 cells)
        row = rs[0, min(int(py // 4), rs.shape[1] - 1)]
        cx = int(px // 4)
        xs = list(range(max(cx - 24, 0), min(cx + 25, rs.shape[2]), 3))
        active = float((rs > 0.5).float().sum() / smp["valid4"].sum())
        print(f"[{k}] image {smp['meta']['image_id']} pointed area {float(pm.sum() / smp['valid4'].sum()):.3f}  "
              f"active(G) {active:.3f}  -> {path}")
        print("     x(cells): " + " ".join(f"{x:4d}" for x in xs))
        print("     R*      : " + " ".join(f"{row[x]:4.2f}" for x in xs))


if __name__ == "__main__":
    main()
