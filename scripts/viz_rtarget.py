"""Visualise the R pseudo-GT profile (lova/data/rtarget.py, the SAME function training uses).

Random samples:
    python scripts/viz_rtarget.py --coco-root datasets/coco --n 6 --out viz_rtarget
Fixed conditions (compare parameters on the same image / instance / pointer):
    python scripts/viz_rtarget.py --coco-root datasets/coco --image-id 139 --instance 0 --pointer 300,200 \
        --r-sigma-in 1.0 --r-gamma 2 --r-sigma-out-frac 0.05 --out viz_rtarget/id139
    python scripts/viz_rtarget.py --data synthetic --n 4 --out viz_rtarget_syn

Each PNG: [image + pointer | R* heatmap (red high, blue low) | G = 1[R* > 0.5] overlay (binary execution area)].
Prints the R* profile along the row through the pointer and mean(G). No augmentation unless --aug.
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
from lova.data.rtarget import downsample, r_profile, sigma_out_px  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", choices=["coco", "synthetic"], default="coco")
    p.add_argument("--coco-root", default="datasets/coco")
    p.add_argument("--split", default="val2017")
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--n", type=int, default=6, help="random samples (ignored with --image-id)")
    p.add_argument("--image-id", type=int, default=None, help="COCO image id (or synthetic index)")
    p.add_argument("--instance", type=int, default=None, help="instance index within the image (annotation order)")
    p.add_argument("--pointer", default=None, help="x,y in INPUT pixels (after resize to --img-size); default: sampled")
    p.add_argument("--pointer-mode", choices=["uniform", "interior"], default="interior")
    p.add_argument("--aug", action="store_true", help="apply training augmentation (random crop/flip/scale)")
    p.add_argument("--seed", type=int, default=0)
    # profile parameters (same names as train.py)
    p.add_argument("--r-profile-mode", choices=["geo", "dt"], default="geo")
    p.add_argument("--r-sigma-in", type=float, default=1.0)
    p.add_argument("--r-gamma", type=float, default=2.0)
    p.add_argument("--r-sigma-out-frac", type=float, default=0.05)
    p.add_argument("--r-b", type=float, default=0.7)
    p.add_argument("--r-gt-stride", type=int, default=2, choices=[1, 2, 4])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="viz_rtarget")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    st = a.r_gt_stride

    if a.data == "synthetic":
        from lova.data.synthetic import SyntheticPointerDataset
        ds = SyntheticPointerDataset(max(a.n, (a.image_id or 0) + 1), min(a.img_size, 256), seed=a.seed,
                                     pointer_mode=a.pointer_mode, mask_stride=st)
        index_of = lambda img_id: img_id
    else:
        from lova.data.coco import COCOPointerDataset
        ds = COCOPointerDataset(a.coco_root, a.split, a.img_size, train=a.aug, pointer_mode=a.pointer_mode, mask_stride=st)
        index_of = lambda img_id: ds.ids.index(img_id)
    s_out = sigma_out_px(ds.size, a.r_sigma_out_frac)
    print(f"mode={a.r_profile_mode} sigma_in={a.r_sigma_in} gamma={a.r_gamma} sigma_out={s_out:.1f}px "
          f"r_b={a.r_b} (dt only) gt_stride={st} aug={a.aug} seed={a.seed}")

    rng = np.random.RandomState(a.seed)
    if a.image_id is not None:
        jobs = [(index_of(a.image_id), a.instance)]
    else:
        jobs = [(int(i), None) for i in rng.choice(len(ds), size=min(a.n, len(ds)), replace=False)]

    for k, (i, inst) in enumerate(jobs):
        torch.manual_seed(a.seed * 7 + i)
        smp = ds.__getitem__(i, pointed=inst)
        if smp is None:
            print(f"[{k}] index {i}: instance vanished after transform")
            continue
        if a.pointer:
            smp["pointer"] = torch.tensor([float(v) for v in a.pointer.split(",")])
        pm = smp["pointed_mask_s"][None].to(a.device)
        ptr = smp["pointer"][None].to(a.device)
        rs_full = r_profile(pm, ptr, a.r_profile_mode, a.r_sigma_in, a.r_gamma, s_out, a.r_b, st)
        rs = (downsample(rs_full, 4 // st)[0] * smp["valid4"].to(a.device)).cpu()  # [1, S/4, S/4]

        img = (smp["image"] * STD + MEAN).clamp(0, 1)
        S = img.shape[-1]
        rr = F.interpolate(rs[None], size=(S, S), mode="bilinear", align_corners=False)[0]
        panel_r = torch.cat([rr, torch.zeros_like(rr), 1 - rr], 0) * 0.6 + img * 0.4
        g = (rr > 0.5).float()
        panel_g = (img * (0.35 + 0.65 * g) + torch.tensor([0.0, 0.3, 0.0]).view(3, 1, 1) * g * 0.3).clamp(0, 1)
        canvas = torch.cat([img, panel_r, panel_g], 2)
        pil = Image.fromarray((canvas.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        d = ImageDraw.Draw(pil)
        px, py = float(smp["pointer"][0]), float(smp["pointer"][1])
        for j in range(3):
            d.ellipse([px + j * S - 6, py - 6, px + j * S + 6, py + 6], outline=(255, 255, 0), width=3)
        tag = f"id{smp['meta']['image_id']}_inst{smp['pointed']}" if a.image_id is not None else f"{k:02d}"
        path = os.path.join(a.out, f"rtarget_{tag}.png")
        pil.save(path)

        row = rs[0, min(int(py // 4), rs.shape[1] - 1)]
        cx = int(px // 4)
        xs = list(range(max(cx - 24, 0), min(cx + 25, rs.shape[2]), 3))
        valid = smp["valid4"]
        area = float((smp["masks4"][smp["pointed"]] > 0.5).float().sum() / valid.sum())
        active = float(((rs > 0.5).float() * valid).sum() / valid.sum())
        print(f"[{k}] image {smp['meta']['image_id']} inst {smp['pointed']} pointer=({px:.0f},{py:.0f})  "
              f"mask area {area:.3f}  active(G) {active:.3f}  -> {path}")
        print("     x(cells): " + " ".join(f"{x:4d}" for x in xs))
        print("     R*      : " + " ".join(f"{row[x]:4.2f}" for x in xs))


if __name__ == "__main__":
    main()
