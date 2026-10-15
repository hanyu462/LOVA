"""COCO segm AP + pointed diagnostics for a checkpoint on the fixed ValSet.

    python evaluate.py --config configs/v0.py --root datasets/coco --ckpt runs/v0/last.pt [--n 500]
    python evaluate.py --config configs/a0.py --root datasets/coco --ckpt runs/a0/last.pt --r-override 1.0   # A0 in its own regime
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lova.engine.checkpoint import load_checkpoint  # noqa: E402
from lova.engine.decode import DecodeCfg  # noqa: E402
from lova.engine.evaluate import EvalCfg, evaluate, format_eval  # noqa: E402
from lova.engine.train import amp_dtype  # noqa: E402
from lova.engine.validate import ValCfg, ValSet, val_loader  # noqa: E402
from lova.models.lova import Lova  # noqa: E402
from train import load_config  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True); p.add_argument("--root", required=True); p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="val2017"); p.add_argument("--n", type=int, default=None); p.add_argument("--device", default=None)
    p.add_argument("--r-override", type=float, default=None, help="constant gate field (1.0 = dense regime for A0)")
    p.add_argument("--score-thr", type=float, default=0.05); p.add_argument("--json", default=None, help="append the result as one JSON line")
    a = p.parse_args()
    cfg = load_config(a.config)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    vcfg = cfg.val if a.n is None else ValCfg(n_images=a.n, seed=cfg.val.seed, batch_size=cfg.val.batch_size, workers=cfg.val.workers)
    vs = ValSet(a.root, a.split, cfg.data, vcfg.n_images, vcfg.seed)
    model = Lova(vs.num_classes, cfg.model).to(dev)
    update = load_checkpoint(a.ckpt, model, device=dev)
    res = evaluate(model, vs, val_loader(vs, vcfg), EvalCfg(decode=DecodeCfg(score_thr=a.score_thr)), a.r_override, amp_dtype(cfg.train))
    name = f"{os.path.basename(os.path.dirname(a.ckpt))} @ {update}" + ("" if a.r_override is None else f" R={a.r_override:g}")
    print(format_eval(res, name))
    if a.json:
        with open(a.json, "a") as f:
            f.write(json.dumps({"ckpt": a.ckpt, "update": update, "r_override": a.r_override, "n": vcfg.n_images, **res}) + "\n")
