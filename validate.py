"""Fast deterministic validation of a checkpoint (no AP: that is the COCO evaluator).

    python validate.py --config configs/v0.py --root datasets/coco --ckpt runs/v0/last.pt [--n 500] [--split val2017]
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lova.engine.checkpoint import load_checkpoint  # noqa: E402
from lova.engine.train import amp_dtype  # noqa: E402
from lova.engine.validate import ValCfg, ValSet, format_val, val_loader, validate  # noqa: E402
from lova.models.lova import Lova  # noqa: E402
from train import load_config  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True); p.add_argument("--root", required=True); p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="val2017"); p.add_argument("--n", type=int, default=None); p.add_argument("--device", default=None)
    a = p.parse_args()
    cfg = load_config(a.config)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    vcfg = cfg.val if a.n is None else ValCfg(n_images=a.n, seed=cfg.val.seed, batch_size=cfg.val.batch_size, workers=cfg.val.workers)
    vs = ValSet(a.root, a.split, cfg.data, vcfg.n_images, vcfg.seed)
    model = Lova(vs.num_classes, cfg.model).to(dev)
    update = load_checkpoint(a.ckpt, model, device=dev)
    print(format_val(validate(model, val_loader(vs, vcfg), cfg.loss, amp_dtype=amp_dtype(cfg.train)), update))
