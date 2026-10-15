"""Train LOVA.

    python train.py --config configs/smoke.py --root datasets/coco --out runs/smoke
    python train.py --config configs/smoke.py --root datasets/coco --out runs/smoke --resume            # continue from out/last.pt
    python train.py --config configs/v0.py --root datasets/coco --out runs/v0 --resume runs/v0/update_010000.pt

A config file defines CFG = RunCfg(...). A fresh run refuses an --out that already holds a last.pt.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lova.engine.loop import RunCfg, fit  # noqa: E402


def load_config(path: str) -> RunCfg:
    spec = importlib.util.spec_from_file_location("run_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = getattr(mod, "CFG", None)
    if not isinstance(cfg, RunCfg):
        raise ValueError(f"{path} must define CFG = RunCfg(...)")
    return cfg


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--resume", nargs="?", const=True, default=None, help="continue from out/last.pt, or from the given checkpoint")
    p.add_argument("--device", default=None)
    p.add_argument("--stop-at", type=int, default=None, help="stop after this many updates (resumable); for smoke tests")
    a = p.parse_args()
    fit(load_config(a.config), a.root, a.out, resume=a.resume, device=a.device, stop_at=a.stop_at)
