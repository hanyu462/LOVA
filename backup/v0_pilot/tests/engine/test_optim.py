"""lova.engine.optim: parameter grouping, warm-up + cosine factors, scheduler == lr_factor, one real step.

    python tests/engine/test_optim.py            # synthetic + the default Lova model's groups
    python tests/engine/test_optim.py --plot     # draw the schedule
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lova.engine.optim import OptimCfg, SchedCfg, build_optimizer, build_scheduler, lr_factor, param_groups  # noqa: E402
from lova.models.lova import Lova  # noqa: E402


def unit_test():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Conv2d(3, 8, 3, bias=True), nn.GroupNorm(4, 8), nn.Linear(8, 2))
    groups = param_groups(m, OptimCfg(weight_decay=0.1))
    names = {id(p): n for n, p in m.named_parameters()}
    decay = sorted(names[id(p)] for p in groups[0]["params"]); nodecay = sorted(names[id(p)] for p in groups[1]["params"])
    assert decay == ["0.weight", "2.weight"] and nodecay == ["0.bias", "1.bias", "1.weight", "2.bias"], (decay, nodecay)
    assert groups[0]["weight_decay"] == 0.1 and groups[1]["weight_decay"] == 0.0
    assert sum(len(g["params"]) for g in groups) == len(list(m.parameters()))        # every parameter in exactly one group
    # schedule factors: 101 updates (0..100), warm-up 10 -> cosine over updates 10..100, min reached AT the final update
    cfg = SchedCfg(total_steps=101, warmup_steps=10, min_lr_ratio=0.1)
    assert abs(lr_factor(0, cfg) - 0.1) < 1e-9 and abs(lr_factor(9, cfg) - 1.0) < 1e-9        # warm-up: 1/W ... 1
    assert abs(lr_factor(10, cfg) - 1.0) < 1e-9                                                  # cosine starts at 1
    assert abs(lr_factor(55, cfg) - (0.1 + 0.9 * 0.5)) < 1e-9                                     # halfway of 10..100: midpoint
    assert abs(lr_factor(100, cfg) - 0.1) < 1e-9 and abs(lr_factor(500, cfg) - 0.1) < 1e-9        # final update = min, then flat
    assert lr_factor(99, cfg) > 0.1                                                               # ... and not before
    f = [lr_factor(s, cfg) for s in range(101)]
    assert all(f[i] <= f[i + 1] + 1e-12 for i in range(9)) and all(f[i] >= f[i + 1] - 1e-12 for i in range(10, 100))
    assert lr_factor(50, SchedCfg(100, 10, kind="constant")) == 1.0
    assert lr_factor(0, SchedCfg(10, 0)) == 1.0 and lr_factor(9, SchedCfg(10, 0)) == 0.01       # no warm-up is fine
    assert lr_factor(5, SchedCfg(6, 5)) == 0.01                                                  # a single cosine update = final
    for bad in (SchedCfg(100, 200), SchedCfg(0, 0), SchedCfg(100, 10, kind="step"), SchedCfg(100, 10, min_lr_ratio=-1),
                SchedCfg(100, 10, min_lr_ratio=2), SchedCfg(100, 100)):
        try:
            lr_factor(0, bad); raise AssertionError("must raise")
        except ValueError:
            pass
    try:
        lr_factor(-1, cfg); raise AssertionError("must raise")
    except ValueError:
        pass
    for bad_o in (OptimCfg(lr=0), OptimCfg(weight_decay=-1), OptimCfg(betas=(0.9, 1.0)), OptimCfg(betas=(0.9,))):
        try:
            param_groups(m, bad_o); raise AssertionError("must raise")
        except ValueError:
            pass
    # scheduler reproduces lr_factor step by step, and an optimizer step moves only parameters with gradient
    opt = build_optimizer(m, OptimCfg(lr=1e-3))
    sch = build_scheduler(opt, cfg)
    for s in range(30):
        assert all(abs(g["lr"] - 1e-3 * lr_factor(s, cfg)) < 1e-12 for g in opt.param_groups), s
        opt.step(); sch.step()
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    m[0].weight.grad = torch.ones_like(m[0].weight)
    opt.step()
    assert not torch.equal(m[0].weight, before["0.weight"]) and torch.equal(m[2].weight, before["2.weight"])
    # the real model: every trainable parameter in exactly one group, representative members where expected
    lova = Lova(80)
    g = param_groups(lova)
    dec_ids, nodec_ids = {id(p) for p in g[0]["params"]}, {id(p) for p in g[1]["params"]}
    assert dec_ids.isdisjoint(nodec_ids) and dec_ids | nodec_ids == {id(p) for p in lova.parameters() if p.requires_grad}
    assert id(lova.stem.body[0][0].weight) in dec_ids and id(lova.backbone.stages[0].blocks[0].body[1][2].weight) in dec_ids   # zero-init conv still decays
    assert id(lova.r_predictor.out.weight) in dec_ids and id(lova.r_predictor.out.bias) in nodec_ids
    assert id(lova.head.heat_out.bias) in nodec_ids and id(lova.stem.body[0][1].weight) in nodec_ids                       # GN affine
    n_dec, n_nodec = sum(p.numel() for p in g[0]["params"]), sum(p.numel() for p in g[1]["params"])
    assert n_dec + n_nodec == sum(p.numel() for p in lova.parameters()) and n_dec > 50 * n_nodec
    print(f"optim unit test OK   Lova(80): decay {n_dec / 1e6:.2f}M, no-decay {n_nodec / 1e6:.4f}M ({len(g[1]['params'])} tensors)")


def plot(cfg: SchedCfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = list(range(cfg.total_steps + 1))
    plt.figure(figsize=(7, 3)); plt.plot(xs, [lr_factor(s, cfg) for s in xs]); plt.xlabel("optimizer step"); plt.ylabel("lr factor")
    plt.title(f"warm-up {cfg.warmup_steps}, {cfg.kind} to {cfg.min_lr_ratio} over {cfg.total_steps}"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("lr_schedule.png", dpi=120); print("saved lr_schedule.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--plot", action="store_true"); p.add_argument("--total", type=int, default=20_000); p.add_argument("--warmup", type=int, default=1_000)
    a = p.parse_args()
    unit_test()
    if a.plot:
        plot(SchedCfg(a.total, a.warmup))
