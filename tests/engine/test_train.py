"""lova.engine.train: one-batch-one-update contract. A normal step (fp32 / bf16), clipping, fp16 overflow
skipping optimizer AND scheduler, non-finite loss failing fast, scheduler counting successful updates only.

    python tests/engine/test_train.py
    python tests/engine/test_train.py --root datasets/coco --steps 20     # real batches, bf16 on the GPU, logs per step
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lova.engine.optim import OptimCfg, SchedCfg, build_optimizer, build_scheduler, lr_factor  # noqa: E402
from lova.engine.train import TrainCfg, amp_dtype, make_scaler, to_device, train_step  # noqa: E402
from lova.losses.total_loss import TotalLossCfg  # noqa: E402
from lova.models.lova import Lova  # noqa: E402
from tests.models.test_lova import SMALL, fake_batch  # noqa: E402

LOG_KEYS = {"total", "r_raw", "heat_raw", "mask_raw", "r", "heat", "mask", "grad_norm", "clipped", "r_budget", "lr", "did_step", "scale"}


def snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def changed(model, snap):
    return any(not torch.equal(p, snap[n]) for n, p in model.named_parameters())


def unit_test():
    torch.manual_seed(0)
    B, K, S = 2, 5, 64
    sched = SchedCfg(total_steps=50, warmup_steps=5)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def fresh(cfg: TrainCfg, opt_kind="adamw", lr=1e-3):
        model = Lova(K, SMALL).to(dev)
        opt = build_optimizer(model, OptimCfg(lr=lr)) if opt_kind == "adamw" else torch.optim.SGD(model.parameters(), lr=lr)
        return model, opt, build_scheduler(opt, sched), make_scaler(cfg, dev)

    # ---- A. normal step in fp32 and bf16: parameters move, scheduler advances exactly once, logs complete ----
    for cfg in (TrainCfg(amp=False), TrainCfg(amp=True, amp_dtype="bf16")):
        model, opt, sch, scaler = fresh(cfg)
        assert scaler is None
        snap = snapshot(model)
        log = train_step(model, fake_batch(B, K, S), opt, sch, TotalLossCfg(), cfg, scaler)
        assert set(log) == LOG_KEYS and torch.isfinite(log["total"]) and log["did_step"] is True
        assert changed(model, snap) and sch.last_epoch == 1
        assert abs(opt.param_groups[0]["lr"] - 1e-3 * lr_factor(1, sched)) < 1e-12 and abs(log["lr"] - 1e-3 * lr_factor(0, sched)) < 1e-12
        assert float(log["grad_norm"]) > 0 and 0 <= float(log["r_budget"]) <= 1
        assert torch.allclose(log["total"], log["r"] + log["heat"] + log["mask"]) and torch.allclose(log["mask"], 3 * log["mask_raw"])
        assert all(not log[k].requires_grad for k in ("total", "r_raw", "heat_raw", "mask_raw", "r", "heat", "mask", "grad_norm", "clipped", "r_budget"))
        assert all(log[k].dim() == 0 for k in ("total", "r_raw", "heat_raw", "mask_raw", "r", "heat", "mask", "grad_norm", "clipped", "r_budget"))
    # ---- B. clipping: pre-clip norm reported, the update uses the clipped gradient (SGD: delta = -lr * g_clipped) ----
    cfg = TrainCfg(amp=False, grad_clip_norm=1.0)
    model, opt, sch, _ = fresh(cfg, "sgd", lr=1.0)
    for p in model.parameters():                                   # make the gradient huge: scale the loss through a hook
        p.register_hook(lambda g: g * 1e4)
    snap = snapshot(model)
    log = train_step(model, fake_batch(B, K, S), opt, sch, TotalLossCfg(), cfg)
    assert float(log["grad_norm"]) > 1.0 and bool(log["clipped"]) is True
    delta = torch.sqrt(sum(((p.detach() - snap[n]) ** 2).sum() for n, p in model.named_parameters()))
    assert abs(float(delta) - log["lr"] * 1.0) < 1e-3, (float(delta), log["lr"])   # ||lr * g_clipped|| == lr * max_norm when clipping was active
    model2, opt2, sch2, _ = fresh(TrainCfg(amp=False, grad_clip_norm=None), "sgd", lr=1.0)
    log2 = train_step(model2, fake_batch(B, K, S), opt2, sch2, TotalLossCfg(), TrainCfg(amp=False, grad_clip_norm=None))
    assert float(log2["grad_norm"]) > 0 and bool(log2["clipped"]) is False   # still measured without clipping, never "clipped"
    model3, opt3, sch3, _ = fresh(TrainCfg(amp=False, grad_clip_norm=1e9))
    assert bool(train_step(model3, fake_batch(B, K, S), opt3, sch3, TotalLossCfg(), TrainCfg(amp=False, grad_clip_norm=1e9))["clipped"]) is False
    # ---- C. fp16 + GradScaler: an overflow skips the optimizer update AND the scheduler, and halves the scale ----
    cfg16 = TrainCfg(amp=True, amp_dtype="fp16")
    model, opt, sch, scaler = fresh(cfg16)
    assert scaler is not None and amp_dtype(cfg16) is torch.float16
    # with the default init scale (65536) the first step or two may overflow on this model; GradScaler handles that
    # by skipping and halving, and every skipped step must leave the scheduler untouched
    ok = 0
    for _ in range(8):
        log = train_step(model, fake_batch(B, K, S), opt, sch, TotalLossCfg(), cfg16, scaler)
        ok += int(log["did_step"])
        assert sch.last_epoch == ok, (sch.last_epoch, ok)
        if ok == 2:
            break
    assert ok == 2, "fp16 never produced a healthy step"
    h = next(iter(model.head.heat_out.parameters())).register_hook(lambda g: g * float("inf"))   # forced overflow
    snap, scale0 = snapshot(model), scaler.get_scale()
    log = train_step(model, fake_batch(B, K, S), opt, sch, TotalLossCfg(), cfg16, scaler)
    h.remove()
    assert log["did_step"] is False and not changed(model, snap) and sch.last_epoch == ok and log["scale"] < scale0
    log = train_step(model, fake_batch(B, K, S), opt, sch, TotalLossCfg(), cfg16, scaler)      # recovers
    assert log["did_step"] is True and sch.last_epoch == ok + 1
    # ---- D. contracts: non-finite loss without a scaler fails fast; scaler / dtype mismatch; bad cfg ----
    model, opt, sch, _ = fresh(TrainCfg(amp=False))
    bad = fake_batch(B, K, S); bad["image"][0, 0, 0, 0] = float("inf")
    try:
        train_step(model, bad, opt, sch, TotalLossCfg(), TrainCfg(amp=False)); raise AssertionError("must raise")
    except FloatingPointError:
        pass
    assert sch.last_epoch == 0                                     # nothing advanced
    # finite loss but a NaN gradient (hook): fp32 and bf16 must refuse, leaving optimizer and scheduler untouched
    for cfg_nf in (TrainCfg(amp=False), TrainCfg(amp=True, amp_dtype="bf16")):
        model, opt, sch, _ = fresh(cfg_nf)
        h = next(iter(model.head.heat_out.parameters())).register_hook(lambda g: g * float("nan"))
        snap = snapshot(model)
        try:
            train_step(model, fake_batch(B, K, S), opt, sch, TotalLossCfg(), cfg_nf); raise AssertionError("must raise")
        except FloatingPointError:
            pass
        h.remove()
        assert not changed(model, snap) and sch.last_epoch == 0
    for args in ((TrainCfg(amp=True, amp_dtype="fp16"), None), (TrainCfg(amp=False), torch.amp.GradScaler(dev))):
        try:
            train_step(model, fake_batch(B, K, S), opt, sch, TotalLossCfg(), args[0], args[1]); raise AssertionError("must raise")
        except ValueError:
            pass
    for bad_cfg in (TrainCfg(amp_dtype="int8"), TrainCfg(grad_clip_norm=0.0), TrainCfg(grad_clip_norm=float("nan")), TrainCfg(grad_clip_norm=float("inf"))):
        try:
            amp_dtype(bad_cfg); raise AssertionError("must raise")
        except ValueError:
            pass
    # ---- E. the schedule follows successful updates: 5 steps -> lr factor of update 5 ----
    model, opt, sch, _ = fresh(TrainCfg(amp=False))
    for _ in range(5):
        train_step(model, fake_batch(B, K, S), opt, sch)
    assert abs(opt.param_groups[0]["lr"] - 1e-3 * lr_factor(5, sched)) < 1e-12
    # to_device moves lists of tensors too
    moved = to_device(fake_batch(B, K, S), dev)
    assert moved["masks_s4"][0].device.type == dev and moved["image"].device.type == dev
    print(f"train_step unit test OK   ({dev}; fp32 / bf16 / fp16+scaler paths)")


def real(a):
    from torch.utils.data import DataLoader
    from lova.data.coco.dataset import CocoDataset, worker_init
    from lova.data.common.build import PipelineCfg, collate
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    ds = CocoDataset(a.root, a.split, PipelineCfg(), train=True)
    loader = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, collate_fn=collate, worker_init_fn=worker_init, drop_last=True, pin_memory=True)
    model = Lova(ds.num_classes).to(dev)
    cfg = TrainCfg(amp=True, amp_dtype=a.amp)
    opt = build_optimizer(model)
    sch = build_scheduler(opt, SchedCfg(total_steps=a.steps * 10, warmup_steps=a.steps))
    scaler = make_scaler(cfg, dev)
    print(f"{dev}, B={a.batch}, amp={a.amp}, clip={cfg.grad_clip_norm}: {a.steps} steps")
    it = iter(loader); t0 = time.perf_counter(); norms, clips = [], []
    for k in range(a.steps):
        log = train_step(model, next(it), opt, sch, TotalLossCfg(), cfg, scaler)
        norms.append(float(log["grad_norm"])); clips.append(bool(log["clipped"]))
        if k % max(a.steps // 10, 1) == 0 or k == a.steps - 1:
            print(f"  step {k:4d}  total {float(log['total']):.3f}  r {float(log['r_raw']):.4f}  heat {float(log['heat_raw']):.3f}  mask {float(log['mask_raw']):.4f}  "
                  f"grad_norm {float(log['grad_norm']):.2f}{'*' if log['clipped'] else ' '} lr {log['lr']:.2e}  budget {float(log['r_budget']):.3f}  step {log['did_step']}")
    if dev == "cuda": torch.cuda.synchronize()
    n = torch.tensor(norms); c = torch.tensor(clips, dtype=torch.float32); half = len(c) // 2
    print(f"  {(time.perf_counter() - t0) / a.steps * 1e3:.0f} ms/step   grad_norm median {float(n.median()):.2f}  p95 {float(n.quantile(0.95)):.2f}  "
          f"clipped {float(c.mean()):.0%} of steps (first half {float(c[:half].mean()):.0%}, second half {float(c[half:].mean()):.0%})   "
          f"max mem {torch.cuda.max_memory_allocated() / 2**30 if dev == 'cuda' else 0:.1f} GB")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=None); p.add_argument("--split", default="val2017")
    p.add_argument("--steps", type=int, default=20); p.add_argument("--batch", type=int, default=8); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "fp32"])
    a = p.parse_args()
    unit_test()
    if a.root:
        real(a)
