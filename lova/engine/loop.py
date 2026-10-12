"""Step 10-6a: the training loop. Wires the parts into one lifecycle; adds no algorithm.

    RunCfg ──fit(cfg, root, out, resume)──>
        CocoDataset(root, split, cfg.data) ──DataLoader (shuffled, restarted when exhausted)──> batches
        Lova(num_classes, cfg.model) + AdamW + schedule (+ GradScaler for fp16)
        while update < total_steps:
            log = train_step(...)                       one batch, one attempt
            if log["did_step"]: update += 1             the loop counts SUCCESSFUL updates (fp16 overflow:
            logger.add(update, log)                      batch consumed, update / schedule / checkpoint untouched)
            if log["did_step"]: checkpointer.step(update, ...)
        on a normal end (total_steps or a deliberate stop_at): one forced checkpoint so that last.pt
        AND the numbered file both hold the final update, whatever the cadences (never on an
        exception: the periodic last.pt is the safe point then)

resume: None -> fresh run (refused if `out` already holds any run artifact: meta.json, train.jsonl or a
checkpoint, so an old run is never silently continued or overwritten); True -> out/last.pt (must
exist); a path -> that file, branching into a NEW `out` (a meta.json is written there); an explicit
path into an existing run directory is refused unless it is that directory's own last.pt, so two
training histories never share a JSONL / checkpoint set.

Known limitation (not for V0): the DataLoader shuffles with the global torch RNG, which the model
initialisation has already consumed, so two architectures with the same seed do not see the same
data order. A dedicated loader generator (with its state in the checkpoint) is the fix when strict
A/B ablations start. The checkpoint
meta must match this run's (data, model, loss, train, optim, sched, num_classes, batch_size, split,
seed): everything that changes what is optimised. Output dir, log cadence, checkpoint cadence and
worker count are not part of it.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader

from ..data.coco.dataset import CocoDataset, worker_init
from ..data.common.build import PipelineCfg, collate
from ..losses.total_loss import TotalLossCfg
from ..models.lova import Lova, LovaCfg
from .checkpoint import LAST, Checkpointer, CkptCfg, latest, load_checkpoint, to_meta
from .logger import LogCfg, TrainLogger
from .optim import OptimCfg, SchedCfg, build_optimizer, build_scheduler
from .train import TrainCfg, make_scaler, train_step


@dataclass(frozen=True)
class RunCfg:
    data: PipelineCfg = field(default_factory=PipelineCfg)
    model: LovaCfg = field(default_factory=LovaCfg)
    loss: TotalLossCfg = field(default_factory=TotalLossCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)
    sched: SchedCfg = field(default_factory=SchedCfg)
    split: str = "train2017"
    batch_size: int = 8
    seed: int = 0
    # not part of the checkpoint meta
    workers: int = 8
    log_every: int = 20
    log_window: int = 100
    ckpt_every: int = 2000
    ckpt_keep: int = 3
    ckpt_every_last: int = 200


def run_meta(cfg: RunCfg, num_classes: int) -> dict:
    return to_meta(data=cfg.data, model=cfg.model, loss=cfg.loss, train=cfg.train, optim=cfg.optim, sched=cfg.sched,
                   num_classes=num_classes, batch_size=cfg.batch_size, split=cfg.split, seed=cfg.seed)


def build_loader(cfg: RunCfg, root: str, device: torch.device) -> tuple[DataLoader, int]:
    ds = CocoDataset(root, cfg.split, cfg.data, train=True)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers, collate_fn=collate,
                        worker_init_fn=worker_init, drop_last=True, pin_memory=device.type == "cuda",
                        persistent_workers=cfg.workers > 0)
    return loader, ds.num_classes


def has_run_artifacts(out: str) -> bool:
    if not os.path.isdir(out):
        return False
    return any(n in ("meta.json", "train.jsonl", LAST) or (n.startswith("update_") and n.endswith(".pt")) for n in os.listdir(out))


def infinite_batches(loader):
    """The schedule is in updates, not epochs: when the loader is exhausted, start it again (reshuffled)."""
    while True:
        yield from loader


def fit(cfg: RunCfg, root: str | None, out: str, resume: bool | str | None = None, device: str | torch.device | None = None,
        stop_at: int | None = None, loader=None, num_classes: int | None = None) -> dict:
    """Train until `update == cfg.sched.total_steps` (or `stop_at`, to interrupt a run on purpose).
    `loader` / `num_classes` override the COCO loader (tests). Returns a summary dict."""
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_path = None
    if resume is True:
        ckpt_path = latest(out)
        if ckpt_path is None:
            raise FileNotFoundError(f"--resume given but {os.path.join(out, LAST)} does not exist")
    elif isinstance(resume, str):
        ckpt_path = os.path.abspath(resume)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(ckpt_path)
        out_last = latest(out)
        if has_run_artifacts(out) and (out_last is None or os.path.abspath(out_last) != ckpt_path):
            raise FileExistsError("resuming from an explicit checkpoint into an existing run directory would mix training "
                                  "histories: use a new --out (or --resume alone to continue this directory's last.pt)")
    elif has_run_artifacts(out):
        raise FileExistsError(f"{out} already contains training artifacts: pass --resume to continue it, or choose another --out")
    os.makedirs(out, exist_ok=True)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if loader is None:
        loader, num_classes = build_loader(cfg, root, device)
    elif num_classes is None:
        raise ValueError("num_classes is required with a custom loader")
    meta = run_meta(cfg, num_classes)
    model = Lova(num_classes, cfg.model).to(device)
    optimizer = build_optimizer(model, cfg.optim)
    scheduler = build_scheduler(optimizer, cfg.sched)
    scaler = make_scaler(cfg.train, device)
    update = 0
    if ckpt_path is not None:
        update = load_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, meta, device=device)
        print(f"resumed from {ckpt_path} at update {update}")
    if not os.path.exists(os.path.join(out, "meta.json")):          # fresh run, or a path resume into a new directory
        with open(os.path.join(out, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
    logger = TrainLogger(LogCfg(cfg.log_every, cfg.log_window, os.path.join(out, "train.jsonl")))
    checkpointer = Checkpointer(CkptCfg(out, cfg.ckpt_every, cfg.ckpt_keep, cfg.ckpt_every_last))
    target = cfg.sched.total_steps if stop_at is None else min(stop_at, cfg.sched.total_steps)
    print(f"{device}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, B={cfg.batch_size}, "
          f"amp={cfg.train.amp_dtype if cfg.train.amp else 'off'}, updates {update} -> {target} (total {cfg.sched.total_steps})")
    batches = infinite_batches(loader)
    t0, start, attempts = time.perf_counter(), update, 0
    try:
        while update < target:
            log = train_step(model, next(batches), optimizer, scheduler, cfg.loss, cfg.train, scaler)
            attempts += 1
            if log["did_step"]:
                update += 1
            logger.add(update, log)
            if log["did_step"]:
                checkpointer.step(update, model, optimizer, scheduler, scaler, meta)
        # the loop ended normally (total_steps reached, or a deliberate stop_at): write this exact update as last.pt
        # AND a numbered file, whatever the cadences (one possibly redundant save per run). Never done on an exception.
        if update > start:
            checkpointer.step(update, model, optimizer, scheduler, scaler, meta, force=True)
    finally:
        logger.close()
    elapsed = time.perf_counter() - t0
    done = update - start
    print(f"done: {done} updates in {elapsed:.0f}s ({elapsed / max(done, 1) * 1e3:.0f} ms/update, {attempts - done} skipped attempts)")
    return {"update": update, "attempts": attempts, "skipped": logger.skipped, "elapsed": elapsed, "last": latest(out)}
