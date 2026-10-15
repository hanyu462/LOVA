"""Step 10-5: checkpoint / resume. One file holds the optimisation state and the process RNG state
required to resume training from a successful optimizer update.

    save_checkpoint(path, model, optimizer, scheduler, scaler, update, meta)
        {"update", "model", "optimizer", "scheduler", "scaler" | None, "rng", "meta"}  written ATOMICALLY:
        torch.save to a temporary file in the same directory, then os.replace -> a crash mid-write
        never corrupts an existing checkpoint
    load_checkpoint(path, model, optimizer, scheduler, scaler, meta) -> update
        two modes: resume (optimizer AND scheduler given; scaler iff the run is fp16) restores every
        state and the RNG; evaluation (model only) restores the weights and ignores the rest.
        All checks (format version, meta, mode, scaler compatibility) run BEFORE any state is touched.
    Checkpointer(cfg).step(update, ...)   last.pt every `every_last` updates, update_XXXXXX.pt every
        `every` updates with the oldest pruned beyond `keep`; `latest(dir)` finds last.pt for a resume

`update` is the number of successful optimizer updates (the schedule's index and the logger's
counter). The process RNG state (torch CPU, every CUDA device, Python random; numpy's is not used
anywhere in lova/) is restored to the save point. Exact continuation of the data order / worker
randomness is NOT guaranteed: DataLoader and sampler state are not checkpointed (a resume restarts
the shuffle), and neither is the logger's moving window.
"""
from __future__ import annotations

import os
import random
import tempfile
from dataclasses import asdict, dataclass, is_dataclass

import torch
import torch.nn as nn

LAST = "last.pt"
FORMAT_VERSION = 1


@dataclass(frozen=True)
class CkptCfg:
    dir: str
    every: int = 2000          # numbered checkpoints
    keep: int = 3              # how many numbered checkpoints to keep
    every_last: int = 200      # how often last.pt is refreshed


def to_meta(**items) -> dict:
    """Plain-dict snapshot of configs (dataclasses are expanded) and scalars, comparable across runs."""
    def conv(v):
        if is_dataclass(v):
            return {k: conv(x) for k, x in asdict(v).items()}
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        if isinstance(v, frozenset):
            return sorted(conv(x) for x in v)
        return v
    return {k: conv(v) for k, v in items.items()}


def rng_state() -> dict:
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python": random.getstate()}


def set_rng_state(state: dict) -> None:
    """RNG state tensors must be CPU ByteTensors even when the checkpoint was mapped to a GPU."""
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"][:torch.cuda.device_count()]])
    random.setstate(state["python"])


def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler, scaler, update: int, meta: dict) -> str:
    """Atomic: the file at `path` is either the previous checkpoint or the complete new one, never partial."""
    payload = {"format_version": FORMAT_VERSION, "update": int(update), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "scheduler": scheduler.state_dict(), "scaler": None if scaler is None else scaler.state_dict(),
               "rng": rng_state(), "meta": meta}
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".pt", dir=d)
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer | None = None, scheduler=None, scaler=None,
                    meta: dict | None = None, strict_meta: bool = True, device: str | torch.device = "cpu") -> int:
    """Returns `update`. Resume: pass optimizer and scheduler (and the scaler iff the run is fp16); every
    state and the RNG are restored. Evaluation: pass the model alone; only the weights are restored and
    the checkpoint's scaler / optimizer state is ignored. `meta` (same to_meta() as at save time) must
    match. All compatibility checks run before any state is modified (a corrupt file or a mismatched
    architecture can still fail inside load_state_dict)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    # ---- checks first, mutation after ----
    if ck.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"checkpoint format {ck.get('format_version')} != {FORMAT_VERSION}")
    if meta is not None and ck["meta"] != meta:
        diff = {k: (ck["meta"].get(k), meta.get(k)) for k in set(ck["meta"]) | set(meta) if ck["meta"].get(k) != meta.get(k)}
        if strict_meta:
            raise ValueError(f"checkpoint meta differs from the current run: {diff}")
        print(f"warning: checkpoint meta differs: {diff}")
    if (optimizer is None) != (scheduler is None):
        raise ValueError("optimizer and scheduler must be given together (resume) or both omitted (evaluation)")
    resume = optimizer is not None
    if scaler is not None and not resume:
        raise ValueError("a scaler is only meaningful for a training resume")
    if resume and (scaler is None) != (ck["scaler"] is None):
        raise ValueError("GradScaler mismatch: the checkpoint was saved " + ("with" if ck["scaler"] is not None else "without")
                         + " a scaler (fp16) but this run is " + ("without" if scaler is None else "with") + " one")
    # ---- mutation ----
    model.load_state_dict(ck["model"])
    if resume:
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        if scaler is not None:
            scaler.load_state_dict(ck["scaler"])
        set_rng_state(ck["rng"])
    return int(ck["update"])


def latest(d: str) -> str | None:
    p = os.path.join(d, LAST)
    return p if os.path.exists(p) else None


class Checkpointer:
    def __init__(self, cfg: CkptCfg):
        if cfg.every <= 0 or cfg.keep <= 0 or cfg.every_last <= 0:
            raise ValueError(f"every / keep / every_last must be > 0, got {cfg}")
        self.cfg = cfg
        os.makedirs(cfg.dir, exist_ok=True)

    def numbered(self) -> list[str]:
        names = sorted(n for n in os.listdir(self.cfg.dir) if n.startswith("update_") and n.endswith(".pt"))
        return [os.path.join(self.cfg.dir, n) for n in names]

    def step(self, update: int, model, optimizer, scheduler, scaler, meta: dict, force: bool = False) -> list[str]:
        """Call after every successful update. Returns the paths written (usually none)."""
        written = []
        if force or update % self.cfg.every_last == 0:
            written.append(save_checkpoint(os.path.join(self.cfg.dir, LAST), model, optimizer, scheduler, scaler, update, meta))
        if force or update % self.cfg.every == 0:
            written.append(save_checkpoint(os.path.join(self.cfg.dir, f"update_{update:06d}.pt"), model, optimizer, scheduler, scaler, update, meta))
            for old in self.numbered()[:-self.cfg.keep]:
                os.remove(old)
        return written
