"""Step 10-4: training logger. Successful-update statistics over a moving window, to stdout and JSONL.

    step_log = train_step(...)   ──logger.add(update, step_log)──>  summary dict every `every` updates, else None

    counted only when step_log["did_step"] is True (a successful optimizer update):
        window averages: total, r_raw, heat_raw, mask_raw, r, heat, mask, grad_norm, r_budget
        clip_fraction  = #(clipped and did_step) / #(did_step) over the same window
    state, never averaged:
        lr       the lr used by the latest update
        scale    the GradScaler scale (fp16) or 1.0
        skipped  attempts skipped since this logger instance was created (fp16 overflow; a resume
                 starts a new count); they never enter the window, an overflow's grad_norm = inf /
                 clipped = True would otherwise poison the statistics

`update` is the number of successful optimizer updates, the same count the schedule and the
checkpoint use; a skipped attempt at an emit boundary never re-emits that update. Tensors are
converted to Python scalars on entry. The JSONL file is opened in append mode so a resume keeps the
earlier lines. The window state is observational and is NOT checkpointed: after a resume it refills.
TODO: `scale` is reported when != 1.0; an fp16 scaler driven all the way down to 1.0 would hide it
(train_step could return None for bf16 / fp32 instead).
"""
from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass

import torch

AVERAGED = ("total", "r_raw", "heat_raw", "mask_raw", "r", "heat", "mask", "grad_norm", "r_budget")


@dataclass(frozen=True)
class LogCfg:
    every: int = 20            # emit a summary every this many successful updates
    window: int = 100          # moving-average window (successful updates)
    jsonl_path: str | None = None


def _scalar(x):
    if torch.is_tensor(x):
        return x.item()
    return x


class TrainLogger:
    def __init__(self, cfg: LogCfg = LogCfg()):
        if cfg.every <= 0 or cfg.window <= 0:
            raise ValueError(f"every and window must be > 0, got {cfg.every} / {cfg.window}")
        self.cfg = cfg
        self.window = {k: deque(maxlen=cfg.window) for k in AVERAGED}
        self.clipped = deque(maxlen=cfg.window)
        self.skipped = 0
        self.lr = float("nan")
        self.scale = 1.0
        self._file = open(cfg.jsonl_path, "a", buffering=1) if cfg.jsonl_path else None

    def add(self, update: int, step_log: dict) -> dict | None:
        """Record one train_step result. `update` = successful updates so far (after this step)."""
        self.scale = float(_scalar(step_log["scale"]))
        if not bool(_scalar(step_log["did_step"])):
            self.skipped += 1                    # a skipped attempt is not an update: lr stays, nothing enters the window
            return None
        self.lr = float(_scalar(step_log["lr"]))
        for k in AVERAGED:
            self.window[k].append(float(_scalar(step_log[k])))
        self.clipped.append(bool(_scalar(step_log["clipped"])))
        if update % self.cfg.every:
            return None
        return self.emit(update)

    def summary(self, update: int) -> dict:
        s = {"update": int(update)}
        for k in AVERAGED:
            w = self.window[k]
            s[k] = sum(w) / len(w) if w else float("nan")
        s["clip_fraction"] = sum(self.clipped) / len(self.clipped) if self.clipped else float("nan")
        s["lr"] = self.lr
        s["skipped"] = self.skipped
        if self.scale != 1.0:
            s["scale"] = self.scale
        return s

    def emit(self, update: int) -> dict:
        s = self.summary(update)
        print(self.format(s), flush=True)
        if self._file is not None:
            self._file.write(json.dumps(s) + "\n")
            self._file.flush()
        return s

    @staticmethod
    def format(s: dict) -> str:
        # weighted terms on stdout so that total ~= r + heat + mask by eye; raw terms are in the JSONL
        line = (f"[{s['update']:06d}] total {s['total']:.3f} | r {s['r']:.3f} heat {s['heat']:.3f} mask {s['mask']:.3f} | "
                f"grad {s['grad_norm']:.1f} | clip {s['clip_fraction']:.0%} | R {s['r_budget']:.3f} | lr {s['lr']:.2e}")
        if s["skipped"]:
            line += f" | skip {s['skipped']}"
        if "scale" in s and not math.isnan(s["scale"]):
            line += f" | scale {s['scale']:.0f}"
        return line

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
