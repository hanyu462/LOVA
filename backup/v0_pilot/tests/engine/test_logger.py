"""lova.engine.logger: window averages, successful-update-only counting, clip fraction, JSONL append.

    python tests/engine/test_logger.py
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lova.engine.logger import AVERAGED, LogCfg, TrainLogger  # noqa: E402


def step_log(total, grad_norm=10.0, clipped=False, did_step=True, lr=1e-4, scale=1.0, r_budget=0.3):
    """A train_step-shaped dict with tensors where train_step returns tensors."""
    t = lambda v: torch.tensor(float(v))
    return {"total": t(total), "r_raw": t(0.1), "heat_raw": t(total - 0.4), "mask_raw": t(0.1), "r": t(0.1), "heat": t(total - 0.4), "mask": t(0.3),
            "grad_norm": t(grad_norm), "clipped": torch.tensor(clipped), "r_budget": t(r_budget), "lr": lr, "did_step": did_step, "scale": scale}


def unit_test():
    # 1. window average of known scalars; summary only every `every`; lr is the latest value, not averaged
    lg = TrainLogger(LogCfg(every=5, window=3))
    out = []
    with redirect_stdout(io.StringIO()):
        for u, v in enumerate([1.0, 2.0, 3.0, 4.0, 5.0], start=1):
            out.append(lg.add(u, step_log(v, lr=u * 1e-5)))
    assert out[:4] == [None] * 4 and out[4] is not None
    s = out[4]
    assert abs(s["total"] - (3.0 + 4.0 + 5.0) / 3) < 1e-6                       # window 3 -> last three (values pass through float32)
    assert abs(s["heat_raw"] - (2.6 + 3.6 + 4.6) / 3) < 1e-6 and abs(s["lr"] - 5e-5) < 1e-15 and s["update"] == 5
    assert all(k in s for k in AVERAGED) and s["clip_fraction"] == 0.0 and s["skipped"] == 0 and "scale" not in s
    assert all(not torch.is_tensor(v) for v in s.values())
    assert abs(s["total"] - (s["r"] + s["heat"] + s["mask"])) < 1e-5            # stdout shows the weighted terms: they add up
    assert TrainLogger.format(s).split("| r ")[1].startswith(f"{s['r']:.3f} heat {s['heat']:.3f} mask {s['mask']:.3f}")
    # 2. a skipped attempt (fp16 overflow: grad_norm inf, clipped True) touches neither the averages nor the clip denominator
    lg = TrainLogger(LogCfg(every=1, window=10))
    with redirect_stdout(io.StringIO()):
        lg.add(1, step_log(2.0))
        r = lg.add(1, step_log(99.0, grad_norm=float("inf"), clipped=True, did_step=False, scale=32768.0))
        assert r is None and lg.skipped == 1
        s = lg.add(2, step_log(4.0, scale=32768.0))
    assert abs(s["total"] - 3.0) < 1e-6 and s["clip_fraction"] == 0.0 and s["skipped"] == 1 and s["scale"] == 32768.0
    assert len(lg.window["grad_norm"]) == 2 and all(v < float("inf") for v in lg.window["grad_norm"])
    # 2b. emit boundary: after update 2 emitted, skipped attempts at the same update must not re-emit, and lr is untouched
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "b.jsonl")
        lg = TrainLogger(LogCfg(every=2, window=10, jsonl_path=path))
        buf = io.StringIO()
        with redirect_stdout(buf):
            assert lg.add(1, step_log(1.0, lr=1e-4)) is None and lg.add(2, step_log(1.0, lr=2e-4)) is not None
            assert lg.add(2, step_log(9.0, grad_norm=float("inf"), clipped=True, did_step=False, lr=3e-4)) is None
            assert lg.add(2, step_log(9.0, grad_norm=float("inf"), clipped=True, did_step=False, lr=3e-4)) is None
        lg.close()
        assert buf.getvalue().count("[000002]") == 1 and len(open(path).read().splitlines()) == 1
        assert lg.skipped == 2 and abs(lg.lr - 2e-4) < 1e-15 and len(lg.clipped) == 2
    # 3. clip fraction over successful updates: True / False / True -> 2/3
    lg = TrainLogger(LogCfg(every=3, window=10))
    with redirect_stdout(io.StringIO()):
        lg.add(1, step_log(1.0, clipped=True)); lg.add(2, step_log(1.0, clipped=False)); s = lg.add(3, step_log(1.0, clipped=True))
    assert abs(s["clip_fraction"] - 2 / 3) < 1e-9
    # 4. JSONL: lines only at multiples of every, parseable, appended across logger instances
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "train.jsonl")
        lg = TrainLogger(LogCfg(every=2, window=10, jsonl_path=path))
        with redirect_stdout(io.StringIO()):
            for u in range(1, 6):
                lg.add(u, step_log(float(u)))
        lg.close()
        lines = [json.loads(l) for l in open(path)]
        assert [l["update"] for l in lines] == [2, 4]
        lg2 = TrainLogger(LogCfg(every=2, window=10, jsonl_path=path))                 # "resume": must append, not truncate
        with redirect_stdout(io.StringIO()):
            lg2.add(6, step_log(6.0))
        lg2.close()
        lines = [json.loads(l) for l in open(path)]
        assert [l["update"] for l in lines] == [2, 4, 6] and abs(lines[-1]["total"] - 6.0) < 1e-6
    # stdout format and config validation
    line = TrainLogger.format(lg.summary(1000) | {"skipped": 2, "scale": 32768.0})
    assert line.startswith("[001000]") and "| skip 2" in line and "| scale 32768" in line
    for bad in (LogCfg(every=0), LogCfg(window=0)):
        try:
            TrainLogger(bad); raise AssertionError("must raise")
        except ValueError:
            pass
    print("logger unit test OK")


if __name__ == "__main__":
    unit_test()
