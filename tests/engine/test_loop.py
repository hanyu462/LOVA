"""lova.engine.loop: the lifecycle on synthetic batches. Run A trains 0 -> 12 and stops; run B resumes 12 -> 20.
Checks: update count, scheduler, lr, JSONL continuity without duplicates, checkpoint retention, resume rules.

    python tests/engine/test_loop.py
    python tests/engine/test_loop.py --root datasets/coco     # the same lifecycle through train.py on real data (configs/smoke.py)
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from dataclasses import replace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lova.engine.checkpoint import load_checkpoint  # noqa: E402
from lova.engine.loop import RunCfg, fit, infinite_batches, run_meta  # noqa: E402
from lova.engine.optim import OptimCfg, SchedCfg, lr_factor  # noqa: E402
from lova.engine.train import TrainCfg  # noqa: E402
from lova.models.lova import Lova  # noqa: E402
from tests.models.test_lova import SMALL, fake_batch  # noqa: E402

K, S, B = 5, 64, 2


class FakeLoader:
    """A 7-batch 'dataset': shorter than the run, so infinite_batches must restart it."""
    def __init__(self, n=7):
        self.n = n
    def __iter__(self):
        for _ in range(self.n):
            yield fake_batch(B, K, S)
    def __len__(self):
        return self.n


def unit_test():
    cfg = replace(RunCfg(), model=SMALL, train=TrainCfg(amp=False), optim=OptimCfg(lr=1e-3), sched=SchedCfg(total_steps=20, warmup_steps=4),
                  batch_size=B, log_every=5, log_window=10, ckpt_every=10, ckpt_keep=2, ckpt_every_last=4)
    assert "data" in run_meta(cfg, K) and "workers" not in run_meta(cfg, K)
    g = infinite_batches(FakeLoader(3))
    assert all(isinstance(next(g), dict) for _ in range(8))                                   # restarts past 3
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "run")
        buf = io.StringIO()
        with redirect_stdout(buf):
            a = fit(cfg, None, out, device="cpu", stop_at=12, loader=FakeLoader(), num_classes=K)
        assert a["update"] == 12 and a["attempts"] == 12 and a["skipped"] == 0
        assert os.path.exists(os.path.join(out, "meta.json")) and json.load(open(os.path.join(out, "meta.json"))) == run_meta(cfg, K)
        files = sorted(os.listdir(out))
        assert "last.pt" in files and "update_000010.pt" in files and "update_000012.pt" in files          # 12: forced at a deliberate stop
        assert load_checkpoint(os.path.join(out, "last.pt"), Lova(K, SMALL)) == 12
        lines = [json.loads(l) for l in open(os.path.join(out, "train.jsonl"))]
        assert [l["update"] for l in lines] == [5, 10]
        # a fresh run into the same dir is refused; --resume without a checkpoint elsewhere is refused
        for kw in (dict(resume=None),):
            try:
                fit(cfg, None, out, device="cpu", loader=FakeLoader(), num_classes=K, **kw); raise AssertionError("must raise")
            except FileExistsError:
                pass
        try:
            fit(cfg, None, os.path.join(d, "empty"), device="cpu", resume=True, loader=FakeLoader(), num_classes=K); raise AssertionError("must raise")
        except FileNotFoundError:
            pass
        # a changed optimisation config cannot resume this run
        try:
            fit(replace(cfg, optim=OptimCfg(lr=5e-4)), None, out, device="cpu", resume=True, loader=FakeLoader(), num_classes=K); raise AssertionError("must raise")
        except ValueError:
            pass
        # run B: resume 12 -> 20
        with redirect_stdout(buf):
            b = fit(cfg, None, out, device="cpu", resume=True, loader=FakeLoader(), num_classes=K)
        assert b["update"] == 20 and b["attempts"] == 8
        model, opt_state = Lova(K, SMALL), torch.load(os.path.join(out, "last.pt"), weights_only=False)
        assert load_checkpoint(os.path.join(out, "last.pt"), model) == 20
        assert opt_state["scheduler"]["last_epoch"] == 20
        assert abs(opt_state["optimizer"]["param_groups"][0]["lr"] - 1e-3 * lr_factor(20, cfg.sched)) < 1e-12   # lr the NEXT update would use
        lines = [json.loads(l) for l in open(os.path.join(out, "train.jsonl"))]
        assert [l["update"] for l in lines] == [5, 10, 15, 20], [l["update"] for l in lines]              # appended, no duplicates
        numbered = sorted(n for n in os.listdir(out) if n.startswith("update_"))
        assert numbered == ["update_000012.pt", "update_000020.pt"], numbered                              # keep 2 (10 pruned)
        assert "resumed from" in buf.getvalue() and "done: 8 updates" in buf.getvalue()
        # resume from an explicit path INTO A NEW directory, with the target already reached: nothing to do, meta.json appears
        out2 = os.path.join(d, "run2")
        with redirect_stdout(io.StringIO()):
            c = fit(cfg, None, out2, device="cpu", resume=os.path.join(out, "update_000020.pt"), loader=FakeLoader(), num_classes=K)
        assert c["update"] == 20 and c["attempts"] == 0 and os.path.exists(os.path.join(out2, "meta.json"))
        # explicit-path resume: allowed into an existing run dir only from its own last.pt; an older numbered file is refused
        with redirect_stdout(io.StringIO()):
            same = fit(cfg, None, out, device="cpu", resume=os.path.join(out, "last.pt"), loader=FakeLoader(), num_classes=K)
        assert same["update"] == 20 and same["attempts"] == 0
        try:
            fit(cfg, None, out, device="cpu", resume=os.path.join(out, "update_000012.pt"), loader=FakeLoader(), num_classes=K); raise AssertionError("must raise")
        except FileExistsError:
            pass
        # fresh-run protection triggers on any artifact, not only last.pt
        out3 = os.path.join(d, "run3"); os.makedirs(out3); open(os.path.join(out3, "train.jsonl"), "w").close()
        try:
            fit(cfg, None, out3, device="cpu", loader=FakeLoader(), num_classes=K); raise AssertionError("must raise")
        except FileExistsError:
            pass
        # the GPT counter-example: numbered cadence hits the final update but last.pt's does not -> last.pt must still be final
        cfg2 = replace(cfg, ckpt_every=10, ckpt_every_last=7, ckpt_keep=3)
        out4 = os.path.join(d, "run4")
        with redirect_stdout(io.StringIO()):
            fit(cfg2, None, out4, device="cpu", stop_at=10, loader=FakeLoader(), num_classes=K)
        assert load_checkpoint(os.path.join(out4, "last.pt"), Lova(K, SMALL)) == 10
        assert load_checkpoint(os.path.join(out4, "update_000010.pt"), Lova(K, SMALL)) == 10
        # seed: two fresh runs with the same seed give identical weights (torch AND python random are seeded)
        outs = []
        for name in ("s1", "s2"):
            o = os.path.join(d, name)
            with redirect_stdout(io.StringIO()):
                fit(cfg, None, o, device="cpu", stop_at=3, loader=FakeLoader(), num_classes=K)
            m = Lova(K, SMALL); load_checkpoint(os.path.join(o, "last.pt"), m); outs.append(m)
        assert all(torch.equal(p, q) for (_, p), (_, q) in zip(outs[0].state_dict().items(), outs[1].state_dict().items()))
    print("loop unit test OK   (12 + 8 == 20 updates; resume rules; JSONL continuity; retention)")


def real(a):
    """train.py end to end on real data: run A stops at 12 updates, run B resumes to 20."""
    py = sys.executable
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "smoke")
        for args in (["--stop-at", "12"], ["--resume"]):
            r = subprocess.run([py, "train.py", "--config", "configs/smoke.py", "--root", a.root, "--out", out] + args, capture_output=True, text=True)
            print("\n".join(l for l in r.stdout.splitlines() if not l.startswith(("loading", "Done", "creating", "index"))))
            assert r.returncode == 0, r.stderr[-2000:]
        lines = [json.loads(l) for l in open(os.path.join(out, "train.jsonl"))]
        assert [l["update"] for l in lines] == [5, 10, 15, 20]
        assert load_checkpoint(os.path.join(out, "last.pt"), Lova(80)) == 20
        print("train.py smoke OK: files", sorted(os.listdir(out)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=None)
    a = p.parse_args()
    unit_test()
    if a.root:
        real(a)
