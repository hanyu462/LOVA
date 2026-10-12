"""lova.engine.checkpoint: round trip, resume equivalence (5 + 5 == 10 updates, bit-identical), RNG restore,
scaler mismatch, meta mismatch, atomic save under a failing writer, numbered pruning.

    python tests/engine/test_checkpoint.py
"""
from __future__ import annotations

import io
import os
import random
import sys
import tempfile
from contextlib import redirect_stdout

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lova.engine.checkpoint import FORMAT_VERSION, CkptCfg, Checkpointer, latest, load_checkpoint, rng_state, save_checkpoint, set_rng_state, to_meta  # noqa: E402
from lova.engine.optim import OptimCfg, SchedCfg, build_optimizer, build_scheduler, lr_factor  # noqa: E402
from lova.engine.train import TrainCfg, make_scaler, train_step  # noqa: E402
from lova.losses.total_loss import TotalLossCfg  # noqa: E402
from lova.models.lova import Lova  # noqa: E402
from tests.models.test_lova import SMALL, fake_batch  # noqa: E402

K, S, B = 5, 64, 2
SCHED = SchedCfg(total_steps=40, warmup_steps=4)
TRAIN = TrainCfg(amp=False)                       # CPU, fp32: deterministic, so "bit-identical" is a fair test


def build(seed: int, cfg=TRAIN):
    torch.manual_seed(seed); random.seed(seed)
    model = Lova(K, SMALL)
    opt = build_optimizer(model, OptimCfg(lr=1e-3))
    return model, opt, build_scheduler(opt, SCHED), make_scaler(cfg, "cpu")


def run(model, opt, sch, scaler, n, cfg=TRAIN):
    for _ in range(n):
        train_step(model, fake_batch(B, K, S), opt, sch, TotalLossCfg(), cfg, scaler)   # fake_batch draws from torch's RNG


def same_params(a, b):
    return all(torch.equal(p, q) for (_, p), (_, q) in zip(a.state_dict().items(), b.state_dict().items()))


def unit_test():
    meta = to_meta(model=SMALL, loss=TotalLossCfg(), train=TRAIN, optim=OptimCfg(lr=1e-3), sched=SCHED, num_classes=K)
    assert isinstance(meta["model"]["backbone"]["widths"], list) and meta["num_classes"] == K   # plain JSON-like dict
    with tempfile.TemporaryDirectory() as d:
        # ---- 1. reference: 10 uninterrupted updates ----
        ref, opt_r, sch_r, _ = build(0)
        run(ref, opt_r, sch_r, None, 10)
        # ---- 2. 5 updates, save, fresh objects (different seed!), load, 5 more: must be bit-identical ----
        m1, o1, s1, _ = build(0)
        run(m1, o1, s1, None, 5)
        path = save_checkpoint(os.path.join(d, "a.pt"), m1, o1, s1, None, 5, meta)
        m2, o2, s2, _ = build(123)
        assert not same_params(m1, m2)
        upd = load_checkpoint(path, m2, o2, s2, None, meta)
        assert upd == 5 and same_params(m1, m2) and s2.last_epoch == 5
        assert abs(o2.param_groups[0]["lr"] - 1e-3 * lr_factor(5, SCHED)) < 1e-12          # the lr of update 5, as the schedule says
        run(m2, o2, s2, None, 5)
        assert same_params(ref, m2), "resumed run diverged from the uninterrupted one"
        assert s2.last_epoch == sch_r.last_epoch == 10
        # ---- 3. RNG restore: the next random draw after a load equals the one after the save ----
        torch.manual_seed(7); random.seed(7)
        st = rng_state(); a = torch.rand(3); pa = random.random()
        set_rng_state(st); b = torch.rand(3); pb = random.random()
        assert torch.equal(a, b) and pa == pb
        # ---- 4. model-only load (evaluation): optimizer / scheduler / RNG untouched ----
        m3, o3, s3, _ = build(5)
        torch.manual_seed(99); before = torch.get_rng_state()
        assert load_checkpoint(path, m3) == 5 and same_params(m1, m3) and s3.last_epoch == 0 and torch.equal(torch.get_rng_state(), before)
        # ---- 5. scaler mismatch both ways; fp16 scaler state round-trips ----
        cfg16 = TrainCfg(amp=True, amp_dtype="fp16")
        m4, o4, s4, sc4 = build(1, cfg16)
        assert sc4 is not None
        sc4.load_state_dict({"scale": 4096.0, "growth_factor": 2.0, "backoff_factor": 0.5, "growth_interval": 2000, "_growth_tracker": 0})
        assert sc4.get_scale() == 4096.0
        p16 = save_checkpoint(os.path.join(d, "fp16.pt"), m4, o4, s4, sc4, 3, meta)
        m5, o5, s5, sc5 = build(2, cfg16)
        assert load_checkpoint(p16, m5, o5, s5, sc5, meta) == 3 and sc5.get_scale() == 4096.0
        # mismatch both ways: ValueError, and NOTHING was modified (checks run before any load)
        m6, o6, s6, sc6 = build(3, cfg16)
        before = {"m": {k: v.clone() for k, v in m6.state_dict().items()}, "o": repr(o6.state_dict()), "s": s6.state_dict()["last_epoch"]}
        for args in ((p16, m6, o6, s6, None), (path, m6, o6, s6, sc6)):
            try:
                load_checkpoint(*args, meta=meta); raise AssertionError("must raise")
            except ValueError:
                pass
            assert all(torch.equal(v, before["m"][k]) for k, v in m6.state_dict().items()) and repr(o6.state_dict()) == before["o"] and s6.last_epoch == before["s"]
        # an fp16 checkpoint loads model-only for evaluation; half a resume (optimizer without scheduler) or a scaler without resume is refused
        m7 = Lova(K, SMALL)
        assert load_checkpoint(p16, m7, meta=meta) == 3 and same_params(m4, m7)
        for kw in (dict(optimizer=o6), dict(scheduler=s6), dict(scaler=sc6)):
            try:
                load_checkpoint(p16, m7, meta=meta, **kw); raise AssertionError("must raise")
            except ValueError:
                pass
        # ---- 6. meta mismatch is refused (strict) or warned (non-strict) ----
        other = to_meta(model=SMALL, loss=TotalLossCfg(w_mask=1.0), train=TRAIN, sched=SCHED, num_classes=K)
        try:
            load_checkpoint(path, m3, meta=other); raise AssertionError("must raise")
        except ValueError:
            pass
        buf = io.StringIO()
        with redirect_stdout(buf):
            assert load_checkpoint(path, m3, meta=other, strict_meta=False) == 5
        assert "warning" in buf.getvalue() and "loss" in buf.getvalue()
        # ---- 7. atomic save: a writer that fails leaves the previous file intact and no temp file behind ----
        import lova.engine.checkpoint as ckmod
        p_last = save_checkpoint(os.path.join(d, "last.pt"), m1, o1, s1, None, 5, meta)
        real_save = ckmod.torch.save
        def failing_save(obj, f, *a, **k):
            with open(f, "wb") as fh:
                fh.write(b"garbage")
            raise OSError("disk full")
        ckmod.torch.save = failing_save
        try:
            try:
                save_checkpoint(p_last, m2, o2, s2, None, 10, meta); raise AssertionError("must raise")
            except OSError:
                pass
        finally:
            ckmod.torch.save = real_save
        assert load_checkpoint(p_last, Lova(K, SMALL)) == 5 and not [n for n in os.listdir(d) if n.startswith(".tmp_")]
        # ---- 7b. format version is checked before anything else ----
        bad = torch.load(path, weights_only=False); bad["format_version"] = FORMAT_VERSION + 1
        torch.save(bad, os.path.join(d, "old.pt"))
        try:
            load_checkpoint(os.path.join(d, "old.pt"), Lova(K, SMALL)); raise AssertionError("must raise")
        except ValueError:
            pass
        # ---- 7c. GPU-mapped load: RNG state tensors come back on the GPU and must still restore ----
        if torch.cuda.is_available():
            m8, o8, s8, _ = build(4)
            m8.cuda()
            assert load_checkpoint(path, m8, o8, s8, None, meta, device="cuda") == 5
            torch.rand(2)                                                    # CPU generator usable after a cuda-mapped restore
        # ---- 8. Checkpointer: last.pt cadence, numbered files, pruning, latest() ----
        cdir = os.path.join(d, "run")
        ck = Checkpointer(CkptCfg(cdir, every=4, keep=2, every_last=2))
        written = {u: ck.step(u, m1, o1, s1, None, meta) for u in range(1, 13)}
        assert all(not written[u] for u in (1, 3, 5)) and len(written[2]) == 1 and len(written[4]) == 2
        assert [os.path.basename(p) for p in ck.numbered()] == ["update_000008.pt", "update_000012.pt"]      # keep 2
        assert latest(cdir) == os.path.join(cdir, "last.pt") and load_checkpoint(latest(cdir), Lova(K, SMALL)) == 12
        assert latest(os.path.join(d, "nowhere")) is None
        assert len(ck.step(13, m1, o1, s1, None, meta, force=True)) == 2
        for bad in (CkptCfg(cdir, every=0), CkptCfg(cdir, keep=0)):
            try:
                Checkpointer(bad); raise AssertionError("must raise")
            except ValueError:
                pass
    print("checkpoint unit test OK   (5 + 5 updates == 10 updates, bit-identical)")


if __name__ == "__main__":
    unit_test()
