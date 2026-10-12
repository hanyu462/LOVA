"""lova.engine.validate: determinism, sweep structure, weighting; real: a fixed val set with fixed pointers.

    python tests/engine/test_validate.py
    python tests/engine/test_validate.py --root datasets/coco --n 32      # build a fixed set twice, check pointers, run a fresh model
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lova.engine.validate import OVERRIDES, ValCfg, ValSet, format_val, val_loader, validate  # noqa: E402
from lova.losses.total_loss import TotalLossCfg  # noqa: E402
from lova.models.lova import Lova  # noqa: E402
from tests.models.test_lova import SMALL, fake_batch  # noqa: E402

K, S = 5, 64


class Batches:
    def __init__(self, sizes, seed=0):
        g = torch.random.get_rng_state(); torch.manual_seed(seed)
        self.items = [fake_batch(b, K, S) for b in sizes]
        torch.random.set_rng_state(g)
    def __iter__(self):
        return iter(self.items)


def unit_test():
    torch.manual_seed(0)
    model = Lova(K, SMALL)
    loader = Batches([2, 2, 1])
    a = validate(model, loader, amp_dtype=None)
    b = validate(model, loader, amp_dtype=None)
    assert a == b, "validation must be deterministic"                                       # same model, same batches
    assert a["n_images"] == 5 and set(a["sweep"]) == {"predicted", "R=0", "R=0.5", "R=1"}
    assert set(a["predicted"]) == {"r_loss", "r_mae", "total", "heat_loss", "mask_loss", "dice", "budget"}
    assert all(set(s) == {"heat_loss", "mask_loss", "dice", "budget"} for s in a["sweep"].values())     # no r terms in the sweep
    # budgets: constants on a half-padded canvas (fake_batch pads the bottom half) are exact
    assert abs(a["sweep"]["R=0"]["budget"]) < 1e-6 and abs(a["sweep"]["R=0.5"]["budget"] - 0.5) < 1e-6 and abs(a["sweep"]["R=1"]["budget"] - 1.0) < 1e-6
    # a fresh model (zero-init blocks): R cannot matter, all conditions equal; after randomising the blocks they differ
    assert all(abs(a["sweep"]["R=0"][k] - a["sweep"]["R=1"][k]) < 1e-6 for k in ("heat_loss", "mask_loss"))
    for p in model.backbone.parameters():
        if p.dim() == 4 and p.abs().sum() == 0: torch.nn.init.normal_(p, std=0.05)
    c = validate(model, loader, amp_dtype=None)
    assert abs(c["sweep"]["R=0"]["mask_loss"] - c["sweep"]["R=1"]["mask_loss"]) > 1e-6
    # weighting: the mean is per image, so [2, 2, 1] batches equal the same five images as [5]
    loader5 = Batches([5]); loader5.items[0] = {k: (torch.cat([bt[k] for bt in loader.items]) if torch.is_tensor(v) else sum((bt[k] for bt in loader.items), []))
                                                for k, v in loader.items[0].items()}
    d = validate(model, loader5, amp_dtype=None)
    assert abs(d["predicted"]["heat_loss"] - c["predicted"]["heat_loss"]) < 2e-2 * max(1.0, abs(c["predicted"]["heat_loss"]))   # focal normalises per batch: close, not equal
    assert abs(d["sweep"]["R=1"]["budget"] - c["sweep"]["R=1"]["budget"]) < 1e-6
    # total is the weighted sum of the predicted-R terms; model is left in its previous mode
    w = TotalLossCfg()
    p = c["predicted"]
    assert abs(p["total"] - (w.w_r * p["r_loss"] + w.w_heat * p["heat_loss"] + w.w_mask * p["mask_loss"])) < 1e-4
    model.train(); validate(model, loader, amp_dtype=None); assert model.training
    model.eval(); validate(model, loader, amp_dtype=None); assert not model.training
    # mode is restored even when validation fails half-way
    class Broken:
        def __iter__(self):
            yield loader.items[0]; raise RuntimeError("boom")
    model.train()
    try:
        validate(model, Broken(), amp_dtype=None); raise AssertionError("must raise")
    except RuntimeError:
        pass
    assert model.training
    # validate() itself does not touch the global RNG (the loader's RNG is covered by the real test)
    torch.manual_seed(123); before = torch.get_rng_state().clone()
    validate(model, loader, amp_dtype=None)
    assert torch.equal(before, torch.get_rng_state())
    assert format_val(c, 1234).startswith("[val @ 001234] 5 images")
    print("validate unit test OK")


def real(a):
    import time
    cfg = ValCfg(n_images=a.n, batch_size=8, workers=4)
    t0 = time.perf_counter()
    vs1 = ValSet(a.root, a.split, __import__("lova.data.common.build", fromlist=["PipelineCfg"]).PipelineCfg(), cfg.n_images, cfg.seed)
    print(f"fixed set: {len(vs1)} images built in {time.perf_counter() - t0:.1f}s, ids {vs1.image_ids[:6]}...")
    vs2 = ValSet(a.root, a.split, __import__("lova.data.common.build", fromlist=["PipelineCfg"]).PipelineCfg(), cfg.n_images, cfg.seed)
    assert vs1.indices == vs2.indices
    for k in (0, len(vs1) // 2, len(vs1) - 1):                                        # same image AND same pointer, twice
        s1, s2 = vs1[k], vs2[k]
        assert s1.image_id == s2.image_id and torch.equal(s1.pointer, s2.pointer) and s1.pointed_idx == s2.pointed_idx
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Lova(vs1.num_classes).to(dev)
    # the whole validation (loader creation, worker seeding, iteration, forward) leaves the global torch RNG untouched
    torch.manual_seed(123); before = torch.get_rng_state().clone()
    loader = val_loader(vs1, cfg)
    assert loader.generator is not None
    t0 = time.perf_counter()
    res = validate(model, loader)
    assert torch.equal(before, torch.get_rng_state()), "validation consumed the global RNG"
    assert validate(model, loader) == res                                              # and is repeatable
    print(format_val(res, 0))
    print(f"  {time.perf_counter() - t0:.1f}s for {len(vs1)} images x {1 + len(OVERRIDES)} conditions on {dev}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=None); p.add_argument("--split", default="val2017"); p.add_argument("--n", type=int, default=32)
    a = p.parse_args()
    unit_test()
    if a.root:
        real(a)
