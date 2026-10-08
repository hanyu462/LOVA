"""CPU smoke test: shapes, gradient paths, all three phases, postprocess, eval.

python tests/smoke_test.py
"""
import os
import subprocess
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lova.data.common import collate  # noqa: E402
from lova.data.synthetic import SyntheticPointerDataset  # noqa: E402
from lova.losses import instance_loss, r_losses  # noqa: E402
from lova.models.head import postprocess  # noqa: E402
from lova.models.model import LOVAv0  # noqa: E402


def test_shapes_and_grads():
    torch.manual_seed(0)
    ds = SyntheticPointerDataset(8, img_size=128)
    batch = collate([ds[i] for i in range(4)])
    model = LOVAv0(ds.num_classes)
    out = model(batch["image"], batch["pointer"])
    B, S = 4, 128
    assert out["r"].shape == (B, 1, S // 4, S // 4)
    assert out["heat_logits"].shape == (B, ds.num_classes, S // 8, S // 8)
    assert out["kernels"].shape == (B, 64, S // 8, S // 8)
    assert out["mask_feat"].shape == (B, 64, S // 4, S // 4)
    feats = model.backbone(model.stem(batch["image"]), out["r"])
    assert [tuple(f.shape[1:]) for f in feats.values()] == [(64, 32, 32), (128, 16, 16), (256, 8, 8), (384, 4, 4)]

    # task loss must reach the R predictor through the gates (no detach anywhere)
    loss = sum(instance_loss(out, batch).values())
    loss.backward()
    g = model.rpred.out.weight.grad
    assert g is not None and g.abs().sum() > 0, "task gradient does not reach R predictor"

    # R must actually change the features: R=0 vs R=1 differ, R=0 is identity through gated blocks
    f0 = model.stem(batch["image"]).detach()
    z = model.backbone(f0, torch.zeros_like(out["r"]))
    o = model.backbone(f0, torch.ones_like(out["r"]))
    assert (z["s32"] - o["s32"]).abs().mean() > 1e-4
    rl = r_losses(out["r"], batch["pointed_mask4"], batch["valid4"])
    assert all(torch.isfinite(v) for v in rl.values())
    res = postprocess({k: v.detach() for k, v in out.items() if k in ("heat_logits", "kernels", "mask_feat")})
    assert len(res) == B
    print("shapes/grad OK; vcompute(R=pred) =", model.backbone.virtual_compute(out["r"].detach()).tolist())

    # binary gate: shared tau, hard mode is exactly 1[R > 0.5], soft_execution differentiable
    mb = LOVAv0(ds.num_classes, gate_mode="binary")
    assert all(blk.tau == 0.5 for st in mb.backbone.stages for blk in st.blocks)
    r = torch.rand(2, 1, 32, 32)
    mb.backbone.gate_mode = "hard"
    g = mb.backbone.stages[0].blocks[0].gate(r, "hard", 0.1)
    assert torch.equal(g, (r > 0.5).float())
    mb.backbone.gate_mode = "binary"
    rr = r.clone().requires_grad_(True)
    mb.backbone.soft_execution(rr).sum().backward()
    assert rr.grad is not None and rr.grad.abs().sum() > 0
    from lova.rsampler import sample_r
    rb = sample_r(batch["pointed_mask4"], kind="binary")
    assert set(rb.unique().tolist()) <= {0.0, 1.0}, "binary sampler must produce {0,1}"

    # R* profile (geo): 1 at the pointer, decaying inside, carried over the boundary, ~0 far away
    from lova.data.rtarget import downsample, r_profile
    ds_s = SyntheticPointerDataset(8, img_size=128, pointer_mode="interior", mask_stride=2)
    bs = collate([ds_s[i] for i in range(4)])
    assert bs["pointed_mask_s"].shape == (4, 1, 64, 64)
    rs = downsample(r_profile(bs["pointed_mask_s"], bs["pointer"], "geo", 1.0, 2.0, 6.0, mask_stride=2), 2)
    m = bs["pointed_mask4"] > 0.5
    assert rs.shape == bs["pointed_mask4"].shape and rs.min() >= 0 and rs.max() <= 1
    assert rs[m].max() > 0.9 and rs[~m].min() < 0.05
    for i in range(4):  # pointer cell is (near) the maximum
        px, py = (bs["pointer"][i] / 4).long().clamp(0, 31).tolist()
        assert rs[i, 0, py, px] > 0.8, rs[i, 0, py, px]
    rd = r_profile(bs["pointed_mask4"], bs["pointer"], "dt", 1.0, 0.7, 12.0, r_b=0.7, mask_stride=4)
    assert rd[m].min() >= 0.7 - 1e-4 and rd[~m].max() <= 0.7 + 1e-4
    # multi-pointer items share image/targets and differ in pointer
    ds2 = SyntheticPointerDataset(8, img_size=128, pointer_mode="interior", pointers_per_image=2)
    item = ds2[0]
    assert isinstance(item, list) and len(item) == 2 and torch.equal(item[0]["image"], item[1]["image"])
    b2 = collate([ds2[0], ds2[1]])
    assert b2["image"].shape[0] == 4


def test_phases_and_eval():
    with tempfile.TemporaryDirectory() as d:
        common = [sys.executable, os.path.join(ROOT, "train.py"), "--data", "synthetic", "--img-size", "128",
                  "--bs", "4", "--workers", "0", "--max-iters", "3", "--log-every", "1", "--warmup", "2",
                  "--synthetic-len", "16", "--device", "cpu"]
        run = lambda *args: subprocess.run(common + list(args), check=True, cwd=ROOT)
        run("--phase", "A", "--out", f"{d}/A")
        run("--phase", "A", "--r-fixed", "1.0", "--out", f"{d}/A1")
        run("--phase", "B", "--init", f"{d}/A/last.pth", "--out", f"{d}/B")
        run("--phase", "B", "--w-task", "0.5", "--init", f"{d}/A/last.pth", "--out", f"{d}/B2")
        run("--phase", "C", "--init", f"{d}/B/last.pth", "--out", f"{d}/C")
        # binary-execution variant: {0,1} sampler, shared tau, budget on mean(g), T annealing in C
        run("--phase", "A", "--gate-mode", "binary", "--out", f"{d}/Ab")
        run("--phase", "C", "--gate-mode", "binary", "--gate-temp-final", "0.05", "--init", f"{d}/Ab/last.pth",
            "--out", f"{d}/Cb")
        # R* profile target + interior pointers + 2 pointers per image (phase B and C)
        run("--phase", "B", "--gate-mode", "binary", "--r-target", "profile", "--pointer-mode", "interior",
            "--pointers-per-image", "2", "--init", f"{d}/Ab/last.pth", "--out", f"{d}/Bp")
        run("--phase", "C", "--gate-mode", "binary", "--r-target", "profile", "--pointer-mode", "interior",
            "--pointers-per-image", "2", "--init", f"{d}/Bp/last.pth", "--out", f"{d}/Cp")
        ev = [sys.executable, os.path.join(ROOT, "eval_pointer.py"), "--ckpt", f"{d}/C/last.pth", "--data",
              "synthetic", "--max-images", "6", "--bs", "4", "--device", "cpu", "--viz", "2"]
        subprocess.run(ev + ["--out", f"{d}/ev"], check=True, cwd=ROOT)
        subprocess.run(ev + ["--out", f"{d}/ev_sweep", "--sweep", "3", "--r-mode", "oracle"], check=True, cwd=ROOT)
        subprocess.run(ev[:2] + ["--ckpt", f"{d}/Cb/last.pth"] + ev[4:] + ["--out", f"{d}/ev_hard", "--gate-mode", "hard"],
                       check=True, cwd=ROOT)
        assert os.path.exists(f"{d}/ev/viz_0000.png")
    print("phases + eval OK")


if __name__ == "__main__":
    test_shapes_and_grads()
    test_phases_and_eval()
