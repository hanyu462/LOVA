"""LOVA V0 training.

Phases (run in order, each initialized from the previous checkpoint):

  A  backbone + neck + head learn instance seg under SAMPLED R fields
     (R predictor unused). --r-fixed 1.0 gives the plain full-compute baseline.
  B  only the R predictor trains: L_R_in + L_R_out + L_budget
     (optionally + task loss through the frozen network, --w-task).
  C  everything joint with R = R_theta(I,p): task + decaying R-sup + budget.

Single GPU:  python train.py --phase A --data coco --coco-root /data/coco --out runs/A
Multi GPU:   torchrun --nproc_per_node 8 train.py ...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from lova.data.common import collate
from lova.data.rtarget import lam_cells, r_profile
from lova.losses import instance_loss, r_losses, r_profile_loss, r_stats
from lova.models.model import LOVAv0
from lova.rsampler import sample_r


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["A", "B", "C"], required=True)
    p.add_argument("--data", choices=["coco", "synthetic"], default="coco")
    p.add_argument("--coco-root", default="data/coco")
    p.add_argument("--train-split", default="train2017", help="COCO split used for training")
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--out", default="runs/exp")
    p.add_argument("--init", default=None, help="checkpoint to initialize model weights from")
    p.add_argument("--resume", default=None, help="checkpoint to resume model+optimizer+step from")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--max-iters", type=int, default=0, help="override epochs if > 0")
    p.add_argument("--bs", type=int, default=16, help="per-GPU batch size")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--clip", type=float, default=10.0)
    p.add_argument("--amp", action="store_true", help="bf16 autocast (CUDA)")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # model
    p.add_argument("--gate-mode", choices=["depth", "binary", "linear", "none"], default="depth",
                   help="depth = continuous block-depth gating (original V0); binary = one shared threshold "
                        "(sigmoid relaxation of G = 1[R>0.5]); V1 hard routing uses the same thresholds")
    p.add_argument("--gate-temp", type=float, default=0.1)
    p.add_argument("--gate-temp-final", type=float, default=None,
                   help="phase C: anneal gate temperature linearly from --gate-temp to this value "
                        "(sharper gate -> smaller gap to V1 hard routing)")
    p.add_argument("--r-sampler", choices=["continuous", "binary"], default=None,
                   help="phase A R-field sampler. default: binary if --gate-mode binary else continuous")
    p.add_argument("--budget-on", choices=["r", "gate"], default=None,
                   help="budget loss on mean(R) or on mean(g(R)) (soft execution map). "
                        "default: gate if --gate-mode binary else r")
    p.add_argument("--transition", choices=["conv", "light"], default="conv",
                   help="stage transition (always-on base path). light = avgpool + 1x1")
    # phase A
    p.add_argument("--r-fixed", type=float, default=None, help="phase A: constant R instead of sampled fields")
    # loss weights
    p.add_argument("--w-heat", type=float, default=1.0)
    p.add_argument("--w-mask", type=float, default=3.0)
    p.add_argument("--w-task", type=float, default=None, help="multiplier on task loss (default A/C=1, B=0)")
    p.add_argument("--w-r-in", type=float, default=1.0)
    p.add_argument("--w-r-out", type=float, default=0.1)
    p.add_argument("--w-budget", type=float, default=1.0)
    p.add_argument("--budget-extra", type=float, default=0.15)
    p.add_argument("--r-sup-floor", type=float, default=0.1,
                   help="phase C: R-sup weights decay linearly to this fraction")
    # R pseudo-GT profile (phase B/C): R* = 1 -> r_b inside the pointed mask, r_b * exp(-d/lambda) outside
    p.add_argument("--r-target", choices=["legacy", "profile"], default="legacy",
                   help="legacy = L_R_in (lower bound) + weak L_R_out; profile = regress R onto R* "
                        "(weight --w-r-in, decays in phase C like legacy). Budget is kept in both")
    p.add_argument("--r-b", type=float, default=0.7, help="profile: R* on the object boundary")
    p.add_argument("--r-gamma", type=float, default=0.7, help="profile: interior ramp exponent")
    p.add_argument("--r-lam-frac", type=float, default=0.05, help="profile: lambda = frac * img_size (px)")
    p.add_argument("--r-profile-mode", choices=["dt", "euclid"], default="dt")
    p.add_argument("--r-loss", choices=["bce", "mse"], default="bce")
    # pointer sampling
    p.add_argument("--pointer-mode", choices=["uniform", "interior"], default="uniform",
                   help="interior = never right at the mask boundary (depth >= 25%% of max)")
    p.add_argument("--pointers-per-image", type=int, default=1,
                   help="K pointers (on different instances) per image in a batch; samples = bs * K")
    # synthetic
    p.add_argument("--synthetic-len", type=int, default=2000)
    a = p.parse_args(argv)
    if a.lr is None:
        a.lr = {"A": 1e-3, "B": 1e-3, "C": 2e-4}[a.phase]
    if a.w_task is None:
        a.w_task = 0.0 if a.phase == "B" else 1.0
    if a.r_sampler is None:
        a.r_sampler = "binary" if a.gate_mode == "binary" else "continuous"
    if a.budget_on is None:
        a.budget_on = "gate" if a.gate_mode == "binary" else "r"
    return a


def build_dataset(a, train=True):
    if a.data == "synthetic":
        from lova.data.synthetic import SyntheticPointerDataset
        return SyntheticPointerDataset(a.synthetic_len if train else 200, a.img_size, seed=0 if train else 1,
                                       pointer_mode=a.pointer_mode, pointers_per_image=a.pointers_per_image)
    from lova.data.coco import COCOPointerDataset
    return COCOPointerDataset(a.coco_root, a.train_split if train else "val2017", a.img_size, train=train,
                              pointer_mode=a.pointer_mode, pointers_per_image=a.pointers_per_image)


def to_device(batch, dev):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(dev, non_blocking=True)
        elif isinstance(v, list) and v and torch.is_tensor(v[0]):
            out[k] = [x.to(dev, non_blocking=True) for x in v]
        else:
            out[k] = v
    return out


def set_trainable(model, phase):
    for n, prm in model.named_parameters():
        if phase == "A":
            prm.requires_grad = not n.startswith("rpred.")
        elif phase == "B":
            prm.requires_grad = n.startswith("rpred.")
        else:
            prm.requires_grad = True


def compute_loss(model, batch, a, progress):
    """progress in [0,1] (used for R-sup decay in phase C)."""
    m = model.module if hasattr(model, "module") else model
    losses, logs = {}, {}
    if a.phase == "C" and a.gate_temp_final is not None:
        m.backbone.gate_temp = a.gate_temp + (a.gate_temp_final - a.gate_temp) * progress
    if a.phase == "A":
        if a.r_fixed is not None:
            r = torch.full_like(batch["pointed_mask4"], a.r_fixed)
        else:
            r = sample_r(batch["pointed_mask4"], kind=a.r_sampler)
        out = model(batch["image"], batch["pointer"], r_override=r)
    elif a.phase == "B" and a.w_task == 0:
        r = model(batch["image"], batch["pointer"], r_only=True)["r"]
        out = None
    else:
        out = model(batch["image"], batch["pointer"])
        r = out["r"]

    if out is not None and a.w_task > 0:
        for k, v in instance_loss(out, batch, a.w_heat, a.w_mask).items():
            losses[k] = a.w_task * v

    if a.phase in ("B", "C"):
        decay = 1.0 if a.phase == "B" else 1.0 - (1.0 - a.r_sup_floor) * progress
        exec_map = None
        if a.budget_on == "gate":
            tau = 0.5 if a.gate_mode == "binary" else m.backbone.stages[0].blocks[0].tau
            exec_map = torch.sigmoid((r.float() - tau) / m.backbone.gate_temp)
        if a.r_target == "profile":
            with torch.no_grad():
                r_star = r_profile(batch["pointed_mask4"], batch["pointer"], a.r_b, a.r_gamma,
                                   lam_cells(a.img_size, a.r_lam_frac), a.r_profile_mode)
            rl = r_losses(r, batch["pointed_mask4"], batch["valid4"], 0.0, 0.0, a.w_budget, a.budget_extra,
                          exec_map=exec_map)
            losses["r_budget"] = rl["r_budget"]
            losses["r_prof"] = a.w_r_in * decay * r_profile_loss(r, r_star, batch["valid4"], a.r_loss)
            with torch.no_grad():
                logs["R_err"] = (((r.float() - r_star).abs() * batch["valid4"]).sum() / batch["valid4"].sum()).item()
        else:
            losses.update(r_losses(r, batch["pointed_mask4"], batch["valid4"],
                                   a.w_r_in * decay, a.w_r_out * decay, a.w_budget, a.budget_extra, exec_map=exec_map))
    logs.update(r_stats(r.detach(), batch["pointed_mask4"], batch["valid4"]))
    with torch.no_grad():
        logs["g_mean"] = m.backbone.soft_execution(r.detach().float()).mean().item()
    logs["vcompute"] = m.backbone.virtual_compute(r.detach(), batch["valid4"]).mean().item()
    if a.phase == "C" and a.gate_temp_final is not None:
        logs["T"] = m.backbone.gate_temp
    return losses, logs


def main(argv=None):
    a = get_args(argv)
    ddp = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    rank = 0
    if ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        a.device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
    dev = torch.device(a.device)
    is_main = rank == 0
    if is_main:
        os.makedirs(a.out, exist_ok=True)
        json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=2)

    ds = build_dataset(a, train=True)
    sampler = DistributedSampler(ds, shuffle=True) if ddp else None
    dl = DataLoader(ds, batch_size=a.bs, shuffle=sampler is None, sampler=sampler, num_workers=a.workers,
                    collate_fn=collate, pin_memory=dev.type == "cuda", drop_last=True,
                    persistent_workers=a.workers > 0)

    model = LOVAv0(ds.num_classes, gate_mode=a.gate_mode, gate_temp=a.gate_temp,
                    transition=a.transition)
    start_step = 0
    ckpt = torch.load(a.resume or a.init, map_location="cpu") if (a.resume or a.init) else None
    if ckpt is not None:
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if is_main and (missing or unexpected):
            print("load_state_dict missing:", missing, "unexpected:", unexpected)
    model.to(dev)
    set_trainable(model, a.phase)
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.ndim > 1]
    no_decay = [p for p in params if p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0}],
                            lr=a.lr)
    if a.resume:
        opt.load_state_dict(ckpt["opt"])
        start_step = ckpt["step"]
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev.index],
                                                          find_unused_parameters=a.phase != "C")

    total = a.max_iters if a.max_iters > 0 else a.epochs * len(dl)
    lr_at = lambda s: a.lr * min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s / total, 1.0)))

    step, epoch, t0 = start_step, 0, time.time()
    model.train()
    while step < total:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in dl:
            if step >= total:
                break
            batch = to_device(batch, dev)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp and dev.type == "cuda"):
                losses, logs = compute_loss(model, batch, a, step / total)
            loss = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(params, a.clip)
            opt.step()
            step += 1
            if is_main and (step % a.log_every == 0 or step == total):
                msg = {"step": step, "lr": f"{lr_at(step):.2e}", "loss": f"{loss.item():.4f}",
                       **{k: f"{v.item():.4f}" for k, v in losses.items()},
                       **{k: f"{v:.3f}" for k, v in logs.items()}, "gnorm": f"{gnorm.item():.2f}",
                       "s/it": f"{(time.time() - t0) / a.log_every:.3f}"}
                print(" ".join(f"{k}={v}" for k, v in msg.items()), flush=True)
                t0 = time.time()
            if is_main and (step % a.save_every == 0 or step == total):
                m = model.module if ddp else model
                torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "step": step,
                            "config": m.config, "args": vars(a)}, os.path.join(a.out, "last.pth"))
        epoch += 1
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
