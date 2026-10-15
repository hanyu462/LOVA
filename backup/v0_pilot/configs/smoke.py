"""End-to-end smoke: the V0 model and data, only the runtime shortened. 20 updates, warm-up 5, checkpoints every
10 (keep 2), last.pt every 5, a log line every 5, validation on 16 fixed images every 10."""
from dataclasses import replace

from lova.engine.loop import RunCfg
from lova.engine.optim import SchedCfg
from lova.engine.validate import ValCfg

CFG = replace(RunCfg(), sched=SchedCfg(total_steps=20, warmup_steps=5), split="val2017",
              log_every=5, log_window=10, ckpt_every=10, ckpt_keep=2, ckpt_every_last=5,
              val_every=10, val=ValCfg(n_images=16, batch_size=8, workers=2))
