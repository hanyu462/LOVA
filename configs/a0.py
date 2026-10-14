"""A0: the dense baseline. Identical to V0 (model, data, losses, schedule) except the backbone always runs on
R = 1 inside the image. The R predictor is still trained by r_loss (nothing else can reach it under an override),
so this run also yields an "R_GT-imitation" predictor for the detach ablation."""
from dataclasses import replace

from lova.engine.loop import RunCfg

CFG = replace(RunCfg(), train_r_override=1.0)
