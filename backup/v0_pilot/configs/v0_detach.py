"""Ablation: joint training but the segmentation losses cannot reach the R predictor (gate runs on r_pred.detach()).
The predictor then learns the pointer prior R_GT only; the backbone adapts to that field."""
from dataclasses import replace

from lova.engine.loop import RunCfg

CFG = replace(RunCfg(), detach_r=True)
