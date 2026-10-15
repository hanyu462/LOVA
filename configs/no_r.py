"""No-R: the pure baseline. No R predictor, no pointer input, no r_loss (w_r = 0); the backbone is the plain residual
network F + Body(F) inside the image. Everything else (stem, backbone, neck, head, segmentation losses, data, schedule,
seed) is the V0 setting. The comparison that answers "does the R mechanism help at all?" is this run vs configs/v0.py.
Unlike configs/a0.py, nothing here touches the shared stem except the segmentation losses."""
from dataclasses import replace

from lova.engine.loop import RunCfg
from lova.losses.total_loss import TotalLossCfg
from lova.models.lova import LovaCfg

CFG = replace(RunCfg(), model=LovaCfg(use_r=False), loss=TotalLossCfg(w_r=0.0))
