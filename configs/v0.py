"""LOVA V0 baseline run: 20k successful updates, warm-up 1k, B=8, bf16, clip 100, loss weights (1, 1, 3), lr 2e-4."""
from lova.engine.loop import RunCfg

CFG = RunCfg()          # every default is the V0 decision documented in its module
