# backup/v0_pilot — snapshot of the V0 pilot (2026-10-15, git tag v0-pilot)

Code as of the end of the V0 pilot: data pipeline, model (stem / R predictor / R-gated backbone / FPN / head),
losses (1, 1, 3), engine (train step, logger, checkpoint, loop, validation, decode, COCO AP evaluator), configs
(v0, a0, no_r, v0_detach, smoke), tests, entry scripts. Reference only; not imported.

runs/: meta + JSONL logs of the two 20k trainings (V0 joint, A0 dense). Checkpoints stay in ../../runs/ (gitignored).
ap.jsonl / ap_selected.jsonl: the final 500-image evaluations.

Headline (fixed 500-image pointer-able subset, 20k updates, single seed):
  V0 predicted R, budget proxy 0.334: AP 8.1, selected pointed IoU 0.597 (best-of-all 0.697), dice 0.724
  A0 dense (R = 1):                   AP 8.1, selected pointed IoU 0.580 (best-of-all 0.685), dice 0.730
  A0 at R = 0.334: AP 2.9; V0 at constant 0.334: AP 7.2; R_GT prior: pointed best 0.836 / global dice 0.552.
Open: pure No-R baseline (configs/no_r.py) not yet trained; detach and w_r = 0 ablations not run;
budget proxy is not FLOPs (dense masking); pointed-only segmentation target not implemented.
