# LOVA — Look Only at Valuable Areas

Pointer-conditioned Resolution Field R + R-gated backbone for instance segmentation.
Rebuilt from scratch, one verified step at a time. Previous iteration (V0, working code) is kept in `backup/v0/`.

## Layout

```
lova/
  data/
    common/  dataset-agnostic pipeline: sample.py (the Sample contract) then, in execution order,
             select → transform → pointer → targets → R*; pure functions on tensors, tested in tests/
    coco/    COCO reader (files + pycocotools -> Sample). Other readers (synthetic, custom) go beside it
  models/    stem, R predictor, R-gated backbone, neck, head
  losses/    task losses (focal, dice), R losses (profile, budget)
  engine/    training / evaluation loops (phases, DDP, logging); no model or data logic here
configs/     experiment settings (one file per run type), so commands stay short and reproducible
scripts/     offline, run-once tools: dataset download, checkpoint conversion, stats
tests/       one script per module that only CALLS it: unit test + visualisation (PNG) where applicable
docs/        design notes and decisions, updated when a step is verified
datasets/    data (gitignored)           runs/  checkpoints + logs (gitignored)
backup/v0/   previous full implementation, reference only (not imported)
```

## Rules

- A module is added only after its test / visualisation has been looked at.
- Tests import the module; they never re-implement its logic.
- `lova/data` modules take tensors, not files; only `load.py` touches COCO / image files.
