# LOVA — Look Only at Valuable Areas

Pointer-conditioned Resolution Field (R) + R-conditioned backbone for COCO instance segmentation.
V0 goal: verify that R controls perception quality. Design: [docs/DESIGN_V0.md](docs/DESIGN_V0.md).

## Requirements

`torch`, `numpy`, `pillow`, `pycocotools` (COCO only).

## Quick check (CPU, synthetic data)

```bash
python tests/smoke_test.py
```

## COCO

Download: `scripts/download_coco.sh <dir> [all|val]` (all = train+val ≈ 19GB).
Local (this Jetson): val2017 + annotations at `~/datasets/coco`. Use `--train-split val2017` for local pipeline tests only.

```
$COCO/annotations/instances_{train,val}2017.json
$COCO/{train,val}2017/*.jpg
```

```bash
torchrun --nproc_per_node 8 train.py --phase A --r-fixed 1.0 --coco-root $COCO --epochs 24 --amp --out runs/A0
torchrun --nproc_per_node 8 train.py --phase A                --coco-root $COCO --epochs 24 --amp --out runs/A
torchrun --nproc_per_node 8 train.py --phase B --init runs/A/last.pth --coco-root $COCO --epochs 3 --amp --out runs/B
torchrun --nproc_per_node 8 train.py --phase C --init runs/B/last.pth --coco-root $COCO --epochs 8 --amp --out runs/C

python eval_pointer.py --ckpt runs/A/last.pth --coco-root $COCO --r-mode const:0.5 --out eval/A_r05
python eval_pointer.py --ckpt runs/C/last.pth --coco-root $COCO --viz 20 --out eval/C_pred
python eval_pointer.py --ckpt runs/C/last.pth --coco-root $COCO --sweep 4 --max-images 1000 --out eval/C_sweep
```

## Binary-execution variant (V0.1, see docs §8b)

R stays a continuous importance field; only the execution is binary (`G = 1[R > 0.5]`, sigmoid-relaxed in training).

```bash
torchrun --nproc_per_node 4 train.py --phase A --gate-mode binary --coco-root $COCO --epochs 24 --bs 32 --amp --out runs/Ab
python eval_pointer.py --ckpt runs/Ab/last.pth --coco-root $COCO --r-mode zeros --out eval/Ab_r0
python eval_pointer.py --ckpt runs/Ab/last.pth --coco-root $COCO --r-mode ones --gate-mode hard --out eval/Ab_r1_hard
```

`--gate-mode binary` implies `--r-sampler binary` and `--budget-on gate`; `--gate-temp-final 0.05` anneals T in phase C.

## R pseudo-GT profile (phase B/C)

Instead of `L_R_in` (lower bound) + weak `L_R_out`, regress R onto a profile built on the fly from the pointed mask:
`R_in = exp(-(d_geodesic/sigma_in)^gamma)` from the pointer inside, boundary value carried outward with
`exp(-d/sigma_out)`, 0 far away. See `lova/data/rtarget.py`.

```bash
python scripts/viz_rtarget.py --coco-root $COCO --n 6 --out viz_rtarget       # look at R* first
python scripts/viz_rtarget.py --coco-root $COCO --image-id 139 --instance 0 --pointer 300,200 --r-sigma-in 0.8 --out viz_rtarget/id139
torchrun --nproc_per_node 4 train.py --phase B --gate-mode binary --r-target profile --pointer-mode interior --pointers-per-image 2 --bs 16 --init runs/Ab/last.pth --coco-root $COCO --epochs 3 --amp --out runs/Bp
torchrun --nproc_per_node 4 train.py --phase C --gate-mode binary --r-target profile --pointer-mode interior --pointers-per-image 2 --bs 16 --gate-temp-final 0.05 --init runs/Bp/last.pth --coco-root $COCO --epochs 8 --amp --out runs/Cp
```

`--pointers-per-image K` puts K pointers (different instances) of the same image in a batch (samples = bs * K).
The profile weight decays in phase C like the legacy R-sup; the budget loss stays.

## Inference with your own pointer

```bash
# one image, pointer in original pixels -> [image | R | masks] PNG + detection list + latency
python infer.py --ckpt runs/C/last.pth --image foo.jpg --pointer 230,180 --out out.png
python infer.py --ckpt runs/C/last.pth --image foo.jpg --pointer 230,180 --r-mode ones   # full-compute control

# browser click demo (no extra deps). On a remote server: ssh -L 7860:localhost:7860 ... first
python demo.py --ckpt runs/C/last.pth --images-dir $COCO/val2017
```

Click anywhere on the image: that point becomes the pointer and the R field / masks update.
V0 gates by masking, so latency is the same for every R.
