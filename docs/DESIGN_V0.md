# LOVA V0 — Look Only at Valuable Areas

## 0. 한 줄 요약

`(I, p) → R_θ(I,p) → R-conditioned backbone → instance segmentation (전체 이미지)`

V0의 유일한 검증 목표는 **"R이 실제 perception quality를 조절할 수 있는가"** 이다.
FLOPs 절감은 V1에서 다룬다. V0의 모든 R 조건화는 **masking(soft gating)** 이며 연산량은 줄지 않는다.

---

## 1. 문제 정의 검토 (확정된 정의)

| 항목 | 확정 내용 |
|---|---|
| Task | COCO instance segmentation (80 class), **이미지 전체**에서 검출 |
| Pointer 역할 | "여기만 감지"가 아니라 "여기에 계산 자원을 더 써라". R을 통해서만 네트워크에 들어감 |
| R의 의미 | 위치별 refinement 양 (perception ON/OFF 아님). Low R = 거칠고 confidence 낮은 인식 |
| 먼 영역 | **Graceful degradation**. 모든 GT instance를 학습에 사용, 먼 instance를 negative로 만들지 않음. confidence가 threshold 아래로 내려가 자연스럽게 사라지는 것은 허용 |
| Shortcut 금지 | head/neck은 R-conditioned backbone 출력만 본다. F0나 별도 backbone에서 head로 가는 경로 없음. head는 R과 pointer도 직접 보지 않는다 |
| Layer 철학 | Low/High branch 없음. 하나의 weight set `A(F; R)`가 위치별 R로 동작을 바꾼다 |
| 제외 | boundary head/loss, occlusion, sampling spacing, 실제 sparse execution |

### 구조적으로 확인해 둔 점

1. **head가 R을 보면 안 된다.** head가 R을 입력으로 받으면 "R이 낮으면 score를 낮춰라"를 직접 배울 수 있어서,
   feature 품질과 무관한 가짜 degradation이 생긴다. V0에서 confidence 저하는 반드시 **feature 품질 저하의 결과**여야 한다.
   또 모든 GT를 positive로 학습하므로 head에는 low-R instance의 score를 일부러 낮출 동기가 없다.
2. **R의 gradient 경로.** class/mask loss가 R predictor를 학습시키려면 routing이 미분 가능해야 한다.
   V0는 soft gate `g(R) = sigmoid((R−τ)/T)`를 쓰므로 task gradient가 R까지 내려간다(smoke test에서 확인).
   V1에서 hard routing으로 바꾸면 STE 등이 필요하다.
3. **compute 제약이 없으면 R → 1로 붕괴한다.** task loss만 보면 모든 곳을 refine하는 것이 항상 유리하다.
   그래서 V0에는 FLOPs 대신 **R budget loss**(이미지 평균 R의 상한)를 둔다. 이것이 R 분포를 "어디에 쓸지 선택하는 문제"로 만든다.
4. **R을 mask copy로 만들지 않기.** 관심 instance에 대해 R을 **하한(R_in → 1)**으로 강하게 지도하고, 바깥은 약한 BCE(warm-up용,
   decay)와 budget만 준다. budget 안에서 task loss가 원하는 곳(pointer에서 먼 중요한 구조)의 R을 올릴 수 있다.
5. **Low R ≠ 정보 0.** gate가 닫히면 해당 위치는 identity로 F_l을 전달하고, stage transition conv(ungated)는 모든 위치에서
   실행된다. 즉 "base perception"은 항상 존재하고 R은 그 위의 refinement 양을 정한다.

---

## 2. V0 아키텍처

```
I [B,3,H,W]                      p [B,2] (x,y px)
   │                               │
 Stem (C_φ)                        │
   │ F0 [B,64,H/4,W/4]              │
   ├────────────── ResolutionPredictor(F0, pointer maps) ── R [B,1,H/4,W/4] ∈ [0,1]
   │                                                         │ (avg-pool to each stage)
 RABackbone(F0, R) ◄─────────────────────────────────────────┘
   │  s4  [B, 64,H/4 ,W/4 ]  stage1: 2 gated blocks
   │  s8  [B,128,H/8 ,W/8 ]  stage2: 2 gated blocks
   │  s16 [B,256,H/16,W/16]  stage3: 3 gated blocks
   │  s32 [B,384,H/32,W/32]  stage4: 2 gated blocks
 FPNLite (R 모름)
   │  inst [B,128,H/8,W/8]   mask [B,128,H/4,W/4]
 InstanceHead (R, p 모름)
      heat_logits [B,80,H/8,W/8]   kernels [B,64,H/8,W/8]   mask_feat [B,64,H/4,W/4]
```

### Tensor shape 표 (S = 512 기준, B = batch)

| Module | 입력 | 출력 |
|---|---|---|
| Stem: ConvGNAct(3→32,s2) → ConvGNAct(32→64,s2) | I `[B,3,512,512]` | F0 `[B,64,128,128]` |
| pointer_maps | p `[B,2]` | `[B,4,h,w]` = (x−p_x, y−p_y, dist, gaussian) |
| ResolutionPredictor | F0 + P `[B,68,128,128]` | R `[B,1,128,128]` |
| Stage1 (no transition, 2 blocks) | `[B,64,128,128]`, R `[B,1,128,128]` | s4 `[B,64,128,128]` |
| Stage2 (PreAct conv s2 + 2 blocks) | s4, R↓2 `[B,1,64,64]` | s8 `[B,128,64,64]` |
| Stage3 (s2 + 3 blocks) | s8, R↓4 `[B,1,32,32]` | s16 `[B,256,32,32]` |
| Stage4 (s2 + 2 blocks) | s16, R↓8 `[B,1,16,16]` | s32 `[B,384,16,16]` |
| FPNLite | s4..s32 | inst `[B,128,64,64]`, mask `[B,128,128,128]` |
| InstanceHead | inst, mask (+coord 2ch) | heat `[B,80,64,64]`, kernels `[B,64,64,64]`, mask_feat `[B,64,128,128]` |
| dynamic mask (per positive / per peak) | kernel `[P,64]`, mask_feat `[64,128,128]` | mask logits `[P,128,128]` |

파라미터 수 (K=80): stem 0.02M · R predictor 0.26M · backbone 10.85M · neck 0.40M · head 1.07M · **합계 12.6M**.

### 정규화

모든 norm은 GroupNorm이다. 작은 batch에서도 안정적이고, V1에서 sparse/tile 실행으로 갈 때도 batch 통계에 의존하지 않는다.
V0에서 GN 통계는 전체 feature map에서 계산되는데, V1 sparse 실행 시에는 tile-local 통계 문제를 다시 검토해야 한다(§7).

---

## 3. Resolution Predictor

```
[F0 (64), P_x, P_y, dist, gauss] @ H/4
  → ConvGNAct(68→64)                         x  (skip, H/4)
  → ConvGNAct s2 → ConvGNAct s2               (H/16)
  → cat pointer maps @H/16 → ConvGNAct d=2 → ConvGNAct d=4   (넓은 context)
  → upsample H/4, cat x → ConvGNAct → 1×1 → sigmoid → R
```

- **pointer-driven**: pointer map이 H/4와 H/16 두 곳에 들어간다.
- **image-driven**: dilated low-res 경로의 receptive field가 넓어서, pointer에서 떨어진 구조의 R도 올릴 수 있다.
- **가볍게 유지**: 약 0.3M params. "싼 coarse perception → 비싼 selective perception" 원칙을 따른다.
- 출력층은 weight 0, bias logit(0.5)로 초기화해서 학습 시작 시 R ≡ 0.5다.
- R은 stride 4로 예측한다. 시각화할 때만 full-res로 upsample한다.

---

## 4. R-conditioned Residual Block과 routing

### Block

```
F_{l+1} = F_l + g_l(R) · Body_l(F_l)
Body_l  = GN→GELU→Conv3×3 → GN→GELU→Conv3×3        (pre-activation)
g_l(R)  = sigmoid((R − τ_l) / T),   T = 0.1
```

- **Pre-activation**이라 `g = 0`이면 출력이 **정확히 identity**다(stream에 activation이 없다).
- **하나의 weight set**이다. Low/High용 별도 layer는 없다.

### Depth routing (R → 실행 깊이)

stage 내 block마다 threshold를 다르게 둔다: `τ_j = (j + 0.5) / depth`.

| stage depth | τ |
|---|---|
| 2 | 0.25, 0.75 |
| 3 | 0.17, 0.50, 0.83 |

- R = 0 → 거의 identity(base path만 실행)
- R = 0.5 → 각 stage의 앞쪽 절반 block만 실행
- R = 1 → 모든 block 실행

즉 R은 연속적인 "몇 번 refine할지" knob이다. V1에서는 `R > τ_l`을 **tile 단위 hard skip**으로 바꾸면 이 구조가
그대로 실제 연산 절감으로 이어진다. `backbone.virtual_compute(R)`가 그 경우의 실행 비율을 미리 계산해 준다(학습 로그의 `vcompute`).

대안 모드: `--gate-mode linear` (g = R), `--gate-mode none` (R 무시, baseline).

### Downsampling routing은 V0에서 보류

stage stride(4/8/16/32)는 고정하고, R은 block 내부의 depth만 제어한다. 이유는 다음과 같다.

- 위치마다 해상도를 다르게 하면 feature map이 비정형(multi-resolution/quadtree)이 된다. 순수 PyTorch에서 검증 가능한
  최소 구조를 벗어난다.
- "R이 quality를 조절하는가"라는 V0 가설은 depth gating만으로 검증할 수 있다.

---

## 5. Instance Segmentation Head (one-stage dense)

SOLOv2 계열의 dynamic kernel에 CenterNet식 center assignment를 결합했다.

- 출력이 stride-8 grid의 위치별로 나오므로 R(x,y)와 공간적으로 직접 대응한다.
  ("이 instance 중심의 R → 이 instance의 score"를 바로 분석할 수 있다.)
- box, RoI, anchor, multi-level assignment가 없다. 구조가 가장 직접적인 형태다.

| 출력 | 설명 |
|---|---|
| heat `[B,K,H/8,W/8]` | class별 center heatmap (objectness × class). CenterNet focal loss |
| kernels `[B,E,H/8,W/8]` | 위치별 1×1 dynamic kernel (E=64). coord channel 추가 |
| mask_feat `[B,E,H/4,W/4]` | 공유 mask feature. coord channel 추가 |

- **Target**: instance centroid → stride-8 cell. gaussian σ = max(0.8, √area/6) (stride-8 cell 단위).
  kernel positive는 center 주변 3×3 중 mask 내부 cell이다(겹치면 작은 instance 우선).
- **Inference**: heat 3×3 max-pool peak → top-100 → dynamic mask → maskness rescoring → Matrix NMS.
- backbone ↔ head 경계는 `feats = backbone(F0, R)` dict 하나다. V1/V2에서 backbone을 교체해도 head는 그대로 쓴다.

---

## 6. Loss와 Training schedule

### Loss

```
L = w_task · (L_focal + 3·L_dice)                       ← 모든 GT instance
  + w_in(t)·L_R_in + w_out(t)·L_R_out + w_budget·L_budget
```

| 항 | 정의 | 의도 |
|---|---|---|
| L_focal | CenterNet penalty-reduced focal, peak 수로 정규화 | objectness/class |
| L_dice | positive별 dice (stride 4) | mask 품질 |
| L_R_in | pointed instance 내부에서 −log R | pointer 대상은 반드시 high R |
| L_R_out | instance를 2px 팽창한 영역 밖에서 −log(1−R), 약하게(0.1) | warm-up. Phase C에서 decay |
| L_budget | `relu(mean_valid(R) − (area_frac(pointed) + 0.15))²` | R → 1 붕괴 방지. 남는 budget은 task가 배분 |

"먼 instance를 검출하면 loss를 준다"는 항은 **없다.**

### Schedule

| Phase | 학습 대상 | R 소스 | Loss | 기본값 |
|---|---|---|---|---|
| **A0** (baseline) | stem, backbone, neck, head | R ≡ 1 (`--r-fixed 1`) | task | full-compute 상한, 이후 비교 기준 |
| **A** | stem, backbone, neck, head | **sampled R field** | task | lr 1e-3, AdamW wd 0.05, cosine, warmup 1k |
| **B** | R predictor만 | R_θ | R_in + R_out + budget (+ 선택: `--w-task`) | lr 1e-3 |
| **C** | 전체 joint | R_θ | task + R-sup(1 → 0.1 linear decay) + budget | lr 2e-4 |

**Phase A의 R sampler**(`lova/rsampler.py`)가 V0의 핵심 장치다. predictor 없이도 backbone이
"R에 따라 quality가 달라지는" 함수가 되도록, 이미지마다 다음 중 하나를 섞는다.

- const (c ~ U(0,1), 30%는 0 또는 1의 극단값 — slimmable의 sandwich rule)
- oracle 형태 (`b + (1−b)·blur(pointed mask)`)
- smooth noise field

**권장 실행 순서와 epoch** (COCO train2017, 512px, 8 GPU × bs16 기준)

- 이 backbone은 ImageNet pretrain이 없는 from-scratch다. GN으로 from-scratch 학습은 가능하지만 길어야 한다.
  A0/A는 **24–36 epoch** 이상을 권장한다.
- 빠른 가설 확인용으로 A를 12 epoch만 돌려도 R별 quality 곡선은 볼 수 있다.
- B: 2–4 epoch, C: 6–12 epoch.

```bash
torchrun --nproc_per_node 8 train.py --phase A --r-fixed 1.0 --coco-root $COCO --epochs 24 --amp --out runs/A0
torchrun --nproc_per_node 8 train.py --phase A                --coco-root $COCO --epochs 24 --amp --out runs/A
torchrun --nproc_per_node 8 train.py --phase B --init runs/A/last.pth --coco-root $COCO --epochs 3 --amp --out runs/B
torchrun --nproc_per_node 8 train.py --phase C --init runs/B/last.pth --coco-root $COCO --epochs 8 --amp --out runs/C
```

---

## 7. 실제 FLOPs 감소 vs 단순 masking

| 요소 | V0 상태 | 실제 연산 감소? | 감소시키려면 |
|---|---|---|---|
| soft gate `g(R)·Body(F)` | 모든 위치에서 Body 계산 후 곱함 | ❌ masking | V1: R > τ인 **tile만 gather → Body → scatter** |
| depth routing τ_l | soft | ❌ (V0) / ✅ (V1 hard) | tile 단위 quantize. block 단위 skip은 이미지 전체 R < τ일 때만 가능 |
| Stage stride | 고정 | — (전 위치 동일) | V2: R 기반 위치별 해상도(coarse tile은 저해상도 처리) |
| R predictor | 항상 실행 | 오버헤드 (+) | 작게 유지. stride 4 → 필요시 stride 8 |
| GroupNorm | 전체 map 통계 | — | sparse 실행 시 tile-local norm 또는 통계 고정 필요 |
| dilation / sampling spacing | 없음 | ❌ (spacing만으로는 FLOPs 동일) | tap 수 n(R)까지 줄여야 감소 (V2) |
| head / neck | 전 위치 실행 | ❌ | V1+: low-R 영역 peak만 계산 등 |

V0 로그의 `vcompute` = "hard tile routing을 했다면 gated block FLOPs 중 실행될 비율"이다. 실측이 아니라 **예상치**다.

---

## 8. V0 검증 계획 (`eval_pointer.py`)

### 측정

GT instance마다 아래 값을 기록한다(→ `instances.csv`).

- `mean_r`: instance 내부 평균 R
- `dist`: pointer → instance 거리 / canvas (pointer가 내부에 있으면 0)
- `best_iou`: 같은 class 예측과의 최대 mask IoU
- `score`: IoU ≥ 0.5인 같은 class 예측의 최대 score
- `detected`: score ≥ `det_thr`(0.3)

이를 R-bin별, 거리-bin별 표로 요약한다. COCO mask AP는 전체와 bin별로 낸다(다른 bin의 GT는 iscrowd로 무시하는 근사).

### 실험 행렬

| 실험 | 명령 | 확인할 것 |
|---|---|---|
| E1 상한 | A0 ckpt, `--r-mode ones` | full-compute AP |
| E2 R 제어성 | A ckpt, `--r-mode const:0/0.25/0.5/0.75/1` | **AP가 R에 대해 단조 증가하는가** (V0 핵심) |
| E3 sampler 비용 | A ckpt `ones` vs E1 | R-conditioning 학습이 R=1 성능을 얼마나 깎는가 |
| E4 oracle | A ckpt `--r-mode oracle` | pointer 대상 품질 ↑, 나머지 ↓ 패턴 |
| E5 예측 R | C ckpt `--r-mode pred` | R-bin / 거리-bin별 score·IoU·det_rate 곡선 |
| E6 pointer gain | C ckpt `--sweep 4` | 같은 instance가 pointed일 때 Δscore, ΔIoU > 0 |
| E7 R의 image-driven 성분 | C ckpt, viz | pointer 밖에서 R이 오르는 위치가 의미 있는가 |
| E8 R 무시 baseline | `--gate-mode none` | gate가 없을 때와 비교 |

### V0 성공 기준 (제안)

1. E2: AP(R=0) < AP(0.5) < AP(1)로 단조 증가하고 그 폭이 의미 있을 것(예: AP 차 ≥ 5–10pt)
2. E5/E6: pointed instance의 score·IoU가 같은 instance의 non-pointed 대비 유의하게 높고, 거리에 따라 단조 감소할 것
3. E3: R=1일 때 성능 손실이 작을 것(A0 대비 1–2 AP 이내)

---

## 8b. V0.1 — continuous R + binary execution (Phase A 결과 후 추가, 2026-10-08)

### 관찰

COCO Phase A(depth gating, continuous sampler) 10k step 체크포인트에서 R을 외부 주입해 보면
R=0 → R=0.25에서는 뚜렷한 차이(작은 물체 소실, score 하락)가 있으나 **R=0.25 이상은 평평**하다.
학습 로그에서도 중간 R 구간(배치 평균 0.33~0.55)과 loss의 상관이 0이다.
τ 스케줄상 R=0.25에서는 각 stage의 첫 블록이 절반 세기(g=0.5)로만 켜지고 뒤쪽 블록은 g≤0.08이므로,
"첫 블록 절반만으로 R=1 품질이 나온다" = 뒤쪽 refinement 블록의 capacity가 실질적으로 쓰이지 않는다.
이유는 §1의 설계 자체에 있다: 모든 GT가 R과 무관하게 positive이므로 네트워크는 R에 둔감해지는 쪽이
loss를 가장 줄이며, 샘플러가 앞쪽 블록을 더 자주 켜 주어 능력이 거기에 몰린다.
E2 다섯 점 AP(val2017 1000장, 24 epoch 완료 체크포인트)로 확정했다:

| R | 0 | 0.25 | 0.5 | 0.75 | 1 |
|---|---|---|---|---|---|
| mask AP | 16.7 | 20.6 | 21.5 | 22.0 | 21.8 |

R=0 vs R=1 차이 5.1 AP(성공 기준 하한 충족), R=0에서도 R=1의 77%(graceful). 이득의 76%가 0→0.25에서 나오고
0.25→1은 1.2 AP. 마지막 블록을 완전히 켜도(0.75→1) 이득 없음. → 계단형, 뒤쪽 블록 미활용 확정.

참고(E5 대조군): R이 상수여도 거리-bin AP는 inside 19.6 → 가장 먼 bin 13.2로 떨어진다(bin별 물체 분포 차이).
Phase C의 거리별 저하는 절대값이 아니라 **이 const 결과와의 차이**로 판단해야 한다.

### 결정: R의 표현과 실행 routing을 분리한다

```
R_θ(I,p) ∈ [0,1]^{H×W}      continuous importance / refinement priority   (그대로 유지)
G = 1[R > τ]                 binary execution map, τ = 0.5, stage 내 블록 공유
F_{l+1} = F_l + G · Body_l(F_l)
학습:  G ≈ sigmoid((R − τ)/T)   (relaxation. V0는 여전히 masking)
V1 :  G = 1[R > τ] hard, STE
```

- compute 손잡이는 "R의 크기"가 아니라 **R > τ인 면적**이 된다. budget loss도 mean(R) 대신
  mean(g(R))에 건다(`--budget-on gate`). "R = τ − ε everywhere" 같은 퇴화 해법을 막는다.
- Phase A 샘플러는 {0,1}만 쓴다(`--r-sampler binary`): const ∈ {0,1}, oracle = pointed mask 팽창(반경 0~8),
  noise = 임계 처리한 이진 패치. backbone이 두 operating point를 확실히 배우게 한다.
  **R predictor(B/C)는 계속 연속**이고 0/1 GT로 당기지 않는다. 값의 크기는 task loss + budget이 정한다.
- Phase C에서 T를 서서히 낮출 수 있다(`--gate-temp-final`). hard routing과의 train/inference 격차를 줄인다.
- 평가 시 `--gate-mode hard`로 1[R>τ] 실행을 미리 재서 격차를 측정한다.

### 용어

V0에서 soft gating인 한 mask 면적을 줄여도 FLOPs는 줄지 않는다. 따라서 V0의 곡선은
**mean(g) ↔ AP** (active-area proxy)라고 부른다. **FLOPs ↔ AP**는 V1 hard routing 구현 후에만 주장한다.
또한 Phase A 체크포인트는 A0가 아니다. A0는 R≡1로만 학습한 별도 baseline이며 E3가 둘의 차이를 잰다.

### V0.1 성공 기준

```
AP(A, R=1) ≈ AP(A0)          (refinement 블록이 실제로 쓰임, 손실 1~2 AP 이내)
AP(A, R=0) < AP(A, R=1)      (차이가 충분히 클 것. 예: 13 vs 19.5)
AP(A, hard) ≈ AP(A, soft)    (train/inference 격차 작음)
```

```bash
torchrun --nproc_per_node 4 train.py --phase A --gate-mode binary --coco-root $COCO --epochs 24 --bs 32 --amp --out runs/Ab
python eval_pointer.py --ckpt runs/Ab/last.pth --coco-root $COCO --r-mode zeros --out eval/Ab_r0
python eval_pointer.py --ckpt runs/Ab/last.pth --coco-root $COCO --r-mode ones  --out eval/Ab_r1
python eval_pointer.py --ckpt runs/Ab/last.pth --coco-root $COCO --r-mode ones --gate-mode hard --out eval/Ab_r1_hard
```

---

## 9. V1 / V2 확장

### V1 — 실제 연산 절감 (같은 R semantics 유지)

- R을 tile 단위(예: stage별 8×8 또는 16×16 cell)로 pooling한다. `R_tile > τ_l`이면 해당 block을 그 tile에서 실행한다.
- **gather → batched conv(halo 포함) → scatter.** 순수 PyTorch(`unfold`/index)로 먼저 구현한 뒤 latency를 실측한다.
- 학습: soft gate를 유지하되 forward는 hard, backward는 straight-through(또는 Gumbel)로 처리한다. train/inference 불일치를 줄인다.
- `L_budget`을 실제 FLOPs 기반 `L_compute = Σ_l FLOPs_l · mean(g_l)`로 교체한다.
- GN을 tile-local 또는 frozen-statistics로 바꾼다.
- R-sup를 점점 약하게 하여 `L_task + λ·L_compute`만으로 R을 latent control variable로 학습하는 실험을 시작한다.

### V2 — 위치별 해상도 / sampling

- R 기반 위치별 해상도: low-R tile은 한 단계 낮은 해상도로 처리한다(quadtree/multi-res tile).
- **RAConv**: 공유 weight `W`, `s(R)` sampling spacing + `n(R)` tap 수. level을 quantize(9/5/3 tap)해서 같은 level끼리 gather한다.
- Triton/CUDA 커널은 이 단계에서 도입한다.
- Boundary head 재도입 → occlusion(front/back) 관계.

---

## 10. 코드 구조

```
lova/
  models/common.py       ConvGNAct, PreActConv, coord_grid
  models/resolution.py   pointer_maps, ResolutionPredictor
  models/backbone.py     Stem, GatedResBlock, RAStage, RABackbone (+virtual_compute)
  models/neck.py         FPNLite
  models/head.py         InstanceHead, dynamic_masks, matrix_nms, postprocess
  models/model.py        LOVAv0 (wiring + 제약)
  data/coco.py           COCOPointerDataset
  data/synthetic.py      SyntheticPointerDataset (smoke/sanity)
  data/targets.py        center/gaussian/positive assignment
  data/common.py         pointer sampling, sample format, collate
  losses.py              focal, dice, R losses
  rsampler.py            Phase A R field sampler
train.py                 Phase A/B/C, DDP(torchrun), bf16
eval_pointer.py          V0 가설 검증 (bin 표, AP, sweep, viz)
tests/smoke_test.py      CPU에서 shape/gradient/전체 파이프라인 확인
```

### 합성 데이터 sanity 결과 (CPU, 128px, Phase A 1500 iter)

| r-mode | score | mIoU | det_rate |
|---|---|---|---|
| const:0 | 0.677 | 0.853 | 0.905 |
| const:0.5 | 0.706 | 0.855 | 0.917 |
| ones | 0.709 | 0.856 | 0.918 |
| oracle: R∈[0.6,0.8) instance | 0.777 | 0.931 | 1.000 |

- 학습이 수렴하고, R에 대해 단조 증가하는 것까지 확인했다.
- 다만 R=0과 R=1의 차이가 작다. 합성 데이터는 색과 모양만으로 풀려서 **always-on base path**
  (stem + stage transition 3개 + neck + head)만으로도 충분하기 때문이다.
- COCO에서는 차이가 커질 것으로 예상하지만, 이것이 V0의 가장 큰 리스크다. 대응 옵션:
  - `--transition light`: transition을 avgpool + 1×1로 약화해 base path를 줄인다.
  - stage depth 증가: gated 비중을 늘린다.
- E2 결과를 보고 결정한다.

### 열린 질문 / 리스크

- **Base path가 너무 강하면 R 효과가 작아진다**(위 sanity 참고). `--transition light`와 비교 실험할 것.

- **R sampler가 R=1 성능을 깎을 수 있다**(E3). 그러면 sampler에서 R=1 비율을 높이거나 A0 → A fine-tune 순서로 바꾼다.
- **Classification은 low-res로도 잘 되는 경향**이 있다. R이 mask 품질에는 영향을 주지만 score에는 덜 영향을 줄 수 있다.
  E5에서 score와 IoU를 따로 보는 이유다.
- **Neck/head의 receptive field**를 통해 high-R 이웃의 feature가 low-R instance로 번진다. 이는 의도된 "부드러운" 효과지만,
  작은 instance일수록 pointer 효과가 더 선명할 것으로 예상된다.
- **Crowd annotation**은 V0에서 background로 취급한다.
