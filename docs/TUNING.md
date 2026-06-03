# TUNING — 하이퍼파라미터 튜닝 가이드

| 항목      | 내용                                                                           |
| --------- | ------------------------------------------------------------------------------ |
| 목적      | Val NDCG@10 기준으로 **체계적·재현 가능한** 튜닝                               |
| 튜닝 지표 | **`val/ndcg_cart_purchase`** (NDCG@10, cart+purchase GT ~1,072명)              |
| 참고 지표 | `val/ndcg_purchase_only` (보조), `leaderboard_proxy/ndcg` (**설정 변경 금지**) |
| 실행      | `python src/train.py ...` (Holdout CV, `cv=single_holdout` 기본)               |
| 관련 문서 | [PLAN.md](PLAN.md) · [OPERATION.md](OPERATION.md) · [README.md](../README.md)  |

---

## 0. 튜닝 원칙 (3줄)

1. **설정 선택은 Val만** — proxy·리더보드로 하이퍼파라미터를 바꾸지 않는다.
2. **한 번에 하나(또는 한 그룹)** — 여러 값을 동시에 바꾸면 원인 분석이 어렵다.
3. **확정 후 Full-train** — 튜닝에서 **`best_epoch` 기록** → `cv=none` + **`train.epochs` 지정** → `full/best.pt`.

---

## 1. 튜닝 단계 로드맵

| Step   | 목표                  | 산출물                                       | 완료 기준                                     |
| ------ | --------------------- | -------------------------------------------- | --------------------------------------------- |
| **S0** | 파이프라인 검증       | SASRec 5 epoch 스모크                        | Val NDCG 출력·ckpt 저장                       |
| **S1** | Loss 가중치           | cart 10/25/50 비교표                         | Val best cart 값 1개 확정                     |
| **S2** | 모델별 기본 학습      | SASRec / TiSASRec / CL4SRec best run         | run별 `tuning/best.pt` + wandb run id         |
| **S3** | 구조 미세 튜닝 (선택) | seq_len, dropout, CL4SRec lmd 등             | Val 개선 ≥ 0.001 또는 시간 대비 무의미면 중단 |
| **S4** | 앙상블 가중치         | `optimize_ensemble.py` 실행 → rank.yaml 갱신 | Val 최고 조합 1개                             |
| **S5** | Proxy sanity (선택)   | `eval_proxy.py` 1~2회                        | 기록만, **설정 변경 없음**                    |
| **S6** | Full-train + 제출     | `full/best.pt`, submission CSV               | 튜닝 `best_epoch` 반영·행 수 6,382,570 검증   |

---

## 2. 우선순위 매트릭스

| 우선순위  | 파라미터 그룹                                 | 설정 위치                                 | Val 영향 | 시간/VRAM       | 권장                             |
| --------- | --------------------------------------------- | ----------------------------------------- | -------- | --------------- | -------------------------------- |
| ★★★       | **loss_weights (cart, purchase)**             | `conf/train/base.yaml`                    | 매우 큼  | 낮음            | **S1 필수**                      |
| ★★★       | **앙상블 weights**                            | `conf/ensemble/rank.yaml`                 | 매우 큼  | 추론만          | **S4 필수**                      |
| ★★☆       | **max_seq_len**                               | `conf/model/*.yaml`                       | 중~큼    | **L²** (50≪100) | **50 기본**, 시간 있으면 100 1회 |
| ★★☆       | **모델 조합** (어떤 모델을 앙상블에 넣을지)   | 실험 설계                                 | 큼       | 높음            | S2 후 결정                       |
| ★★☆       | **CL4SRec lmd, aug / FEARec freq_mask_ratio** | `conf/model/cl4srec.yaml` · `fearec.yaml` | 중~큼    | 중              | 희소성·주파수 대응               |
| ★★☆       | **TiSASRec time_span**                        | `conf/model/tisasrec.yaml`                | 중       | 중              | CV=4.12, 넓게 유지               |
| ★☆☆       | lr, patience, epochs                          | `conf/train/base.yaml`                    | 중       | 중              | Val 정체 시 조정                 |
| ★☆☆       | hidden_size, n_layers, inner_size             | `conf/model/*.yaml`                       | 중       | 높음            | 기본값(256/3/512) 우선           |
| ★☆☆       | dropout                                       | `conf/model/*.yaml`                       | 소~중    | 낮음            | 0.3 vs 0.5                       |
| 제출 직전 | **cart boost_to_top_n**                       | 후처리 (PLAN 6-B)                         | 중~큼    | 낮음            | Full-train 후 Val로 0/1/3        |

---

## 3. 공통 학습 하이퍼파라미터

**파일:** `conf/train/base.yaml` · Hydra 오버라이드 `train.*`

| 파라미터                  | 현재값                       | 탐색 후보            | CLI 예시                           | 비고                             |
| ------------------------- | ---------------------------- | -------------------- | ---------------------------------- | -------------------------------- |
| `loss_weights.view`       | 1.0                          | 고정                 | —                                  | 기준 스케일                      |
| `loss_weights.cart`       | **25.0**                     | **10, 25, 50**       | `train.loss_weights.cart=30`       | **1순위 sweep**                  |
| `loss_weights.purchase`   | 50.0                         | 25, 50, 100          | `train.loss_weights.purchase=50`   | cart와 비율 유지                 |
| `lr`                      | 0.001                        | 0.0005, 0.001, 0.002 | `train.lr=0.0005`                  | NaN 시 ↓                         |
| `epochs`                  | 20                           | 20 ~ 50              | `train.epochs=30`                  | early stop과 병행                |
| `early_stopping_patience` | 5                            | 5, 10, 20            | `train.early_stopping_patience=10` | Val plateau 시                   |
| `weight_decay`            | 1e-4                         | 고정                 | —                                  |                                  |
| `train_batch_size`        | 모델별 (`conf/model/*.yaml`) | 모델별 상한          | 아래 §5                            | OOM 시 `train.train_batch_size=` |
| `amp`                     | bf16                         | 고정                 | —                                  | RTX 3090                         |

**S1 권장 grid (cart만, SASRec 빠른 비교):**

| Run ID | cart | purchase | CLI                                                                                                         |
| ------ | ---- | -------- | ----------------------------------------------------------------------------------------------------------- |
| T1-01  | 10   | 50       | `python src/train.py model=sasrec train.loss_weights.cart=10 wandb.name=tune_cart10 wandb.tags=[sasrec,s1]` |
| T1-02  | 25   | 50       | `python src/train.py model=sasrec train.loss_weights.cart=25 wandb.name=tune_cart25 wandb.tags=[sasrec,s1]` |
| T1-03  | 50   | 50       | `python src/train.py model=sasrec train.loss_weights.cart=50 wandb.name=tune_cart50 wandb.tags=[sasrec,s1]` |

→ Val best인 cart 값을 **S2~S4 전 모델에 동일 적용**.

---

## 4. 모델 구조 하이퍼파라미터

### 4-1. 공통 (SASRec / TiSASRec / CL4SRec)

**파일:** `conf/model/<model>.yaml` · 오버라이드 `model.*`

| 파라미터           | 현재값                  | 탐색 후보   | 우선순위 | CLI 예시                                              |
| ------------------ | ----------------------- | ----------- | -------- | ----------------------------------------------------- |
| `max_seq_len`      | **50**                  | **50, 100** | ★★☆      | `model.max_seq_len=50`                                |
| `hidden_size`      | 256                     | 128, 256    | ★☆☆      | `model.hidden_size=256`                               |
| `n_layers`         | 3                       | 2, 3        | ★☆☆      | `model.n_layers=3`                                    |
| `n_heads`          | 4                       | 2, 4        | ★☆☆      | `model.n_heads=4`                                     |
| `inner_size`       | 512                     | 256, 512    | ★☆☆      | `model.inner_size=512`                                |
| `hidden_dropout`   | 0.5                     | 0.3, 0.5    | ★☆☆      | `model.hidden_dropout=0.5`                            |
| `attn_dropout`     | 0.5                     | 0.3, 0.5    | ★☆☆      | `model.attn_dropout=0.5`                              |
| `train_batch_size` | 모델별 (4096/1024/2048) | OOM 시 ↓    | ★☆☆      | `train.train_batch_size=512` (yaml 기본값 오버라이드) |

**max_seq_len 참고 (EDA):** 중앙값 6 · p90 29 · p99 100 · 50 초과 유저 ~4%. **학습 시간 L²** → 기본 **50** 권장.

### 4-2. TiSASRec 전용

| 파라미터    | 현재값 | 탐색 후보      | CLI 예시              |
| ----------- | ------ | -------------- | --------------------- |
| `time_span` | 512    | 256, 512, 1024 | `model.time_span=512` |

### 4-3. CL4SRec 전용

| 파라미터            | 현재값 | 탐색 후보      | CLI 예시                      |
| ------------------- | ------ | -------------- | ----------------------------- |
| `lmd`               | 0.1    | 0.05, 0.1, 0.2 | `model.lmd=0.1`               |
| `tau`               | 1.0    | 0.5, 1.0, 2.0  | `model.tau=1.0`               |
| `aug_crop_ratio`    | 0.2    | 0.1 ~ 0.3      | `model.aug_crop_ratio=0.2`    |
| `aug_mask_ratio`    | 0.2    | 0.1 ~ 0.3      | `model.aug_mask_ratio=0.2`    |
| `aug_reorder_ratio` | 0.2    | 0.1 ~ 0.3      | `model.aug_reorder_ratio=0.2` |

### 4-4. FEARec 전용

| 파라미터          | 현재값 | 탐색 후보      | CLI 예시                    |
| ----------------- | ------ | -------------- | --------------------------- |
| `lmd`             | 0.1    | 0.05, 0.1, 0.2 | `model.lmd=0.1`             |
| `tau`             | 1.0    | 0.5, 1.0       | `model.tau=1.0`             |
| `freq_mask_ratio` | 0.3    | 0.1, 0.3, 0.5  | `model.freq_mask_ratio=0.3` |

### 4-5. SAFERec 전용

| 파라미터         | 현재값 | 탐색 후보   | CLI 예시                  |
| ---------------- | ------ | ----------- | ------------------------- |
| `n_freq_buckets` | 64     | 32, 64, 128 | `model.n_freq_buckets=64` |

### 4-6. 앙상블 가중치 (optimize_ensemble.py 자동 탐색)

**파일:** `conf/ensemble/rank.yaml`  
`optimize_ensemble.py` 실행 시 Val NDCG@10 기준 랜덤 서치 300회로 자동 갱신됨.

현재 초기값:

| 모델     | 가중치 |
| -------- | ------ |
| sasrec   | 1.00   |
| tisasrec | 0.35   |
| cl4srec  | 0.20   |
| fearec   | 0.30   |
| bsarec   | 0.25   |
| saferec  | 0.20   |
| mbstr    | 0.40   |
| tifu_knn | 0.15   |

```bash
# 자동 최적화 (300회, seed=42)
python src/optimize_ensemble.py

# 시도 횟수 늘리기 (conf/config.yaml n_trials=300 기본)
python src/optimize_ensemble.py n_trials=500 seed=0
```

---

## 5. 모델별 배치·VRAM (RTX 3090, BF16, max_seq_len=50)

| 모델     | 기본 `train_batch_size` | VRAM (대략)                 | 설정 위치 / OOM 시 CLI                                    |
| -------- | ----------------------- | --------------------------- | --------------------------------------------------------- |
| SASRec   | **4096**                | ~6 GB                       | `conf/model/sasrec.yaml` · `train.train_batch_size=2048`  |
| TiSASRec | **1024**                | ~9 GB (time_matrix [B,L,L]) | `conf/model/tisasrec.yaml` · `train.train_batch_size=512` |
| CL4SRec  | **2048**                | ~10 GB (대조 뷰 2개)        | `conf/model/cl4srec.yaml` · `train.train_batch_size=1024` |
| FEARec   | **2048**                | ~10 GB (FFT 증강 뷰)        | `conf/model/fearec.yaml` · `train.train_batch_size=1024`  |
| BSARec   | **4096**                | ~6 GB                       | `conf/model/bsarec.yaml` · `train.train_batch_size=2048`  |
| SAFERec  | **4096**                | ~6 GB                       | `conf/model/saferec.yaml` · `train.train_batch_size=2048` |
| MB-STR   | **4096**                | ~6 GB                       | `conf/model/mbstr.yaml` · `train.train_batch_size=2048`   |
| TIFU-KNN | — (CPU)                 | < 1 GB                      | `src/train_tifu.py` 단독 실행                             |

OOM 시: batch **절반** → 재실행. batch는 Val 점수에도 영향하므로 **OOM 해결용**으로만 변경.

---

## 6. S2 — 모델별 기본 튜닝 Runbook

S1에서 확정한 `train.loss_weights.cart`를 아래에 반영 (`CART=25` 예시).

| Run   | 모델     | 명령 (요약)                                                     | 체크포인트                                      |
| ----- | -------- | --------------------------------------------------------------- | ----------------------------------------------- |
| T2-01 | SASRec   | `python src/train.py model=sasrec train.loss_weights.cart=25`   | `outputs/sasrec/runNNN_YYMMDD/tuning/best.pt`   |
| T2-02 | TiSASRec | `python src/train.py model=tisasrec train.loss_weights.cart=25` | `outputs/tisasrec/runNNN_YYMMDD/tuning/best.pt` |
| T2-03 | CL4SRec  | `python src/train.py model=cl4srec train.loss_weights.cart=25`  | `outputs/cl4srec/runNNN_YYMMDD/tuning/best.pt`  |
| T2-04 | FEARec   | `python src/train.py model=fearec train.loss_weights.cart=25`   | `outputs/fearec/runNNN_YYMMDD/tuning/best.pt`   |
| T2-05 | BSARec   | `python src/train.py model=bsarec train.loss_weights.cart=25`   | `outputs/bsarec/runNNN_YYMMDD/tuning/best.pt`   |
| T2-06 | SAFERec  | `python src/train.py model=saferec train.loss_weights.cart=25`  | `outputs/saferec/runNNN_YYMMDD/tuning/best.pt`  |
| T2-07 | MB-STR   | `python src/train.py model=mbstr train.loss_weights.cart=25`    | `outputs/mbstr/runNNN_YYMMDD/tuning/best.pt`    |

**run 폴더:** 튜닝마다 `run001`, `run002` … **자동 증가**. wandb run name에 `tune_fearec_s2` 등 태그 권장.

---

## 7. S3 — 선택적 구조 튜닝 (시간 있을 때)

한 모델·한 파라미터씩만 변경.

| 실험       | 변경                          | 비교 기준                 |
| ---------- | ----------------------------- | ------------------------- |
| T3-seq50   | `model.max_seq_len=50` (기본) | T2 best                   |
| T3-seq100  | `model.max_seq_len=100`       | T2 best (시간 4배 가까움) |
| T3-drop03  | `model.hidden_dropout=0.3`    | T2 best                   |
| T3-cl4-lmd | `model.lmd=0.05` / `0.2`      | T2-03 best                |

**중단 규칙:** Val 개선 **< 0.001** 이면 해당 축 튜닝 중단.

---

## 8. S4 — 앙상블 가중치

**파일:** `conf/ensemble/rank.yaml` · `src/optimize_ensemble.py`

가중치는 `optimize_ensemble.py`가 Val NDCG@10 기준 랜덤 서치로 자동 탐색하여 갱신합니다.  
체크포인트가 없는 모델은 자동 스킵됩니다.

```bash
# 자동 최적화 후 앙상블 제출
python src/optimize_ensemble.py
python src/ensemble_submit.py
```

수동 오버라이드가 필요한 경우 `conf/ensemble/rank.yaml`을 직접 편집 후 `ensemble_submit.py`를 재실행합니다.

---

## 9. S5 — Proxy (참고만)

```bash
# tuning ckpt (기본)
python src/eval_proxy.py model=tisasrec

# ckpt·seq_len 명시 (CL4SRec 등)
python src/eval_proxy.py model=cl4srec model.max_seq_len=100 \
  ckpt_path=outputs/cl4srec/run001_260520/tuning/best.pt

# Full-train ckpt (eval_proxy는 기본 tuning — full은 ckpt_path로)
python src/eval_proxy.py model=sasrec \
  ckpt_path=outputs/sasrec/run003_260520/full/best.pt
```

| Val vs Proxy | 해석              | 조치                                         |
| ------------ | ----------------- | -------------------------------------------- |
| Val↑ Proxy↑  | 양호              | Full-train 진행                              |
| Val↑ Proxy↓  | 평시 과적합 가능  | TIFU↓, cart boost 검토 (**Val 기준은 유지**) |
| Val↓ Proxy↑  | Val 과소평가 가능 | Full-train·후처리 비중 ↑                     |

---

## 10. S6 — Full-train & 제출

튜닝 확정 후 **동일 하이퍼파라미터·튜닝 `best_epoch`** 로 전 기간 학습.

### 10-1. `best_epoch` 기록 (S2~S3 완료 시 필수)

튜닝(Holdout CV) run마다 아래를 **실험 테이블(§11)에 기록**한다.

| 항목                          | 확인 위치                                                                  |
| ----------------------------- | -------------------------------------------------------------------------- |
| `best_epoch`                  | wandb `best_epoch` · 콘솔 early stop 로그 · `tuning/best.pt` 내 `epoch` 키 |
| `best val/ndcg_cart_purchase` | wandb `val/best_ndcg_cart_purchase`                                        |
| early stop epoch              | 콘솔 `Early stopping at epoch …` (**Full-train epoch로 쓰지 않음**)        |

> ⚠️ **`best_epoch` ≠ early stop epoch.** early stop은 Val이 더 이상 오르지 않아 멈춘 시점이며, Full-train에는 **Val NDCG가 최고였던 epoch**를 사용한다.

### 10-2. Full-train epoch 설정 (`cv=none`)

`cv=none` 시 `src/train.py` 동작:

- Train/Val 분할 **없음** → Val NDCG·early stopping **없음** (`early_stopping_patience` 무시)
- `train_loss`만 로깅 → **마지막 epoch** 가중치를 `full/best.pt`에 저장
- 기본 `train.epochs=20`을 그대로 쓰면 튜닝 best와 **불일치**할 수 있음 → **반드시 튜닝 `best_epoch` 반영**

**epoch 수 변환 (코드는 0-index):**

| 튜닝 로그 / ckpt                               | Full-train CLI   |
| ---------------------------------------------- | ---------------- |
| `best_epoch=6` (0-index, epoch 0~6까지 7 step) | `train.epochs=7` |
| ckpt `epoch: 6`                                | `train.epochs=7` |

확실하지 않으면 `tuning/best.pt`의 `epoch` 값을 확인한 뒤 **`train.epochs = epoch + 1`** 로 지정한다.

```bash
# 예: 튜닝 best_epoch=6 (0-index) → Full-train 7 epoch
python src/train.py model=tisasrec cv=none train.epochs=7 \
  wandb.tags=[fulltrain] wandb.name=tisasrec_full_ep7

python src/submit.py model=tisasrec
python src/ensemble_submit.py
```

**선택 (시간·GPU 여유 시):** `best_epoch ± 1~2` 로 Full-train을 2~3회 돌린 뒤 `eval_proxy.py`로 sanity check (Proxy 점수로 **설정 변경하지 않음**).

| 산출물    | 경로                                           |
| --------- | ---------------------------------------------- |
| 튜닝 ckpt | `outputs/<model>/runNNN_YYMMDD/tuning/best.pt` |
| 제출 ckpt | `outputs/<model>/runNNN_YYMMDD/full/best.pt`   |

---

## 11. 실험 기록 테이블 (복사해서 사용)

wandb 또는 스프레드시트에 아래 컬럼으로 기록.

| exp_id | step | model    | changed_param | value         | val/ndcg_cp | best_epoch | full_train_epochs | run_dir       | wandb_run | notes                               |
| ------ | ---- | -------- | ------------- | ------------- | ----------- | ---------- | ----------------- | ------------- | --------- | ----------------------------------- |
| T1-01  | S1   | sasrec   | cart          | 10            |             |            |                   | run003_260520 |           |                                     |
| T1-02  | S1   | sasrec   | cart          | 25            | 0.1541      | 10         | **11**            | run004_260520 | 9tzv2dyc  | **best cart** · full=`best_epoch+1` |
| T2-02  | S2   | tisasrec | —             | defaults      | 0.1501      | 1          | **2**             | run001_260520 | pzjvy1b6  | early stop ep6 ≠ full epoch         |
| E1     | S4   | ensemble | weights       | 0.4/0.25/0.35 |             |            |                   | —             |           |                                     |

**changed_param:** 실제로 바꾼 것만 적기 (`cart`, `max_seq_len`, `ensemble`, `-`).

---

## 12. 튜닝하지 않는 것

| 항목                     | 이유                     |
| ------------------------ | ------------------------ |
| Holdout 기간 (Feb 09~22) | 대회·PLAN 설계 고정      |
| `gt_mode: cart_purchase` | 튜닝 목적함수 정의       |
| `leaderboard_proxy` 점수 | 참고만, **설정 변경 ❌** |
| `val/ndcg_purchase_only` | GT ~37명, 분산 과다      |
| `amp: bf16`              | 이미 활성, 3090 최적     |
| `initializer std=0.02`   | 코드 고정, 관례값        |

---

## 13. CLI 치트시트

Hydra 오버라이드: `키=값` (**`=` 앞뒤 공백 없음**). `wandb.name` 미지정 시 `{model}_seed{seed}`.

```bash
# S1 — cart sweep
python src/train.py model=sasrec train.loss_weights.cart=25 wandb.name=tune_s1_cart25 wandb.tags=[sasrec,s1]

# S2 — 전체 모델 튜닝 (배치·VRAM 주의)
python src/train.py model=sasrec
python src/train.py model=tisasrec
python src/train.py model=cl4srec model.max_seq_len=100
python src/train.py model=fearec
python src/train.py model=bsarec
python src/train.py model=saferec
python src/train.py model=mbstr

# S3 — seq (시간 주의)
python src/train.py model=tisasrec model.max_seq_len=50

# S4 — 앙상블 가중치 자동 최적화
python src/optimize_ensemble.py

# S5 — proxy (튜닝 후 1회)
python src/eval_proxy.py model=tisasrec

# S6 — full-train 및 제출 (train.epochs = 튜닝 best_epoch + 1)
python src/train.py model=tisasrec cv=none train.epochs=7 wandb.tags=[fulltrain]
python src/train_tifu.py          # TIFU-KNN 예측 재생성
python src/ensemble_submit.py     # cart boost 자동 포함
```

---

## 14. 관련 문서

- [OPERATION.md](OPERATION.md) — Phase별 운영·체크리스트
- [PLAN.md](PLAN.md) — EDA 근거·모델 우선순위
- [README.md](../README.md) — 파이프라인 개요

---

## 15. Feature Engineering 검토

> **검토 일자**: 2026-05-21  
> **결론**: 구현 복잡도 대비 기대 효과 순서 — `cat_l2/brand 임베딩 > price tier 임베딩 > 기타`. 현재 미완인 앙상블 튜닝·MB-STR/BSARec 실험이 우선, Feature Engineering은 후순위.

### 현재 이미 활용 중인 피처 신호

핵심 behavioral signal은 이미 모델별로 커버되고 있습니다.

| 피처                                          | 활용 모델 | 구현 위치                                                 |
| --------------------------------------------- | --------- | --------------------------------------------------------- |
| 아이템 간 시간 간격                           | TiSASRec  | `data/features.py` `build_time_seq`                       |
| view 빈도 버킷 (log-scale 64bucket)           | SAFERec   | `data/features.py` `compute_item_freq` / `build_freq_seq` |
| view·cart·purchase 행동 타입 (padding_idx=3)  | MB-STR    | `data/features.py` `build_behavior_seq`                   |
| event-type 손실 가중치 (cart=25, purchase=50) | 전 모델   | `conf/train/base.yaml` `loss_weights`                     |

---

### 추가 피처 엔지니어링 검토

#### ① category + brand 사이드 피처 — 기대 효과 **높음**

- **근거**: PLAN.md에 `ItemEmbedding` 설계(`cat_l2` 16dim + `cat_l3` 8dim + `brand` 32dim)가 이미 존재하나 미구현. `cat_l2` 17종(의류 유형)은 아웃핏 완성 패턴(jacket→shirt→trousers→belt)을 명시적으로 학습 가능. 희소 아이템의 표현 품질도 개선됨.
- **주의**: 아이템당 평균 283회 등장(8.35M ÷ 29,502)으로 item ID 임베딩이 카테고리 공동 출현 패턴을 이미 암묵적으로 학습함 → 한계 효과는 생각보다 작을 수 있음.
- **구현 방향**: 아이템 임베딩에 cat/brand 임베딩을 합산 (PLAN.md §메타데이터 활용 전략 참조).

```python
# 아이템 임베딩 확장 예시 (SASRec/TiSASRec에서 item_emb 교체)
item_repr = item_emb(item_id) + cat2_emb(cat2) + brand_emb(brand)
```

#### ② 가격대(price tier) 임베딩 — 기대 효과 **중간**

- **근거**: 데이터에 `price` 컬럼 존재하나 현재 미사용. 의류 도메인에서 유저의 가격 민감도는 강한 취향 신호 (저가/중가/고가 분위수 버킷화 → item side info 주입).
- **구현 방향**: `price`를 log-scale 또는 분위수(예: 10 buckets)로 버킷화 → `nn.Embedding(n_price_buckets, hidden_size)` → item 임베딩에 합산.

```python
# price 버킷화 예시
price_bucket = pd.cut(df['price'], bins=10, labels=False).fillna(0).astype(int)
```

#### ③ 유저 수준 집계 통계 — 기대 효과 **낮음**

- **한계**: purchase 이벤트가 0.02%로 극히 적어 유저별 카트 전환율·평균 구매 간격 등 통계가 대부분 유저에서 불안정함. 노이즈가 신호보다 클 가능성이 높음.
- **판단**: 구현 대비 효과 불투명, 후순위.

#### ④ 아이템 전역 인기도·추세 — 기대 효과 **낮음**

- **한계**: TIFU-KNN이 시간 감쇠 기반 아이템 스코어로 이미 커버. 추가 중복 효과 제한적.

#### ⑤ 세션 내 파생변수 — 기대 효과 **매우 낮음**

- **한계**: 세션의 99.76%가 단일 이벤트 타입. 세션 단위 피처는 대부분 trivial.

---

### 구현 우선순위 및 권장 순서

| 순위 | 피처                                    | 구현 복잡도             | 기대 효과 | 판단                         |
| ---- | --------------------------------------- | ----------------------- | --------- | ---------------------------- |
| 1    | `cat_l2` + `brand` 아이템 사이드 임베딩 | 중 (아키텍처 수정 필요) | 중~높음   | **앙상블 튜닝 완료 후 검토** |
| 2    | `price` 분위수 버킷 임베딩              | 낮음                    | 중        | cat/brand 실험 후 추가 비교  |
| 3    | 유저 집계 통계 / 세션 피처              | 높음                    | 낮음      | 보류                         |

**실행 순서 권장**: S2~S4(앙상블 가중치 최적화) 완료 → 성능 정체 시 cat_l2/brand 임베딩 실험 → Val NDCG 개선 ≥ 0.002이면 Full-train 반영.

| 문서 이력  |                                                                                                                                                          |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| 2026-05-20 | 초안 — Val 기준 체계적 튜닝 테이블                                                                                                                       |
| 2026-05-21 | S6 Full-train `best_epoch` 가이드 · CLI(`wandb.name`, eval_proxy ckpt) 코드 정합                                                                         |
| 2026-05-21 | 체크포인트 경로 형식 반영 (runNNN_YYMMDD/tuning                                                                                                          | full), 전체 구현 완료 기준 정리 |
| 2026-05-21 | 버그 수정 반영: MB-STR 패딩(0→3), saferec·mbstr submit/proxy 지원, TIFU-KNN top_k 보장, ensemble active_models 동적화, optimize_ensemble torch seed 추가 |
