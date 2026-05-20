# OPERATION — 추천 시스템 운영 가이드

| 항목 | 내용 |
|------|------|
| 대상 독자 | 실험·학습·제출을 수행하는 운영자 |
| 전제 | [`PLAN.md`](PLAN.md)에 정의된 **모든 기능이 구현·연결 완료**된 상태 |
| 관련 문서 | [PLAN.md](PLAN.md) (설계·EDA), [README.md](../README.md) (개요), [TUNING.md](TUNING.md) (튜닝) |
| 작업 디렉터리 | **항상 레포 루트** `recsys/` |

---

## 0. 운영 흐름 한눈에

대회 제출까지의 **권장 순서**입니다. 번호는 Phase와 대응합니다.

```
Phase 0  환경·데이터 확인
    ↓
Phase 1  검증 파이프라인 스모크 (SASRec 짧은 학습)
    ↓
Phase 2  베이스라인 모델 튜닝 (SASRec·TiSASRec·CL4SRec·FEARec, Val NDCG@10 기준)
    ↓
Phase 2b leaderboard_proxy 참고 평가 (선택, 설정 변경 금지)
    ↓
Phase 3  확장 모델 실험 (BSARec·SAFERec·MB-STR)
    ↓
Phase 4  TIFU-KNN 예측 생성 + optimize_ensemble.py로 앙상블 가중치 자동 최적화
    ↓
Phase 5  Full-train (cv=none, 전 기간, 선정 모델 전체)
    ↓
Phase 6  추론·앙상블·cart boost 후처리·제출 CSV 생성
          (cart boost는 ensemble_submit.py에 통합 — 별도 단계 불필요)
    ↓
대회 플랫폼 제출
```

**핵심 원칙**

| 구분 | 기간 | 역할 |
|------|------|------|
| **Train** | 11/01 ~ 02/08 | 학습 데이터 |
| **Val** | 02/09 ~ 02/22 | **튜닝·Early stopping·모델 선택** (`val/ndcg_cart_purchase`) |
| **leaderboard_proxy** | 02/23 ~ 02/29 | **참고용** 오프라인 NDCG (튜닝·가중치 변경 ❌) |
| **Full-train** | 11/01 ~ 02/29 | **최종 제출용** 체크포인트 (`cv=none`) |

---

## 공통 사전 준비

### 환경

```bash
cd /data/ephemeral/home/recsys

python -V                    # 3.10+ 권장
pip install -r requirements.txt

cp .env.template .env
# .env 편집: WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT
```

### GPU·VRAM 확인

```bash
nvidia-smi
```

| 모델 | 기본 `train_batch_size` | VRAM (BF16, max_seq_len=50, 대략) | 설정 위치 |
|------|-------------------------|-----------------------------------|-----------|
| SASRec | 4096 | ~6 GB | `conf/model/sasrec.yaml` |
| TiSASRec | 1024 | ~9 GB (time_matrix [B,L,L]) | `conf/model/tisasrec.yaml` |
| CL4SRec | 2048 | ~10 GB (대조 뷰 2개) | `conf/model/cl4srec.yaml` |
| FEARec | 2048 | ~10 GB (FFT 증강 뷰) | `conf/model/fearec.yaml` |
| BSARec | 4096 | ~6 GB | `conf/model/bsarec.yaml` |
| SAFERec | 4096 | ~6 GB | `conf/model/saferec.yaml` |
| MB-STR | 4096 | ~6 GB | `conf/model/mbstr.yaml` |
| TIFU-KNN | — (CPU) | < 1 GB | `src/train_tifu.py` |

OOM 시: `train.train_batch_size=` 로 **절반**씩 낮춰 재실행 (`conf/train/base.yaml`은 `${model.train_batch_size}` 참조).

### Hydra 실행 규칙

- **cwd = `recsys/`** (프로젝트 루트)
- 설정 오버라이드: `키=값` 형태 (`train.lr=0.0005`, `cv=none`)
- 실행 설정 스냅샷: `outputs/YYYY-MM-DD/HH-MM-SS/.hydra/config.yaml`
- Best 체크포인트: `outputs/<model>/runNNN_YYMMDD/tuning/best.pt` (튜닝·run **자동 증가**) / `.../full/best.pt` (Full-train·**최신 run**)

### wandb

```bash
# 오프라인/비활성
python src/train.py model=sasrec wandb.enabled=false

# 태그·run 이름 (권장)
python src/train.py model=tisasrec \
  wandb.tags=[tisasrec,phase2,tune] \
  wandb.name=tisasrec_seq50_ep300
```

**모니터링 지표**

| 지표 | 용도 |
|------|------|
| `val/ndcg_cart_purchase` | **튜닝·Early stopping 기준** |
| `val/ndcg_purchase_only` | 보조 (표본 ~37명, 참고만) |
| `val/gt_user_count` | GT 커버리지 |
| `leaderboard_proxy/ndcg` | 참고 (`cv.run_leaderboard_proxy=true`) |
| `train_loss` | 과적합 |
| `val/best_ndcg_cart_purchase` | Run 종료 시 best 요약 |

---

## Phase 0: 환경·데이터 파악

**목표**: 학습·제출 전 데이터·스키마·제약을 확인한다.

**예상 소요**: 약 0.5~1일

### 0-1. 학습 데이터 배치

```bash
ls -lh data/train.parquet data/sample_submission.csv
```

`data/train.parquet`가 없으면 대회에서 받은 파일을 `data/`에 둔다.

### 0-2. 스키마·기간·이벤트 분포 확인

```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/train.parquet')
df['event_time'] = pd.to_datetime(df['event_time'], utc=True)
print('rows:', len(df))
print('users:', df['user_id'].nunique())
print('items:', df['item_id'].nunique())
print('period:', df['event_time'].min(), '~', df['event_time'].max())
print(df['event_type'].value_counts(normalize=True))
"
```

**기대값 (PLAN EDA)**

- 유저 **638,257** / 아이템 **29,502** / 기간 **2019-11-01 ~ 2020-02-29**
- view ~99.78% / cart ~0.20% / purchase ~0.02%

### 0-3. 제출 포맷 검증

```bash
python -c "
import pandas as pd
sub = pd.read_csv('data/sample_submission.csv')
n_users = sub['user_id'].nunique()
assert len(sub) == n_users * 10
print('OK:', len(sub), 'rows,', n_users, 'users')
"
```

### 0-4. (선택) EDA 노트북

```bash
jupyter notebook EDA/eda.ipynb
# 또는 baseline_code/EDA.ipynb
```

### 0-5. (선택) ALS 베이스라인 — 협업필터 강도 파악

```bash
cd baseline_code
pip install -r requirements.txt
python train_als.py
cd ..
```

### Phase 0 완료 체크리스트

- [ ] `train.parquet` 존재, 기간·행 수 일치
- [ ] 제출 유저 ⊆ 학습 유저 (콜드 유저 없음)
- [ ] `sample_submission.csv` 롱 포맷 (유저당 10행)
- [ ] GPU·wandb·`.env` 설정 완료

---

## Phase 1: 검증 프로토콜 확립 (스모크)

**목표**: Holdout CV + NDCG@10 + wandb + 체크포인트 파이프라인이 **누수 없이** 동작함을 확인한다.

**예상 소요**: 약 0.5~1일

### 1-1. 짧은 SASRec 스모크 (5 epoch)

```bash
python src/train.py \
  model=sasrec \
  model.max_seq_len=50 \
  model.hidden_size=64 \
  model.n_layers=2 \
  train.epochs=5 \
  train.early_stopping_patience=99 \
  wandb.tags=[smoke,phase1] \
  wandb.name=sasrec_smoke5ep
```

**확인 사항**

- 콘솔에 `ndcg_cp` (cart+purchase NDCG) 출력
- `outputs/sasrec/runNNN_YYMMDD/tuning/best.pt` 생성
- wandb에 `val/ndcg_cart_purchase` 곡선
- `outputs/<날짜>/<시간>/.hydra/config.yaml` 저장

### 1-2. Holdout 분할 상수 확인

`src/cv/holdout.py`:

- `VAL_START = 2020-02-09`
- `VAL_END   = 2020-02-23` (exclusive → Val은 **02/22까지**)

### Phase 1 완료 체크리스트

- [ ] Val GT `cart_purchase` 유저 수 ~1,000명대 (`val/gt_user_count`)
- [ ] Early stopping이 `val/ndcg_cart_purchase` 기준으로 best 저장
- [ ] 재실행 시 동일 `seed=42`로 유사 점수 재현

---

## Phase 2: 베이스라인 강화 (튜닝)

**목표**: SASRec · TiSASRec · CL4SRec · (선택) FEARec을 **Val NDCG@10**으로 튜닝하고 best 설정을 확정한다.

**예상 소요**: 약 2~4일 (모델·sweep 수에 따라)

**공통 CV 모드** (기본값 `conf/cv/single_holdout.yaml`):

```bash
# cv 명시 생략 가능 (defaults에 single_holdout)
python src/train.py model=<이름> ...
```

### 2-1. SASRec 전체 튜닝

```bash
python src/train.py \
  model=sasrec \
  train.epochs=300 \
  train.early_stopping_patience=20 \
  wandb.tags=[sasrec,phase2,tune] \
  wandb.name=sasrec_full
```

**탐색 예시 (수동 grid)**

```bash
# max_seq_len
for L in 50 100; do
  python src/train.py model=sasrec model.max_seq_len=$L \
    wandb.name=sasrec_len${L}
done

# loss cart 가중 (10~50)
for C in 10 25 50; do
  python src/train.py model=sasrec \
    train.loss_weights.cart=$C \
    wandb.name=sasrec_cart${C}
done
```

### 2-2. TiSASRec (시간 간격 CV=4.12 — 필수)

```bash
python src/train.py \
  model=tisasrec \
  model.time_span=512 \
  train.epochs=300 \
  train.early_stopping_patience=20 \
  wandb.tags=[tisasrec,phase2,tune] \
  wandb.name=tisasrec_full
```

### 2-3. CL4SRec (희소성 대응)

```bash
python src/train.py \
  model=cl4srec \
  model.lmd=0.1 \
  train.epochs=300 \
  train.early_stopping_patience=20 \
  wandb.tags=[cl4srec,phase2,tune] \
  wandb.name=cl4srec_full
```

### 2-4. FEARec (FFT 주파수 증강 + InfoNCE 대조학습)

```bash
python src/train.py \
  model=fearec \
  train.epochs=300 \
  train.early_stopping_patience=20 \
  wandb.tags=[fearec,phase2,tune] \
  wandb.name=fearec_full
```

### 2-5. (선택) spike ablation

Feb 27~29 구매 spike 포함 여부 비교:

```bash
# 기본: spike 포함
python src/train.py model=tisasrec data=base

# ablation: spike 제거
python src/train.py model=tisasrec data=spike_excluded \
  wandb.tags=[ablation,spike_excluded]
```

### 2-6. wandb Sweep (loss·seq_len 등)

`conf/sweep/`에 sweep YAML이 있다고 가정 (PLAN 완료 시):

```bash
wandb sweep conf/sweep/cart_weight.yaml
wandb agent <entity>/recsys-2026/<sweep_id>
```

또는 단일 파라미터 CLI 오버라이드로 grid 수행 (위 2-1 예시).

### Phase 2 완료 체크리스트

- [ ] 모델별 `outputs/<model>/runNNN_YYMMDD/tuning/best.pt` 및 wandb best `val/ndcg_cart_purchase` 기록
- [ ] Val 기준 **상위 1~2개 설정** 후보 목록 정리 (Hydra config 스냅샷 경로 기록)
- [ ] purchase-only NDCG만 보고 모델을 고르지 않았는지 확인

---

## Phase 2b: leaderboard_proxy (참고 평가)

**목표**: 확정 후보 설정에 대해 **2/23~29 구간** 오프라인 NDCG를 **1~2회**만 확인한다.

**주의**: proxy 점수로 하이퍼파라미터·앙상블 가중치를 **변경하지 않는다**.

```bash
python src/train.py \
  model=tisasrec \
  cv.run_leaderboard_proxy=true \
  train.epochs=1 \
  train.early_stopping_patience=99 \
  wandb.tags=[proxy,sanity] \
  wandb.name=tisasrec_proxy_check
```

**이미 학습된 체크포인트만** 평가 (재학습 없음):

```bash
python src/eval_proxy.py model=tisasrec
python src/eval_proxy.py model=tisasrec ckpt_path=outputs/tisasrec/runNNN_YYMMDD/tuning/best.pt
python src/eval_proxy.py model=cl4srec wandb.enabled=true wandb.name=cl4srec_proxy_only
```

- `cv=none` Full-train 체크포인트도 사용 가능
- 기본 경로: 최신 run의 `tuning/best.pt` (`eval_proxy`) · `full/best.pt` (`submit`)

**해석**

| 패턴 | 의미 | 조치 |
|------|------|------|
| Val↑ Proxy↑ | 평시·스파이크 주간 모두 양호 | Full-train 후보로 유지 |
| Val↑ Proxy↓ | 평소 패턴에 과적합 가능 | TIFU 비중↓, MB-STR·cart 부스트 검토 |
| Val↓ Proxy↑ | Val 과소평가 가능 | Full-train·후처리 비중↑ (설정은 Val 기준 유지) |

---

## Phase 3: 확장 모델 실험

**목표**: BSARec · SAFERec · MB-STR 등 PLAN 우선순위 모델을 동일 Holdout으로 비교한다.

**예상 소요**: 약 3~5일

### 3-1. BSARec

```bash
python src/train.py \
  model=bsarec \
  train.train_batch_size=4096 \
  train.epochs=300 \
  wandb.tags=[bsarec,phase3] \
  wandb.name=bsarec_full
```

### 3-2. SAFERec (view 빈도 피처)

```bash
python src/train.py \
  model=saferec \
  train.train_batch_size=4096 \
  wandb.tags=[saferec,phase3] \
  wandb.name=saferec_full
```

### 3-3. MB-STR (cart→purchase 직접 모델링 — 권장)

```bash
python src/train.py \
  model=mbstr \
  train.train_batch_size=4096 \
  train.loss_weights.cart=25.0 \
  train.loss_weights.purchase=50.0 \
  wandb.tags=[mbstr,phase3] \
  wandb.name=mbstr_full
```

### 3-4. 모델 선택 기준

wandb Table 또는 스프레드시트에 정리:

| model | val/ndcg_cart_purchase | proxy/ndcg (참고) | 비고 |
|-------|------------------------|-------------------|------|
| tisasrec | | | 시간 CV=4.12 |
| cl4srec | | | 희소성 |
| bsarec | | | 주파수 |
| mbstr | | | cart→purchase |

**Full-train 대상**: Val 상위 **3~4개** 딥 모델 + (선택) TIFU-KNN.

---

## Phase 4: TIFU-KNN · 앙상블 가중치 최적화

**목표**: TIFU-KNN 예측 생성 후, **Val NDCG@10** 기준 랜덤 서치로 최적 앙상블 가중치를 탐색한다.

**예상 소요**: 약 1~2일

### 4-1. TIFU-KNN 예측 생성 (전체 학습 데이터 기준)

```bash
# Full-train 제출용 예측 생성 (전체 df 기준)
python src/train_tifu.py
# → outputs/tifu_knn/preds.pkl
```

하이퍼파라미터는 `conf/ensemble/rank.yaml`의 `tifu_group_count / tifu_decay_within / tifu_decay_across` 로 제어.

### 4-2. 앙상블 가중치 자동 최적화 (Val 기준 랜덤 서치)

```bash
# 300회(기본) 랜덤 서치 → conf/ensemble/rank.yaml 자동 업데이트
python src/optimize_ensemble.py

# 시도 횟수·시드 변경
python src/optimize_ensemble.py n_trials=500 seed=0
```

현재 초기값 [`conf/ensemble/rank.yaml`](../conf/ensemble/rank.yaml):

```yaml
weights:
  sasrec:   1.00
  tisasrec: 0.35
  cl4srec:  0.20
  fearec:   0.30
  bsarec:   0.25
  saferec:  0.20
  mbstr:    0.40
  tifu_knn: 0.15
top_k: 10
cart_boost: true
tifu_group_count: 7
tifu_decay_within: 0.9
tifu_decay_across: 0.7
```

`optimize_ensemble.py`는 체크포인트가 없는 모델을 자동 스킵하고 사용 가능한 모델만으로 최적화를 수행합니다.

### 4-3. 앙상블 결과 확인

```bash
# 최적화된 가중치로 앙상블 제출 파일 생성
python src/ensemble_submit.py
```

### Phase 4 완료 체크리스트

- [ ] `outputs/tifu_knn/preds.pkl` 생성 완료
- [ ] `optimize_ensemble.py` 실행 후 `conf/ensemble/rank.yaml` 갱신
- [ ] Val만으로 가중치 확정, Proxy는 기록만
- [ ] `cart_boost: true` (기본값) 유지 확인

---

## Phase 5: Full-train (최종 제출용 학습)

**목표**: Phase 2~4에서 확정한 설정으로 **11/01~02/29 전체**를 학습한다.

**예상 소요**: 모델당 수 시간~1일 (epoch·GPU에 따라)

### 5-1. CV 끄고 전 기간 학습

```bash
# 모델별 Full-train (병렬 GPU가 있으면 동시 실행 가능)
python src/train.py model=sasrec   cv=none wandb.tags=[fulltrain]
python src/train.py model=tisasrec cv=none wandb.tags=[fulltrain]
python src/train.py model=cl4srec  cv=none wandb.tags=[fulltrain]
python src/train.py model=fearec   cv=none wandb.tags=[fulltrain]
python src/train.py model=bsarec   cv=none wandb.tags=[fulltrain]
python src/train.py model=saferec  cv=none wandb.tags=[fulltrain]
python src/train.py model=mbstr    cv=none wandb.tags=[fulltrain]
```

**동작 (`cv=none`)**

- Train/Val 분할 없음 → **전체 120일** 시퀀스 학습
- Val NDCG 없음 → 마지막 epoch(또는 설정된 epoch) 체크포인트를 `outputs/<model>/runNNN_YYMMDD/full/best.pt`에 저장
- Feb 27~29 **spike 포함** (기본 `data=base`)

### 5-2. TIFU-KNN Full 예측 (전체 df)

```bash
# Full-train 후 전체 df로 TIFU-KNN 예측 재생성
python src/train_tifu.py
# → outputs/tifu_knn/preds.pkl (전체 유저 대상)
```

### 5-3. 배치 스크립트 예시

```bash
bash run_tisasrec_cl4srec.sh   # Val 모드 예시 — Full-train용으로 cv=none 추가 권장
```

Full-train 전용 예:

```bash
#!/bin/bash
set -e
cd /data/ephemeral/home/recsys

python src/train.py model=tisasrec cv=none train.epochs=300
python src/train.py model=cl4srec  cv=none train.epochs=300
```

### Phase 5 완료 체크리스트

- [ ] 제출에 쓸 모든 모델의 `outputs/<model>/runNNN_YYMMDD/full/best.pt`가 **Full-train**으로 갱신됨
- [ ] Val용 체크포인트를 제출에 쓰지 않았는지 확인 (경로·wandb tag `fulltrain`)
- [ ] `exclude_spike_purchase: false` (기본) 유지

---

## Phase 6: 추론·제출

**목표**: 638,257 유저 × Top-10 롱 포맷 CSV를 생성하고 검증한 뒤 대회에 제출한다.

**예상 소요**: 약 0.5~1일

### 6-1. 단일 모델 제출

```bash
python src/submit.py \
  model=tisasrec \
  ckpt_path=outputs/tisasrec/runNNN_YYMMDD/full/best.pt
# → outputs/submission_tisasrec.csv
```

### 6-2. 랭크 앙상블 제출

```bash
python src/ensemble_submit.py
# → outputs/submission_ensemble_<모델목록>.csv
```

가중치를 직접 조정하려면 `conf/ensemble/rank.yaml`을 편집 후 재실행합니다.

**사전 조건**: `outputs/<model>/runNNN_YYMMDD/full/best.pt` 존재 (Full-train 완료 필수).

### 6-3. 제출 파일 검증

`ensemble_submit.py`가 내부적으로 `validate_submission`을 호출하지만, 직접 확인할 수도 있습니다.

```bash
# 실제 파일명은 사용된 모델 목록에 따라 달라짐
ls outputs/submission_ensemble_*.csv

python -c "
import pandas as pd, glob
p = sorted(glob.glob('outputs/submission_ensemble_*.csv'))[-1]
df = pd.read_csv(p)
n = df['user_id'].nunique()
assert len(df) == n * 10, f'행 수 불일치: {len(df)} != {n*10}'
assert (df.groupby('user_id').size() == 10).all()
assert (df.groupby('user_id')['item_id'].nunique() == 10).all()
print('검증 통과:', p, len(df), 'rows')
"
```

### 6-4. 역매핑 샘플 수동 확인

```bash
python -c "
import pandas as pd
sub = pd.read_csv('outputs/submission_tisasrec.csv').head(20)
train = pd.read_parquet('data/train.parquet', columns=['user_id','item_id'])
print(sub)
print('user in train:', sub['user_id'].iloc[0] in set(train['user_id']))
"
```

### Phase 6 완료 체크리스트

- [ ] 행 수 **6,382,570** (= 638,257 × 10)
- [ ] 유저당 정확히 10행, item_id 중복 없음
- [ ] UUID가 train과 일치 (내부 idx 그대로 제출 ❌)

---

## Phase 6-B: cart 후처리 부스트

**목표**: “장바구니에 담았으나 아직 구매하지 않은” 아이템을 Top-K 상위로 끌어올린다.

**구현**: `ensemble_submit.py` 내 `_cart_boost()` 함수로 **통합 완료**.  
`conf/ensemble/rank.yaml`의 `cart_boost: true` (기본값)로 활성화.

### cart boost 동작 원리

1. 전체 df에서 유저별 carted-but-not-purchased 아이템 집합 계산
2. 앙상블 예측 리스트에서 cart 아이템을 앞으로 이동 (예측에 있으면 순서 유지, 없으면 item_id 정렬로 추가)
3. 나머지 예측 아이템으로 Top-10 채움

### cart boost 끄기 (비교 실험용)

```bash
# conf/ensemble/rank.yaml에서 cart_boost: false 설정 후
python src/ensemble_submit.py
```

**누수 방지**: `ensemble_submit.py`는 전체 df를 cart_boost 기준으로 사용.  
Val 평가(`optimize_ensemble.py`)에서는 train_df(Val 이전)만 사용.

### Phase 6-B 완료 체크리스트

- [ ] `conf/ensemble/rank.yaml`에 `cart_boost: true` 확인
- [ ] 앙상블 실행 로그에 “Cart boost 후처리 중...” 출력 확인
- [ ] 부스트 후에도 10개·중복 없음 유지 (`validate_submission` 통과)

---

## 최종 제출 Runbook (요약)

한 번의 제출 사이클을 처음부터 끝까지 정리한 체크리스트입니다.

```bash
cd /data/ephemeral/home/recsys

# ① 튜닝 (이미 완료했다면 생략)
python src/train.py model=sasrec   wandb.name=tune_final
python src/train.py model=tisasrec wandb.name=tune_final
python src/train.py model=cl4srec  wandb.name=tune_final
python src/train.py model=mbstr    wandb.name=tune_final
# 추가 모델: fearec / bsarec / saferec

# ② 앙상블 가중치 최적화 (Val 기준)
python src/optimize_ensemble.py

# ③ proxy sanity (선택, 1회)
python src/eval_proxy.py model=tisasrec

# ④ Full-train (앙상블에 사용할 모든 모델)
python src/train.py model=tisasrec cv=none
python src/train.py model=cl4srec  cv=none
python src/train.py model=mbstr    cv=none

# ⑤ TIFU-KNN 예측 재생성 (전체 df 기준)
python src/train_tifu.py

# ⑥ 앙상블 제출 (cart boost 자동 적용)
python src/ensemble_submit.py
# → outputs/submission_ensemble_<사용된모델>.csv

# ⑦ 검증 후 대회 사이트 업로드
```

---

## 장애 대응 (Troubleshooting)

| 증상 | 원인 | 조치 |
|------|------|------|
| CUDA OOM | batch 과대 | `train.train_batch_size` 절반, TiSASRec≤1024, CL4SRec≤2048 |
| NaN loss | mask 이중 적용 등 | PLAN: key_padding_mask 제거·causal only 확인 |
| Val NDCG 비정상적으로 높음 | val_df로 시퀀스 빌드 (누수) | `build_sequences(train_df)`만 사용 |
| 제출 행 수 불일치 | 유저 누락·중복 | `validate_submission` 재실행 |
| wandb 미로깅 | API 키·enabled | `.env`, `wandb.enabled=true` |
| Hydra config not found | cwd 오류 | 반드시 `recsys/`에서 실행 |
| 앙상블 스킵 | ckpt 없음 | Phase 5 Full-train 완료 여부 확인 |

---

## CLI 빠른 참조

### 학습

| 목적 | 명령 |
|------|------|
| 기본 튜닝 | `python src/train.py model=<name>` |
| Full-train | `python src/train.py model=<name> cv=none` |
| Proxy (학습 직후) | `python src/train.py model=<name> cv.run_leaderboard_proxy=true` |
| Proxy만 (ckpt) | `python src/eval_proxy.py model=<name> [ckpt_path=...]` |
| spike 제외 | `python src/train.py data=spike_excluded` |
| wandb 끄기 | `python src/train.py wandb.enabled=false` |

### 제출 및 앙상블

| 목적 | 명령 |
|------|------|
| 단일 모델 | `python src/submit.py model=<name>` |
| TIFU-KNN 예측 생성 | `python src/train_tifu.py` |
| 앙상블 가중치 최적화 | `python src/optimize_ensemble.py` |
| 앙상블 (cart boost 포함) | `python src/ensemble_submit.py` |

### 지원 `model=` 이름 (구현 완료)

`sasrec` · `tisasrec` · `cl4srec` · `fearec` · `bsarec` · `saferec` · `mbstr`  
비신경망: `tifu_knn` (`train_tifu.py` 단독 실행, 앙상블에서 자동 사용)

---

## 산출물 경로

| 경로 | 설명 |
|------|------|
| `outputs/<model>/runNNN_YYMMDD/tuning/best.pt` | 튜닝 ckpt |
| `outputs/<model>/runNNN_YYMMDD/full/best.pt` | Full-train ckpt |
| `outputs/submission_<model>.csv` | 단일 모델 제출 |
| `outputs/submission_ensemble_*.csv` | 앙상블 제출 |
| `outputs/YYYY-MM-DD/HH-MM-SS/.hydra/` | 실행별 설정 스냅샷 |
| `outputs/tifu_knn/preds.pkl` | TIFU-KNN 예측 캐시 (`train_tifu.py`) |
| wandb project `recsys-2026` | 실험 이력 |

---

## 리더보드 직전 최종 점검

[`PLAN.md` 리더보드 체크리스트](PLAN.md#리더보드와-직결되는-체크리스트) 요약:

- [ ] Full-train = Nov~Feb 29, spike **포함** (`exclude_spike_purchase: false`)
- [ ] 튜닝 기준 = `val/ndcg_cart_purchase` (cart+purchase GT)
- [ ] proxy로 설정을 바꾸지 않았음
- [ ] 제출 6,382,570행, 유저당 10개, item 중복 없음
- [ ] `user_id` / `item_id` 원본 UUID
- [ ] BF16 AMP, cart_boost **전체 df 기준** (Full-train 체크포인트 사용 시 누수 없음)
- [ ] wandb에 최종 run·config 경로 기록

---

## 문서 이력

| 날짜 | 내용 |
|------|------|
| 2026-05-20 | 초안 — PLAN.md Phase 0~6-B 기준 운영 가이드 작성 |
| 2026-05-20 | 8개 모델 반영 (FEARec·BSARec·SAFERec·MB-STR·TIFU-KNN), train_tifu.py·optimize_ensemble.py 추가, cart boost 통합 업데이트 |
| 2026-05-21 | 체크포인트 경로 `outputs/<model>/best.pt` → `outputs/<model>/runNNN_YYMMDD/{tuning\|full}/best.pt` 형식으로 전면 반영 |
