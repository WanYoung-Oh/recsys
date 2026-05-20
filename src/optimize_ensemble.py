"""앙상블 가중치 최적화 스크립트.

각 모델의 val 예측을 메모리에 올린 뒤 랜덤/그리드 서치로
val NDCG@10 (cart+purchase) 기준 최적 가중치를 탐색하고
conf/ensemble/rank.yaml을 업데이트한다.

실행:
  python src/optimize_ensemble.py
  python src/optimize_ensemble.py n_trials=500 seed=0
"""

from __future__ import annotations

import os
import sys
import random
import numpy as np
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_val_preds(model_name: str, val_seqs, val_user_ids, gt_cp,
                    idx2item, device, amp_dtype, cfg_base, df=None,
                    item2idx=None) -> dict | None:
    """체크포인트 로드 → val 예측 dict 반환. 체크포인트 없으면 None."""
    from util.paths import get_checkpoint_path
    from models import build_model
    from data.features import (build_time_seq, build_freq_seq, build_behavior_seq,
                                compute_item_freq)
    from data.dataset import build_sequences, build_val_sequences

    def load_cfg(name):
        model_cfg = OmegaConf.load(ROOT / "conf" / "model" / f"{name}.yaml")
        data_cfg  = OmegaConf.load(ROOT / "conf" / "data"  / "base.yaml")
        train_cfg = OmegaConf.load(ROOT / "conf" / "train" / "base.yaml")
        data_cfg.data_dir = str(ROOT / data_cfg.data_dir)
        return OmegaConf.create({
            "model": model_cfg, "data": data_cfg, "train": train_cfg,
            "cv": {"enabled": False}, "run_id": None, "run_date": None,
            "ckpt_path": None, "checkpoint": {"load_phase": "full"},
        })

    m_cfg    = load_cfg(model_name)
    ckpt_p   = get_checkpoint_path(m_cfg, str(ROOT), for_training=False)
    if not ckpt_p.exists():
        print(f"  ⚠ {model_name} 체크포인트 없음 — 스킵")
        return None

    model = build_model(m_cfg.model, cfg_base["n_items"]).to(device)
    ckpt  = torch.load(ckpt_p, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    seq_len = m_cfg.model.max_seq_len
    m_seqs     = build_sequences(df, item2idx, seq_len)
    m_val_seqs = build_val_sequences(m_seqs, seq_len)

    extra: dict = {}
    if model_name == "tisasrec":
        extra["time_sequences"] = build_time_seq(m_seqs, seq_len)
    elif model_name == "saferec":
        n_buck = getattr(m_cfg.model, "n_freq_buckets", 64)
        freq   = compute_item_freq(df, item2idx, n_buck)
        extra["freq_sequences"] = build_freq_seq(m_seqs, seq_len, freq)
    elif model_name == "mbstr":
        extra["behavior_sequences"] = build_behavior_seq(m_seqs, seq_len)

    from inference import generate_predictions
    preds = generate_predictions(
        model, val_user_ids.tolist(), m_val_seqs, idx2item, device,
        batch_size=cfg_base["eval_batch_size"],
        amp_dtype=amp_dtype,
        **extra,
    )
    del model
    torch.cuda.empty_cache()
    return preds


def _tifu_val_preds(seqs, val_user_ids, idx2item) -> dict:
    from models.tifu_knn import TIFUKNN
    tifu = TIFUKNN()
    tifu.fit(seqs, idx2item)
    return tifu.predict_all(val_user_ids.tolist())


def _score(preds_dict: dict, weights: dict, gt_cp: dict,
           top_k: int = 10) -> float:
    from ensemble import rank_ensemble
    from metrics  import ndcg_at_k
    ens = rank_ensemble(preds_dict, weights, top_k=top_k)
    scores = [ndcg_at_k(ens.get(uid, []), gt_cp[uid], top_k)
              for uid in gt_cp if uid in ens]
    return float(np.mean(scores)) if scores else 0.0


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if cfg.train.amp == "bf16" else torch.float16

    orig_cwd = hydra.utils.get_original_cwd()
    cfg.data.data_dir = os.path.join(orig_cwd, cfg.data.data_dir)

    from data.dataset import load_data, build_vocab, build_sequences, build_val_sequences
    from cv.holdout   import make_holdout, build_gt

    print("▶ 데이터 로드 중...")
    df = load_data(cfg)
    item2idx, user2idx, idx2item = build_vocab(df)
    n_items = len(item2idx)

    train_df, val_df = make_holdout(df)
    gt_cp        = build_gt(val_df, mode="cart_purchase")
    val_user_ids = np.array(list(val_df["user_id"].unique()))
    print(f"  GT 유저 수: {len(gt_cp):,}")

    base_seq = build_sequences(train_df, item2idx, cfg.model.max_seq_len)
    val_seqs = build_val_sequences(base_seq, cfg.model.max_seq_len)

    cfg_base = {
        "n_items":        n_items,
        "eval_batch_size": cfg.train.eval_batch_size,
    }

    ensemble_cfg = OmegaConf.load(ROOT / "conf" / "ensemble" / "rank.yaml")
    candidate_models = [m for m in ensemble_cfg.weights if m != "tifu_knn"]

    print("▶ 모델별 val 예측 수집 중...")
    all_preds: dict[str, dict] = {}
    for mn in candidate_models:
        p = _load_val_preds(mn, val_seqs, val_user_ids, gt_cp,
                            idx2item, device, amp_dtype, cfg_base,
                            df=train_df, item2idx=item2idx)
        if p is not None:
            all_preds[mn] = p

    all_preds["tifu_knn"] = _tifu_val_preds(base_seq, val_user_ids, idx2item)

    if not all_preds:
        raise RuntimeError("예측 가능한 모델이 없습니다.")

    active = list(all_preds.keys())
    print(f"  최적화 대상 모델: {active}")

    n_trials = cfg.get("n_trials", 300)
    seed     = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    best_score  = -1.0
    best_weights = {m: 1.0 for m in active}

    print(f"▶ 랜덤 서치 {n_trials}회 시작...")
    for t in range(n_trials):
        w = {m: random.uniform(0.05, 1.0) for m in active}
        s = _score(all_preds, w, gt_cp)
        if s > best_score:
            best_score   = s
            best_weights = w
            print(f"  [{t:4d}] NDCG@10={s:.4f}  {w}")

    print(f"\n▶ 최적 가중치 (NDCG@10={best_score:.4f}):")
    for m, w in sorted(best_weights.items()):
        print(f"  {m}: {w:.4f}")

    # conf/ensemble/rank.yaml 업데이트
    rank_cfg_path = ROOT / "conf" / "ensemble" / "rank.yaml"
    rank_cfg = OmegaConf.load(rank_cfg_path)
    for m, w in best_weights.items():
        rank_cfg.weights[m] = round(float(w), 4)
    # 기존 가중치 키 중 active에 없는 것은 유지
    OmegaConf.save(rank_cfg, rank_cfg_path)
    print(f"▶ conf/ensemble/rank.yaml 업데이트 완료")


if __name__ == "__main__":
    main()
