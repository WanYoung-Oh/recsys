"""3개 모델 체크포인트 로드 → 랭크 앙상블 → 제출 CSV 저장.

실행:
  python src/ensemble_submit.py
"""

import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_cfg(model_name: str, run_id=None, checkpoint_phase: str = "full"):
    model_cfg = OmegaConf.load(ROOT / "conf" / "model" / f"{model_name}.yaml")
    data_cfg  = OmegaConf.load(ROOT / "conf" / "data"  / "base.yaml")
    train_cfg = OmegaConf.load(ROOT / "conf" / "train" / "base.yaml")
    data_cfg.data_dir = str(ROOT / data_cfg.data_dir)
    return OmegaConf.create({
        "model": model_cfg,
        "data":  data_cfg,
        "train": train_cfg,
        "cv": {"enabled": False},
        "run_id": run_id,
        "run_date": None,
        "ckpt_path": None,
        "checkpoint": {"load_phase": checkpoint_phase},
    })


def resolve_ensemble_ckpt(model_name: str, ensemble_cfg) -> Path:
    from util.paths import get_checkpoint_path

    run_id = ensemble_cfg.get("run_id")
    phase = ensemble_cfg.get("checkpoint_phase", "full")
    cfg = load_cfg(model_name, run_id=run_id, checkpoint_phase=phase)
    return get_checkpoint_path(cfg, str(ROOT), for_training=False, load_phase=phase)


def _cart_boost(
    preds: dict,
    df,
    item2idx: dict,
    idx2item: dict,
    top_k: int = 10,
) -> dict:
    """carted-but-not-purchased 아이템을 예측 상위로 이동.

    전체 기간 기준으로 cart는 있으나 purchase가 없는 아이템을 유저별로 수집,
    해당 아이템을 앙상블 예측 리스트 맨 앞에 배치하고 나머지를 채워 top_k를 맞춤.
    """
    import pandas as pd

    cart_df     = df[df["event_type"] == "cart"]
    purchase_df = df[df["event_type"] == "purchase"]

    carted    = cart_df.groupby("user_id")["item_id"].apply(set).to_dict()
    purchased = purchase_df.groupby("user_id")["item_id"].apply(set).to_dict()

    boosted: dict = {}
    for uid, ranked in preds.items():
        cart_items = carted.get(uid, set()) - purchased.get(uid, set())
        if not cart_items:
            boosted[uid] = ranked
            continue
        # cart 아이템 중 현재 예측에 있는 것 우선, 없으면 item_id 정렬로 순서 결정
        in_pred    = [i for i in ranked if i in cart_items]
        not_in_pred = sorted(cart_items - set(ranked))
        boost_list = in_pred + not_in_pred
        remaining  = [i for i in ranked if i not in cart_items]
        merged     = boost_list + remaining
        boosted[uid] = merged[:top_k]
    return boosted


def main():
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16

    ensemble_cfg = OmegaConf.load(ROOT / "conf" / "ensemble" / "rank.yaml")
    # 현재 학습 완료된 모델만 사용 (tifu_knn 제외)
    active_models = ["sasrec", "tisasrec", "cl4srec"]
    raw_weights   = {m: ensemble_cfg.weights[m] for m in active_models}
    total = sum(raw_weights.values())
    weights = {m: w / total for m, w in raw_weights.items()}
    print(f"▶ 앙상블 가중치 (정규화): { {m: f'{w:.3f}' for m, w in weights.items()} }")

    # ── 데이터 공통 로드 ──────────────────────────────────────────
    from data.dataset import load_data, build_vocab, build_sequences, build_val_sequences
    from data.features import (build_time_seq, build_freq_seq, build_behavior_seq,
                                compute_item_freq)
    from models        import build_model
    from inference     import generate_predictions, generate_submission_long, validate_submission
    from ensemble      import rank_ensemble

    cfg = load_cfg("sasrec")   # data/train cfg는 모델 무관하게 동일
    print("▶ 데이터 로드 중...")
    df = load_data(cfg)
    item2idx, user2idx, idx2item = build_vocab(df)
    n_items      = len(item2idx)
    all_user_ids = list(user2idx.keys())
    print(f"  유저 {len(user2idx):,} / 아이템 {n_items:,}")

    base_seq_len = cfg.model.max_seq_len  # sasrec 기준 50
    seqs     = build_sequences(df, item2idx, base_seq_len)
    val_seqs = build_val_sequences(seqs, base_seq_len)

    # ── TIFU-KNN (non-neural) ────────────────────────────────────
    tifu_preds: dict = {}
    if "tifu_knn" in active_models:
        from models.tifu_knn import TIFUKNN
        print("▶ TIFU-KNN 예측 중...")
        tifu = TIFUKNN(
            group_count=ensemble_cfg.get("tifu_group_count", 7),
            decay_within=ensemble_cfg.get("tifu_decay_within", 0.9),
            decay_across=ensemble_cfg.get("tifu_decay_across", 0.7),
        )
        tifu.fit(seqs, idx2item)
        tifu_preds = tifu.predict_all(all_user_ids, top_k=ensemble_cfg.top_k)

    # ── 모델별 추론 ───────────────────────────────────────────────
    model_preds = {}
    for model_name in active_models:
        if model_name == "tifu_knn":
            if tifu_preds:
                model_preds[model_name] = tifu_preds
            continue

        ckpt_path = resolve_ensemble_ckpt(model_name, ensemble_cfg)
        if not ckpt_path.exists():
            print(f"  ⚠ {model_name} 체크포인트 없음 ({ckpt_path}) — 스킵")
            continue

        model_cfg = load_cfg(model_name)
        model = build_model(model_cfg.model, n_items).to(device)
        ckpt  = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        ndcg = ckpt.get("ndcg", 0.0)
        print(f"▶ {model_name} 추론 중... (best val NDCG@10={ndcg:.4f})")

        # max_seq_len이 다르면 전용 시퀀스 빌드
        model_seq_len = model_cfg.model.max_seq_len
        if model_seq_len != base_seq_len:
            m_seqs     = build_sequences(df, item2idx, model_seq_len)
            m_val_seqs = build_val_sequences(m_seqs, model_seq_len)
        else:
            m_seqs     = seqs
            m_val_seqs = val_seqs

        # 모델별 보조 시퀀스 준비
        extra_kwargs: dict = {}
        if model_name == "tisasrec":
            extra_kwargs["time_sequences"] = build_time_seq(m_seqs, model_seq_len)
        elif model_name == "saferec":
            n_buckets = getattr(model_cfg.model, "n_freq_buckets", 64)
            item_freq = compute_item_freq(df, item2idx, n_buckets)
            extra_kwargs["freq_sequences"] = build_freq_seq(m_seqs, model_seq_len, item_freq)
        elif model_name == "mbstr":
            extra_kwargs["behavior_sequences"] = build_behavior_seq(m_seqs, model_seq_len)

        preds = generate_predictions(
            model, all_user_ids, m_val_seqs, idx2item, device,
            batch_size=cfg.train.eval_batch_size,
            amp_dtype=amp_dtype,
            **extra_kwargs,
        )
        model_preds[model_name] = preds
        del model
        torch.cuda.empty_cache()

    if not model_preds:
        raise RuntimeError("추론 가능한 모델이 없습니다. 체크포인트를 확인하세요.")

    used_models = list(model_preds.keys())
    if len(used_models) < len(active_models):
        # 스킵된 모델 제외 후 가중치 재정규화
        weights = {m: weights[m] for m in used_models}
        total   = sum(weights.values())
        weights = {m: w / total for m, w in weights.items()}
        print(f"  재정규화 가중치: { {m: f'{w:.3f}' for m, w in weights.items()} }")

    # ── 앙상블 ────────────────────────────────────────────────────
    print("▶ 랭크 앙상블 중...")
    ensemble_preds = rank_ensemble(model_preds, weights, top_k=ensemble_cfg.top_k)

    # ── cart boost 후처리 ─────────────────────────────────────────
    # 유저의 cart/purchase 이벤트 아이템 중 아직 구매하지 않은 것을 상위로 올림
    if ensemble_cfg.get("cart_boost", True):
        print("▶ Cart boost 후처리 중...")
        ensemble_preds = _cart_boost(ensemble_preds, df, item2idx, idx2item,
                                     top_k=ensemble_cfg.top_k)

    sub_df   = generate_submission_long(ensemble_preds, all_user_ids)
    validate_submission(sub_df, n_users=len(all_user_ids))

    model_tag = "_".join(used_models)
    out_path  = ROOT / "outputs" / f"submission_ensemble_{model_tag}.csv"
    sub_df.to_csv(out_path, index=False)
    print(f"▶ 제출 파일 저장: {out_path}  ({len(sub_df):,}행)")


if __name__ == "__main__":
    main()
