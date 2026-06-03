
import os
import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omegaconf import OmegaConf
from data.dataset import load_data, build_vocab, build_sequences, make_padded_seq
from data.features import build_behavior_seq, build_time_seq
from cv.holdout import make_holdout, build_gt
from inference import generate_predictions, generate_submission_long, validate_submission
from models import build_model
from models.tifu_knn import TIFUKNN
from util.paths import get_checkpoint_path
from metrics import ndcg_at_k

# importance=0 으로 확인된 in_ 피처는 학습·추론 모두 제외
_SKIP_IN_MODELS = {"mbstr", "tisasrec"}

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DTYPE = torch.bfloat16

# 실제 존재하는 체크포인트 경로 오버라이드
# 없으면 util.paths.get_checkpoint_path 로 자동 탐색
CKPT_OVERRIDES = {
    ("bsarec",   "tuning"): ROOT / "outputs/bsarec/run001_260526/tuning/best.pt",
    ("bsarec",   "full"):   ROOT / "outputs/bsarec/run002_260526/full/best.pt",

    ("tisasrec", "tuning"): ROOT / "outputs/tisasrec/run001_260520/tuning/best.pt",
    ("tisasrec", "full"):   ROOT / "outputs/tisasrec/run002_260526/full/best.pt",

    ("mbstr",    "tuning"): ROOT / "outputs/mbstr/run001_260526/tuning/best.pt",
    ("mbstr",    "full"):   ROOT / "outputs/mbstr/run003_260526/full/best.pt",
}


def load_cfg(model_name: str, load_phase: str = "full"):
    model_cfg = OmegaConf.load(ROOT / "conf" / "model" / f"{model_name}.yaml")
    data_cfg  = OmegaConf.load(ROOT / "conf" / "data"  / "base.yaml")
    train_cfg = OmegaConf.load(ROOT / "conf" / "train" / "base.yaml")
    data_cfg.data_dir = str(ROOT / data_cfg.data_dir)
    return OmegaConf.create({
        "model":      model_cfg,
        "data":       data_cfg,
        "train":      train_cfg,
        "cv":         {"enabled": load_phase == "tuning"},
        "run_id":     None,
        "run_date":   None,
        "ckpt_path":  None,
        "checkpoint": {"load_phase": load_phase},
    })


def resolve_ckpt(model_name: str, load_phase: str) -> Path:
    override = CKPT_OVERRIDES.get((model_name, load_phase))
    if override and Path(override).exists():
        return Path(override)
    cfg = load_cfg(model_name, load_phase)
    return get_checkpoint_path(cfg, str(ROOT), for_training=False, load_phase=load_phase)


def get_sample_user_order(df) -> list:
    p = ROOT / "data" / "sample_submission.csv"
    if p.exists():
        s = pd.read_csv(p)
        return list(s["user_id"].unique())
    return sorted(df["user_id"].unique().tolist())


def build_all_user_val_inputs(
    hist_df, item2idx, user_ids,
    max_seq_len: int = 50,
    need_behavior: bool = False,
    need_time: bool = False,
):
    seqs = build_sequences(hist_df, item2idx, max_seq_len)

    raw_beh  = build_behavior_seq(seqs, max_seq_len) if need_behavior else {}
    raw_time = build_time_seq(seqs, max_seq_len)     if need_time     else {}

    zero_seq = torch.zeros(max_seq_len, dtype=torch.long)

    val_seqs  = {}
    beh_seqs  = {}
    time_seqs = {}

    for uid in user_ids:
        items = seqs.get(uid, {}).get("items", [])
        val_seqs[uid] = make_padded_seq(items, max_seq_len)
        if need_behavior:
            beh_seqs[uid]  = raw_beh.get(uid,  zero_seq.clone())
        if need_time:
            time_seqs[uid] = raw_time.get(uid, zero_seq.clone())

    return val_seqs, beh_seqs, time_seqs


def collect_nn_preds(model_name, phase, hist_df, item2idx, idx2item, user_ids, batch_size=64):
    try:
        ckpt_path = resolve_ckpt(model_name, phase)
        if not ckpt_path.exists():
            print(f"[SKIP] {model_name} {phase} ckpt not found: {ckpt_path}")
            return None
    except Exception as e:
        print(f"[SKIP] {model_name} {phase} ckpt resolve failed: {e}")
        return None

    print(f"[LOAD] {model_name} ({phase}) -> {ckpt_path}")
    cfg   = load_cfg(model_name, phase)
    model = build_model(cfg.model, N_ITEMS).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    need_behavior = (model_name == "mbstr")
    need_time     = (model_name == "tisasrec")

    val_seqs, beh_seqs, time_seqs = build_all_user_val_inputs(
        hist_df, item2idx, user_ids,
        max_seq_len=cfg.model.max_seq_len,
        need_behavior=need_behavior,
        need_time=need_time,
    )

    kwargs = {}
    if need_behavior:
        kwargs["behavior_sequences"] = beh_seqs
    if need_time:
        kwargs["time_sequences"] = time_seqs

    # TiSASRec: [B,L,L,hd] 시간 행렬 2개 → 배치당 메모리 L배, 512 상한
    # max_seq_len > 50 모델(cl4srec=100): FFN 텐서가 비례해서 커지므로 배치 축소
    BASE_SEQ_LEN = 50
    if model_name == "tisasrec":
        bs = min(batch_size, 512)
    elif cfg.model.max_seq_len > BASE_SEQ_LEN:
        bs = batch_size // (cfg.model.max_seq_len // BASE_SEQ_LEN)
    else:
        bs = batch_size

    try:
        preds = generate_predictions(
            model=model,
            user_ids=user_ids,
            val_sequences=val_seqs,
            item_idx2id=idx2item,
            device=DEVICE,
            batch_size=bs,
            top_k=10,
            amp_dtype=AMP_DTYPE,
            **kwargs,
        )
    finally:
        del model
        torch.cuda.empty_cache()

    return preds


def collect_tifu_preds(hist_df, item2idx, idx2item, user_ids) -> dict:
    print("[LOAD] tifu_knn")
    seqs = build_sequences(hist_df, item2idx, max_seq_len=50)
    tifu = TIFUKNN(group_count=7, decay_within=0.9, decay_across=0.7)
    tifu.fit(seqs, idx2item)
    return tifu.predict_all(user_ids, top_k=10)


def build_history_maps(hist_df) -> dict:
    max_time    = hist_df["event_time"].max()
    recent14    = max_time - pd.Timedelta(days=14)
    purchase_df = hist_df[hist_df["event_type"] == "purchase"]

    user_total_s    = hist_df.groupby("user_id").size()
    user_purchase_s = purchase_df.groupby("user_id").size().reindex(user_total_s.index, fill_value=0)

    item_total_s    = hist_df.groupby("item_id").size()
    item_purchase_s = purchase_df.groupby("item_id").size().reindex(item_total_s.index, fill_value=0)

    return {
        "user_total_cnt":      user_total_s.to_dict(),
        "user_recent14_cnt":   hist_df[hist_df["event_time"] >= recent14].groupby("user_id").size().to_dict(),
        "item_pop":            item_total_s.to_dict(),
        "item_recent14_pop":   hist_df[hist_df["event_time"] >= recent14].groupby("item_id").size().to_dict(),
        "viewed_map":          hist_df.groupby("user_id")["item_id"].apply(set).to_dict(),
        "carted_map":          hist_df[hist_df["event_type"] == "cart"].groupby("user_id")["item_id"].apply(set).to_dict(),
        "purchased_map":       purchase_df.groupby("user_id")["item_id"].apply(set).to_dict(),
        # 신규: binary seen_before → 횟수로 확장
        "user_item_view_cnt":  hist_df.groupby(["user_id", "item_id"]).size().to_dict(),
        # 신규: purchase 관점 인기도 (view만 많고 구매 없는 아이템 구분)
        "item_purchase_ratio": (item_purchase_s / item_total_s).to_dict(),
        # 신규: 유저의 실제 구매 성향 ("사는 사람"인지 여부)
        "user_purchase_ratio": (user_purchase_s / user_total_s).to_dict(),
    }


def rank_lookup(item_list: list) -> dict:
    return {iid: r + 1 for r, iid in enumerate(item_list)}


def build_candidate_rows(user_ids, preds_by_model, hist_maps, gt_map=None):
    model_names = list(preds_by_model.keys())
    rank_maps   = {
        m: {uid: rank_lookup(items) for uid, items in pred.items()}
        for m, pred in preds_by_model.items()
    }

    rows   = []
    groups = []

    for uid in user_ids:
        cand_set = set()
        for m in model_names:
            cand_set.update(preds_by_model[m].get(uid, []))
        if not cand_set:
            continue

        u_total      = hist_maps["user_total_cnt"].get(uid, 0)
        u_r14        = hist_maps["user_recent14_cnt"].get(uid, 0)
        u_pur_ratio  = hist_maps["user_purchase_ratio"].get(uid, 0.0)
        view_cnt_map = hist_maps["user_item_view_cnt"]

        local_rows = []
        for iid in cand_set:
            feat = {
                "user_total_cnt_log1p":    np.log1p(u_total),
                "user_recent14_cnt_log1p": np.log1p(u_r14),
                "item_pop_log1p":          np.log1p(hist_maps["item_pop"].get(iid, 0)),
                "item_recent14_pop_log1p": np.log1p(hist_maps["item_recent14_pop"].get(iid, 0)),
                "user_item_view_cnt":      view_cnt_map.get((uid, iid), 0),
                "item_purchase_ratio":     hist_maps["item_purchase_ratio"].get(iid, 0.0),
                "user_purchase_ratio":     u_pur_ratio,
                "model_hit_count":         0,
            }
            for m in model_names:
                r    = rank_maps[m].get(uid, {}).get(iid, 999)
                in_m = int(r != 999)
                if m not in _SKIP_IN_MODELS:
                    feat[f"in_{m}"] = in_m
                feat[f"rank_{m}"] = r if in_m else 999
                feat[f"rr_{m}"]   = (1.0 / r) if in_m else 0.0
                feat["model_hit_count"] += in_m

            feat["_user_id"] = uid
            feat["_item_id"] = iid
            feat["_label"]   = int(iid in gt_map.get(uid, set())) if gt_map is not None else 0
            local_rows.append(feat)

        if local_rows:
            rows.extend(local_rows)
            groups.append(len(local_rows))

    df           = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if not c.startswith("_")]
    return df, groups, feature_cols


def train_valid_split_users(users, valid_ratio: float = 0.2):
    users = list(users)
    random.shuffle(users)
    n_valid     = max(1, int(len(users) * valid_ratio))
    train_users = users[n_valid:]
    valid_users = users[:n_valid]
    return train_users, valid_users


def eval_grouped_ndcg(df_scored, top_k: int = 10) -> float:
    scores = []
    for uid, g in df_scored.groupby("_user_id", sort=False):
        g          = g.sort_values("pred", ascending=False)
        pred_items = g["_item_id"].tolist()[:top_k]
        gt_items   = set(g[g["_label"] == 1]["_item_id"].tolist())
        if gt_items:
            scores.append(ndcg_at_k(pred_items, gt_items, top_k))
    return float(np.mean(scores)) if scores else 0.0


def main():
    global N_ITEMS

    ensemble_cfg = OmegaConf.load(ROOT / "conf" / "ensemble" / "rank.yaml")
    nn_models    = [m for m in ensemble_cfg.weights.keys() if m != "tifu_knn"]
    print(f"▶ 리랭커 대상 NN 모델: {nn_models}")

    print("=== load full data ===")
    base_cfg = load_cfg(nn_models[0], "full")
    df = load_data(base_cfg)
    item2idx, user2idx, idx2item = build_vocab(df)
    N_ITEMS = len(item2idx)

    sample_user_order = get_sample_user_order(df)
    print(f"full users = {len(sample_user_order):,}, items = {N_ITEMS:,}")

    print("=== holdout split ===")
    train_df, val_df = make_holdout(df)
    gt_cp = build_gt(val_df, mode="cart_purchase")

    train_hist_maps = build_history_maps(train_df)
    full_hist_maps  = build_history_maps(df)

    val_users = sorted(set(gt_cp.keys()) & set(train_df["user_id"].unique()))
    print(f"meta-train users with GT+history = {len(val_users):,}")

    # ── 튜닝 예측 수집 (리랭커 학습용) ──────────────────────────────
    print("=== collect tuning predictions for reranker training ===")
    preds_val = {}
    for model_name in nn_models:
        p = collect_nn_preds(model_name, "tuning", train_df, item2idx, idx2item, val_users, batch_size=64)
        if p is not None:
            preds_val[model_name] = p
    preds_val["tifu_knn"] = collect_tifu_preds(train_df, item2idx, idx2item, val_users)
    print("active tuning models =", list(preds_val.keys()))

    # ── 메타 데이터셋 빌드 ──────────────────────────────────────────
    print("=== build meta dataset ===")
    meta_df, _meta_groups, feat_cols = build_candidate_rows(
        val_users, preds_val, train_hist_maps, gt_map=gt_cp
    )
    print(f"meta_df shape = {meta_df.shape}, feature count = {len(feat_cols)}")

    train_users_meta, valid_users_meta = train_valid_split_users(val_users, valid_ratio=0.2)
    tr_df = meta_df[meta_df["_user_id"].isin(train_users_meta)].copy()
    va_df = meta_df[meta_df["_user_id"].isin(valid_users_meta)].copy()

    tr_groups = tr_df.groupby("_user_id").size().tolist()
    va_groups = va_df.groupby("_user_id").size().tolist()

    X_tr, y_tr = tr_df[feat_cols], tr_df["_label"]
    X_va, y_va = va_df[feat_cols], va_df["_label"]
    print(f"train rows = {len(tr_df)}, valid rows = {len(va_df)}")

    # ── LightGBM LambdaMART 학습 ────────────────────────────────────
    try:
        import lightgbm as lgb
    except Exception as e:
        raise RuntimeError("lightgbm import failed. pip install lightgbm 필요") from e

    print("=== train LightGBM LambdaMART ===")
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="gbdt",
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        n_estimators=500,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=os.cpu_count() or 4,
    )
    ranker.fit(
        X_tr, y_tr,
        group=tr_groups,
        eval_set=[(X_va, y_va)],
        eval_group=[va_groups],
        eval_at=[10],
        callbacks=[
            lgb.early_stopping(50, verbose=True),
            lgb.log_evaluation(20),
        ],
    )

    va_df = va_df.copy()
    va_df["pred"] = ranker.predict(X_va, num_iteration=ranker.best_iteration_)
    offline_ndcg  = eval_grouped_ndcg(va_df, top_k=10)
    print(f"offline valid ndcg@10 = {offline_ndcg:.6f}")

    feat_imp_path = ROOT / "outputs" / "reranker_feature_importance.csv"
    pd.DataFrame({
        "feature":    feat_cols,
        "importance": ranker.feature_importances_,
    }).sort_values("importance", ascending=False).to_csv(feat_imp_path, index=False)
    print("saved feature importance ->", feat_imp_path)

    # ── 전체 유저 예측 수집 (제출용) ────────────────────────────────
    print("=== collect full predictions for submission ===")
    preds_full        = {}
    active_full_models = []
    for model_name in nn_models:
        p = collect_nn_preds(model_name, "full", df, item2idx, idx2item, sample_user_order, batch_size=64)
        if p is not None:
            preds_full[model_name] = p
            active_full_models.append(model_name)
    preds_full["tifu_knn"] = collect_tifu_preds(df, item2idx, idx2item, sample_user_order)
    active_full_models.append("tifu_knn")
    print("active full models =", active_full_models)

    # ── 전체 유저 리랭킹 ────────────────────────────────────────────
    print("=== rerank full users ===")
    model_names    = list(preds_full.keys())
    rank_maps_full = {
        m: {uid: rank_lookup(items) for uid, items in pred.items()}
        for m, pred in preds_full.items()
    }

    RERANK_BATCH     = 2000
    out_pred         = {}
    full_view_cnt_map = full_hist_maps["user_item_view_cnt"]

    batches = range(0, len(sample_user_order), RERANK_BATCH)
    for batch_start in tqdm(batches, desc="reranking", unit="batch"):
        batch_uids = sample_user_order[batch_start : batch_start + RERANK_BATCH]

        rows     = []
        uid_list = []
        iid_list = []

        for uid in batch_uids:
            cand_set = set()
            for m in model_names:
                cand_set.update(preds_full[m].get(uid, []))

            u_total     = full_hist_maps["user_total_cnt"].get(uid, 0)
            u_r14       = full_hist_maps["user_recent14_cnt"].get(uid, 0)
            u_pur_ratio = full_hist_maps["user_purchase_ratio"].get(uid, 0.0)

            for iid in cand_set:
                feat = {
                    "user_total_cnt_log1p":    np.log1p(u_total),
                    "user_recent14_cnt_log1p": np.log1p(u_r14),
                    "item_pop_log1p":          np.log1p(full_hist_maps["item_pop"].get(iid, 0)),
                    "item_recent14_pop_log1p": np.log1p(full_hist_maps["item_recent14_pop"].get(iid, 0)),
                    "user_item_view_cnt":      full_view_cnt_map.get((uid, iid), 0),
                    "item_purchase_ratio":     full_hist_maps["item_purchase_ratio"].get(iid, 0.0),
                    "user_purchase_ratio":     u_pur_ratio,
                    "model_hit_count":         0,
                }
                for m in model_names:
                    r    = rank_maps_full[m].get(uid, {}).get(iid, 999)
                    in_m = int(r != 999)
                    if m not in _SKIP_IN_MODELS:
                        feat[f"in_{m}"] = in_m
                    feat[f"rank_{m}"] = r if in_m else 999
                    feat[f"rr_{m}"]   = (1.0 / r) if in_m else 0.0
                    feat["model_hit_count"] += in_m
                rows.append(feat)
                uid_list.append(uid)
                iid_list.append(iid)

        if not rows:
            for uid in batch_uids:
                out_pred[uid] = []
            continue

        X_batch  = pd.DataFrame(rows)[feat_cols]
        scores   = ranker.predict(X_batch, num_iteration=ranker.best_iteration_)
        uid_arr  = np.array(uid_list)
        iid_arr  = np.array(iid_list)

        for uid in batch_uids:
            mask = uid_arr == uid
            if not mask.any():
                out_pred[uid] = []
                continue

            ranked = [iid for _, iid in sorted(
                zip(scores[mask], iid_arr[mask].tolist()),
                key=lambda z: z[0], reverse=True,
            )]

            seen_items = set()
            final      = []
            for iid in ranked:
                if iid not in seen_items:
                    final.append(iid)
                    seen_items.add(iid)
                if len(final) == 10:
                    break

            # top10 보장: 후보 부족 시 각 모델 예측에서 채움
            if len(final) < 10:
                for m in model_names:
                    for iid in preds_full[m].get(uid, []):
                        if iid not in seen_items:
                            final.append(iid)
                            seen_items.add(iid)
                        if len(final) == 10:
                            break
                    if len(final) == 10:
                        break

            out_pred[uid] = final[:10]

    print("=== save submission ===")
    sub_df = generate_submission_long(out_pred, sample_user_order, top_k=10)
    validate_submission(sub_df, n_users=len(sample_user_order), top_k=10)

    out_path = ROOT / "outputs" / "submission_reranker_lgbm.csv"
    sub_df.to_csv(out_path, index=False)
    print("saved submission ->", out_path)


if __name__ == "__main__":
    main()
