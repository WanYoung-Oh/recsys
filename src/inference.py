import torch
import pandas as pd
from typing import Dict, List


def generate_predictions(
    model,
    user_ids: List[str],
    val_sequences: Dict[str, torch.Tensor],
    item_idx2id: Dict[int, str],
    device: torch.device,
    batch_size: int = 32768,
    top_k: int = 10,
    amp_dtype=torch.bfloat16,
    time_sequences: Dict[str, torch.Tensor] = None,      # TiSASRec
    freq_sequences: Dict[str, torch.Tensor] = None,      # SAFERec
    behavior_sequences: Dict[str, torch.Tensor] = None,  # MBSTRec
) -> Dict[str, List[str]]:
    """모든 user에 대해 top_k 아이템 예측.

    반환: {user_id: [item_id, ...]} (길이 top_k)
    """
    model.eval()
    results: Dict[str, List[str]] = {}

    for i in range(0, len(user_ids), batch_size):
        batch_uids = user_ids[i : i + batch_size]
        seqs = torch.stack([val_sequences[uid] for uid in batch_uids]).to(device)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
            if time_sequences is not None:
                times = torch.stack([time_sequences[uid] for uid in batch_uids]).to(device)
                scores = model.full_sort_predict(seqs, times)
            elif freq_sequences is not None:
                freqs = torch.stack([freq_sequences[uid] for uid in batch_uids]).to(device)
                scores = model.full_sort_predict(seqs, freqs)
            elif behavior_sequences is not None:
                behaviors = torch.stack([behavior_sequences[uid] for uid in batch_uids]).to(device)
                scores = model.full_sort_predict(seqs, behaviors)
            else:
                scores = model.full_sort_predict(seqs)

        top_indices = scores.topk(top_k, dim=-1).indices.cpu().tolist()
        for uid, idx_row in zip(batch_uids, top_indices):
            results[uid] = [item_idx2id[idx + 1] for idx in idx_row]

    return results


def generate_submission_long(
    predictions: Dict[str, List[str]],
    all_user_ids: List[str],
    top_k: int = 10,
) -> pd.DataFrame:
    """제출 파일 롱 포맷 생성: (user_id, item_id) × n_users × top_k."""
    missing = [uid for uid in all_user_ids if uid not in predictions]
    if missing:
        raise ValueError(f"predictions에 없는 유저 {len(missing)}명: {missing[:5]} ...")

    rows = []
    for uid in all_user_ids:
        for iid in predictions[uid]:
            rows.append({"user_id": uid, "item_id": iid})
    return pd.DataFrame(rows)


def validate_submission(sub_df: pd.DataFrame, n_users: int = 638257, top_k: int = 10):
    assert len(sub_df) == n_users * top_k, f"행 수 불일치: {len(sub_df)}"
    vc = sub_df.groupby("user_id").size()
    assert (vc == top_k).all(), "유저당 행 수가 10이 아닌 유저 존재"
    dup = sub_df.groupby("user_id")["item_id"].nunique()
    assert (dup == top_k).all(), "유저별 item_id 중복 존재"
    print("제출 파일 검증 통과")
