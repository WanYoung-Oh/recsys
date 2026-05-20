import numpy as np
import torch
from typing import Dict, List, Set


def ndcg_at_k(predicted: List, actual: Set, k: int = 10) -> float:
    dcg  = sum(1.0 / np.log2(i + 2) for i, item in enumerate(predicted[:k]) if item in actual)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual), k)))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(
    model,
    val_sequences: Dict,      # {user_id: padded_seq tensor [L]}
    val_user_ids: np.ndarray,
    gt_dict: Dict[str, Set],  # {user_id: set of item_ids}
    item_idx2id: Dict[int, str],
    device: torch.device,
    batch_size: int = 32768,
    k: int = 10,
    amp_dtype=torch.bfloat16,
    val_times: Dict = None,      # TiSASRec용 {user_id: padded_time tensor [L]}
    val_freqs: Dict = None,      # SAFERec용 {user_id: padded_freq tensor [L]}
    val_behaviors: Dict = None,  # MBSTRec용 {user_id: padded_behavior tensor [L]}
) -> float:
    """val_user_ids 중 gt_dict에 있는 유저만 대상으로 평균 NDCG@10 계산."""
    model.eval()
    scores_list = []

    active_users = [uid for uid in val_user_ids if uid in gt_dict and uid in val_sequences]
    if not active_users:
        return 0.0

    all_top_items: Dict[str, List[str]] = {}
    for i in range(0, len(active_users), batch_size):
        batch_uids = active_users[i : i + batch_size]
        seqs = torch.stack([val_sequences[uid] for uid in batch_uids]).to(device)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
            if val_times is not None:
                times = torch.stack([val_times[uid] for uid in batch_uids]).to(device)
                item_scores = model.full_sort_predict(seqs, times)
            elif val_freqs is not None:
                freqs = torch.stack([val_freqs[uid] for uid in batch_uids]).to(device)
                item_scores = model.full_sort_predict(seqs, freqs)
            elif val_behaviors is not None:
                behaviors = torch.stack([val_behaviors[uid] for uid in batch_uids]).to(device)
                item_scores = model.full_sort_predict(seqs, behaviors)
            else:
                item_scores = model.full_sort_predict(seqs)

        top_indices = item_scores.topk(k, dim=-1).indices.cpu().tolist()
        for uid, idx_row in zip(batch_uids, top_indices):
            all_top_items[uid] = [item_idx2id[idx + 1] for idx in idx_row]

    for uid in active_users:
        preds = all_top_items.get(uid, [])
        scores_list.append(ndcg_at_k(preds, gt_dict[uid], k))

    return float(np.mean(scores_list)) if scores_list else 0.0
