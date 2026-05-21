"""leaderboard_proxy (Feb 23~29) 오프라인 NDCG@10 평가."""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import DictConfig


def evaluate_leaderboard_proxy(
    model: torch.nn.Module,
    df,
    item2idx: dict,
    idx2item: dict,
    cfg: DictConfig,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[float, int]:
    """전체 df 시퀀스 기준 proxy GT로 NDCG@10 계산 (train.py와 동일 로직)."""
    from cv.holdout import build_leaderboard_proxy_gt
    from data.dataset import build_sequences, build_val_sequences
    from metrics import evaluate

    proxy_gt = build_leaderboard_proxy_gt(df)
    proxy_users = np.array(list(proxy_gt.keys()))
    n_gt = len(proxy_gt)

    full_seqs = build_sequences(df, item2idx, cfg.model.max_seq_len)
    full_val_seqs = build_val_sequences(full_seqs, cfg.model.max_seq_len)

    model_name = cfg.model.name
    proxy_times = proxy_freqs = proxy_behaviors = None

    if model_name == "tisasrec":
        from data.features import build_time_seq
        proxy_times = build_time_seq(full_seqs, cfg.model.max_seq_len)
    elif model_name == "saferec":
        from data.features import build_freq_seq, compute_item_freq
        item_freq = compute_item_freq(df, item2idx, cfg.model.n_freq_buckets)
        proxy_freqs = build_freq_seq(full_seqs, cfg.model.max_seq_len, item_freq)
    elif model_name == "mbstr":
        from data.features import build_behavior_seq
        proxy_behaviors = build_behavior_seq(full_seqs, cfg.model.max_seq_len)

    model.eval()
    proxy_ndcg = evaluate(
        model,
        full_val_seqs,
        proxy_users,
        proxy_gt,
        idx2item,
        device,
        cfg.train.eval_batch_size,
        amp_dtype=amp_dtype,
        val_times=proxy_times,
        val_freqs=proxy_freqs,
        val_behaviors=proxy_behaviors,
    )
    return proxy_ndcg, n_gt
