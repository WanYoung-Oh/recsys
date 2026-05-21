import math
import numpy as np
import torch
from typing import Dict, List
from .dataset import make_padded_seq


def build_time_seq(
    seqs: Dict[str, Dict],
    max_seq_len: int = 50,
) -> Dict[str, torch.Tensor]:
    """TiSASRec용 — 유저별 padded time tensor (unix 초).

    반환: {user_id: tensor [max_seq_len]}  (0 = padding)
    """
    result = {}
    for uid, data in seqs.items():
        times = data["times"]  # unix 초 리스트
        times_trunc = times[-max_seq_len:]
        pad = [0] * (max_seq_len - len(times_trunc))
        result[uid] = torch.tensor(pad + times_trunc, dtype=torch.long)
    return result


# ── SAFERec: 아이템 view 빈도 피처 ────────────────────────────────────────


def compute_item_freq(
    df,
    item2idx: Dict[str, int],
    n_buckets: int = 64,
) -> Dict[int, int]:
    """학습 df에서 아이템별 view 횟수를 log-scale 버킷으로 변환.

    반환: {item_idx(1-based): bucket_idx(0..n_buckets-1)}
    """
    view_df = df[df["event_type"] == "view"]
    counts = view_df["item_id"].value_counts().to_dict()
    max_count = max(counts.values()) if counts else 1

    bucket_map: Dict[int, int] = {}
    for item_id, idx in item2idx.items():
        cnt = counts.get(item_id, 0)
        if cnt == 0:
            bucket = 0
        else:
            bucket = min(n_buckets - 1, int(math.log1p(cnt) / math.log1p(max_count) * (n_buckets - 1)))
        bucket_map[idx] = bucket
    return bucket_map


def build_freq_seq(
    seqs: Dict[str, Dict],
    max_seq_len: int,
    item_freq: Dict[int, int],
) -> Dict[str, torch.Tensor]:
    """SAFERec 추론용 — 유저별 padded freq-bucket tensor."""
    result = {}
    for uid, data in seqs.items():
        items = data["items"][-max_seq_len:]
        freqs = [item_freq.get(i, 0) for i in items]
        pad = [0] * (max_seq_len - len(freqs))
        result[uid] = torch.tensor(pad + freqs, dtype=torch.long)
    return result


class SeqTrainDatasetWithFreq(torch.utils.data.Dataset):
    """SAFERec 학습용: (seq, freq_seq, pos, neg, weight) 반환."""

    def __init__(self, seqs, n_items, event_weights, max_seq_len, item_freq):
        from .dataset import SeqTrainDataset
        self._base = SeqTrainDataset(seqs, n_items, event_weights, max_seq_len)
        self._seqs = seqs
        self._max_seq_len = max_seq_len
        self._item_freq = item_freq

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        u_idx, k, weight = self._base._samples[idx]
        uid   = self._base._uid_list[u_idx]
        iidx  = self._seqs[uid]["items"]

        seq_items = iidx[:k]
        pos = iidx[k]
        neg = np.random.randint(1, self._base.n_items + 1)
        freq_items = [self._item_freq.get(i, 0) for i in seq_items]

        return (
            make_padded_seq(seq_items, self._max_seq_len),
            make_padded_seq(freq_items, self._max_seq_len),
            torch.tensor(pos,    dtype=torch.long),
            torch.tensor(neg,    dtype=torch.long),
            torch.tensor(weight, dtype=torch.float),
        )


# ── MB-STR: 행동 타입 피처 ────────────────────────────────────────────────

_BEHAVIOR_MAP = {"view": 0, "cart": 1, "purchase": 2, "pad": 3}  # must match models/mbstr.BEHAVIOR_MAP


def build_behavior_seq(
    seqs: Dict[str, Dict],
    max_seq_len: int,
) -> Dict[str, torch.Tensor]:
    """MBSTRec 추론용 — 유저별 padded behavior-type tensor (pad=3)."""
    result = {}
    for uid, data in seqs.items():
        types = data["types"][-max_seq_len:]
        idxs  = [_BEHAVIOR_MAP.get(t, 0) for t in types]
        pad   = [3] * (max_seq_len - len(idxs))   # 3 = pad token
        result[uid] = torch.tensor(pad + idxs, dtype=torch.long)
    return result


class SeqTrainDatasetWithBehavior(torch.utils.data.Dataset):
    """MBSTRec 학습용: (seq, behavior_seq, pos, neg, weight) 반환."""

    def __init__(self, seqs, n_items, event_weights, max_seq_len):
        from .dataset import SeqTrainDataset
        self._base = SeqTrainDataset(seqs, n_items, event_weights, max_seq_len)
        self._seqs = seqs
        self._max_seq_len = max_seq_len

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        u_idx, k, weight = self._base._samples[idx]
        uid   = self._base._uid_list[u_idx]
        data  = self._seqs[uid]
        iidx  = data["items"]
        types = data["types"]

        seq_items = iidx[:k]
        seq_types = [_BEHAVIOR_MAP.get(t, 0) for t in types[:k]]
        pos = iidx[k]
        neg = np.random.randint(1, self._base.n_items + 1)

        pad_len = self._max_seq_len - len(seq_types)
        behavior_tensor = torch.tensor([3] * pad_len + seq_types, dtype=torch.long)

        return (
            make_padded_seq(seq_items, self._max_seq_len),
            behavior_tensor,
            torch.tensor(pos,    dtype=torch.long),
            torch.tensor(neg,    dtype=torch.long),
            torch.tensor(weight, dtype=torch.float),
        )


# ── TiSASRec ─────────────────────────────────────────────────────────────


class SeqTrainDatasetWithTime(torch.utils.data.Dataset):
    """TiSASRec 학습용: (seq, times, pos, neg, weight) 반환."""

    def __init__(self, seqs, n_items, event_weights, max_seq_len=50):
        from .dataset import SeqTrainDataset
        self._base = SeqTrainDataset(seqs, n_items, event_weights, max_seq_len)
        self._seqs = seqs
        self._max_seq_len = max_seq_len

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        u_idx, k, weight = self._base._samples[idx]
        uid  = self._base._uid_list[u_idx]
        data = self._seqs[uid]
        iidx  = data["items"]
        times = data["times"]

        seq_items = iidx[:k]
        seq_times = times[:k]
        pos = iidx[k]
        neg = np.random.randint(1, self._base.n_items + 1)

        padded_items = make_padded_seq(seq_items, self._max_seq_len)
        padded_times = make_padded_seq(seq_times, self._max_seq_len)

        return (
            padded_items,
            padded_times,
            torch.tensor(pos,    dtype=torch.long),
            torch.tensor(neg,    dtype=torch.long),
            torch.tensor(weight, dtype=torch.float),
        )
