import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple


def load_data(cfg) -> pd.DataFrame:
    """parquet 로드 및 기본 전처리."""
    path = f"{cfg.data.data_dir}/{cfg.data.train_file}"
    df = pd.read_parquet(path)
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    df = df.sort_values(["user_id", "event_time"]).reset_index(drop=True)

    if cfg.data.exclude_spike_purchase:
        spike_dates = set(cfg.data.spike_purchase_dates)
        mask = (df["event_type"] == "purchase") & (
            df["event_time"].dt.date.astype(str).isin(spike_dates)
        )
        df = df[~mask].reset_index(drop=True)

    return df


def build_vocab(df: pd.DataFrame):
    """전체 df 기준 item/user vocab 생성 (val 아이템 누락 방지)."""
    item_ids = sorted(df["item_id"].unique())
    user_ids = sorted(df["user_id"].unique())
    # 1-based: 0은 padding token
    item2idx = {iid: idx + 1 for idx, iid in enumerate(item_ids)}
    user2idx = {uid: idx     for idx, uid in enumerate(user_ids)}
    idx2item = {idx + 1: iid for idx, iid in enumerate(item_ids)}
    return item2idx, user2idx, idx2item


def build_sequences(
    df: pd.DataFrame,
    item2idx: Dict[str, int],
    max_seq_len: int = 50,
) -> Dict[str, Dict]:
    """train_df 기준으로만 호출 (leakage 방지).

    반환: {user_id: {'items': [item_idx, ...], 'types': [event_type, ...], 'times': [unix_sec, ...]}}
    """
    df = df.sort_values(["user_id", "event_time"])
    seqs: Dict[str, Dict] = {}

    for uid, group in df.groupby("user_id", sort=False):
        items = group["item_id"].tolist()
        types = group["event_type"].tolist()
        times = (group["event_time"].astype("int64") // 10**9).tolist()

        iidx = [item2idx.get(i, 0) for i in items]

        seqs[uid] = {
            "items": iidx[-max_seq_len:],
            "types": types[-max_seq_len:],
            "times": times[-max_seq_len:],
        }
    return seqs


def make_padded_seq(item_seq: List[int], max_seq_len: int) -> torch.Tensor:
    """item 인덱스 리스트 → 좌측 패딩된 tensor [max_seq_len]."""
    seq = item_seq[-max_seq_len:]
    pad = [0] * (max_seq_len - len(seq))
    return torch.tensor(pad + seq, dtype=torch.long)


def build_val_sequences(
    seqs: Dict[str, Dict],
    max_seq_len: int = 50,
) -> Dict[str, torch.Tensor]:
    """evaluate() 입력용 — 유저별 padded seq tensor."""
    return {
        uid: make_padded_seq(data["items"], max_seq_len)
        for uid, data in seqs.items()
    }


class SeqTrainDataset(Dataset):
    """SASRec/TiSASRec 학습 데이터셋.

    각 인터랙션 위치 k에서 (seq[:k], pos=item[k], neg=random) 샘플 생성.
    event_type에 따라 loss weight 부여.
    """

    def __init__(
        self,
        seqs: Dict[str, Dict],
        n_items: int,
        event_weights: Dict[str, float],
        max_seq_len: int = 50,
    ):
        self.seqs      = seqs
        self.n_items   = n_items
        self.weights   = event_weights
        self.max_seq_len = max_seq_len

        # (uid, position_k, loss_weight) 인덱스 사전 계산
        uid_list = list(seqs.keys())
        self._uid_list = uid_list
        self._samples: List[Tuple[int, int, float]] = []  # (user_int_idx, k, weight)

        for u_idx, uid in enumerate(uid_list):
            data = seqs[uid]
            iidx = data["items"]
            types = data["types"]
            for k in range(1, len(iidx)):
                if iidx[k] == 0:   # unknown item → skip
                    continue
                w = self.weights.get(types[k], 1.0)
                self._samples.append((u_idx, k, w))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        u_idx, k, weight = self._samples[idx]
        uid  = self._uid_list[u_idx]
        data = self.seqs[uid]
        iidx = data["items"]

        seq = iidx[:k]   # 위치 k 이전 시퀀스
        pos = iidx[k]    # 타겟 아이템

        padded = make_padded_seq(seq, self.max_seq_len)
        neg = np.random.randint(1, self.n_items + 1)

        return (
            padded,
            torch.tensor(pos,    dtype=torch.long),
            torch.tensor(neg,    dtype=torch.long),
            torch.tensor(weight, dtype=torch.float),
        )
