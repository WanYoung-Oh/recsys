import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from .sasrec import SASRec


def _crop(seq: torch.Tensor, ratio: float) -> torch.Tensor:
    """비패딩 구간에서 연속 crop."""
    lengths = (seq != 0).sum(dim=1)  # [B]
    out = seq.clone()
    for i in range(seq.size(0)):
        L = lengths[i].item()
        if L <= 1:
            continue
        crop_len = max(1, int(L * (1 - ratio)))
        start = random.randint(0, L - crop_len)
        # 비패딩 구간에서 crop_len 개 선택 후 재패딩
        nonpad_start = seq.size(1) - L
        selected = seq[i, nonpad_start + start : nonpad_start + start + crop_len]
        new_seq = torch.zeros_like(seq[i])
        new_seq[seq.size(1) - crop_len:] = selected
        out[i] = new_seq
    return out


def _mask(seq: torch.Tensor, ratio: float, n_items: int) -> torch.Tensor:
    """비패딩 위치 중 일부를 랜덤 아이템으로 교체."""
    out = seq.clone()
    pad_mask = seq == 0
    rand = torch.rand_like(seq.float())
    rand[pad_mask] = 1.0
    mask_pos = rand < ratio
    n_mask = int(mask_pos.sum().item())
    if n_mask > 0:
        out[mask_pos] = torch.randint(1, n_items + 1, (n_mask,), device=seq.device)
    return out


def _reorder(seq: torch.Tensor, ratio: float) -> torch.Tensor:
    """비패딩 구간의 부분 시퀀스를 랜덤 셔플."""
    lengths = (seq != 0).sum(dim=1)
    out = seq.clone()
    for i in range(seq.size(0)):
        L = lengths[i].item()
        if L <= 2:
            continue
        reorder_len = max(2, int(L * ratio))
        nonpad_start = seq.size(1) - L
        start = random.randint(0, L - reorder_len)
        s = nonpad_start + start
        e = s + reorder_len
        perm = torch.randperm(e - s, device=seq.device)
        out[i, s:e] = out[i, s:e][perm]
    return out


class CL4SRec(nn.Module):
    """CL4SRec: Contrastive Learning for Sequential Recommendation (ICDE 2022).

    SASRec 인코더 위에 crop/mask/reorder augmentation 기반 대조학습 추가.
    """

    def __init__(self, cfg: DictConfig, n_items: int):
        super().__init__()
        self.encoder  = SASRec(cfg, n_items)
        self.n_items  = n_items
        self.lmd      = cfg.lmd   # 대조학습 손실 가중치
        self.tau      = cfg.tau   # 온도 파라미터
        self.crop_r   = cfg.aug_crop_ratio
        self.mask_r   = cfg.aug_mask_ratio
        self.reorder_r = cfg.aug_reorder_ratio

        # 속성 위임
        self.hidden_size = self.encoder.hidden_size
        self.max_seq_len = self.encoder.max_seq_len

    def _aug(self, seq: torch.Tensor) -> torch.Tensor:
        """3종 augmentation 중 랜덤 1개 적용."""
        choice = random.randint(0, 2)
        if choice == 0:
            return _crop(seq, self.crop_r)
        elif choice == 1:
            return _mask(seq, self.mask_r, self.n_items)
        else:
            return _reorder(seq, self.reorder_r)

    def _contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """InfoNCE loss (in-batch negatives)."""
        z1 = F.normalize(z1, dim=-1)  # [B, H]
        z2 = F.normalize(z2, dim=-1)
        sim = torch.matmul(z1, z2.T) / self.tau  # [B, B]
        labels = torch.arange(z1.size(0), device=z1.device)
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2

    def calculate_loss(self, seq, pos, neg, weights) -> torch.Tensor:
        # 추천 손실
        rec_loss = self.encoder.calculate_loss(seq, pos, neg, weights)

        # 대조학습 손실
        aug1 = self._aug(seq)
        aug2 = self._aug(seq)
        h1 = self.encoder._encode(aug1)
        h2 = self.encoder._encode(aug2)
        z1 = self.encoder._last_hidden(aug1, h1)
        z2 = self.encoder._last_hidden(aug2, h2)
        cl_loss = self._contrastive_loss(z1, z2)

        return rec_loss + self.lmd * cl_loss

    def full_sort_predict(self, seq: torch.Tensor) -> torch.Tensor:
        return self.encoder.full_sort_predict(seq)
