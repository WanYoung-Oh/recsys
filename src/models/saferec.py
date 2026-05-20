import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from .sasrec import SASRec


class SAFERec(nn.Module):
    """SAFERec: Self-Attentive Frequency-Enhanced Recommendation.

    SASRec + 아이템 view 빈도 임베딩.
    각 아이템의 전체 학습 데이터 내 view 횟수를 log-scale 버킷으로 이산화하여
    아이템 임베딩에 더함. 빈도 높은 인기 아이템과 롱테일 아이템을 구분.
    """

    def __init__(self, cfg: DictConfig, n_items: int):
        super().__init__()
        self.encoder        = SASRec(cfg, n_items)
        self.n_freq_buckets = cfg.n_freq_buckets
        self.freq_emb = nn.Embedding(cfg.n_freq_buckets, cfg.hidden_size)
        nn.init.normal_(self.freq_emb.weight, std=0.02)

        self.hidden_size = self.encoder.hidden_size
        self.max_seq_len = self.encoder.max_seq_len

    def _encode(self, seq: torch.Tensor, freq_seq: torch.Tensor) -> torch.Tensor:
        B, L = seq.shape
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, -1)

        x = (
            self.encoder.item_emb(seq)
            + self.encoder.pos_emb(pos)
            + self.freq_emb(freq_seq)
        )
        x = self.encoder.emb_dropout(x)

        causal_mask = torch.triu(
            torch.ones(L, L, device=seq.device, dtype=torch.bool), diagonal=1
        )
        for layer in self.encoder.layers:
            x = layer(x, causal_mask)
        return self.encoder.ln_out(x)

    def calculate_loss(self, seq, freq_seq, pos, neg, weights) -> torch.Tensor:
        h = self._encode(seq, freq_seq)
        h_last = self.encoder._last_hidden(seq, h)
        pos_score = (h_last * self.encoder.item_emb(pos)).sum(-1)
        neg_score = (h_last * self.encoder.item_emb(neg)).sum(-1)
        return (-F.logsigmoid(pos_score - neg_score) * weights).mean()

    def full_sort_predict(self, seq: torch.Tensor,
                          freq_seq: torch.Tensor = None) -> torch.Tensor:
        if freq_seq is None:
            freq_seq = torch.zeros_like(seq)
        h = self._encode(seq, freq_seq)
        h_last = self.encoder._last_hidden(seq, h)
        return h_last @ self.encoder.item_emb.weight[1:].T
