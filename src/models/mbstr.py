import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from .sasrec import SASRec

# view=0, cart=1, purchase=2, padding=3
BEHAVIOR_MAP = {"view": 0, "cart": 1, "purchase": 2, "pad": 3}
N_BEHAVIORS = 4


class MBSTRec(nn.Module):
    """Multi-Behavior Sequential Transformer Recommender (simplified).

    SASRec + view/cart/purchase 행동 타입 임베딩.
    각 시퀀스 위치에 아이템 임베딩 + 위치 임베딩 + 행동 타입 임베딩을 합산.
    cart/purchase 위치가 loss weight를 통해 추가로 강조됨.
    """

    def __init__(self, cfg: DictConfig, n_items: int):
        super().__init__()
        self.encoder = SASRec(cfg, n_items)
        # padding_idx=3 → 패딩 위치의 grad 차단
        self.behavior_emb = nn.Embedding(N_BEHAVIORS, cfg.hidden_size, padding_idx=3)
        nn.init.normal_(self.behavior_emb.weight, std=0.02)
        self.behavior_emb.weight.data[3].zero_()

        self.hidden_size = self.encoder.hidden_size
        self.max_seq_len = self.encoder.max_seq_len

    def _encode(self, seq: torch.Tensor, behavior_seq: torch.Tensor) -> torch.Tensor:
        B, L = seq.shape
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, -1)

        x = (
            self.encoder.item_emb(seq)
            + self.encoder.pos_emb(pos)
            + self.behavior_emb(behavior_seq)
        )
        x = self.encoder.emb_dropout(x)

        causal_mask = torch.triu(
            torch.ones(L, L, device=seq.device, dtype=torch.bool), diagonal=1
        )
        for layer in self.encoder.layers:
            x = layer(x, causal_mask)
        return self.encoder.ln_out(x)

    def calculate_loss(self, seq, behavior_seq, pos, neg, weights) -> torch.Tensor:
        h = self._encode(seq, behavior_seq)
        h_last = self.encoder._last_hidden(seq, h)
        pos_score = (h_last * self.encoder.item_emb(pos)).sum(-1)
        neg_score = (h_last * self.encoder.item_emb(neg)).sum(-1)
        return (-F.logsigmoid(pos_score - neg_score) * weights).mean()

    def full_sort_predict(self, seq: torch.Tensor,
                          behavior_seq: torch.Tensor = None) -> torch.Tensor:
        if behavior_seq is None:
            behavior_seq = torch.zeros_like(seq)
        h = self._encode(seq, behavior_seq)
        h_last = self.encoder._last_hidden(seq, h)
        return h_last @ self.encoder.item_emb.weight[1:].T
