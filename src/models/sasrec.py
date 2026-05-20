import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class SASRecLayer(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, inner_size: int,
                 hidden_dropout: float, attn_dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_size, n_heads, dropout=attn_dropout, batch_first=True
        )
        self.ln1 = nn.LayerNorm(hidden_size, eps=1e-8)
        self.ln2 = nn.LayerNorm(hidden_size, eps=1e-8)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, inner_size),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(inner_size, hidden_size),
            nn.Dropout(hidden_dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # key_padding_mask 미사용: left-pad + causal mask 조합에서 초기 패딩 위치가
        # softmax(-inf,...) = NaN을 유발하므로 causal mask만 적용 (RecBole SASRec 방식)
        residual = x
        x_ln = self.ln1(x)
        x, _ = self.attn(
            x_ln, x_ln, x_ln,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = residual + x

        residual = x
        x = self.ffn(self.ln2(x))
        return residual + x


class SASRec(nn.Module):
    """SASRec: Self-Attentive Sequential Recommendation (ICDM 2018)."""

    def __init__(self, cfg: DictConfig, n_items: int):
        super().__init__()
        self.n_items     = n_items
        self.hidden_size = cfg.hidden_size
        self.max_seq_len = cfg.max_seq_len

        # 0 = padding token, items are 1-indexed
        self.item_emb = nn.Embedding(n_items + 1, cfg.hidden_size, padding_idx=0)
        self.pos_emb  = nn.Embedding(cfg.max_seq_len, cfg.hidden_size)
        self.emb_dropout = nn.Dropout(cfg.hidden_dropout)

        self.layers = nn.ModuleList([
            SASRecLayer(cfg.hidden_size, cfg.n_heads, cfg.inner_size,
                        cfg.hidden_dropout, cfg.attn_dropout)
            for _ in range(cfg.n_layers)
        ])
        self.ln_out = nn.LayerNorm(cfg.hidden_size, eps=1e-8)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: [B, L] → [B, L, H] 인코딩."""
        B, L = seq.shape
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, -1)

        x = self.item_emb(seq) + self.pos_emb(pos)
        x = self.emb_dropout(x)

        # causal mask만 사용 (key_padding_mask 제거 — NaN 방지)
        causal_mask = torch.triu(
            torch.ones(L, L, device=seq.device, dtype=torch.bool), diagonal=1
        )  # [L, L]

        for layer in self.layers:
            x = layer(x, causal_mask)

        return self.ln_out(x)  # [B, L, H]

    def _last_hidden(self, seq: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """seq의 마지막 비패딩 위치 hidden state 반환: [B, H]."""
        lengths = (seq != 0).sum(dim=1) - 1   # [B] 0-indexed last position
        lengths = lengths.clamp(min=0)
        idx = lengths.view(-1, 1, 1).expand(-1, 1, self.hidden_size)
        return h.gather(1, idx).squeeze(1)    # [B, H]

    def calculate_loss(
        self,
        seq: torch.Tensor,       # [B, L]
        pos: torch.Tensor,       # [B]
        neg: torch.Tensor,       # [B]
        weights: torch.Tensor,   # [B]
    ) -> torch.Tensor:
        h = self._encode(seq)                        # [B, L, H]
        h_last = self._last_hidden(seq, h)           # [B, H]

        pos_emb = self.item_emb(pos)                 # [B, H]
        neg_emb = self.item_emb(neg)                 # [B, H]

        pos_score = (h_last * pos_emb).sum(-1)       # [B]
        neg_score = (h_last * neg_emb).sum(-1)       # [B]

        loss = (-F.logsigmoid(pos_score - neg_score) * weights).mean()
        return loss

    def full_sort_predict(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: [B, L] → [B, n_items] 점수 (아이템 인덱스 0-based, 즉 item_idx - 1)."""
        h = self._encode(seq)                        # [B, L, H]
        h_last = self._last_hidden(seq, h)           # [B, H]

        # items are indexed 1..n_items; skip padding row (index 0)
        all_emb = self.item_emb.weight[1:]           # [n_items, H]
        return h_last @ all_emb.T                    # [B, n_items]
