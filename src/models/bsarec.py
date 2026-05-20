import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from .sasrec import SASRecLayer


class BSARecLayer(nn.Module):
    """BSARec 레이어: Self-Attention + 주파수 도메인 저역통과 필터 혼합.

    출력 = sigmoid(α) * h_attn + (1 - sigmoid(α)) * h_freq
    α 는 학습 가능 파라미터.
    """

    def __init__(self, hidden_size: int, n_heads: int, inner_size: int,
                 hidden_dropout: float, attn_dropout: float):
        super().__init__()
        self.sa = SASRecLayer(hidden_size, n_heads, inner_size, hidden_dropout, attn_dropout)
        self.alpha = nn.Parameter(torch.tensor(0.0))   # sigmoid(0) = 0.5 초기값
        self.freq_proj = nn.Linear(hidden_size, hidden_size)
        self.freq_ln   = nn.LayerNorm(hidden_size, eps=1e-8)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h_attn = self.sa(x, attn_mask)                    # [B, L, H]

        # FFT 저역통과: 상위 절반 주파수 제거
        f = torch.fft.rfft(h_attn, dim=1, norm="ortho")   # [B, L//2+1, H]
        keep = max(1, f.shape[1] // 2)
        f_low = f.clone()
        f_low[:, keep:] = 0
        h_freq = torch.fft.irfft(f_low, n=h_attn.shape[1], dim=1, norm="ortho")
        h_freq = self.freq_ln(self.freq_proj(h_freq))

        a = torch.sigmoid(self.alpha)
        return a * h_attn + (1 - a) * h_freq


class BSARec(nn.Module):
    """BSARec: Behavior Sequence-Aware Recommendation (AAAI 2024).

    SASRec 레이어를 BSARecLayer로 교체하여 주파수·시간 도메인 정보를 혼합.
    """

    def __init__(self, cfg: DictConfig, n_items: int):
        super().__init__()
        self.n_items     = n_items
        self.hidden_size = cfg.hidden_size
        self.max_seq_len = cfg.max_seq_len

        self.item_emb    = nn.Embedding(n_items + 1, cfg.hidden_size, padding_idx=0)
        self.pos_emb     = nn.Embedding(cfg.max_seq_len, cfg.hidden_size)
        self.emb_dropout = nn.Dropout(cfg.hidden_dropout)

        self.layers = nn.ModuleList([
            BSARecLayer(cfg.hidden_size, cfg.n_heads, cfg.inner_size,
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
        B, L = seq.shape
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, -1)
        x = self.emb_dropout(self.item_emb(seq) + self.pos_emb(pos))
        causal_mask = torch.triu(
            torch.ones(L, L, device=seq.device, dtype=torch.bool), diagonal=1
        )
        for layer in self.layers:
            x = layer(x, causal_mask)
        return self.ln_out(x)

    def _last_hidden(self, seq: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        lengths = (seq != 0).sum(dim=1) - 1
        lengths = lengths.clamp(min=0)
        idx = lengths.view(-1, 1, 1).expand(-1, 1, self.hidden_size)
        return h.gather(1, idx).squeeze(1)

    def calculate_loss(self, seq, pos, neg, weights) -> torch.Tensor:
        h = self._encode(seq)
        h_last = self._last_hidden(seq, h)
        pos_score = (h_last * self.item_emb(pos)).sum(-1)
        neg_score = (h_last * self.item_emb(neg)).sum(-1)
        return (-F.logsigmoid(pos_score - neg_score) * weights).mean()

    def full_sort_predict(self, seq: torch.Tensor) -> torch.Tensor:
        h = self._encode(seq)
        h_last = self._last_hidden(seq, h)
        return h_last @ self.item_emb.weight[1:].T
