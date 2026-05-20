import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class TiSASRecLayer(nn.Module):
    """시간 간격을 attention weight에 직접 반영하는 self-attention layer."""

    def __init__(self, hidden_size: int, n_heads: int, inner_size: int,
                 hidden_dropout: float, attn_dropout: float, time_span: int):
        super().__init__()
        assert hidden_size % n_heads == 0
        self.n_heads   = n_heads
        self.head_dim  = hidden_size // n_heads
        self.scale     = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        # 시간 간격 임베딩: 버킷 수 = time_span + 1 (0 = same-time)
        self.time_emb_k = nn.Embedding(time_span + 1, self.head_dim)
        self.time_emb_q = nn.Embedding(time_span + 1, self.head_dim)

        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(hidden_dropout)

        self.ln1 = nn.LayerNorm(hidden_size, eps=1e-8)
        self.ln2 = nn.LayerNorm(hidden_size, eps=1e-8)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, inner_size),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(inner_size, hidden_size),
            nn.Dropout(hidden_dropout),
        )
        self.time_span = time_span

    def _time_bucket(self, time_matrix: torch.Tensor) -> torch.Tensor:
        """시간 간격(초) → 버킷 인덱스 [0, time_span] (long tensor)."""
        pos_mask = time_matrix > 0
        bucket = torch.zeros_like(time_matrix, dtype=torch.long)
        vals = (
            torch.log1p(time_matrix[pos_mask].float()) /
            math.log(1 + 86400 * 120) *   # 120일 최대값으로 정규화
            self.time_span
        ).long().clamp(1, self.time_span)
        bucket[pos_mask] = vals
        return bucket

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor,
                time_matrix: torch.Tensor) -> torch.Tensor:
        # x: [B, L, H], time_matrix: [B, L, L] 시간 간격(초)
        B, L, H = x.shape
        nh, hd = self.n_heads, self.head_dim

        residual = x
        x_ln = self.ln1(x)

        Q = self.q_proj(x_ln).view(B, L, nh, hd).transpose(1, 2)  # [B, nh, L, hd]
        K = self.k_proj(x_ln).view(B, L, nh, hd).transpose(1, 2)
        V = self.v_proj(x_ln).view(B, L, nh, hd).transpose(1, 2)

        # 시간 간격 임베딩: [B, L, L, hd]
        time_bucket = self._time_bucket(time_matrix)           # [B, L, L]
        T_k = self.time_emb_k(time_bucket)                    # [B, L, L, hd]
        T_q = self.time_emb_q(time_bucket)                    # [B, L, L, hd]

        # Content attention: [B, nh, L, L]
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Time-aware bias (TiSASRec 논문 수식):
        # Q*T_k^T + Q*T_q^T (symmetric time bias)
        # [B, nh, L, hd] x [B, L, L, hd]^T → [B, nh, L, L]
        Q_exp = Q.permute(0, 2, 1, 3)  # [B, L, nh, hd]
        time_bias_k = torch.einsum("blhd,bljd->bhlj", Q_exp, T_k)
        time_bias_q = torch.einsum("blhd,bljd->bhjl", Q_exp, T_q)
        attn = attn + (time_bias_k + time_bias_q) / self.scale

        # causal mask만 적용 (key_padding_mask 제거 — NaN 방지)
        if causal_mask is not None:
            attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = self.attn_drop(F.softmax(attn, dim=-1))

        out = torch.matmul(attn, V)                            # [B, nh, L, hd]
        out = out.transpose(1, 2).contiguous().view(B, L, H)
        out = self.proj_drop(self.o_proj(out))
        x = residual + out

        residual = x
        x = residual + self.ffn(self.ln2(x))
        return x


class TiSASRec(nn.Module):
    """TiSASRec: Time Interval Aware Self-Attention (WSDM 2020)."""

    def __init__(self, cfg: DictConfig, n_items: int):
        super().__init__()
        self.n_items     = n_items
        self.hidden_size = cfg.hidden_size
        self.max_seq_len = cfg.max_seq_len
        self.time_span   = cfg.time_span

        self.item_emb = nn.Embedding(n_items + 1, cfg.hidden_size, padding_idx=0)
        self.pos_emb  = nn.Embedding(cfg.max_seq_len, cfg.hidden_size)
        self.emb_dropout = nn.Dropout(cfg.hidden_dropout)

        self.layers = nn.ModuleList([
            TiSASRecLayer(cfg.hidden_size, cfg.n_heads, cfg.inner_size,
                          cfg.hidden_dropout, cfg.attn_dropout, cfg.time_span)
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

    def _build_time_matrix(self, seq: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """times: [B, L] unix 초 → time_matrix: [B, L, L] 간격(초, 절대값).
        패딩 위치(seq==0)는 0으로 마스킹 — times=0과 실제 타임스탬프(~1.57e9)가
        섞이면 간격이 최대값(~18,200일)으로 계산되어 time bucket이 오염됨.
        """
        t = times.float()                              # [B, L]
        mat = (t.unsqueeze(2) - t.unsqueeze(1)).abs() # [B, L, L]
        pad = (seq == 0)                               # [B, L]
        mat = mat.masked_fill(pad.unsqueeze(1) | pad.unsqueeze(2), 0)
        return mat

    def _encode(self, seq: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        B, L = seq.shape
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, -1)

        x = self.item_emb(seq) + self.pos_emb(pos)
        x = self.emb_dropout(x)

        causal_mask = torch.triu(
            torch.ones(L, L, device=seq.device, dtype=torch.bool), diagonal=1
        )
        time_matrix = self._build_time_matrix(seq, times)  # [B, L, L]

        for layer in self.layers:
            x = layer(x, causal_mask, time_matrix)

        return self.ln_out(x)

    def _last_hidden(self, seq: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        lengths = (seq != 0).sum(dim=1) - 1
        lengths = lengths.clamp(min=0)
        idx = lengths.view(-1, 1, 1).expand(-1, 1, self.hidden_size)
        return h.gather(1, idx).squeeze(1)

    def calculate_loss(self, seq, times, pos, neg, weights) -> torch.Tensor:
        h = self._encode(seq, times)
        h_last = self._last_hidden(seq, h)

        pos_score = (h_last * self.item_emb(pos)).sum(-1)
        neg_score = (h_last * self.item_emb(neg)).sum(-1)
        return (-F.logsigmoid(pos_score - neg_score) * weights).mean()

    def full_sort_predict(self, seq: torch.Tensor,
                          times: torch.Tensor = None) -> torch.Tensor:
        if times is None:
            times = torch.zeros_like(seq)
        h = self._encode(seq, times)
        h_last = self._last_hidden(seq, h)
        return h_last @ self.item_emb.weight[1:].T
