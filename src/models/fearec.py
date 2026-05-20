import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from .sasrec import SASRec


class FEARec(nn.Module):
    """FEARec: Frequency Enhanced Augmented Contrastive Learning (SIGIR 2023).

    SASRec 인코더 위에 FFT 주파수 도메인 증강 + InfoNCE 대조학습 추가.
    증강 뷰: 아이템 임베딩에 FFT → 랜덤 주파수 마스킹 → IFFT → 동일 트랜스포머 통과.
    """

    def __init__(self, cfg: DictConfig, n_items: int):
        super().__init__()
        self.encoder = SASRec(cfg, n_items)
        self.lmd = cfg.lmd
        self.tau = cfg.tau
        self.freq_mask_ratio = cfg.freq_mask_ratio
        self.hidden_size = self.encoder.hidden_size
        self.max_seq_len = self.encoder.max_seq_len

    def _freq_augment(self, emb: torch.Tensor) -> torch.Tensor:
        """[B, L, H] 임베딩에 FFT 주파수 마스킹 적용 후 IFFT."""
        f = torch.fft.rfft(emb, dim=1, norm="ortho")   # [B, L//2+1, H]
        n_freq = f.shape[1]
        n_mask = max(1, int(n_freq * self.freq_mask_ratio))
        aug = f.clone()
        for i in range(f.shape[0]):
            idx = torch.randperm(n_freq, device=f.device)[:n_mask]
            aug[i, idx] = 0
        return torch.fft.irfft(aug, n=emb.shape[1], dim=1, norm="ortho")

    def _encode_raw(self, item_emb: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        """사전 계산된 item_emb [B,L,H]를 SASRec 레이어에 통과."""
        B, L, _ = item_emb.shape
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, -1)
        x = self.encoder.emb_dropout(item_emb + self.encoder.pos_emb(pos))
        causal_mask = torch.triu(
            torch.ones(L, L, device=seq.device, dtype=torch.bool), diagonal=1
        )
        for layer in self.encoder.layers:
            x = layer(x, causal_mask)
        return self.encoder.ln_out(x)

    def _contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        sim = z1 @ z2.T / self.tau
        labels = torch.arange(z1.size(0), device=z1.device)
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2

    def calculate_loss(self, seq, pos, neg, weights) -> torch.Tensor:
        rec_loss = self.encoder.calculate_loss(seq, pos, neg, weights)

        item_emb = self.encoder.item_emb(seq)               # [B, L, H]
        aug_emb  = self._freq_augment(item_emb)             # [B, L, H]

        h_orig = self._encode_raw(item_emb, seq)
        h_aug  = self._encode_raw(aug_emb,  seq)
        z1 = self.encoder._last_hidden(seq, h_orig)
        z2 = self.encoder._last_hidden(seq, h_aug)
        cl_loss = self._contrastive_loss(z1, z2)

        return rec_loss + self.lmd * cl_loss

    def full_sort_predict(self, seq: torch.Tensor) -> torch.Tensor:
        return self.encoder.full_sort_predict(seq)
