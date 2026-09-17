from __future__ import annotations

import torch
from torch import nn


class AttentionFusion(nn.Module):
    """Manuscript-aligned concat -> scale -> Linear(256) -> 4-head MHA -> LN.

    The manuscript describes MHA after the three vectors have already been
    concatenated into a single 404-D vector. In PyTorch this corresponds to a
    sequence length of one. This is implemented literally for reproducibility;
    it should be interpreted as attention-based feature refinement, not as
    inter-token/cross-modality attention.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        h = self.proj(x)
        seq = h.unsqueeze(1)  # [B,1,H]
        attn, _ = self.attn(seq, seq, seq, need_weights=False)
        out = self.norm(seq + self.dropout(attn)).squeeze(1)
        return out
