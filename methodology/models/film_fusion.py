"""Fusion of the two branches: cross-attention and FiLM.

Cross-attention: each time step of the satellite series queries the four static-fact
tokens, so the series can pick up the field context that matters at that point in the
season. (With only one static vector as key, attention weights are always 1 and the
layer degenerates to adding a constant; four separate tokens avoid that.)

FiLM: the static summary scales and shifts the pooled temporal features,
    H_fused = gamma(h_static) * h_temp + beta(h_static)
gamma starts at 1 and beta at 0, so training begins from the plain temporal features.
"""
from __future__ import annotations

import torch
from torch import nn


class StaticCrossAttention(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor, static_tokens: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(tokens, static_tokens, static_tokens, need_weights=False)
        return self.norm(tokens + out)


class FiLM(nn.Module):
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.to_gamma = nn.Linear(d_model, d_model)
        self.to_beta = nn.Linear(d_model, d_model)
        for lin in (self.to_gamma, self.to_beta):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, h_temp: torch.Tensor, h_static: torch.Tensor) -> torch.Tensor:
        gamma = 1.0 + self.to_gamma(h_static)
        beta = self.to_beta(h_static)
        return gamma * h_temp + beta


class ConcatFusion(nn.Module):
    """Plain concatenation, for comparison."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.proj = nn.Linear(2 * d_model, d_model)

    def forward(self, h_temp: torch.Tensor, h_static: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([h_temp, h_static], dim=-1))
