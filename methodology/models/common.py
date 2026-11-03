"""Shared building blocks: day positional encoding, the temporal Transformer, masked max-pooling."""
from __future__ import annotations

import math

import torch
from torch import nn

DAY_SHIFT = 366  # day offsets run from about -366 to 365; shifted to be non-negative


class DayPositionalEncoding(nn.Module):
    """Sinusoidal encoding of each observation's day offset (not its position in the list).

    Observations are irregular (clouds, missing dates), so the encoding uses the actual
    acquisition day rather than the index in the sequence.
    """

    def __init__(self, d_model: int, period: float = 10000.0):
        super().__init__()
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(period) / d_model))
        self.register_buffer("div", div, persistent=False)
        self.d_model = d_model

    def forward(self, days: torch.Tensor) -> torch.Tensor:  # (B, T) -> (B, T, d)
        pos = (days.float() + DAY_SHIFT).unsqueeze(-1) * self.div
        pe = torch.zeros(*days.shape, self.d_model, device=days.device)
        pe[..., 0::2] = torch.sin(pos)
        pe[..., 1::2] = torch.cos(pos)
        return pe


class TemporalEncoder(nn.Module):
    """Per-step projection + Transformer encoder (Russwurm & Koerner style; 5 blocks, d=64)."""

    def __init__(self, in_channels: int, d_model: int = 64, n_heads: int = 2, n_layers: int = 5,
                 d_ff: int = 128, dropout: float = 0.1, positional_encoding: bool = True):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_channels)
        self.in_proj = nn.Linear(in_channels, d_model)
        self.act = nn.ReLU()
        self.pe = DayPositionalEncoding(d_model) if positional_encoding else None
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, activation="relu",
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, days: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.act(self.in_proj(self.in_norm(x)))
        if self.pe is not None:
            h = h + self.pe(days).to(h.dtype)
        h = self.encoder(h, src_key_padding_mask=~mask)
        return self.out_norm(h)  # (B, T, d)


def masked_max_pool(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Max over valid time steps only. Every sample has at least one valid step."""
    h = h.masked_fill(~mask.unsqueeze(-1), torch.finfo(h.dtype).min)
    return h.max(dim=1).values
