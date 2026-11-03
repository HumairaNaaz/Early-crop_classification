"""Run A: baseline in the style of Gene-Mola et al. (2026).

One Transformer over the time series, with the ancillary data (elevation, slope,
irrigation system, municipality, previous-year crop) appended as extra columns to
EVERY time step. This is the "crude mixing" our dual-branch model replaces.

Settings are provisional until checked against the baseline paper.
"""
from __future__ import annotations

import torch
from torch import nn

from models.common import TemporalEncoder, masked_max_pool


class BaselineTransformer(nn.Module):
    def __init__(self, num_classes: int, n_sreg: int, n_mun: int, in_channels: int = 12,
                 d_model: int = 64, n_heads: int = 2, n_layers: int = 5, d_ff: int = 128,
                 dropout: float = 0.1, positional_encoding: bool = True, anc_emb_dim: int = 8):
        super().__init__()
        self.num_classes = num_classes
        self.sreg_emb = nn.Embedding(n_sreg + 1, anc_emb_dim, padding_idx=0)
        self.mun_emb = nn.Embedding(n_mun + 1, anc_emb_dim, padding_idx=0)
        self.prev_emb = nn.Embedding(num_classes + 1, anc_emb_dim, padding_idx=0)
        n_anc = 2 + 3 * anc_emb_dim
        self.encoder = TemporalEncoder(in_channels + n_anc, d_model, n_heads, n_layers, d_ff,
                                       dropout, positional_encoding)
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(d_model, num_classes))

    def forward(self, batch: dict) -> dict:
        x, days, mask = batch["x"], batch["days"], batch["mask"]
        cat = batch["static_cat"]
        prev_idx = (batch["y_prev"] + 1).clamp(min=0)
        anc = torch.cat([
            batch["static_cont"],
            self.sreg_emb(cat[:, 0]),
            self.mun_emb(cat[:, 1]),
            self.prev_emb(prev_idx),
        ], dim=-1)
        anc = anc.unsqueeze(1).expand(-1, x.shape[1], -1)
        h = self.encoder(torch.cat([x, anc], dim=-1), days, mask)
        logits = self.head(masked_max_pool(h, mask))
        zeros = torch.zeros(x.shape[0], device=x.device)
        return {"logits": logits, "gate": zeros, "divergence": zeros}
