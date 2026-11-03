"""The two parallel branches of the proposed model.

Branch 1 (temporal): Transformer over the Sentinel-2 series -> per-step tokens; pooled to 64-d.
Branch 2 (static):  elevation, slope, irrigation system and municipality -> a 64-d summary,
                    plus one token per static fact for cross-attention.
"""
from __future__ import annotations

import torch
from torch import nn

from models.common import TemporalEncoder


class TemporalBranch(TemporalEncoder):
    """Branch 1. Same encoder as the baseline, but fed only the 12 time-series channels."""


class StaticBranch(nn.Module):
    N_TOKENS = 4  # elevation, slope, irrigation system, municipality

    def __init__(self, n_sreg: int, n_mun: int, d_model: int = 64, emb_dim: int = 16,
                 dropout: float = 0.1):
        super().__init__()
        self.sreg_emb = nn.Embedding(n_sreg + 1, emb_dim, padding_idx=0)
        self.mun_emb = nn.Embedding(n_mun + 1, emb_dim, padding_idx=0)
        self.cont_proj = nn.Linear(2, 32)
        self.mlp = nn.Sequential(
            nn.Linear(32 + 2 * emb_dim, d_model), nn.LayerNorm(d_model), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_model, d_model),
        )
        # One token per static fact (for cross-attention), each with a learned type embedding.
        self.elev_tok = nn.Linear(1, d_model)
        self.slope_tok = nn.Linear(1, d_model)
        self.sreg_tok = nn.Linear(emb_dim, d_model)
        self.mun_tok = nn.Linear(emb_dim, d_model)
        self.type_emb = nn.Parameter(torch.zeros(self.N_TOKENS, d_model))
        nn.init.normal_(self.type_emb, std=0.02)

    def forward(self, static_cont: torch.Tensor, static_cat: torch.Tensor):
        e_sreg = self.sreg_emb(static_cat[:, 0])
        e_mun = self.mun_emb(static_cat[:, 1])
        h_static = self.mlp(torch.cat([torch.relu(self.cont_proj(static_cont)), e_sreg, e_mun], dim=-1))
        tokens = torch.stack([
            self.elev_tok(static_cont[:, :1]),
            self.slope_tok(static_cont[:, 1:2]),
            self.sreg_tok(e_sreg),
            self.mun_tok(e_mun),
        ], dim=1) + self.type_emb  # (B, 4, d)
        return h_static, tokens
