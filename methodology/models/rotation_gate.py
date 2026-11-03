"""Rotation-Aware Adaptive Gate.

For a field whose previous-year crop p is known, compare each current-season
observation x_t (10 normalised bands) with the expected profile of crop p at that
time of year, mu_p(t), learned from the 2022 training fields:

    D_t   = 1 - cos(x_t, mu_p(t))              divergence per observation
    D_rot = mean of D_t over this season's observations so far
    gamma = sigmoid( W_g [h_temp | e_prev | has_obs] + b  -  beta * D_rot ),  beta > 0

gamma is in [0, 1] and scales the previous-crop embedding: when the field looks
different from last year's crop, gamma drops and the model leans on the satellite signal.
gamma is forced to 0 when the previous crop is unknown.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

N_BANDS = 10
PROFILE_BIN_DAYS = 5


class RotationGate(nn.Module):
    def __init__(self, profiles: torch.Tensor, d_model: int = 64):
        super().__init__()
        profiles = torch.as_tensor(profiles, dtype=torch.float32)  # (C, n_bins, 10)
        self.register_buffer("profiles", profiles)
        self.n_bins = profiles.shape[1]
        self.lin = nn.Linear(2 * d_model + 1, 1)
        # softplus(0.5413) = 1.0, so beta starts at 1
        self.beta_raw = nn.Parameter(torch.tensor(0.5413))

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self.beta_raw)

    def divergence(self, x, days, mask, y_prev):
        known = y_prev >= 0
        cur = mask & (days >= 0) & known.unsqueeze(1)
        bins = (days.clamp(min=0) // PROFILE_BIN_DAYS).clamp(max=self.n_bins - 1)
        p = y_prev.clamp(min=0)
        expected = self.profiles[p.unsqueeze(1), bins]                  # (B, T, 10)
        obs = x[..., :N_BANDS].float()
        cos = F.cosine_similarity(obs, expected, dim=-1, eps=1e-6)       # (B, T)
        d = (1.0 - cos) * cur
        n = cur.sum(dim=1)
        d_rot = d.sum(dim=1) / n.clamp(min=1)
        return d_rot, (n > 0).float(), known.float()

    def forward(self, x, days, mask, y_prev, h_temp, e_prev):
        d_rot, has_obs, known = self.divergence(x, days, mask, y_prev)
        logit = self.lin(torch.cat([h_temp.float(), e_prev.float(), has_obs.unsqueeze(-1)], dim=-1)).squeeze(-1)
        gamma = torch.sigmoid(logit - self.beta * d_rot) * known
        return gamma, d_rot
