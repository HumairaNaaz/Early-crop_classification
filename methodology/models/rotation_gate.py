"""Rotation-aware adaptive gating.

Expected profiles are computed from the 2022 TRAINING split only and have shape
(C, n_bins, 10). Inputs to ``RotationGate`` follow the methodology pipeline:

    x       : (B, T, 12), first 10 channels are normalised Sentinel-2 bands
    days    : (B, T), integer day offsets (day 0 = 1 January)
    mask    : (B, T), True for real observations
    y_prev  : (B,), previous-crop class indices in [0, C-1], or -1 if unknown
    h_temp  : (B, d_model), temporal representation
    e_prev  : (B, d_model), previous-crop embedding

The gate returns ``(gamma, d_rot)`` where gamma and divergence have shape (B,).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

N_BANDS = 10
PROFILE_BIN_DAYS = 5


def divergence(
    x: torch.Tensor,
    days: torch.Tensor,
    mask: torch.Tensor,
    y_prev: torch.Tensor,
    profiles: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate mean cosine divergence from the previous crop's expected profile.

    Returns ``(d_rot, has_obs, known)``. ``d_rot`` is zero when no eligible
    observations exist; ``has_obs`` distinguishes that case from true zero
    divergence. Unknown/out-of-range previous-crop labels never index profiles.
    """
    profiles = torch.as_tensor(profiles, device=x.device, dtype=torch.float32)
    if profiles.ndim != 3 or profiles.shape[-1] != N_BANDS:
        raise ValueError(
            f"profiles must have shape (num_classes, num_bins, {N_BANDS}); "
            f"got {tuple(profiles.shape)}"
        )
    if profiles.shape[0] < 1 or profiles.shape[1] < 1:
        raise ValueError("profiles must contain at least one class and one time bin")
    if not torch.isfinite(profiles).all():
        raise ValueError("profiles contain non-finite values")
    if x.ndim != 3 or x.shape[-1] < N_BANDS:
        raise ValueError(f"x must have shape (B, T, >= {N_BANDS})")
    if days.shape != x.shape[:2] or mask.shape != x.shape[:2]:
        raise ValueError("days and mask must both have shape (B, T)")
    if y_prev.ndim != 1 or y_prev.shape[0] != x.shape[0]:
        raise ValueError("y_prev must have shape (B,)")

    days = days.to(device=x.device, dtype=torch.long)
    mask = mask.to(device=x.device, dtype=torch.bool)
    y_prev = y_prev.to(device=x.device, dtype=torch.long)
    n_classes, n_bins, _ = profiles.shape

    known = (y_prev >= 0) & (y_prev < n_classes)
    valid = mask & (days >= 0) & known.unsqueeze(1)

    # Clamp only for safe indexing. Invalid observations are removed by `valid`.
    safe_days = days.clamp(min=0)
    bins = (safe_days // PROFILE_BIN_DAYS).clamp(max=n_bins - 1)
    safe_labels = y_prev.clamp(min=0, max=n_classes - 1)
    expected = profiles[safe_labels.unsqueeze(1), bins]  # (B, T, 10)
    observed = x[..., :N_BANDS].to(dtype=torch.float32)

    finite_obs = torch.isfinite(observed).all(dim=-1)
    valid = valid & finite_obs
    # Replace non-finite values before cosine_similarity; the validity mask
    # ensures these positions contribute exactly zero.
    observed = torch.nan_to_num(observed, nan=0.0, posinf=0.0, neginf=0.0)
    cosine = F.cosine_similarity(observed, expected, dim=-1, eps=1e-6)
    per_step = (1.0 - cosine).clamp(min=0.0, max=2.0)
    per_step = per_step * valid.to(per_step.dtype)

    counts = valid.sum(dim=1)
    has_obs = counts > 0
    d_rot = per_step.sum(dim=1) / counts.clamp_min(1).to(per_step.dtype)
    # Guard against tiny floating-point excursions outside cosine-distance bounds.
    d_rot = d_rot.clamp(0.0, 2.0)
    return d_rot, has_obs.to(torch.float32), known.to(torch.float32)


class RotationGate(nn.Module):
    """Learn a gate that suppresses the previous-crop embedding under divergence."""

    def __init__(self, profiles: torch.Tensor, d_model: int = 64):
        super().__init__()
        profiles = torch.as_tensor(profiles, dtype=torch.float32)
        if profiles.ndim != 3 or profiles.shape[-1] != N_BANDS:
            raise ValueError(
                f"profiles must have shape (num_classes, num_bins, {N_BANDS})"
            )
        if profiles.shape[0] < 1 or profiles.shape[1] < 1:
            raise ValueError("profiles must contain at least one class and one time bin")
        if not torch.isfinite(profiles).all():
            raise ValueError("profiles contain non-finite values")
        if d_model <= 0:
            raise ValueError("d_model must be positive")

        self.register_buffer("profiles", profiles.contiguous())
        self.d_model = int(d_model)
        self.lin = nn.Linear(2 * self.d_model + 1, 1)
        # softplus(0.5413) is approximately 1, giving a positive initial sensitivity.
        self.beta_raw = nn.Parameter(torch.tensor(0.5413, dtype=torch.float32))

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self.beta_raw)

    def divergence(self, x, days, mask, y_prev):
        return divergence(x, days, mask, y_prev, self.profiles)

    def forward(self, x, days, mask, y_prev, h_temp, e_prev):
        if h_temp.ndim != 2 or e_prev.ndim != 2:
            raise ValueError("h_temp and e_prev must have shape (B, d_model)")
        if h_temp.shape != e_prev.shape or h_temp.shape[1] != self.d_model:
            raise ValueError(
                f"h_temp and e_prev must both have shape (B, {self.d_model})"
            )
        if h_temp.shape[0] != x.shape[0]:
            raise ValueError("embedding batch size must match x")

        d_rot, has_obs, known = self.divergence(x, days, mask, y_prev)
        gate_features = torch.cat(
            [h_temp.float(), e_prev.float(), has_obs.unsqueeze(-1)], dim=-1
        )
        logit = self.lin(gate_features).squeeze(-1)
        gamma = torch.sigmoid(logit - self.beta * d_rot)
        gamma = gamma * known * has_obs
        return gamma, d_rot
