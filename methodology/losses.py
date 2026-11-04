
"""Loss functions for early crop classification.

Supports standard cross-entropy and class-balanced focal loss.
Class counts must be calculated from the training split only.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def effective_number_weights(
    class_counts,
    beta: float = 0.999,
) -> torch.Tensor:
    """Calculate normalized class-balanced weights.

    Uses the effective number of samples:
        E_n = (1 - beta^n) / (1 - beta)

    The resulting class weights are normalized to sum to num_classes.
    """

    counts = torch.as_tensor(class_counts, dtype=torch.float32)

    if counts.ndim != 1 or counts.numel() == 0:
        raise ValueError("class_counts must be a non-empty 1D sequence")

    if not torch.isfinite(counts).all():
        raise ValueError("class_counts must contain finite values")

    if (counts < 0).any():
        raise ValueError("class_counts cannot be negative")

    if (counts == 0).any():
        raise ValueError("Every class must have at least one training sample")

    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in the range [0, 1)")

    if beta == 0.0:
        weights = torch.ones_like(counts)
    else:
        weights = (1.0 - beta) / (
            1.0 - torch.pow(torch.tensor(beta), counts)
        )

    weights = weights / weights.mean()

    return weights


class ClassBalancedFocalLoss(nn.Module):
    """Class-balanced focal loss for multi-class classification."""

    def __init__(
        self,
        class_counts,
        beta: float = 0.999,
        gamma: float = 1.0,
    ):
        super().__init__()

        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be finite and non-negative")

        weights = effective_number_weights(class_counts, beta)

        self.register_buffer("class_weights", weights)
        self.gamma = float(gamma)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        if logits.ndim != 2:
            raise ValueError("logits must have shape (B, C)")

        if targets.ndim != 1:
            raise ValueError("targets must have shape (B,)")

        if logits.shape[0] != targets.shape[0]:
            raise ValueError("logits and targets must have matching batch sizes")

        if logits.shape[1] != self.class_weights.numel():
            raise ValueError("logit class count does not match class_weights")

        if targets.dtype != torch.long:
            raise ValueError("targets must have dtype torch.long")

        if targets.numel() == 0:
            raise ValueError("targets cannot be empty")

        if (targets < 0).any() or (targets >= logits.shape[1]).any():
            raise ValueError("targets contain invalid class indices")

        log_probs = F.log_softmax(logits.float(), dim=1)

        log_pt = log_probs.gather(
            dim=1,
            index=targets.unsqueeze(1),
        ).squeeze(1)

        pt = log_pt.exp()

        alpha = self.class_weights[targets]

        focal_factor = (1.0 - pt).pow(self.gamma)

        loss = -alpha * focal_factor * log_pt

        return loss.mean()


def build_loss(
    loss_cfg: dict,
    class_counts,
) -> nn.Module:
    """Construct the configured classification loss."""

    loss_type = loss_cfg["type"].lower()

    if loss_type == "ce":
        return nn.CrossEntropyLoss()

    if loss_type == "cb_focal":
        return ClassBalancedFocalLoss(
            class_counts=class_counts,
            beta=float(loss_cfg["cb_beta"]),
            gamma=float(loss_cfg["focal_gamma"]),
        )

    raise ValueError(f"Unknown loss type: {loss_type!r}")
