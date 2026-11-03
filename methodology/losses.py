"""Loss functions: plain cross-entropy (runs A-C) and Class-Balanced Focal Loss (run D).

Class-Balanced Focal Loss (Cui et al. 2019 + Lin et al. 2017):
    alpha_c = (1 - beta) / (1 - beta ** n_c)        effective-number weight of class c
    L       = - alpha_y * (1 - p_y) ** gamma * log p_y
alpha is normalised to sum to the number of classes.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def class_balanced_weights(counts, beta: float = 0.9999) -> np.ndarray:
    counts = np.maximum(np.asarray(counts, dtype=np.float64), 1.0)
    effective = 1.0 - np.power(beta, counts)
    w = (1.0 - beta) / effective
    return (w / w.sum() * len(counts)).astype(np.float32)


class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, class_counts, beta: float = 0.9999, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(class_balanced_weights(class_counts, beta)))
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits.float(), dim=-1)
        logp_y = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        p_y = logp_y.exp()
        loss = -self.alpha[target] * (1.0 - p_y).pow(self.gamma) * logp_y
        return loss.mean()


def build_loss(loss_cfg: dict, class_counts) -> nn.Module:
    if loss_cfg["type"] == "ce":
        return nn.CrossEntropyLoss()
    if loss_cfg["type"] == "cb_focal":
        return ClassBalancedFocalLoss(class_counts, loss_cfg["cb_beta"], loss_cfg["focal_gamma"])
    raise ValueError(f"unknown loss {loss_cfg['type']!r}")
