"""OWNER: Model Lead (Anshika). Reproduction of Gene-Mola et al. (2026).
Gate for Phase 2: reproduction must match the authors' reported behaviour first.
Interface: forward(x_temporal (B,T,C_in), mask (B,T), static_feats ...) -> logits (B,18)."""
import torch.nn as nn


class BaselineTransformer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("TODO(Anshika)")
