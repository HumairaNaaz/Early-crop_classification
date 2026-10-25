"""OWNER: Model Lead (Anshika). Temporal branch, static branch, FiLM / cross-attention fusion.

Contract:
  TemporalBranch: (B,T,12), mask (B,T)            -> h_temp   (B,64)
  StaticBranch:   cont (B,n_cont), irr (B,), mun (B,) -> h_stat (B,64)
  FiLMFusion:     h_temp (B,64), h_stat (B,64)    -> (B,64)   [scale s(h_stat) * h_temp + o(h_stat)]
  DualBranchModel.forward(batch, gate=None)        -> logits (B,18)
"""
import torch.nn as nn


class TemporalBranch(nn.Module):
    def __init__(self, *a, **k):
        super().__init__(); raise NotImplementedError("TODO(Anshika)")


class StaticBranch(nn.Module):
    def __init__(self, *a, **k):
        super().__init__(); raise NotImplementedError("TODO(Anshika)")


class FiLMFusion(nn.Module):
    def __init__(self, *a, **k):
        super().__init__(); raise NotImplementedError("TODO(Anshika)")


class DualBranchModel(nn.Module):
    def __init__(self, *a, **k):
        super().__init__(); raise NotImplementedError("TODO(Anshika)")
