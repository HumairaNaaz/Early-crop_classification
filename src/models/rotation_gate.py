"""OWNER: Gate & Loss Lead (Deepasha).

Contract:
  expected_profiles: computed ONCE from 2022 training fields only (no test leakage).
  divergence(x_temporal (B,T,12), mask, y_prev (B,)) -> D_rot (B,)   in [0, 2]
  RotationGate.forward(h_temp (B,64), e_prior (B,64), d_rot (B,))    -> g (B,1) in [0,1]
Log gate values for rotated vs non-rotated fields (needed for a paper figure).
"""
import torch.nn as nn


def divergence(*a, **k):
    raise NotImplementedError("TODO(Deepasha)")


class RotationGate(nn.Module):
    def __init__(self, *a, **k):
        super().__init__(); raise NotImplementedError("TODO(Deepasha)")
