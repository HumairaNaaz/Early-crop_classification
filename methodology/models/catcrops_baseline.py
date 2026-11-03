"""Run A (v2): faithful re-implementation of the CatCrops baseline (Gene-Mola et al., 2026).

Mirrors catcrops/models/TransformerModel.py (itself BreizhCrops' TransformerEncoder) and the
input format of catcrops/transforms.py in the official repository
(https://github.com/UEA-Irta/CatCrops):

    input       12 channels per time step: 10 bands, cloud probability, day of year
    ancillary   each ancillary value is appended as ONE EXTRA TIME STEP whose 12 channels all
                hold that value (np.vstack in transforms.py): previous crop, irrigation system,
                municipality, elevation, slope, scaled as in the official code
    model       Linear(12 -> 64) -> ReLU -> 5 post-norm Transformer encoder layers
                (2 heads, feed-forward 128, dropout 0.018) + final LayerNorm
                -> max over time -> ReLU -> Linear(64 -> 18)
    no positional encoding (time enters through the day-of-year channel)

One deliberate difference: the official code feeds a zero-filled 365-day daily grid ("sparse"
type C input); here the same 365-day window is given as the list of observed dates with a
padding mask, so empty days are skipped rather than entered as zeros. Band scaling differs only
by a per-channel affine transform, which the first linear layer absorbs.
"""
from __future__ import annotations

import torch
from torch import nn

from models.common import masked_max_pool

OFFICIAL_DROPOUT = 0.017998950510888446  # default in the official TransformerModel


class CatCropsTransformer(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 12, d_model: int = 64, n_heads: int = 2,
                 n_layers: int = 5, d_ff: int = 128, dropout: float = OFFICIAL_DROPOUT):
        super().__init__()
        self.num_classes = num_classes
        self.inlinear = nn.Linear(in_channels, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, activation="relu",
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, norm=nn.LayerNorm(d_model),
                                             enable_nested_tensor=False)
        self.outlinear = nn.Linear(d_model, num_classes)

    def forward(self, batch: dict) -> dict:
        x, mask, anc = batch["x"], batch["mask"], batch["anc"]
        B, _, C = x.shape
        anc_steps = anc.to(x.dtype).unsqueeze(-1).expand(B, anc.shape[1], C)  # one step per value
        seq = torch.cat([x, anc_steps], dim=1)
        valid = torch.cat([mask, torch.ones(B, anc.shape[1], dtype=torch.bool, device=x.device)], dim=1)
        h = torch.relu(self.inlinear(seq))
        h = self.encoder(h, src_key_padding_mask=~valid)
        logits = self.outlinear(torch.relu(masked_max_pool(h, valid)))
        zeros = torch.zeros(B, device=x.device)
        return {"logits": logits, "gate": zeros, "divergence": zeros}
