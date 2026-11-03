"""The proposed dual-branch model (runs B, C, D) and a factory for every run.

    Branch 1  temporal Transformer -> tokens -(cross-attention to static tokens)-> max-pool -> h_temp
    Branch 2  static MLP -> h_static (+ 4 static tokens)
    Fusion    H = FiLM(h_temp; h_static) + gamma * e_prev
    Head      MLP -> 18 classes

gate = "none"     previous crop not used
gate = "fixed"    previous crop always added (gamma = 1 when known)      -> run B
gate = "rotation" gamma from the Rotation-Aware Adaptive Gate             -> runs C, D
"""
from __future__ import annotations

import torch
from torch import nn

from models.baseline import BaselineTransformer
from models.catcrops_baseline import CatCropsTransformer
from models.branches import StaticBranch, TemporalBranch
from models.common import masked_max_pool
from models.film_fusion import FiLM, ConcatFusion, StaticCrossAttention
from models.rotation_gate import RotationGate

GATE_MODES = ("none", "fixed", "rotation")


class DualBranchModel(nn.Module):
    def __init__(self, num_classes: int, n_sreg: int, n_mun: int, profiles, in_channels: int = 12,
                 d_model: int = 64, n_heads: int = 2, n_layers: int = 5, d_ff: int = 128,
                 dropout: float = 0.1, positional_encoding: bool = True,
                 use_cross_attention: bool = True, fusion: str = "film", gate: str = "rotation",
                 static_emb_dim: int = 16):
        super().__init__()
        if gate not in GATE_MODES:
            raise ValueError(f"gate must be one of {GATE_MODES}")
        self.num_classes = num_classes
        self.gate_mode = gate
        self.temporal = TemporalBranch(in_channels, d_model, n_heads, n_layers, d_ff, dropout,
                                       positional_encoding)
        self.static = StaticBranch(n_sreg, n_mun, d_model, static_emb_dim, dropout)
        self.cross = StaticCrossAttention(d_model, n_heads, dropout) if use_cross_attention else None
        if fusion == "film":
            self.fusion = FiLM(d_model)
        elif fusion == "concat":
            self.fusion = ConcatFusion(d_model)
        else:
            raise ValueError("fusion must be 'film' or 'concat'")
        self.prev_emb = nn.Embedding(num_classes + 1, d_model, padding_idx=0) if gate != "none" else None
        self.gate = RotationGate(profiles, d_model) if gate == "rotation" else None
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_model, num_classes),
        )

    def forward(self, batch: dict) -> dict:
        x, days, mask = batch["x"], batch["days"], batch["mask"]
        y_prev = batch["y_prev"]
        B = x.shape[0]

        tokens = self.temporal(x, days, mask)
        h_static, static_tokens = self.static(batch["static_cont"], batch["static_cat"])
        if self.cross is not None:
            tokens = self.cross(tokens, static_tokens)
        h_temp = masked_max_pool(tokens, mask)
        fused = self.fusion(h_temp, h_static)

        gate = torch.zeros(B, device=x.device)
        div = torch.zeros(B, device=x.device)
        if self.gate_mode != "none":
            e_prev = self.prev_emb((y_prev + 1).clamp(min=0))
            if self.gate_mode == "fixed":
                gate = (y_prev >= 0).float()
            else:
                gate, div = self.gate(x, days, mask, y_prev, h_temp, e_prev)
            fused = fused + gate.unsqueeze(-1).to(fused.dtype) * e_prev.to(fused.dtype)

        return {"logits": self.head(fused), "gate": gate, "divergence": div}


def build_model(model_cfg: dict, stats: dict, num_classes: int) -> nn.Module:
    n_sreg = len(stats["sreg_vocab"])
    n_mun = len(stats["mun_vocab"])
    common = dict(
        d_model=model_cfg["d_model"], n_heads=model_cfg["n_heads"], n_layers=model_cfg["n_layers"],
        d_ff=model_cfg["d_ff"], dropout=model_cfg["dropout"],
        positional_encoding=model_cfg["positional_encoding"],
    )
    if model_cfg["type"] == "baseline":
        return BaselineTransformer(num_classes, n_sreg, n_mun, **common)
    if model_cfg["type"] == "catcrops_baseline":  # faithful official baseline (v2 run A)
        return CatCropsTransformer(num_classes, d_model=model_cfg["d_model"], n_heads=model_cfg["n_heads"],
                                   n_layers=model_cfg["n_layers"], d_ff=model_cfg["d_ff"],
                                   dropout=model_cfg["dropout"])
    if model_cfg["type"] == "dual_branch":
        return DualBranchModel(
            num_classes, n_sreg, n_mun, stats["profiles"], **common,
            use_cross_attention=model_cfg["use_cross_attention"], fusion=model_cfg["fusion"],
            gate=model_cfg["gate"], static_emb_dim=model_cfg["static_emb_dim"],
        )
    raise ValueError(f"unknown model type {model_cfg['type']!r}")
