"""YAML configs with inheritance: a config may name a "base:" file whose values it overrides."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

REQUIRED = {
    "data": ["aggregation", "multi_year", "prior_dropout", "min_cutoff_doy", "max_cutoff_doy"],
    "model": ["type", "d_model", "n_heads", "n_layers", "d_ff", "dropout", "positional_encoding",
              "use_cross_attention", "fusion", "gate", "static_emb_dim"],
    "loss": ["type", "cb_beta", "focal_gamma"],
    "train": ["epochs", "batch_size", "lr", "weight_decay", "patience", "num_workers",
              "eval_cutoffs"],
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    if "base" in cfg:
        base = load_config(path.parent / cfg.pop("base"))
        cfg = deep_merge(base, cfg)
    return cfg


def validate(cfg: dict) -> None:
    for section, keys in REQUIRED.items():
        missing = [k for k in keys if k not in cfg.get(section, {})]
        if missing:
            raise KeyError(f"config section '{section}' is missing {missing}")
    if "name" not in cfg:
        raise KeyError("config needs a 'name'")
