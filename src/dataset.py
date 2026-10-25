"""OWNER: Data Lead (Ishu). Sparse multi-year loading + random day-of-year cropping.

Batch contract (see docs/INTERFACE_CONTRACT.md):
    x_temporal : float  (B, T, 12)   padded; plus mask (B, T) bool, True = valid
    x_static   : float  (B, n_cont)  and categorical ids: irr (B,), mun (B,)
    y_prev     : long   (B,)         previous-year crop id (0..17)
    y          : long   (B,)         current-year crop id (0..17)
"""
from __future__ import annotations


class CatCropsDataset:
    def __init__(self, root: str, year: int, max_doy: int | None = None):
        raise NotImplementedError("TODO(Ishu)")

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
