"""Load cached years, make the train/validation split, and compute training statistics.

Torch-free: everything here is numpy, so it can be tested without a GPU.

Statistics are computed on the 2022 TRAINING split only (never validation or 2023):
  * per-feature mean/std of the 10 bands + cloud probability (current-year observations)
  * mean/std of elevation and slope
  * vocabularies for the irrigation-system code (sreg_code) and municipality code;
    index 0 is reserved for "unknown / not seen in training"
  * class counts (for the class-balanced focal loss)
  * expected spectral profiles: the mean normalised band values of each crop class in
    each 5-day bin of the year. The rotation gate compares a field's current signal with
    the profile of last year's crop.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

N_BANDS = 10  # first 10 features are spectral bands; feature 11 is cloud probability
PROFILE_BIN_DAYS = 5
N_PROFILE_BINS = 74  # covers day offsets 0..369


@dataclass
class YearCache:
    year: int
    field_id: np.ndarray
    label: np.ndarray
    prev_label: np.ndarray
    offsets: np.ndarray
    days: np.ndarray
    values_mean: np.ndarray
    values_cw: np.ndarray
    elev: np.ndarray
    slope: np.ndarray
    sreg_code: np.ndarray
    mun_code: np.ndarray
    meta: dict

    @classmethod
    def load(cls, cache_dir: str | Path, year: int) -> "YearCache":
        cache_dir = Path(cache_dir)
        with np.load(cache_dir / f"{year}.npz") as z:
            arrays = {k: z[k] for k in z.files}
        meta_path = cache_dir / f"{year}.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return cls(year=year, meta=meta, **arrays)

    def __len__(self) -> int:
        return len(self.field_id)

    def values(self, aggregation: str) -> np.ndarray:
        if aggregation == "mean":
            return self.values_mean
        if aggregation == "cloud_weighted":
            return self.values_cw
        raise ValueError(f"unknown aggregation {aggregation!r}")

    def field_obs(self, i: int, aggregation: str = "mean"):
        s, e = self.offsets[i], self.offsets[i + 1]
        return self.days[s:e], self.values(aggregation)[s:e]

    def rotation_group(self) -> np.ndarray:
        """'same', 'changed' or 'unknown' per field (previous vs current crop)."""
        g = np.where(self.prev_label < 0, "unknown",
                     np.where(self.prev_label == self.label, "same", "changed"))
        return g.astype(object)


def make_split(cache: YearCache, val_fraction: float = 0.10, seed: int = 42):
    """Stratified random split by field (same recipe as the team's dataset work)."""
    idx = np.arange(len(cache))
    train_idx, val_idx = train_test_split(
        idx, test_size=val_fraction, random_state=seed, stratify=cache.label
    )
    return np.sort(train_idx), np.sort(val_idx)


def _obs_mask(cache: YearCache, fields: np.ndarray) -> np.ndarray:
    """Boolean mask over all observations that belong to the given fields."""
    owner = np.repeat(np.arange(len(cache)), np.diff(cache.offsets))
    sel = np.zeros(len(cache), dtype=bool)
    sel[fields] = True
    return sel[owner], owner


def _vocab(codes: np.ndarray) -> dict[int, int]:
    known = sorted(int(c) for c in np.unique(codes) if c >= 0)
    return {c: i + 1 for i, c in enumerate(known)}  # 0 = unknown


def compute_stats(cache: YearCache, train_idx: np.ndarray, num_classes: int,
                  aggregation: str = "mean") -> dict:
    obs_sel, owner = _obs_mask(cache, train_idx)
    cur = obs_sel & (cache.days >= 0)
    vals = cache.values(aggregation)[cur].astype(np.float32)
    feat_mean = vals.mean(axis=0, dtype=np.float64)
    feat_std = vals.std(axis=0, dtype=np.float64)
    feat_std[feat_std < 1e-6] = 1.0

    def mstd(x):
        x = x[train_idx].astype(np.float64)
        x = x[np.isfinite(x)]
        m, s = (x.mean(), x.std()) if len(x) else (0.0, 1.0)
        return float(m), float(s if s > 1e-6 else 1.0)

    counts = np.bincount(cache.label[train_idx], minlength=num_classes)

    # Expected profiles in normalised band space: [class, bin, band]
    norm = ((vals[:, :N_BANDS] - feat_mean[:N_BANDS]) / feat_std[:N_BANDS]).astype(np.float32)
    del vals
    lab = cache.label[owner[cur]].astype(np.int64)
    bins = np.clip(cache.days[cur].astype(np.int64) // PROFILE_BIN_DAYS, 0, N_PROFILE_BINS - 1)
    flat = lab * N_PROFILE_BINS + bins
    total = num_classes * N_PROFILE_BINS
    sums = np.stack([np.bincount(flat, weights=norm[:, b], minlength=total) for b in range(N_BANDS)], axis=1)
    n = np.bincount(flat, minlength=total).astype(np.float64)
    prof = sums / np.maximum(n, 1)[:, None]
    prof = prof.reshape(num_classes, N_PROFILE_BINS, N_BANDS)
    n = n.reshape(num_classes, N_PROFILE_BINS)
    # Fill empty bins (rare classes) from the nearest filled bin of the same class.
    for c in range(num_classes):
        filled = np.flatnonzero(n[c] > 0)
        if len(filled) == 0:
            continue
        for b in np.flatnonzero(n[c] == 0):
            prof[c, b] = prof[c, filled[np.argmin(np.abs(filled - b))]]

    return {
        "aggregation": aggregation,
        "feat_mean": feat_mean.tolist(),
        "feat_std": feat_std.tolist(),
        "elev": mstd(cache.elev),
        "slope": mstd(cache.slope),
        "sreg_vocab": _vocab(cache.sreg_code[train_idx]),
        "mun_vocab": _vocab(cache.mun_code[train_idx]),
        "class_counts": counts.tolist(),
        "profiles": prof.astype(np.float32),
        "profile_counts": n.astype(np.int64),
    }


def save_stats(stats: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    js = {k: v for k, v in stats.items() if k not in ("profiles", "profile_counts")}
    js["sreg_vocab"] = {str(k): v for k, v in stats["sreg_vocab"].items()}
    js["mun_vocab"] = {str(k): v for k, v in stats["mun_vocab"].items()}
    path.with_suffix(".json").write_text(json.dumps(js, indent=2))
    np.savez(path.with_suffix(".npz"), profiles=stats["profiles"], profile_counts=stats["profile_counts"])


def load_stats(path: str | Path) -> dict:
    path = Path(path)
    js = json.loads(path.with_suffix(".json").read_text())
    js["sreg_vocab"] = {int(k): v for k, v in js["sreg_vocab"].items()}
    js["mun_vocab"] = {int(k): v for k, v in js["mun_vocab"].items()}
    js["elev"], js["slope"] = tuple(js["elev"]), tuple(js["slope"])
    with np.load(path.with_suffix(".npz")) as z:
        js["profiles"], js["profile_counts"] = z["profiles"], z["profile_counts"]
    return js
