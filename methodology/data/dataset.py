"""PyTorch dataset over a cached year.

Each item is one field's time series up to a cutoff date:
  * all previous-year observations (day offset < 0) when multi_year is on
  * current-year observations with day-of-year <= cutoff_doy
    (day offset 0 = 1 January = day-of-year 1, so the test is day_offset < cutoff_doy)
  * with window_days=365 only the 365 days before the cutoff are kept
    (days in (cutoff - 366, cutoff)), exactly as the CatCrops code's apply_datecrop does

Training draws a random cutoff per sample (random day-of-year cropping), so one model
learns to predict at any point in the season. Evaluation uses a fixed cutoff.

Per-step features (12 channels): 10 normalised bands, normalised cloud probability,
and day offset / 365.

"anc" holds the five ancillary values in the scaling the CatCrops code uses (previous crop
class / number of classes, irrigation code / max code, municipality code min-max scaled,
elevation / 1500 m, slope / 54 deg; -1 when unknown). The faithful baseline adds each of
them as one extra time step; the dual-branch models use static_cont / static_cat instead.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from data.cache import YearCache

BENCHMARK_CUTOFFS = {"01Mar": 60, "01Jun": 152, "01Sep": 244, "01Dec": 335}
FULL_YEAR = 366
N_CHANNELS = 12


class CropDataset(Dataset):
    def __init__(self, cache: YearCache, indices, stats: dict, *, train: bool,
                 cutoff_doy: int | None = None, multi_year: bool = True,
                 aggregation: str = "mean", prior_dropout: float = 0.0,
                 min_cutoff_doy: int = 1, max_cutoff_doy: int = 365,
                 window_days: int | None = None):
        self.cache = cache
        self.indices = np.asarray(indices, dtype=np.int64)
        self.train = train
        self.cutoff = FULL_YEAR if cutoff_doy is None else int(cutoff_doy)
        self.multi_year = multi_year
        self.prior_dropout = float(prior_dropout) if train else 0.0
        self.min_cut, self.max_cut = int(min_cutoff_doy), int(max_cutoff_doy)
        self.window = None if window_days in (None, 0) else int(window_days)
        self.values = cache.values(aggregation)

        self.feat_mean = np.asarray(stats["feat_mean"], dtype=np.float32)
        self.feat_std = np.asarray(stats["feat_std"], dtype=np.float32)
        em, es = stats["elev"]
        sm, ss = stats["slope"]
        self.static_cont = np.stack([
            np.nan_to_num((cache.elev - em) / es, nan=0.0),
            np.nan_to_num((cache.slope - sm) / ss, nan=0.0),
        ], axis=1).astype(np.float32)
        sv, mv = stats["sreg_vocab"], stats["mun_vocab"]
        self.static_cat = np.stack([
            np.array([sv.get(int(c), 0) for c in cache.sreg_code], dtype=np.int64),
            np.array([mv.get(int(c), 0) for c in cache.mun_code], dtype=np.int64),
        ], axis=1)
        self.num_classes = len(stats["class_counts"])
        self.anc_static = ancillary_scalars(cache, sv, mv)

    def __len__(self) -> int:
        return len(self.indices)

    def _cutoff(self) -> int:
        if self.train:
            return int(torch.randint(self.min_cut, self.max_cut + 1, (1,)).item())
        return self.cutoff

    def __getitem__(self, i: int) -> dict:
        f = int(self.indices[i])
        s, e = self.cache.offsets[f], self.cache.offsets[f + 1]
        days = self.cache.days[s:e].astype(np.int64)
        x = self.values[s:e].astype(np.float32)

        cutoff = self._cutoff()
        keep = days < cutoff
        if self.window is not None:
            keep &= days > cutoff - (self.window + 1)
        if not self.multi_year:
            keep &= days >= 0
        days, x = days[keep], x[keep]
        if len(days) == 0:  # nothing observed yet: one neutral step (normalises to zeros)
            days = np.array([max(cutoff - 1, 0)], dtype=np.int64)
            x = self.feat_mean[None, :].copy()

        xn = (x - self.feat_mean) / self.feat_std
        feats = np.concatenate([xn, (days / 365.0)[:, None].astype(np.float32)], axis=1)

        y_prev = int(self.cache.prev_label[f])
        if self.prior_dropout > 0 and torch.rand(1).item() < self.prior_dropout:
            y_prev = -1

        prev_val = (y_prev + 1) / self.num_classes if y_prev >= 0 else -1.0
        anc = np.concatenate([[prev_val], self.anc_static[f]]).astype(np.float32)

        return {
            "x": torch.from_numpy(feats.astype(np.float32)),
            "anc": torch.from_numpy(anc),
            "days": torch.from_numpy(days),
            "y": int(self.cache.label[f]),
            "y_prev": y_prev,
            "static_cont": torch.from_numpy(self.static_cont[f]),
            "static_cat": torch.from_numpy(self.static_cat[f]),
            "index": f,
            "cutoff": cutoff,
        }


def collate(batch: list[dict]) -> dict:
    B = len(batch)
    T = max(b["x"].shape[0] for b in batch)
    x = torch.zeros(B, T, N_CHANNELS)
    days = torch.zeros(B, T, dtype=torch.long)
    mask = torch.zeros(B, T, dtype=torch.bool)
    for i, b in enumerate(batch):
        n = b["x"].shape[0]
        x[i, :n] = b["x"]
        days[i, :n] = b["days"]
        mask[i, :n] = True
    return {
        "x": x,
        "days": days,
        "mask": mask,
        "y": torch.tensor([b["y"] for b in batch], dtype=torch.long),
        "y_prev": torch.tensor([b["y_prev"] for b in batch], dtype=torch.long),
        "static_cont": torch.stack([b["static_cont"] for b in batch]),
        "static_cat": torch.stack([b["static_cat"] for b in batch]),
        "anc": torch.stack([b["anc"] for b in batch]),
        "index": torch.tensor([b["index"] for b in batch], dtype=torch.long),
        "cutoff": torch.tensor([b["cutoff"] for b in batch], dtype=torch.long),
    }


MAX_ELEV, MAX_SLOPE = 1500.0, 54.0  # CatCrops code: elev / 1500, slope / 54


def ancillary_scalars(cache: YearCache, sreg_vocab: dict, mun_vocab: dict) -> np.ndarray:
    """(n_fields, 4): irrigation, municipality, elevation, slope scaled as in the CatCrops code.
    The code ranges come from the training-year vocabularies; unknown values become -1."""
    sreg = cache.sreg_code.astype(np.float64)
    mun = cache.mun_code.astype(np.float64)
    smax = float(max(max(sreg_vocab), 1)) if sreg_vocab else 1.0
    mmin = float(min(mun_vocab)) if mun_vocab else 0.0
    mmax = float(max(mun_vocab)) if mun_vocab else 1.0
    sreg_v = np.where(sreg >= 0, sreg / smax, -1.0)
    mun_v = np.where(mun >= 0, (mun - mmin) / max(mmax - mmin, 1.0), -1.0)
    elev_v = np.nan_to_num(cache.elev.astype(np.float64) / MAX_ELEV, nan=-1.0)
    slope_v = np.nan_to_num(cache.slope.astype(np.float64) / MAX_SLOPE, nan=-1.0)
    return np.stack([sreg_v, mun_v, elev_v, slope_v], axis=1).astype(np.float32)


def loader_workers(requested: int) -> int:
    """DataLoader worker count for this machine.

    Windows starts each worker by copying the whole dataset (the cached arrays, a few GB)
    into a new process, so there the data is loaded in the main process instead.
    Set the CROP_NUM_WORKERS environment variable to override."""
    env = os.environ.get("CROP_NUM_WORKERS")
    if env not in (None, ""):
        return int(env)
    return 0 if os.name == "nt" else requested
