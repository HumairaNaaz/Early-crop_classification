"""Build a tiny fake CatCrops dataset with the same folder layout and columns.

Each class gets its own spectral curve over the year, so a model can learn it.
Some dates are seen by two overlapping tiles with different cloud probability,
which exercises the per-date merging.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_CLASS_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20]
BANDS12 = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]


def class_curve(raw_class: int, doy: np.ndarray) -> np.ndarray:
    """(len(doy), 12) reflectances; peak day and height depend on the class."""
    peak = 40 + 15 * (raw_class % 20)
    height = 0.2 + 0.03 * (raw_class % 7)
    green = height * np.exp(-((doy - peak) / 45.0) ** 2)
    base = np.linspace(0.05, 0.35, 12)
    return (base[None, :] + green[:, None] * np.linspace(1.0, -0.5, 12)[None, :]) * 10000


def make_fake_catcrops(root: str | Path, fields_per_class: int = 10, seed: int = 0,
                       years=(2022, 2023)) -> Path:
    rng = np.random.default_rng(seed)
    root = Path(root) / "catcrops_dataset"
    next_id = 1000
    for year in years:
        csv_dir = root / str(year) / "L2A" / "lleida" / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for raw in RAW_CLASS_IDS:
            for _ in range(fields_per_class):
                fid = next_id
                next_id += 1
                prev_raw = raw if rng.random() < 0.7 else int(rng.choice(RAW_CLASS_IDS))
                code = 100 + raw            # one crop code per class
                pcode = 100 + prev_raw
                # Two years of observations every 5 days, with occasional tile duplicates.
                obs = []
                for y, crop in ((year - 1, prev_raw), (year, raw)):
                    dates = pd.date_range(f"{y}-01-02", f"{y}-12-28", freq="5D")
                    doy = dates.day_of_year.to_numpy()
                    refl = class_curve(crop, doy) + rng.normal(0, 30, (len(doy), 12))
                    for d, r in zip(dates, refl):
                        n_tiles = 2 if rng.random() < 0.25 else 1
                        for t in range(n_tiles):
                            cp = float(rng.choice([0, 5, 20, 80]))
                            noise = 400 * (cp / 100.0)
                            obs.append([f"S2_{d:%Y%m%d}_T31TC{'GF'[t]}"] + list(r + noise)
                                       + [cp, d.strftime("%Y-%m-%d"), raw, fid])
                cols = ["nom_id_imatge"] + BANDS12 + ["CP", "doa", "Label", "id"]
                pd.DataFrame(obs, columns=cols).to_csv(csv_dir / f"{fid}.csv", index=False)
                rows.append({
                    "id": fid, "CODE_CULTU": code, "classid": raw, "classname": f"class_{raw}",
                    "region": "lleida", "var_code": 1, "pcrop_code": pcode if rng.random() > 0.02 else np.nan,
                    "pvar_code": 1, "sreg_code": int(rng.integers(1, 5)),
                    "mun_code": int(rng.integers(25000, 25010)),
                    "elev": float(rng.uniform(100, 600)), "slope": float(rng.uniform(0, 15)),
                    "sequencelength": len(obs),
                })
        pd.DataFrame(rows).to_csv(root / str(year) / "L2A" / "lleida.csv", index=False)
    return root
