"""Convert the CatCrops per-field CSV files into one cached array file per year.

Reading ~265,000 small CSV files on every epoch is what made earlier runs take
~31 min/epoch. This script reads them once and writes, for each year:

    <out_dir>/<year>.npz   ragged arrays (all fields' observations concatenated)
    <out_dir>/<year>.json  metadata (class names, counts, sanity checks)

Each field keeps observations from the PREVIOUS year and the CURRENT year
(the baseline's "sparse multi-year" input). Time is stored as a signed day
offset from 1 January of the current year: previous-year dates are negative
(-365..-1), current-year dates are 0..365.

Two per-date aggregations of overlapping Sentinel-2 tiles are stored side by side:
    values_mean   plain mean over tiles seen on the same date (default)
    values_cw     cloud-weighted mean (switchable extra fix 3)

This module does not import torch, so it can be tested anywhere.

Usage (Colab):
    python -m data.preprocess --root /root/.cache/kagglehub/.../catcrops_dataset \
        --out /content/drive/MyDrive/crop-methodology/cache --years 2022 2023
"""
from __future__ import annotations

import argparse
import json
import os
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

# Ten Sentinel-2 bands used by the baseline (10 m and 20 m bands; B1, B9 dropped).
BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
FEATURES = BANDS + ["CP"]  # 11 per-observation features; day offset is added later

# Raw class ids dropped so that 18 classes remain, matching the baseline.
# 10 = "Altres extensius" (112 fields in 2022), 17 = "Cebes" (258 fields).
# TODO: confirm against the baseline paper's class table.
EXCLUDED_CLASS_IDS = (10, 17)
EXPECTED_NUM_CLASSES = 18

# Cloud probability is stored on a 0-100 scale in CatCrops.
CP_SCALE = 100.0
# Cloud-weighted merging: a view above this cloud probability is dropped when a
# clearer view of the same field exists on the same date.
CW_DROP_THRESHOLD = 50.0
CW_MIN_WEIGHT = 0.01


# --------------------------------------------------------------------------- paths
# The dataset can be read from an extracted folder or straight from the downloaded zip archive
# (Kaggle's download is 14 GB zipped but 30 GB unpacked, more than a laptop may have free).
# A zip root is passed around as the string "zip::<archive path>::<folder prefix inside the zip>",
# so it can be sent to worker processes; each process opens the archive once.
ZIP_TAG = "zip::"
_ZIP_HANDLES: dict[str, zipfile.ZipFile] = {}


def _find_in_zip(archive: Path) -> str | None:
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if name.endswith("2022/L2A/lleida.csv"):
                return name[: -len("2022/L2A/lleida.csv")]
    return None


def find_dataset_root(path: str | Path) -> str:
    """Return the dataset root: a folder containing the '2022' and '2023' year folders,
    or a zip root for an archive that contains them."""
    path = Path(path)
    if path.is_file() and zipfile.is_zipfile(path):
        prefix = _find_in_zip(path)
        if prefix is not None:
            return f"{ZIP_TAG}{path}::{prefix}"
    else:
        candidates = [path, path / "catcrops_dataset"]
        candidates += [p for p in path.rglob("catcrops_dataset") if p.is_dir()]
        for c in candidates:
            if (c / "2022" / "L2A").is_dir():
                return str(c)
    raise FileNotFoundError(f"No CatCrops dataset (with 2022/L2A) found under {path}")


def _open(root: str, rel: str):
    """Open a file of the dataset (binary). Raises FileNotFoundError if it is not there."""
    if str(root).startswith(ZIP_TAG):
        archive, prefix = str(root)[len(ZIP_TAG):].rsplit("::", 1)
        z = _ZIP_HANDLES.get(archive)
        if z is None:
            z = _ZIP_HANDLES[archive] = zipfile.ZipFile(archive)
        try:
            return z.open(prefix + rel)
        except KeyError:
            raise FileNotFoundError(rel) from None
    return open(Path(root) / rel, "rb")


def index_path(year: int) -> str:
    return f"{year}/L2A/lleida.csv"


def field_csv_path(year: int, field_id) -> str:
    return f"{year}/L2A/lleida/csv/{int(field_id)}.csv"


# --------------------------------------------------------------------------- index
def load_index(root: str, year: int) -> pd.DataFrame:
    with _open(root, index_path(year)) as fh:
        df = pd.read_csv(fh)
    df.columns = df.columns.astype(str).str.strip()
    df = df.dropna(subset=["id", "classid"]).copy()
    df["id"] = df["id"].astype(np.int64)
    df["classid"] = df["classid"].astype(int)
    return df


def build_class_mapping(index: pd.DataFrame) -> tuple[dict[int, int], list[str]]:
    """Raw class id -> model label 0..17, plus class names in model order."""
    names = (
        index[["classid", "classname"]]
        .drop_duplicates("classid")
        .set_index("classid")["classname"]
    )
    kept = sorted(c for c in names.index if c not in EXCLUDED_CLASS_IDS)
    if len(kept) != EXPECTED_NUM_CLASSES:
        raise ValueError(f"Expected {EXPECTED_NUM_CLASSES} classes, found {len(kept)}: {kept}")
    mapping = {int(raw): i for i, raw in enumerate(kept)}
    return mapping, [str(names[raw]) for raw in kept]


def build_crop_code_map(indices: list[pd.DataFrame]) -> dict[int, int]:
    """Declared crop code (CODE_CULTU) -> raw class id, keeping unambiguous codes only."""
    pairs = (
        pd.concat([d[["CODE_CULTU", "classid"]] for d in indices], ignore_index=True)
        .dropna()
        .astype({"CODE_CULTU": np.int64, "classid": int})
        .drop_duplicates()
    )
    n_classes = pairs.groupby("CODE_CULTU")["classid"].nunique()
    safe = pairs[pairs["CODE_CULTU"].isin(n_classes[n_classes == 1].index)]
    return {int(c): int(k) for c, k in zip(safe["CODE_CULTU"], safe["classid"])}


def previous_crop_labels(index, code_map, class_mapping) -> np.ndarray:
    """pcrop_code -> model label, or -1 when unknown, unmapped or an excluded class."""
    out = np.full(len(index), -1, dtype=np.int16)
    for i, code in enumerate(index["pcrop_code"].to_numpy()):
        if pd.isna(code):
            continue
        raw = code_map.get(int(code))
        if raw is not None and raw in class_mapping:
            out[i] = class_mapping[raw]
    return out


# --------------------------------------------------------------------------- fields
def aggregate_field(df: pd.DataFrame, year: int):
    """One field's raw rows -> (day_offset int16 [n], values_mean f32 [n,11], values_cw f32 [n,11])."""
    missing = [c for c in FEATURES + ["doa"] if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing}")

    dates = pd.to_datetime(df["doa"], errors="coerce")
    vals = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    keep = dates.notna() & dates.dt.year.isin([year - 1, year])
    dates, vals = dates[keep], vals[keep]
    empty = (np.zeros(0, np.int16), np.zeros((0, 11), np.float32), np.zeros((0, 11), np.float32))
    if len(vals) == 0:
        return empty

    jan1 = pd.Timestamp(year=year, month=1, day=1)
    day = (dates - jan1).dt.days.to_numpy()
    v = vals.to_numpy(dtype=np.float64)
    cp = np.nan_to_num(v[:, -1], nan=CP_SCALE)  # unknown cloud = assume cloudy

    uniq, inv = np.unique(day, return_inverse=True)
    n = len(uniq)
    finite = np.isfinite(v)
    v0 = np.where(finite, v, 0.0)

    def per_date_sum(weights):  # weights: (rows, 11) -> (dates, 11)
        return np.stack([np.bincount(inv, weights=weights[:, k], minlength=n) for k in range(v.shape[1])], axis=1)

    # Plain mean over the tiles seen on each date (ignoring missing values).
    cnt = per_date_sum(finite.astype(np.float64))
    vm = np.where(cnt > 0, per_date_sum(v0) / np.maximum(cnt, 1), np.nan)

    # Cloud-weighted mean: drop views above the threshold when a clearer view exists on that date,
    # then weight the rest by (1 - cloud probability).
    clear = cp <= CW_DROP_THRESHOLD
    has_clear = np.bincount(inv, weights=clear.astype(np.float64), minlength=n) > 0
    use = clear | ~has_clear[inv]
    w = np.clip(1.0 - cp / CP_SCALE, CW_MIN_WEIGHT, 1.0) * use
    wf = finite * w[:, None]
    den = per_date_sum(wf)
    vc = np.where(den > 0, per_date_sum(v0 * wf) / np.maximum(den, 1e-12), np.nan)

    vm = np.nan_to_num(vm.astype(np.float32), nan=0.0)
    vc = np.nan_to_num(vc.astype(np.float32), nan=0.0)
    return uniq.astype(np.int16), vm, vc


def _read_one(args):
    root, year, fid = args
    try:
        fh = _open(root, field_csv_path(year, fid))
    except FileNotFoundError:
        return fid, None, "missing_file"
    try:
        with fh:
            return fid, aggregate_field(pd.read_csv(fh), year), None
    except Exception as exc:  # noqa: BLE001 - logged in metadata
        return fid, None, f"error: {exc}"


# --------------------------------------------------------------------------- main
def preprocess_year(root, year, out_dir, class_mapping, class_names, code_map, workers=None):
    root, out_dir = str(root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    index = load_index(root, year)
    n_raw = len(index)
    index = index[~index["classid"].isin(EXCLUDED_CLASS_IDS)].reset_index(drop=True)
    unknown = set(index["classid"]) - set(class_mapping)
    if unknown:
        raise ValueError(f"{year}: classes not in the mapping: {sorted(unknown)}")

    ids = index["id"].to_numpy()
    jobs = [(root, year, int(f)) for f in ids]
    workers = workers or max(1, (os.cpu_count() or 2))
    if root.startswith(ZIP_TAG):
        workers = min(workers, 8)  # each worker holds the archive's file list (~150 MB) in memory
    results = {}
    problems = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for fid, res, err in pool.map(_read_one, jobs, chunksize=256):
            if err is not None or res is None or len(res[0]) == 0:
                problems[fid] = err or "no_observations"
            else:
                results[fid] = res

    keep = np.array([int(f) in results for f in ids])
    index = index[keep].reset_index(drop=True)
    ids = index["id"].to_numpy()

    lengths = np.array([len(results[int(f)][0]) for f in ids], dtype=np.int64)
    offsets = np.zeros(len(ids) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    days = np.concatenate([results[int(f)][0] for f in ids])
    values_mean = np.concatenate([results[int(f)][1] for f in ids]).astype(np.float16)
    values_cw = np.concatenate([results[int(f)][2] for f in ids]).astype(np.float16)

    prev_label = previous_crop_labels(index, code_map, class_mapping)

    def col(name, default, dtype):
        if name not in index.columns:
            return np.full(len(index), default, dtype=dtype)
        return index[name].fillna(default).to_numpy(dtype=dtype)

    np.savez(
        out_dir / f"{year}.npz",
        field_id=ids.astype(np.int64),
        label=index["classid"].map(class_mapping).to_numpy(dtype=np.int16),
        prev_label=prev_label,
        offsets=offsets,
        days=days,
        values_mean=values_mean,
        values_cw=values_cw,
        elev=col("elev", np.nan, np.float32),
        slope=col("slope", np.nan, np.float32),
        sreg_code=col("sreg_code", -1, np.int64),
        mun_code=col("mun_code", -1, np.int64),
    )

    cur = days >= 0
    meta = {
        "year": year,
        "features": FEATURES,
        "class_names": class_names,
        "class_mapping_raw_to_model": {str(k): v for k, v in class_mapping.items()},
        "excluded_class_ids": list(EXCLUDED_CLASS_IDS),
        "fields_in_index": int(n_raw),
        "fields_after_class_filter": int(len(keep)),
        "fields_kept": int(len(ids)),
        "fields_dropped": {str(k): v for k, v in list(problems.items())[:50]},
        "n_dropped": len(problems),
        "observations": int(len(days)),
        "obs_previous_year": int((~cur).sum()),
        "obs_current_year": int(cur.sum()),
        "max_sequence_length": int(lengths.max()),
        "median_sequence_length": float(np.median(lengths)),
        "cp_max": float(values_mean[:, -1].astype(np.float32).max()),
        "prev_label_unknown": int((prev_label < 0).sum()),
        "label_counts": np.bincount(index["classid"].map(class_mapping), minlength=len(class_names)).tolist(),
        "seconds": round(time.time() - t0, 1),
    }
    (out_dir / f"{year}.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="CatCrops dataset folder (or a parent of it), or the downloaded zip archive")
    ap.add_argument("--out", required=True, help="output cache folder")
    ap.add_argument("--years", type=int, nargs="+", default=[2022, 2023])
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="rebuild even if the cache exists")
    args = ap.parse_args(argv)

    root = find_dataset_root(args.root)
    indices = {y: load_index(root, y) for y in args.years}
    train_year = min(args.years)
    class_mapping, class_names = build_class_mapping(indices[train_year])
    code_map = build_crop_code_map(list(indices.values()))

    for y in args.years:
        if (Path(args.out) / f"{y}.npz").exists() and not args.force:
            print(f"{y}: cache exists, skipping (use --force to rebuild)")
            continue
        meta = preprocess_year(root, y, args.out, class_mapping, class_names, code_map, args.workers)
        print(f"{y}: kept {meta['fields_kept']} fields, {meta['observations']} observations "
              f"({meta['obs_previous_year']} previous-year), dropped {meta['n_dropped']}, "
              f"max length {meta['max_sequence_length']}, {meta['seconds']} s")


if __name__ == "__main__":
    main()
