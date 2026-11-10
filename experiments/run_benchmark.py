"""OWNER: Eval Lead (Divyanshi). Evaluate the ablation ladder A-D at the 4 dates, 3 seeds each.
A baseline | B + dual-branch fusion | C + rotation gate | D + CB focal loss.

Training is done by methodology/run_experiments.py, which also writes one prediction file per
run and date. This script reads those files (no retraining, no GPU) and computes the full
metric set with src/metrics.py:

    python experiments/run_benchmark.py --runs <runs dir> --year 2023 --out <runs dir>/benchmark

Input  (per run):  <runs>/<name>_seed<k>/predictions/<year>_<date>.csv.gz
                   columns used: y_true, y_pred, rotation_group ("changed" = rotated field)
Output (in --out): benchmark_metrics.json      every metric incl. confusion matrices
                   benchmark_per_seed.csv      one row per run x date (headline metrics)
                   benchmark_summary.csv       mean / std / n over seeds per config x date
                   benchmark_per_class_f1.csv  per-class F1 + support, mean over seeds

Dates: 1 Mar = day 60, 1 Jun = 152, 1 Sep = 244, 1 Dec = 335. A model only sees observations
with day-of-year < cutoff; that cut is applied by the dataset (methodology/data/dataset.py),
not here. The 2023 test year is never used for model selection (early stopping used a
stratified 10 % hold-out of 2022).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import compute_metrics  # noqa: E402

DATES = {"01Mar": 60, "01Jun": 152, "01Sep": 244, "01Dec": 335}
LADDER = ["A_baseline", "B_fusion", "C_gate", "D_full"]
HEADLINE = ["accuracy", "macro_f1", "weighted_f1", "rotated_accuracy", "rotated_macro_f1",
            "rare_class_f1"]


def find_runs(runs_dir: Path, year: int) -> list[tuple[str, int, Path]]:
    """(config name, seed, prediction folder) for every run that has predictions for `year`."""
    found = []
    for pred_dir in sorted(Path(runs_dir).glob("*_seed*/predictions")):
        if not any(pred_dir.glob(f"{year}_*.csv.gz")):
            continue
        name, seed = pred_dir.parent.name.rsplit("_seed", 1)
        found.append((name, int(seed), pred_dir))
    return found


def evaluate_run(pred_dir: Path, year: int, n_classes: int | None = None) -> dict:
    """compute_metrics for each available date of one run."""
    result = {}
    for date, cutoff in DATES.items():
        f = pred_dir / f"{year}_{date}.csv.gz"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        if n_classes is None:
            n_classes = sum(c.startswith("p_") for c in df.columns) or 18
        m = compute_metrics(df["y_true"].to_numpy(), df["y_pred"].to_numpy(),
                            rotation_mask=(df["rotation_group"] == "changed").to_numpy(),
                            n_classes=n_classes)
        m["cutoff_doy"] = cutoff
        m["unknown_prev_n"] = int((df["rotation_group"] == "unknown").sum())
        result[date] = m
    return result


def check_against_engine(pred_dir: Path, year: int, evaluated: dict) -> float | None:
    """Largest |difference| to methodology/predict.py's own accuracy / macro-F1 / weighted-F1,
    so the two code paths are proven to agree. None if that file is missing."""
    f = pred_dir / f"metrics_{year}.json"
    if not f.exists():
        return None
    ref = json.loads(f.read_text())["dates"]
    diffs = [abs(evaluated[d][k] - ref[d]["overall"][k])
             for d in evaluated if d in ref for k in ("accuracy", "macro_f1", "weighted_f1")]
    return max(diffs) if diffs else None


def build_tables(all_metrics: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, class_rows = [], []
    for run, dates in all_metrics.items():
        name, seed = run.rsplit("_seed", 1)
        for date, m in dates.items():
            rows.append({"config": name, "seed": int(seed), "date": date,
                         "cutoff_doy": m["cutoff_doy"], "n": m["n"],
                         **{k: m[k] for k in HEADLINE}, "rotated_n": m["rotated_n"]})
            for c, (f1, s) in enumerate(zip(m["per_class_f1"], m["per_class_support"])):
                class_rows.append({"config": name, "seed": int(seed), "date": date,
                                   "cutoff_doy": m["cutoff_doy"], "class_id": c, "f1": f1, "support": s})
    per_seed = pd.DataFrame(rows)
    if per_seed.empty:
        return per_seed, per_seed, pd.DataFrame()
    g = per_seed.groupby(["config", "date", "cutoff_doy"])[HEADLINE]
    summary = g.agg(["mean", "std", "count"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index().sort_values(["config", "cutoff_doy"])
    per_class = (pd.DataFrame(class_rows).groupby(["config", "date", "cutoff_doy", "class_id"])
                 .agg(f1_mean=("f1", "mean"), f1_std=("f1", "std"), support=("support", "first"))
                 .reset_index())
    return per_seed, summary, per_class


def run(runs_dir, year: int, out_dir) -> dict:
    runs_dir, out_dir = Path(runs_dir), Path(out_dir)
    runs = find_runs(runs_dir, year)
    if not runs:
        raise FileNotFoundError(f"no <run>/predictions/{year}_*.csv.gz files under {runs_dir}")
    all_metrics, worst = {}, 0.0
    for name, seed, pred_dir in runs:
        evaluated = evaluate_run(pred_dir, year)
        all_metrics[f"{name}_seed{seed}"] = evaluated
        diff = check_against_engine(pred_dir, year, evaluated)
        if diff is not None:
            worst = max(worst, diff)
            if diff > 1e-6:
                print(f"WARNING {name}_seed{seed}: differs from predict.py by {diff:.2e}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark_metrics.json").write_text(json.dumps(all_metrics))
    per_seed, summary, per_class = build_tables(all_metrics)
    per_seed.to_csv(out_dir / "benchmark_per_seed.csv", index=False)
    summary.to_csv(out_dir / "benchmark_summary.csv", index=False)
    per_class.to_csv(out_dir / "benchmark_per_class_f1.csv", index=False)
    print(f"{len(runs)} runs evaluated; max difference to predict.py metrics: {worst:.2e}")
    cols = ["config", "date", "accuracy_mean", "macro_f1_mean", "rotated_accuracy_mean",
            "rare_class_f1_mean", "macro_f1_count"]
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return all_metrics


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="folder with <name>_seed<k>/ run folders")
    ap.add_argument("--year", type=int, default=2023, help="test year (default 2023)")
    ap.add_argument("--out", default=None, help="output folder (default <runs>/benchmark)")
    args = ap.parse_args(argv)
    return run(args.runs, args.year, args.out or Path(args.runs) / "benchmark")


if __name__ == "__main__":
    main()
