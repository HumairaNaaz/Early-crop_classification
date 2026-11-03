"""Run the ablation ladder (A -> D) over several seeds, predict 2023, and build the results table.

    python run_experiments.py --cache <cache> --out <runs> --seeds 0 1 2 --max-minutes 150

Safe to rerun at any time: finished runs are skipped, unfinished runs resume from their
last checkpoint, and predictions are only made once per finished run. When --max-minutes
is reached the script stops; run the same cell again in a new Colab session.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

import predict
import train

LADDER = ["A_baseline", "B_fusion", "C_gate", "D_full"]
EXTRAS = ["D_full_prior_dropout", "D_full_cloud_weighted"]


def summarize(out: Path, test_year: int = 2023) -> pd.DataFrame | None:
    rows = []
    for mfile in sorted(out.glob(f"*_seed*/predictions/metrics_{test_year}.json")):
        m = json.loads(mfile.read_text())
        name, seed = m["run"].rsplit("_seed", 1)
        for date, e in m["dates"].items():
            rows.append({"run": name, "seed": int(seed), "date": date,
                         "accuracy": e["overall"]["accuracy"], "macro_f1": e["overall"]["macro_f1"],
                         "weighted_f1": e["overall"]["weighted_f1"],
                         "changed_macro_f1": e["changed"].get("macro_f1"),
                         "changed_accuracy": e["changed"].get("accuracy"),
                         "same_macro_f1": e["same"].get("macro_f1")})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.to_csv(out / "results_per_seed.csv", index=False)
    metrics = ["accuracy", "macro_f1", "weighted_f1", "changed_accuracy", "changed_macro_f1", "same_macro_f1"]
    g = df.groupby(["run", "date"])[metrics].agg(["mean", "std", "count"])
    g.columns = [f"{m}_{s}" for m, s in g.columns]
    g = g.reset_index()
    g.to_csv(out / "results_summary.csv", index=False)
    return g


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", nargs="+", default=LADDER, help=f"default: {LADDER}; extras: {EXTRAS}")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-minutes", type=float, default=None, help="total time budget for this session")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--config-dir", default=None, help="folder with the run configs (default: configs/)")
    args = ap.parse_args(argv)

    cfg_dir = Path(args.config_dir) if args.config_dir else Path(__file__).resolve().parent / "configs"
    out = Path(args.out)
    t0 = time.time()
    # Seed-major order: all four configs get seed 0 first, so a full A-D comparison exists early.
    for seed in args.seeds:
        for name in args.configs:
            left = None
            if args.max_minutes:
                left = args.max_minutes - (time.time() - t0) / 60
                if left < 5:
                    print("Session time budget used up. Rerun this cell to continue.")
                    summarize(out)
                    return
            targs = ["--config", str(cfg_dir / f"{name}.yaml"), "--cache", args.cache,
                     "--out", str(out), "--seed", str(seed)]
            if left is not None:
                targs += ["--max-minutes", str(left)]
            if args.epochs:
                targs += ["--epochs", str(args.epochs)]
            res = train.main(targs)
            if res.get("status") == "paused":
                print("Paused mid-run (time budget). Rerun this cell to continue.")
                summarize(out)
                return
            run_dir = out / f"{name}_seed{seed}"
            if not (run_dir / "predictions" / "metrics_2023.json").exists():
                predict.main(["--run", str(run_dir), "--cache", args.cache])
            summarize(out)
    table = summarize(out)
    print("All requested runs finished.")
    if table is not None:
        cols = ["run", "date", "accuracy_mean", "macro_f1_mean", "changed_macro_f1_mean", "accuracy_count"]
        print(table[cols].to_string(index=False))


if __name__ == "__main__":
    main()
