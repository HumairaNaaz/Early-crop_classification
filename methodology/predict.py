"""Predict every test-year field at the four benchmark dates with a finished run.

    python predict.py --run <runs dir>/C_gate_seed0 --cache <cache dir> --year 2023

Writes into <run>/predictions/:
    <year>_<date>.csv.gz   one row per field: field_id, cutoff_doy, y_true, y_pred, y_prev,
                           rotation_group, gate, divergence, p_0..p_17 (class probabilities)
    metrics_<year>.json    accuracy / macro-F1 / weighted-F1 per date, overall and per rotation
                           group (same crop, changed crop, unknown previous crop)

These files are what the evaluation stage (per-class metrics, confusion matrices,
significance tests) works from, so no retraining is needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from data.cache import YearCache, load_stats
from data.dataset import BENCHMARK_CUTOFFS, CropDataset, collate, loader_workers
from engine import metrics, predict
from models import build_model


def run(args) -> dict:
    run_dir = Path(args.run)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    stats = load_stats(run_dir / "stats")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    num_classes = ck["num_classes"]
    model = build_model(cfg["model"], stats, num_classes)
    model.load_state_dict(ck["model"])
    model.to(device)

    year = args.year or cfg["test_year"]
    cache = YearCache.load(args.cache, year)
    idx = np.arange(len(cache))
    groups = cache.rotation_group()
    out_dir = run_dir / "predictions"
    out_dir.mkdir(exist_ok=True)

    all_metrics = {"run": run_dir.name, "year": year, "best_epoch": ck["epoch"], "dates": {}}
    for name, cutoff in BENCHMARK_CUTOFFS.items():
        ds = CropDataset(cache, idx, stats, train=False, cutoff_doy=cutoff,
                         multi_year=cfg["data"]["multi_year"], aggregation=cfg["data"]["aggregation"],
                         window_days=cfg["data"].get("window_days"))
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=loader_workers(args.num_workers),
                            collate_fn=collate, pin_memory=device.type == "cuda")
        res = predict(model, loader, device)
        f = res["index"]
        df = pd.DataFrame({
            "field_id": cache.field_id[f], "cutoff_doy": cutoff,
            "y_true": res["y"], "y_pred": res["pred"], "y_prev": res["y_prev"],
            "rotation_group": groups[f], "gate": res["gate"], "divergence": res["divergence"],
        })
        probs = pd.DataFrame(res["probs"], columns=[f"p_{k}" for k in range(num_classes)])
        pd.concat([df, probs], axis=1).to_csv(out_dir / f"{year}_{name}.csv.gz", index=False)

        entry = {"overall": metrics(df["y_true"], df["y_pred"])}
        for g in ("same", "changed", "unknown"):
            sel = df["rotation_group"] == g
            entry[g] = metrics(df.loc[sel, "y_true"], df.loc[sel, "y_pred"])
            entry[g]["mean_gate"] = float(df.loc[sel, "gate"].mean()) if sel.any() else None
        all_metrics["dates"][name] = entry
        o = entry["overall"]
        print(f"{name} (doy {cutoff}): acc {o['accuracy']:.4f}  macro-F1 {o['macro_f1']:.4f}  "
              f"changed-crop macro-F1 {entry['changed'].get('macro_f1', float('nan')):.4f}", flush=True)

    (out_dir / f"metrics_{year}.json").write_text(json.dumps(all_metrics, indent=2))
    return all_metrics


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="a finished run folder")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default=None)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
