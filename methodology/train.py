"""Train one run (one config, one seed) with checkpoint/resume.

    python train.py --config configs/C_gate.yaml --cache <cache dir> --out <runs dir> --seed 0

Writes <out>/<name>_seed<seed>/:
    config.yaml          resolved config
    stats.json / .npz    normalisation, vocabularies, class counts, expected profiles
    last.pt              latest checkpoint (resumes automatically after a Colab disconnect)
    best.pt              best validation checkpoint
    history.csv          per-epoch metrics
    summary.json         written when the run finishes

Model selection: mean validation macro-F1 over the four benchmark cutoffs
(1 Mar, 1 Jun, 1 Sep, 1 Dec), not accuracy.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from config import load_config, validate
from data.cache import YearCache, compute_stats, load_stats, make_split, save_stats
from data.dataset import CropDataset, collate, loader_workers
from engine import metrics, predict, train_one_epoch
from losses import build_loss
from models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_stats(cfg, cache, train_idx, num_classes, cache_dir: Path, run_dir: Path,
              shareable: bool = True) -> dict:
    """Stats depend only on the split and aggregation; computed once and shared by runs.
    Smoke runs on a subset (--limit) never write or read the shared copy."""
    agg = cfg["data"]["aggregation"]
    if not shareable:
        stats = compute_stats(cache, train_idx, num_classes, agg)
        save_stats(stats, run_dir / "stats")
        return load_stats(run_dir / "stats")
    shared = cache_dir / f"stats_{agg}_split{cfg['split_seed']}"
    if shared.with_suffix(".json").exists():
        stats = load_stats(shared)
    else:
        stats = compute_stats(cache, train_idx, num_classes, agg)
        save_stats(stats, shared)
        stats = load_stats(shared)
    save_stats(stats, run_dir / "stats")
    return stats


def save_atomic(obj, path: Path) -> None:
    """Write a checkpoint to a temporary file first, so a crash or power cut mid-write
    never leaves a half-written last.pt behind."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def make_loader(ds, batch_size, shuffle, num_workers, device):
    num_workers = loader_workers(num_workers)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate, pin_memory=device.type == "cuda",
                      persistent_workers=num_workers > 0, drop_last=False)


def run(args) -> dict:
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    validate(cfg)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cache_dir, out_dir = Path(args.cache), Path(args.out)
    run_dir = out_dir / f"{cfg['name']}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.force:
        for f in ("summary.json", "last.pt", "best.pt", "history.csv"):
            (run_dir / f).unlink(missing_ok=True)
    if (run_dir / "summary.json").exists():
        print(f"{run_dir.name}: already finished, skipping (use --force to retrain)")
        return json.loads((run_dir / "summary.json").read_text())
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    set_seed(args.seed)

    cache = YearCache.load(cache_dir, cfg["train_year"])
    num_classes = len(cache.meta["class_names"]) if cache.meta else int(cache.label.max()) + 1
    train_idx, val_idx = make_split(cache, cfg["val_fraction"], cfg["split_seed"])
    if args.limit:  # quick smoke runs
        rng = np.random.default_rng(0)
        train_idx = np.sort(rng.choice(train_idx, min(args.limit, len(train_idx)), replace=False))
        val_idx = np.sort(rng.choice(val_idx, min(max(args.limit // 5, 1), len(val_idx)), replace=False))
    stats = get_stats(cfg, cache, train_idx, num_classes, cache_dir, run_dir, shareable=not args.limit)

    d, t = cfg["data"], cfg["train"]
    window = d.get("window_days")
    ds_train = CropDataset(cache, train_idx, stats, train=True, multi_year=d["multi_year"],
                           aggregation=d["aggregation"], prior_dropout=d["prior_dropout"],
                           min_cutoff_doy=d["min_cutoff_doy"], max_cutoff_doy=d["max_cutoff_doy"],
                           window_days=window)
    train_loader = make_loader(ds_train, t["batch_size"], True, t["num_workers"], device)
    val_loaders = {
        c: make_loader(CropDataset(cache, val_idx, stats, train=False, cutoff_doy=c,
                                   multi_year=d["multi_year"], aggregation=d["aggregation"],
                                   window_days=window),
                       t["batch_size"] * 2, False, t["num_workers"], device)
        for c in t["eval_cutoffs"]
    }

    model = build_model(cfg["model"], stats, num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    loss_fn = build_loss(cfg["loss"], stats["class_counts"]).to(device)
    # v1 runs: AdamW + halve the LR on plateaus. v2 runs follow the CatCrops code: Adam, constant LR.
    opt_cls = torch.optim.Adam if t.get("optimizer", "adamw") == "adam" else torch.optim.AdamW
    optimizer = opt_cls(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    plateau = t.get("lr_schedule", "plateau") == "plateau"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                                           patience=2 if plateau else 10**9)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch, best, bad = 1, -1.0, 0
    last = run_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best, bad = ck["epoch"] + 1, ck["best"], ck["bad_epochs"]
        torch.set_rng_state(ck["torch_rng"])
        print(f"Resumed {run_dir.name} from epoch {ck['epoch']} (best val score {best:.4f})")
    else:
        print(f"Starting {run_dir.name}: {n_params:,} parameters, {len(train_idx):,} train / "
              f"{len(val_idx):,} val fields, device {device}")

    hist_path = run_dir / "history.csv"
    t_start = time.time()
    stopped_early = False
    for epoch in range(start_epoch, t["epochs"] + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, device,
                             log_every=args.log_every)
        row = {"epoch": epoch, **tr, "lr": optimizer.param_groups[0]["lr"]}
        f1s = []
        for c, loader in val_loaders.items():
            res = predict(model, loader, device)
            m = metrics(res["y"], res["pred"])
            row[f"val_acc_{c}"] = m["accuracy"]
            row[f"val_macro_f1_{c}"] = m["macro_f1"]
            row[f"val_gate_mean_{c}"] = float(res["gate"].mean())
            f1s.append(m["macro_f1"])
        score = float(np.mean(f1s))
        row["val_score"] = score
        row["seconds"] = round(time.time() - t0, 1)
        scheduler.step(score)

        improved = score > best
        if improved:
            best, bad = score, 0
            save_atomic({"model": model.state_dict(), "epoch": epoch, "val_score": score,
                         "config": cfg, "num_classes": num_classes}, run_dir / "best.pt")
        else:
            bad += 1
        save_atomic({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                     "epoch": epoch, "best": best, "bad_epochs": bad,
                     "torch_rng": torch.get_rng_state()}, last)

        new_file = not hist_path.exists()
        with hist_path.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if new_file:
                w.writeheader()
            w.writerow(row)
        print(f"Epoch {epoch:02d} | loss {tr['train_loss']:.4f} | val macro-F1 by date "
              + " ".join(f"{c}:{row[f'val_macro_f1_{c}']:.3f}" for c in val_loaders)
              + f" | score {score:.4f}{' *best*' if improved else ''} | {row['seconds']:.0f}s", flush=True)

        if bad >= t["patience"]:
            print(f"Early stopping: no improvement for {bad} epochs")
            stopped_early = True
            break
        if args.max_minutes and (time.time() - t_start) / 60 > args.max_minutes:
            print("Time budget reached; rerun the same command to resume.")
            return {"status": "paused", "epoch": epoch, "best_val_score": best}

    best_ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    summary = {"name": cfg["name"], "seed": args.seed, "best_epoch": best_ck["epoch"],
               "best_val_score": best, "params": n_params, "stopped_early": stopped_early}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Finished {run_dir.name}: best epoch {summary['best_epoch']}, val score {best:.4f}")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", required=True, help="folder with 2022.npz / 2023.npz")
    ap.add_argument("--out", required=True, help="folder for run outputs (use Google Drive on Colab)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None, help="override config epochs")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="use only N training fields (smoke test)")
    ap.add_argument("--max-minutes", type=float, default=None, help="pause after this many minutes")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--force", action="store_true")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
