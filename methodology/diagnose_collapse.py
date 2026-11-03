"""Why did run D (v1, seed 2) collapse? Compare its checkpoints with a healthy D run.

    python diagnose_collapse.py --runs runs --cache cache --out logs/diagnose_D.json

For each checkpoint, on 2,000 validation fields at the four benchmark cutoffs, it measures:
  * how much the prediction depends on the cutoff date (a collapsed model ignores the series)
  * the size of each term in  H = FiLM(h_temp; h_static) + gate * e_prev :
      |gamma * h_temp| (satellite signal after FiLM scaling), |beta| (static shift),
      |gate * e_prev| (previous crop)
  * the FiLM scale gamma = 1 + to_gamma(h_static) (near 0 means the series is switched off)
  * the share of input-projection units that are never active ("dead" ReLUs)
  * the most common predicted class and its share
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.cache import YearCache, load_stats, make_split
from data.dataset import BENCHMARK_CUTOFFS, CropDataset, collate
from models import build_model
from models.common import masked_max_pool


@torch.no_grad()
def inspect(run_dir: Path, ckpt: str, cache, val_idx, device) -> dict:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    stats = load_stats(run_dir / "stats")
    ck = torch.load(run_dir / ckpt, map_location="cpu", weights_only=False)
    model = build_model(cfg["model"], stats, len(stats["class_counts"]))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    if not hasattr(model, "fusion"):
        return {"skipped": "not a dual-branch model"}

    preds, out = {}, {"checkpoint": ckpt, "epoch": int(ck.get("epoch", -1))}
    terms = {"gamma_h_temp": [], "beta": [], "gate_e_prev": [], "gamma_mean": [], "gamma_abs_min": [],
             "dead_units": []}
    for name, cutoff in BENCHMARK_CUTOFFS.items():
        ds = CropDataset(cache, val_idx, stats, train=False, cutoff_doy=cutoff,
                         multi_year=cfg["data"]["multi_year"], aggregation=cfg["data"]["aggregation"],
                         window_days=cfg["data"].get("window_days"))
        p_all = []
        for batch in DataLoader(ds, batch_size=500, collate_fn=collate):
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            t = model.temporal
            pre = t.in_proj(t.in_norm(b["x"]))
            active = ((pre > 0) & b["mask"].unsqueeze(-1)).any(dim=(0, 1))
            terms["dead_units"].append(1.0 - active.float().mean().item())
            tokens = t(b["x"], b["days"], b["mask"])
            h_static, static_tokens = model.static(b["static_cont"], b["static_cat"])
            if model.cross is not None:
                tokens = model.cross(tokens, static_tokens)
            h_temp = masked_max_pool(tokens, b["mask"])
            gamma = 1.0 + model.fusion.to_gamma(h_static)
            beta = model.fusion.to_beta(h_static)
            terms["gamma_h_temp"].append((gamma * h_temp).norm(dim=-1).mean().item())
            terms["beta"].append(beta.norm(dim=-1).mean().item())
            terms["gamma_mean"].append(gamma.mean().item())
            terms["gamma_abs_min"].append(gamma.abs().mean(dim=-1).min().item())
            o = model(b)
            if model.prev_emb is not None:
                e_prev = model.prev_emb((b["y_prev"] + 1).clamp(min=0))
                terms["gate_e_prev"].append((o["gate"].unsqueeze(-1) * e_prev).norm(dim=-1).mean().item())
            p_all.append(o["logits"].argmax(-1).cpu().numpy())
        preds[name] = np.concatenate(p_all)

    names = list(preds)
    same = [float((preds[a] == preds[b]).mean()) for a, b in zip(names, names[1:])]
    first = preds[names[0]]
    top = np.bincount(first).argmax()
    out.update({
        "share_same_prediction_between_consecutive_dates": same,
        "most_common_class": int(top), "most_common_class_share": float((first == top).mean()),
        "n_distinct_predicted_classes": int(len(np.unique(first))),
        **{k: float(np.mean(v)) for k, v in terms.items() if v},
    })
    return out


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="logs/diagnose_D.json")
    ap.add_argument("--fields", type=int, default=2000)
    args = ap.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = YearCache.load(args.cache, 2022)
    _, val_idx = make_split(cache, 0.10, 42)
    val_idx = np.sort(np.random.default_rng(0).choice(val_idx, min(args.fields, len(val_idx)), replace=False))
    report = {}
    for run, ckpts in (("D_full_seed2", ("best.pt", "last.pt")), ("D_full_seed0", ("best.pt",)),
                       ("C_gate_seed2", ("best.pt",))):
        run_dir = Path(args.runs) / run
        for ck in ckpts:
            if (run_dir / ck).exists():
                print(f"diagnosing {run}/{ck} ...", flush=True)
                report[f"{run}/{ck}"] = inspect(run_dir, ck, cache, val_idx, device)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
