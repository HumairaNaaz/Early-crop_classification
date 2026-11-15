"""OWNER: Eval Lead (Divyanshi). F1-vs-day-of-year plots, confusion matrices, gate-value plots.
Every figure is saved as a publication-ready PDF plus a 300-DPI PNG preview in paper/figures/.

    python experiments/visualize_results.py --runs <runs dir> --out paper/figures

Reads what experiments/run_benchmark.py wrote to <runs>/benchmark/ (summary CSV and metrics
JSON) plus the per-field prediction files of the gate runs. Writes:

    f1_vs_doy.pdf              macro-F1, rare-class F1 and rotated-field accuracy vs cutoff day,
                               one line per ablation run, mean +/- std over seeds
    confusion_<run>_<date>.pdf row-normalised confusion matrix, seeds pooled
    per_class_f1_<run>_<date>.pdf  per-class F1 (mean +/- std over seeds), rare classes highlighted
    ablation_macro_f1.pdf      A-D ablation comparison: macro-F1 at every evaluation date
    gate_values.pdf            rotation-gate value for rotated vs non-rotated fields, per date

Colours: first four slots of the validated categorical palette (blue, orange, aqua, yellow) in
fixed order, one per ablation run; lines also differ by marker and are labelled directly, and
benchmark_summary.csv is the table view of the same numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import load_rare_class_ids  # noqa: E402

LADDER = ["A_baseline", "B_fusion", "C_gate", "D_full"]
SERIES = {
    "A_baseline": ("#2a78d6", "o"), "B_fusion": ("#eb6834", "s"),
    "C_gate": ("#1baf7a", "^"), "D_full": ("#eda100", "D"),
}
LABELS = {"A_baseline": "A baseline", "B_fusion": "B + fusion", "C_gate": "C + gate", "D_full": "D + CB focal"}
DATE_LABELS = {"01Mar": "1 Mar", "01Jun": "1 Jun", "01Sep": "1 Sep", "01Dec": "1 Dec"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2de", "#fcfcfb"
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"]
GROUP_COLORS = {"same": "#2a78d6", "changed": "#eb6834"}


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 8, "axes.titlesize": 9, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "legend.frameon": False, "pdf.fonttype": 42,
    })


def _save(fig, path: Path) -> Path:
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    return path


def _spread(values, gap):
    """Nudge label positions apart so direct labels never overlap (mean position preserved)."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    placed = values[order].copy()
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + gap)
    placed += values.mean() - placed.mean()
    out = np.empty_like(placed)
    out[order] = placed
    return out


def plot_f1_vs_doy(summary: pd.DataFrame, out: Path) -> Path:
    panels = [("macro_f1", "Macro-F1"), ("rare_class_f1", "Rare-class F1"),
              ("rotated_accuracy", "Accuracy, rotated fields")]
    cfgs = [c for c in LADDER if c in set(summary["config"])] or sorted(set(summary["config"]))
    cutoffs = sorted(summary["cutoff_doy"].unique())
    ticks = {int(r.cutoff_doy): DATE_LABELS.get(r.date, r.date) for r in summary.drop_duplicates("cutoff_doy").itertuples()}
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2, 2.7), constrained_layout=True)
    for ax, (metric, title) in zip(axes, panels):
        ends = []
        for cfg in cfgs:
            d = summary[summary["config"] == cfg].sort_values("cutoff_doy")
            color, marker = SERIES.get(cfg, ("#6250d6", "v"))
            mean = d[f"{metric}_mean"].to_numpy(float)
            std = np.nan_to_num(d[f"{metric}_std"].to_numpy(float))
            x = d["cutoff_doy"].to_numpy()
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
            ax.plot(x, mean, color=color, marker=marker, markersize=5, linewidth=2,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=LABELS.get(cfg, cfg))
            ends.append(mean[-1])
        span = np.nanmax(ends) - np.nanmin(ends) if len(ends) > 1 else 0.1
        ys = _spread(ends, max(span, 0.05) * 0.18)
        for cfg, y in zip(cfgs, ys):
            ax.text(cutoffs[-1] + 8, y, LABELS.get(cfg, cfg).split(" ", 1)[0], va="center", fontsize=7,
                    color=INK2)
        ax.set_title(title)
        ax.set_xticks(cutoffs)
        ax.set_xticklabels([ticks.get(int(c), str(c)) for c in cutoffs])
        ax.set_xlim(cutoffs[0] - 12, cutoffs[-1] + 40)
        ax.set_xlabel("Cutoff date")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper right", ncol=len(cfgs), fontsize=7.5)
    path = out / "f1_vs_doy.pdf"
    _save(fig, path)
    plt.close(fig)
    return path


def plot_confusion(all_metrics: dict, config: str, date: str, out: Path) -> Path | None:
    cms = [np.array(m[date]["confusion_matrix"]) for run, m in all_metrics.items()
           if run.rsplit("_seed", 1)[0] == config and date in m]
    if not cms:
        return None
    cm = np.sum(cms, axis=0).astype(float)
    rows = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
    n = cm.shape[0]
    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(5.2, 4.8), constrained_layout=True)
    im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            if norm[i, j] >= 0.10:
                ax.text(j, i, f"{norm[i, j] * 100:.0f}", ha="center", va="center", fontsize=5,
                        color="#ffffff" if norm[i, j] > 0.5 else INK)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.tick_params(labelsize=6)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"{LABELS.get(config, config)}, {DATE_LABELS.get(date, date)}: % of true class "
                 f"({len(cms)} seeds pooled)", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(labelsize=6)
    path = out / f"confusion_{config}_{date}.pdf"
    _save(fig, path)
    plt.close(fig)
    return path


def plot_gate_values(runs_dir: Path, year: int, configs: list[str], out: Path) -> Path | None:
    dates = list(DATE_LABELS)
    data = {}
    for cfg in configs:
        for d in dates:
            frames = [pd.read_csv(f, usecols=["gate", "rotation_group"])
                      for f in sorted(runs_dir.glob(f"{cfg}_seed*/predictions/{year}_{d}.csv.gz"))]
            if frames:
                data[(cfg, d)] = pd.concat(frames)
    cfgs = [c for c in configs if any(k[0] == c for k in data)]
    if not cfgs:
        return None
    fig, axes = plt.subplots(1, len(cfgs), figsize=(3.6 * len(cfgs), 2.8), constrained_layout=True,
                             sharey=True, squeeze=False)
    for ax, cfg in zip(axes[0], cfgs):
        for gi, (group, offset) in enumerate((("same", -0.2), ("changed", 0.2))):
            vals, pos = [], []
            for di, d in enumerate(dates):
                df = data.get((cfg, d))
                if df is None:
                    continue
                v = df.loc[df["rotation_group"] == group, "gate"].to_numpy(float)
                if v.size:
                    vals.append(v)
                    pos.append(di + offset)
            if not vals:
                continue
            color = GROUP_COLORS[group]
            bp = ax.boxplot(vals, positions=pos, widths=0.32, patch_artist=True, showfliers=False,
                            medianprops={"color": INK, "linewidth": 1.5},
                            whiskerprops={"color": color}, capprops={"color": color})
            for box in bp["boxes"]:
                box.set(facecolor=color, alpha=0.55, edgecolor=color)
            ax.plot([], [], color=color, linewidth=6, alpha=0.7,
                    label="same crop as 2022" if group == "same" else "crop changed")
        ax.set_xticks(range(len(dates)))
        ax.set_xticklabels([DATE_LABELS[d] for d in dates])
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{LABELS.get(cfg, cfg)}: gate value")
    axes[0][0].set_ylabel("Gate value g")
    handles, labels = axes[0][-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=7, frameon=False)
    path = out / "gate_values.pdf"
    _save(fig, path)
    plt.close(fig)
    return path


def plot_per_class_f1(per_class: pd.DataFrame, config: str, date: str, out: Path,
                      rare_ids) -> Path | None:
    d = per_class[(per_class["config"] == config) & (per_class["date"] == date)].sort_values("class_id")
    if d.empty:
        return None
    rare = {int(c) for c in rare_ids}
    ids = d["class_id"].to_numpy(int)
    mean = d["f1_mean"].to_numpy(float)
    std = np.nan_to_num(d["f1_std"].to_numpy(float))
    support = d["support"].to_numpy(int)
    colors = [GROUP_COLORS["changed"] if c in rare else GROUP_COLORS["same"] for c in ids]
    fig, ax = plt.subplots(figsize=(5.2, 0.28 * len(ids) + 1.4), constrained_layout=True)
    y = np.arange(len(ids))
    ax.barh(y, mean, xerr=std, color=colors,
            error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 2})
    ax.set_yticks(y)
    ax.set_yticklabels([f"class {c}" for c in ids])
    ax.set_ylim(len(ids) + 1.0, -0.6)
    ax.set_xlim(0, 1.2)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("F1 (mean \u00b1 std over seeds)")
    for yi, sup in enumerate(support):
        ax.text(1.02, yi, f"n={sup}", va="center", fontsize=6, color=INK2)
    ax.legend(handles=[Patch(color=GROUP_COLORS["same"], label="other classes"),
                       Patch(color=GROUP_COLORS["changed"], label="rare (<0.2% of 2022 train)")],
              loc="lower left", ncol=2, fontsize=6)
    ax.set_title(f"Per-class F1: {LABELS.get(config, config)}, {DATE_LABELS.get(date, date)}")
    path = out / f"per_class_f1_{config}_{date}.pdf"
    _save(fig, path)
    plt.close(fig)
    return path


def plot_ablation(summary: pd.DataFrame, out: Path) -> Path | None:
    configs = [c for c in LADDER if c in set(summary["config"])]
    if not configs:
        return None
    dates = summary.drop_duplicates("date").sort_values("cutoff_doy")["date"].tolist()
    width = 0.8 / len(configs)
    x = np.arange(len(dates))
    fig, ax = plt.subplots(figsize=(5.2, 3.0), constrained_layout=True)
    for k, cfg in enumerate(configs):
        d = summary[summary["config"] == cfg].set_index("date").reindex(dates)
        mean = d["macro_f1_mean"].to_numpy(float)
        std = np.nan_to_num(d["macro_f1_std"].to_numpy(float))
        ax.bar(x + (k - (len(configs) - 1) / 2) * width, mean, width, yerr=std,
               color=SERIES[cfg][0], label=LABELS[cfg],
               error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 2})
    ax.set_xticks(x)
    ax.set_xticklabels([DATE_LABELS.get(d, d) for d in dates])
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("Evaluation date (observations up to this day)")
    ax.set_ylabel("Macro-F1 (mean \u00b1 std over seeds)")
    ax.legend(loc="upper left", ncol=2, fontsize=6)
    ax.set_title("Ablation A-D: macro-F1 by evaluation date")
    path = out / "ablation_macro_f1.pdf"
    _save(fig, path)
    plt.close(fig)
    return path


def run(runs_dir, out_dir, year: int = 2023, cm_configs=("A_baseline", "D_full"),
        cm_dates=("01Mar", "01Dec"), gate_configs=("C_gate", "D_full"),
        pc_configs=("A_baseline", "D_full"), pc_date="01Dec") -> list[Path]:
    runs_dir, out_dir = Path(runs_dir), Path(out_dir)
    bench = runs_dir / "benchmark"
    if not (bench / "benchmark_summary.csv").exists():
        raise FileNotFoundError(f"{bench}/benchmark_summary.csv missing - run experiments/run_benchmark.py first")
    out_dir.mkdir(parents=True, exist_ok=True)
    _style()
    summary = pd.read_csv(bench / "benchmark_summary.csv")
    all_metrics = json.loads((bench / "benchmark_metrics.json").read_text())
    written = [plot_f1_vs_doy(summary, out_dir)]
    for cfg in cm_configs:
        for d in cm_dates:
            p = plot_confusion(all_metrics, cfg, d, out_dir)
            if p:
                written.append(p)
    pc_file = bench / "benchmark_per_class_f1.csv"
    if pc_file.exists():
        per_class = pd.read_csv(pc_file)
        rare_ids = load_rare_class_ids()
        for cfg in pc_configs:
            p = plot_per_class_f1(per_class, cfg, pc_date, out_dir, rare_ids)
            if p:
                written.append(p)
    p = plot_ablation(summary, out_dir)
    if p:
        written.append(p)
    p = plot_gate_values(runs_dir, year, list(gate_configs), out_dir)
    if p:
        written.append(p)
    for w in written:
        print("wrote", w)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="runs folder (contains benchmark/ and <run>_seed<k>/)")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--cm-configs", nargs="+", default=["A_baseline", "D_full"])
    ap.add_argument("--cm-dates", nargs="+", default=["01Mar", "01Dec"])
    ap.add_argument("--gate-configs", nargs="+", default=["C_gate", "D_full"])
    ap.add_argument("--pc-configs", nargs="+", default=["A_baseline", "D_full"])
    ap.add_argument("--pc-date", default="01Dec")
    a = ap.parse_args(argv)
    return run(a.runs, a.out, a.year, a.cm_configs, a.cm_dates, a.gate_configs,
               a.pc_configs, a.pc_date)


if __name__ == "__main__":
    main()
