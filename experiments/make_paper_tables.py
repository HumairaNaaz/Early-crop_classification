"""OWNER: Eval Lead (Divyanshi). Turns the benchmark CSVs into LaTeX tables for the paper.

    python experiments/make_paper_tables.py --bench <runs dir>/benchmark --out paper/tables_generated.tex

Tables written (mean +/- standard deviation over seeds, same layout as Tables II-V):
    rare-class F1 per run and date, weighted-F1 per run and date, and per-class F1 of the
    baseline (A) versus the full model (D) at 1 December, with the rare classes marked.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import load_rare_class_ids  # noqa: E402

DATES = ["01Mar", "01Jun", "01Sep", "01Dec"]
DATE_HEAD = {"01Mar": "1 Mar", "01Jun": "1 Jun", "01Sep": "1 Sep", "01Dec": "1 Dec"}
RUNS = {"A_baseline": "A", "B_fusion": "B", "C_gate": "C", "D_full": "D"}


def _cell(mean: float, std: float, bold: bool) -> str:
    text = f"{mean:.3f}$\\pm${0.0 if pd.isna(std) else std:.3f}"
    return f"\\textbf{{{text}}}" if bold else text


def metric_table(summary: pd.DataFrame, metric: str, caption: str, label: str) -> str:
    best = {d: summary[summary["date"] == d][f"{metric}_mean"].max() for d in DATES}
    lines = [r"\begin{table}[t]", r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\centering", r"\begin{tabular}{lcccc}", r"\hline",
             "Run & " + " & ".join(DATE_HEAD[d] for d in DATES) + r" \\", r"\hline"]
    for cfg, letter in RUNS.items():
        cells = []
        for d in DATES:
            row = summary[(summary["config"] == cfg) & (summary["date"] == d)]
            if row.empty:
                cells.append("--")
                continue
            m, s = float(row[f"{metric}_mean"].iloc[0]), float(row[f"{metric}_std"].iloc[0])
            cells.append(_cell(m, s, abs(m - best[d]) < 1e-12))
        lines.append(f"{letter} & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def per_class_table(per_class: pd.DataFrame, date: str = "01Dec", cfgs=("A_baseline", "D_full")) -> str:
    rare = {int(c) for c in load_rare_class_ids()}
    d = per_class[per_class["date"] == date]
    first = d[d["config"] == cfgs[0]].set_index("class_id")
    second = d[d["config"] == cfgs[1]].set_index("class_id")
    lines = [r"\begin{table}[t]",
             r"\caption{Per-class F1 on the 2023 test year at " + DATE_HEAD[date]
             + r" for the baseline (A) and the full model (D), mean $\pm$ standard deviation over "
             r"three seeds. Rare classes (frequency below 0.2\% of the 2022 training fields) are "
             r"marked with an asterisk; $n$ is the number of test fields.}",
             r"\label{tab:per_class}", r"\centering", r"\footnotesize",
             r"\begin{tabular}{lrcc}", r"\hline",
             r"Class & $n$ & A & D \\", r"\hline"]
    for c in sorted(first.index):
        name = f"{c}" + ("*" if c in rare else "")
        a, b = first.loc[c], second.loc[c]
        lines.append(f"{name} & {int(a['support'])} & {_cell(a['f1_mean'], a['f1_std'], False)} & "
                     f"{_cell(b['f1_mean'], b['f1_std'], False)}" + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def run(bench_dir, out_file) -> Path:
    bench_dir, out_file = Path(bench_dir), Path(out_file)
    summary = pd.read_csv(bench_dir / "benchmark_summary.csv")
    per_class = pd.read_csv(bench_dir / "benchmark_per_class_f1.csv")
    parts = [
        metric_table(summary, "rare_class_f1",
                     r"Macro-F1 over the five rare classes (frequency below 0.2\% of the 2022 training "
                     r"fields) on the 2023 test year; runs as in Table~II.",
                     "tab:rare"),
        metric_table(summary, "weighted_f1",
                     r"Weighted-F1 on the 2023 test year (classes weighted by their number of test "
                     r"fields); runs as in Table~II.", "tab:weighted"),
        per_class_table(per_class),
    ]
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"wrote {out_file}")
    return out_file


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True, help="benchmark folder written by run_benchmark.py")
    ap.add_argument("--out", default="paper/tables_generated.tex")
    args = ap.parse_args(argv)
    return run(args.bench, args.out)


if __name__ == "__main__":
    main()
