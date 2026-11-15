"""run_benchmark.py and visualize_results.py on synthetic prediction files that use the exact
layout methodology/predict.py writes. Needs pandas (+ matplotlib for the figures)."""
import json

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from experiments import run_benchmark  # noqa: E402

N = 18
DATES = {"01Mar": 60, "01Jun": 152, "01Sep": 244, "01Dec": 335}


def _engine_metrics(y, p):
    """What methodology/engine.metrics reports (sklearn defaults)."""
    sk = pytest.importorskip("sklearn.metrics")
    return {"n": len(y), "accuracy": float(sk.accuracy_score(y, p)),
            "macro_f1": float(sk.f1_score(y, p, average="macro", zero_division=0)),
            "weighted_f1": float(sk.f1_score(y, p, average="weighted", zero_division=0))}


def make_runs(root, configs=("A_baseline", "D_full"), seeds=(0, 1), n=600, with_engine=True):
    for ci, cfg in enumerate(configs):
        for seed in seeds:
            rng = np.random.default_rng(100 * ci + seed)
            pdir = root / f"{cfg}_seed{seed}" / "predictions"
            pdir.mkdir(parents=True)
            engine = {"run": f"{cfg}_seed{seed}", "year": 2023, "dates": {}}
            for di, (date, cutoff) in enumerate(DATES.items()):
                y = rng.integers(0, N, n)
                acc = 0.5 + 0.1 * di + 0.05 * ci
                p = np.where(rng.random(n) < acc, y, rng.integers(0, N, n))
                prev = np.where(rng.random(n) < 0.03, -1, np.where(rng.random(n) < 0.7, y, rng.integers(0, N, n)))
                group = np.where(prev < 0, "unknown", np.where(prev == y, "same", "changed"))
                df = pd.DataFrame({"field_id": np.arange(n), "cutoff_doy": cutoff, "y_true": y, "y_pred": p,
                                   "y_prev": prev, "rotation_group": group,
                                   "gate": np.where(group == "changed", 0.4, 0.8) + 0.05 * rng.standard_normal(n),
                                   "divergence": rng.random(n)})
                probs = pd.DataFrame(rng.random((n, N)), columns=[f"p_{k}" for k in range(N)])
                pd.concat([df, probs], axis=1).to_csv(pdir / f"2023_{date}.csv.gz", index=False)
                if with_engine:
                    engine["dates"][date] = {"overall": _engine_metrics(y, p)}
            if with_engine:
                (pdir / "metrics_2023.json").write_text(json.dumps(engine))


def test_benchmark_outputs_and_agreement_with_engine(tmp_path):
    make_runs(tmp_path)
    metrics = run_benchmark.run(tmp_path, 2023, tmp_path / "benchmark")
    assert set(metrics) == {"A_baseline_seed0", "A_baseline_seed1", "D_full_seed0", "D_full_seed1"}
    bench = tmp_path / "benchmark"
    summary = pd.read_csv(bench / "benchmark_summary.csv")
    assert len(summary) == 2 * 4
    assert (summary["macro_f1_count"] == 2).all()
    per_seed = pd.read_csv(bench / "benchmark_per_seed.csv")
    assert {"rotated_accuracy", "rare_class_f1", "macro_f1"} <= set(per_seed.columns)
    assert per_seed["rotated_accuracy"].notna().all()
    per_class = pd.read_csv(bench / "benchmark_per_class_f1.csv")
    assert per_class["class_id"].nunique() == N
    for run, dates in metrics.items():
        diff = run_benchmark.check_against_engine(tmp_path / run / "predictions", 2023, dates)
        assert diff is not None and diff < 1e-9


def test_benchmark_without_predictions_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_benchmark.run(tmp_path, 2023, tmp_path / "o")


def test_figures_are_written(tmp_path):
    pytest.importorskip("matplotlib")
    from experiments import visualize_results
    make_runs(tmp_path, with_engine=False)
    run_benchmark.run(tmp_path, 2023, tmp_path / "benchmark")
    out = tmp_path / "figs"
    written = visualize_results.run(tmp_path, out, cm_configs=("D_full",), cm_dates=("01Mar",),
                                    gate_configs=("D_full",), pc_configs=("D_full",))
    names = {p.name for p in written}
    assert names == {"f1_vs_doy.pdf", "confusion_D_full_01Mar.pdf", "gate_values.pdf",
                     "per_class_f1_D_full_01Dec.pdf", "ablation_macro_f1.pdf"}
    assert all(p.stat().st_size > 1000 for p in written)
    for pdf in written:
        png = pdf.with_suffix(".png")
        assert png.exists() and png.stat().st_size > 1000
    from matplotlib.image import imread
    assert imread(out / "ablation_macro_f1.png").shape[1] >= 5.2 * 300 - 5
