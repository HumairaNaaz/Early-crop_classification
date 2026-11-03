"""End-to-end smoke test: train 2 epochs, resume, predict 2023 (CPU, synthetic data)."""
import json
from pathlib import Path

import pandas as pd

import predict
import train


CFG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_train_resume_predict(fake_cache, tmp_path):
    out = tmp_path / "runs"
    common = ["--config", str(CFG_DIR / "D_full.yaml"), "--cache", str(fake_cache), "--out", str(out),
              "--device", "cpu", "--log-every", "0"]
    # 1 epoch, then "disconnect", then resume to 2 epochs
    s1 = train.main(common + ["--epochs", "1"])
    run_dir = out / "D_full_seed0"
    assert (run_dir / "best.pt").exists() and (run_dir / "last.pt").exists()
    (run_dir / "summary.json").unlink()  # pretend the session died before finishing
    s2 = train.main(common + ["--epochs", "2"])
    hist = pd.read_csv(run_dir / "history.csv")
    assert list(hist["epoch"]) == [1, 2], "resume must continue at epoch 2"
    assert s2["best_epoch"] in (1, 2)

    m = predict.main(["--run", str(run_dir), "--cache", str(fake_cache), "--year", "2023",
                      "--device", "cpu", "--num-workers", "0"])
    assert set(m["dates"]) == {"01Mar", "01Jun", "01Sep", "01Dec"}
    df = pd.read_csv(run_dir / "predictions" / "2023_01Jun.csv.gz")
    assert len(df) == 180 and {"y_true", "y_pred", "rotation_group", "gate", "p_17"} <= set(df.columns)
    assert json.loads((run_dir / "predictions" / "metrics_2023.json").read_text())["year"] == 2023


def test_baseline_trains(fake_cache, tmp_path):
    s = train.main(["--config", str(CFG_DIR / "A_baseline.yaml"), "--cache", str(fake_cache),
                    "--out", str(tmp_path), "--device", "cpu", "--epochs", "1", "--log-every", "0"])
    assert s["best_epoch"] == 1


def test_v2_runs_train_and_predict(fake_cache, tmp_path):
    """v2 protocol (Adam, constant LR, 365-day window): faithful baseline and fixed run D."""
    for name in ("A_baseline", "D_full"):
        s = train.main(["--config", str(CFG_DIR / "v2" / f"{name}.yaml"), "--cache", str(fake_cache),
                        "--out", str(tmp_path), "--device", "cpu", "--epochs", "1", "--log-every", "0"])
        assert s["best_epoch"] == 1
        m = predict.main(["--run", str(tmp_path / f"{name}_seed0"), "--cache", str(fake_cache),
                          "--device", "cpu", "--num-workers", "0"])
        assert set(m["dates"]) == {"01Mar", "01Jun", "01Sep", "01Dec"}
