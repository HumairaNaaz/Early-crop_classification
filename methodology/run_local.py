"""Run the whole methodology on this computer's NVIDIA GPU, from raw data to the results table.

    python run_local.py                 # everything (v1 configs)
    python run_local.py --seeds 0       # only seed 0
    python run_local.py --skip-tests    # skip the test suite

Steps (each one is skipped when its output already exists, so rerunning after a crash,
a restart or Ctrl+C simply continues where it stopped):

    1. check the GPU                         logs/run_local.log, logs/status.json
    2. test suite on synthetic data          (stops here if any test fails)
    3. download CatCrops from Kaggle         raw/        (14 GB zip, only once; read without unpacking)
    4. build the cached arrays               cache/2022.npz, cache/2023.npz
    5. smoke run: 2,000 fields, 1 epoch      smoke/      (times an epoch)
    6. ablation ladder A -> D, seeds 0 1 2   runs/<run>_seed<k>/  (checkpoints, predictions)
    7. results table                         results/results_summary.csv, results/RESULTS.md

raw/, cache/, smoke/, runs/ and logs/ are large or machine-specific and are ignored by git;
results/ is small and meant to be committed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS, RAW, CACHE, SMOKE, RUNS, RESULTS = (ROOT / d for d in ("logs", "raw", "cache", "smoke", "runs", "results"))
LOG_FILE, STATUS_FILE = LOGS / "run_local.log", LOGS / "status.json"
KAGGLE_DATASET = "irtaremotesensing/catcrops-dataset"
MIN_FREE_GB_FOR_DOWNLOAD = 50
DATE_ORDER = ["01Mar", "01Jun", "01Sep", "01Dec"]


# --------------------------------------------------------------------------- logging
class Tee:
    """Send everything printed to the console and to logs/run_local.log."""

    def __init__(self, stream, path: Path):
        self.stream, self.file = stream, path.open("a", encoding="utf-8", buffering=1)

    def write(self, s):
        self.stream.write(s)
        self.file.write(s)
        return len(s)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def isatty(self):
        return False


def say(msg: str) -> None:
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def status(stage: str, **extra) -> None:
    STATUS_FILE.write_text(json.dumps({"stage": stage, "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                                       **extra}, indent=2), encoding="utf-8")


def run_streamed(cmd: list[str], env: dict | None = None) -> int:
    """Run a command, echoing its output into the console and the log."""
    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", env=env)
    for line in proc.stdout:
        sys.stdout.write(line)
    sys.stdout.flush()
    return proc.wait()


def keep_awake() -> None:
    """Stop Windows from sleeping while this script runs (no admin rights needed).
    It cannot stop sleep when a laptop lid is closed, if Windows is set to sleep on lid close."""
    if os.name == "nt":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


# --------------------------------------------------------------------------- steps
def check_gpu(allow_cpu: bool) -> None:
    import torch
    say(f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name}, {p.total_memory / 2**30:.1f} GB memory")
        status("gpu_ok", gpu=p.name)
    elif not allow_cpu:
        raise SystemExit("No CUDA GPU visible to PyTorch. Nothing was run. (Use --allow-cpu to force CPU.)")


def run_tests() -> None:
    status("tests")
    say("Step 2: test suite")
    if run_streamed([sys.executable, "-m", "pytest", "-q", "tests"]) != 0:
        status("tests_failed")
        raise SystemExit("Tests failed, so no training was started. See the log above.")


def find_archive() -> Path | None:
    """A complete Kaggle download already in raw/ (kagglehub saves it as '<version>.archive')."""
    import zipfile
    for a in sorted(RAW.glob("datasets/*/*/*.archive")) + sorted(RAW.glob("*.zip")):
        if zipfile.is_zipfile(a):
            return a
    return None


def build_cache(raw_root: str | None) -> None:
    if (CACHE / "2022.npz").exists() and (CACHE / "2023.npz").exists():
        say("Steps 3-4: cache already built, skipping download and preprocessing")
        return
    env = dict(os.environ, KAGGLEHUB_CACHE=str(RAW))
    if raw_root is None:
        archive = find_archive()
        if archive is not None:
            # Read the zip directly: unpacked, CatCrops needs about 30 GB more disk space.
            say(f"Step 3: using the downloaded archive {archive} (read without unpacking)")
            partial = archive.parent / "versions"
            if partial.exists():
                say(f"Removing the partly unpacked copy in {partial} to free disk space")
                shutil.rmtree(partial, ignore_errors=True)
            raw_root = str(archive)
    if raw_root is None:
        free_gb = shutil.disk_usage(ROOT).free / 2**30
        say(f"Step 3: downloading CatCrops from Kaggle into {RAW} (14 GB zipped, 30 GB unpacked; "
            f"{free_gb:.0f} GB free)")
        if free_gb < MIN_FREE_GB_FOR_DOWNLOAD:
            raise SystemExit(f"Only {free_gb:.0f} GB free on this drive; downloading and unpacking need about "
                             f"{MIN_FREE_GB_FOR_DOWNLOAD} GB. Free some space, or download the Kaggle zip "
                             f"yourself into the raw folder and rerun (it is then read without unpacking).")
        status("downloading", target=str(RAW))
        code = ("import kagglehub; p = kagglehub.dataset_download(%r); "
                "open(%r, 'w', encoding='utf-8').write(p)" % (KAGGLE_DATASET, str(LOGS / "raw_path.txt")))
        # The download's progress bar goes straight to the console (not the log).
        if subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env).returncode != 0:
            raise SystemExit("Kaggle download failed (see above). Rerun to retry.")
        raw_root = (LOGS / "raw_path.txt").read_text(encoding="utf-8").strip()
    say(f"Step 4: building the cache from {raw_root} (about 15-40 minutes)")
    status("preprocessing", raw=raw_root)
    if run_streamed([sys.executable, "-m", "data.preprocess", "--root", raw_root, "--out", str(CACHE)],
                    env=env) != 0:
        raise SystemExit("Preprocessing failed (see above).")


def smoke_run() -> None:
    done = SMOKE / "done.json"
    if done.exists():
        say("Step 5: smoke run already done: " + done.read_text(encoding="utf-8"))
        return
    import train
    say("Step 5: smoke run (D_full, 2,000 fields, 1 epoch)")
    status("smoke_run")
    t0 = time.time()
    train.main(["--config", str(ROOT / "configs" / "D_full.yaml"), "--cache", str(CACHE),
                "--out", str(SMOKE), "--limit", "2000", "--epochs", "1", "--force"])
    done.write_text(json.dumps({"seconds": round(time.time() - t0, 1)}), encoding="utf-8")


def diagnose() -> None:
    import diagnose_collapse
    out = LOGS / "diagnose_D.json"
    if out.exists():
        say(f"Step 5b: collapse diagnosis already in {out}")
        return
    say("Step 5b: diagnosing the collapsed v1 run D (seed 2)")
    status("diagnosing")
    diagnose_collapse.main(["--runs", str(ROOT / "runs"), "--cache", str(CACHE), "--out", str(out)])


def ladder(configs: list[str], seeds: list[int], runs: Path, config_dir: Path | None) -> None:
    import run_experiments
    say(f"Step 6: ablation ladder {configs} x seeds {seeds} -> {runs}")
    status("training", configs=configs, seeds=seeds, runs=str(runs))
    argv = ["--cache", str(CACHE), "--out", str(runs), "--configs", *configs, "--seeds", *map(str, seeds)]
    if config_dir is not None:
        argv += ["--config-dir", str(config_dir)]
    run_experiments.main(argv)


def write_results(runs: Path, results: Path, title_note: str = "") -> None:
    import pandas as pd
    src = runs / "results_summary.csv"
    if not src.exists():
        say("Step 7: no finished runs to summarise yet")
        return
    results.mkdir(parents=True, exist_ok=True)
    for name in ("results_summary.csv", "results_per_seed.csv"):
        if (runs / name).exists():
            shutil.copy2(runs / name, results / name)
    for mfile in runs.glob("*_seed*/predictions/metrics_2023.json"):
        shutil.copy2(mfile, results / f"metrics_2023_{mfile.parent.parent.name}.json")

    t = pd.read_csv(src)
    t["_d"] = t["date"].map({d: i for i, d in enumerate(DATE_ORDER)})
    t = t.sort_values(["run", "_d"])

    def cell(r, m):
        sd = r[f"{m}_std"]
        return f"{r[f'{m}_mean']:.4f}" + ("" if pd.isna(sd) else f" ± {sd:.4f}")

    lines = [f"# Results: 2023 test year (CatCrops, Lleida){title_note}", "",
             "Mean ± standard deviation over seeds; `n` = number of seeds finished. "
             f"Generated by `run_local.py` from `{runs.name}/results_summary.csv`.", ""]
    for metric, title in (("accuracy", "Overall accuracy"), ("macro_f1", "Macro-F1"),
                          ("changed_macro_f1", "Macro-F1 on fields whose crop changed from 2022")):
        lines += [f"## {title}", "", "| Run | n | " + " | ".join(DATE_ORDER) + " |",
                  "| --- | --- |" + " --- |" * len(DATE_ORDER)]
        for run, g in t.groupby("run", sort=True):
            g = g.set_index("date")
            vals = [cell(g.loc[d], metric) if d in g.index else "" for d in DATE_ORDER]
            lines.append(f"| {run} | {int(g[f'{metric}_count'].max())} | " + " | ".join(vals) + " |")
        lines.append("")
    (results / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    say(f"Step 7: results written to {results}")


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--configs", nargs="+", default=["A_baseline", "B_fusion", "C_gate", "D_full"])
    ap.add_argument("--raw-root", default=None, help="an already downloaded CatCrops folder (skips Kaggle)")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--config-dir", default=None, help="run configs folder, e.g. configs/v2")
    ap.add_argument("--runs-dir", default="runs", help="output folder for this ladder, e.g. runs_v2")
    ap.add_argument("--results-dir", default="results", help="where the tables go, e.g. results/v2")
    ap.add_argument("--diagnose", action="store_true", help="first explain the collapsed v1 run D")
    args = ap.parse_args(argv)

    os.chdir(ROOT)
    os.environ["PYTHONUNBUFFERED"] = "1"   # child processes (tests, preprocessing) log line by line
    sys.path.insert(0, str(ROOT))
    LOGS.mkdir(exist_ok=True)
    sys.stdout = Tee(sys.stdout, LOG_FILE)
    sys.stderr = Tee(sys.stderr, LOG_FILE)
    say(f"run_local.py started in {ROOT}")
    keep_awake()
    t0 = time.time()
    try:
        check_gpu(args.allow_cpu)
        if not args.skip_tests:
            run_tests()
        build_cache(args.raw_root)
        smoke_run()
        if args.diagnose:
            try:
                diagnose()
            except Exception:  # the diagnosis must never block the training runs
                traceback.print_exc()
                say("Diagnosis failed (see above); continuing with training.")
        runs, results = ROOT / args.runs_dir, ROOT / args.results_dir
        config_dir = ROOT / args.config_dir if args.config_dir else None
        note = f" ({args.config_dir})" if args.config_dir else ""
        ladder(args.configs, args.seeds, runs, config_dir)
        write_results(runs, results, note)
    except SystemExit as e:
        if e.code not in (0, None):
            say(f"STOPPED: {e.code}")
            status("stopped", reason=str(e.code))
            return 1
        raise
    except KeyboardInterrupt:
        say("Interrupted. Run the same command again to continue from the last saved epoch.")
        status("interrupted")
        return 1
    except Exception:
        traceback.print_exc()
        status("error", traceback=traceback.format_exc()[-3000:])
        return 1
    hours = (time.time() - t0) / 3600
    say(f"Everything finished in {hours:.1f} h.")
    status("finished", hours=round(hours, 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
