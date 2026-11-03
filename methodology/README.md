# Early Crop Type Classification: Dual-Branch Fusion and Rotation-Aware Gating

Methodology code for the Minor Project *Enhanced Early Crop Type Classification via Dual-Branch
Cross-Attention and Rotation-Aware Adaptive Gating* (B.Tech CSE-AI, IGDTUW). Supervisor: Dr. Himanshu Mittal.

The task is to name the crop in each field from Sentinel-2 time series as early in the season as
possible. Data: CatCrops, Lleida (Catalonia), trained on 2022 and tested on 2023, 18 crop classes.
Baseline: Gené-Mola et al., *Early crop type classification and mapping combining remote sensing
and ancillary data*, Computers and Electronics in Agriculture, 2026.

## What differs from the baseline

| | Baseline (run A) | Proposed (runs B-D) |
| --- | --- | --- |
| Static field facts | each fact appended as one extra time step (official code) | separate static branch running in parallel with the time-series branch |
| Fusion | none (one input table) | cross-attention from time steps to static tokens, then FiLM (static facts scale and shift the temporal features) |
| Previous-year crop | always used | Rotation-Aware Adaptive Gate lowers its weight when this season's signal diverges from last year's crop |
| Loss | cross-entropy | Class-Balanced Focal Loss (run D) |

Ablation ladder, each step adds one change:

| Run | Config | Adds |
| --- | --- | --- |
| A | `configs/v2/A_baseline.yaml` | faithful baseline reproduction |
| B | `configs/v2/B_fusion.yaml` | dual-branch + cross-attention + FiLM (previous crop always added) |
| C | `configs/v2/C_gate.yaml` | rotation gate |
| D | `configs/v2/D_full.yaml` | Class-Balanced Focal Loss (full model) |

Switchable extras, off in A-D: `D_full_prior_dropout.yaml` (hide the previous crop for 30% of training
fields) and `D_full_cloud_weighted.yaml` (cloud-weighted merging of overlapping tiles).

## Model

```
Sentinel-2 series (12 ch x T) --> Branch 1: Transformer, 5 blocks, d=64 --+--> cross-attention --> max-pool --> h_temp
static facts (elev, slope,    --> Branch 2: embeddings + MLP ------------+--> h_static, 4 static tokens
  irrigation, municipality)
previous-year crop            --> embedding e_prev
                                  rotation gate: gamma = sigmoid(W[h_temp | e_prev] - beta * D_rot)

H = FiLM(h_temp; h_static) + gamma * e_prev   -->   MLP   -->   18 classes
```

* 12 input channels per observation: 10 bands (B2-B8, B8A, B11, B12), cloud probability, day offset / 365.
* Previous-year observations are kept with negative day offsets; time is also encoded with a sinusoidal
  encoding of the acquisition day.
* D_rot = mean over this season's observations of 1 - cos(x_t, mu_p(t)), where mu_p(t) is the average
  2022 training profile of last year's crop p at that time of year.
* Training draws a random cutoff day per sample, so one model predicts at any date. Evaluation uses
  1 Mar, 1 Jun, 1 Sep and 1 Dec. Model selection uses mean validation macro-F1 over those dates.

## Layout

```
data/preprocess.py     CatCrops CSVs -> one cached array file per year (run once)
data/cache.py          loading, 90/10 stratified split (seed 42), training statistics, expected profiles
data/dataset.py        PyTorch dataset with cutoff dates, collate with padding masks
models/                baseline.py, branches.py, film_fusion.py, rotation_gate.py, full_model.py
losses.py              cross-entropy and Class-Balanced Focal Loss
train.py               one run (config + seed) with checkpoint / resume
predict.py             2023 predictions at the 4 dates + metrics per rotation group
run_experiments.py     the whole ladder over seeds, then results_summary.csv
configs/               base.yaml + one file per run
tests/                 pytest suite on a synthetic CatCrops copy
notebooks/colab_run.ipynb   end-to-end Colab workflow
run_local.py           end-to-end local workflow (tests, data, smoke run, ladder, results)
setup_env.py           installs the GPU build of PyTorch and the requirements into the active environment
METHODOLOGY.md         plain-language plan of everything implemented
paper/methodology.tex  methodology section for the paper
```

## Running

**Final results (v2 protocol), with an NVIDIA GPU:**

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
python setup_env.py               # GPU build of PyTorch + requirements
python run_local.py --config-dir configs/v2 --runs-dir runs_v2 --results-dir results/v2 --configs D_full A_baseline B_fusion C_gate --diagnose
```

`run_local.py` does every step and skips any that are already done:
* GPU check and tests;
* download CatCrops from Kaggle and build the cache (the zip is read without unpacking);
* the ladder A-D x seeds 0, 1, 2 under the official CatCrops training setup (`configs/v2`, faithful baseline A);
* `results/v2/RESULTS.md`.

Run the same command again after any interruption and it continues from the last saved epoch. Progress is written to
`logs/run_local.log`. `METHODOLOGY.md` explains everything implemented and the results (section 9).

The underlying steps, run by hand:

```bash
pip install -r requirements.txt
python -m pytest -q tests
python -m data.preprocess --root <catcrops_dataset folder or the downloaded zip> --out cache
python run_experiments.py --cache cache --out runs_v2 --config-dir configs/v2 --seeds 0 1 2
```

`configs/*.yaml` and `runs/` / `results/` (top level) hold the earlier v1 ladder, which is superseded and kept for
reference. `notebooks/colab_run.ipynb` is the Colab version of the v1 workflow.

## Outputs for evaluation

For each run, `runs_v2/<run>_seed<k>/predictions/` holds one CSV per date with:
* every 2023 field's true label, prediction and previous crop;
* its rotation group (same / changed / unknown), gate value and divergence;
* the 18 class probabilities.

It also holds `metrics_2023.json`. Per-class metrics, confusion matrices and significance tests can be computed from
these files without retraining. The large folders (`raw/`, `cache/`, `runs*/`, `handover_deepasha/`) are not in git.

## Results (2023, mean over 3 seeds)

| Run | Accuracy, 1 Mar / 1 Jun / 1 Sep / 1 Dec | Macro-F1, 1 Mar / 1 Jun / 1 Sep / 1 Dec |
| --- | --- | --- |
| A faithful baseline | 0.786 / 0.831 / 0.872 / 0.882 | 0.462 / 0.501 / 0.555 / 0.576 |
| B dual-branch + FiLM | 0.787 / 0.823 / 0.847 / 0.848 | 0.465 / 0.503 / 0.525 / 0.529 |
| C + rotation gate | 0.770 / 0.815 / 0.842 / 0.845 | 0.453 / 0.499 / 0.523 / 0.527 |
| D + CB focal loss | 0.791 / 0.829 / 0.849 / 0.852 | 0.467 / 0.514 / 0.541 / 0.545 |

Standard deviations and changed-crop results are in `results/v2/RESULTS.md` and `METHODOLOGY.md`.

## Baseline check

Run A follows the official code at https://github.com/UEA-Irta/CatCrops: the same 18 classes and field counts, the
same inputs, architecture and training setup. The comparison is in `METHODOLOGY.md`, section 10.
