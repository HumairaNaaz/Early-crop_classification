# Methodology implementation plan

**Project:** Enhanced Early Crop Type Classification via Dual-Branch Cross-Attention and Rotation-Aware Adaptive Gating
**Team:** Minor Project, B.Tech CSE-AI, IGDTUW. Supervisor: Dr. Himanshu Mittal
**This part:** methodology (Anshika Rana). Dataset stage: Ishu. Evaluation: Deepasha. Paper drafting: Humaira and Divyanshi.

This file describes exactly what the code in this repository implements, in the order it runs.
The LaTeX version for the paper is `paper/methodology.tex`.

---

## 1. Goal

Name the crop growing in each field from its Sentinel-2 satellite time series **as early in the season as
possible**, and do better than the baseline (Gené-Mola et al., *Computers and Electronics in Agriculture*, 2026)
at every date. The proposal changes three things in the baseline:

1. **Static field facts get their own branch.** The baseline copies elevation, slope, irrigation and municipality
   onto every time step. Here a second branch processes them, and they are fused with cross-attention and FiLM.
2. **Last year's crop is gated.** The baseline always trusts the previous crop. Here a rotation-aware gate turns it
   down when this season's signal no longer looks like last year's crop, for example when the farmer rotated crops.
3. **Rare crops get more weight.** The cross-entropy loss is replaced by a class-balanced focal loss.

## 2. Data

| Item | Setting |
| --- | --- |
| Dataset | CatCrops (IRTA), Kaggle `irtaremotesensing/catcrops-dataset`, Lleida region |
| Train / validation | 2022 fields, stratified 90/10 split by crop, seed 42 (same split as the dataset stage) |
| Test | 2023 fields, used only for final testing |
| Classes | 18 crop classes (raw class ids 10 "Altres extensius" and 17 "Cebes" removed) |
| Per observation | 10 bands (B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12) + cloud probability |
| Static facts | elevation, slope, irrigation system, municipality |
| Previous crop | the field's crop in the previous year (`pcrop_code` in the dataset index); "unknown" when missing or not one of the 18 classes |

**Preprocessing** (`data/preprocess.py`, run once): the ~265,000 per-field CSV files are read once and written to
one array file per year (`cache/2022.npz`, `cache/2023.npz`). When overlapping satellite tiles see a field on the
same day, the views are averaged. Observations from the previous year are kept with negative day numbers, so the
model sees last season too, as in the baseline's multi-year input.

**Normalisation** (`data/cache.py`): band means and standard deviations, elevation and slope statistics, category
vocabularies and class counts are computed on the 2022 training fields only.

## 3. Input to the models

Each observation becomes 12 numbers: the 10 standardised bands, the cloud probability and the day of year divided
by 365. A field is the list of all its observations before the **cutoff day** τ.

* **Training:** τ is drawn at random for every field and every epoch, so one model learns to predict on any date.
* **Evaluation:** τ = 1 March (day 60), 1 June (152), 1 September (244) and 1 December (335).
* Nothing after the cutoff is ever shown to the model (checked by a test).

## 4. Models (the ablation ladder)

Each run changes **exactly one thing** relative to the run before it, so every gain can be attributed.

| Run | Config | What it is |
| --- | --- | --- |
| A | `configs/A_baseline.yaml` | Baseline reproduction: one Transformer; static facts and previous crop appended to every time step; cross-entropy |
| B | `configs/B_fusion.yaml` | Dual-branch model with cross-attention and FiLM; previous crop always added (gate fixed at 1) |
| C | `configs/C_gate.yaml` | B + rotation-aware adaptive gate |
| D | `configs/D_full.yaml` | C + class-balanced focal loss (the full proposed model) |

Optional extras, switched off in A to D: `D_full_prior_dropout.yaml` (hide the previous crop for 30% of training
fields) and `D_full_cloud_weighted.yaml` (cloud-weighted merging of overlapping tiles).

### 4.1 Temporal branch (all runs)
Linear projection of the 12 input numbers to width 64, plus a sinusoidal encoding of the acquisition day, then a
Transformer encoder: 5 blocks, 2 attention heads, feed-forward width 128, dropout 0.1. Padding is masked.

### 4.2 Static branch (B, C, D)
Elevation and slope are standardised; irrigation system and municipality are embedded. An MLP gives a static
summary `h_static` (64 numbers), and each of the 4 facts is also turned into its own token.

### 4.3 Fusion (B, C, D)
* **Cross-attention:** every time step attends to the 4 static tokens, so for example a dip in reflectance can be
  read differently for an irrigated field than for a dry one.
* **Max-pooling** over valid time steps gives `h_temp`.
* **FiLM:** `h = (1 + φ(h_static)) * h_temp + ψ(h_static)`. φ and ψ start at zero, so training starts from the
  plain temporal features and learns how much the static facts should change them.

### 4.4 Rotation-aware adaptive gate (C, D)
For each crop c and each 5-day period of the year, the average band profile μ_c is computed from the 2022 training
fields. For a field whose previous crop is p:

```
D_rot = mean over this season's observations of  1 - cos(x_t, μ_p(day of x_t))
γ     = sigmoid( W [h_temp | e_prev | has_observations] + b  -  λ · D_rot ),   λ = softplus(λ0) > 0
H     = FiLM output + γ · e_prev
```

`e_prev` is a learned embedding of the previous crop. γ = 0 when the previous crop is unknown. When the field looks
like last year's crop, D_rot is small and γ stays high; when it looks different (crop rotated), γ drops and the
model relies on the satellite signal.

### 4.5 Classifier and loss
An MLP head outputs probabilities for the 18 classes.
* Runs A, B, C: cross-entropy.
* Run D: class-balanced focal loss, with class weights from the effective number of samples (β = 0.9999) and
  focusing exponent 2.

## 5. Training protocol (identical for every run)

| Setting | Value |
| --- | --- |
| Optimiser | AdamW, learning rate 1e-3, weight decay 1e-4 |
| Batch size | 256 |
| Epochs | up to 30; learning rate halved when the validation score plateaus (ReduceLROnPlateau, patience 2) |
| Early stopping | patience 6 epochs |
| Model selection | best mean validation macro-F1 over the 4 dates (not accuracy, so rare crops count) |
| Precision | mixed precision (float16) on GPU, gradient clipping at 1.0 |
| Seeds | 0, 1, 2 for every run (12 trainings in total) |
| Checkpoints | `last.pt` after every epoch (training resumes after any interruption), `best.pt` for the best epoch |

## 6. What is produced for the evaluation stage

For each finished run, `runs/<run>_seed<k>/predictions/` contains:

* `2023_01Mar.csv.gz`, `2023_01Jun.csv.gz`, `2023_01Sep.csv.gz`, `2023_01Dec.csv.gz`: one row per 2023 field with
  the true crop, the predicted crop, the previous crop, the rotation group (same crop / changed crop / unknown),
  the gate value γ, the divergence D_rot and all 18 class probabilities.
* `metrics_2023.json`: accuracy, macro-F1 and weighted-F1 per date, overall and for each rotation group.

`results/results_summary.csv` and `results/RESULTS.md` give mean ± standard deviation over the three seeds. From the
prediction files, Deepasha can compute per-class F1, confusion matrices, early-date curves and significance tests
(e.g. McNemar between runs) without retraining anything.

## 7. Checks built into the code

`tests/` (run automatically before training) checks, on a synthetic copy of CatCrops:
no observation after the cutoff is used; previous-year observations are kept or dropped as configured; padding never
changes a prediction; every parameter of every model receives gradient; the gate is 0 when the previous crop is
unknown and lies in [0, 1]; the gate's divergence is higher when the previous crop is wrong; FiLM starts as the
identity; class-balanced weights favour rare classes; and focal loss reduces to cross-entropy when it should.

## 8. How it is run

**With an NVIDIA GPU:** create a virtual environment, run `python setup_env.py` (installs the GPU build of PyTorch
and the requirements), then run

    python run_local.py --config-dir configs/v2 --runs-dir runs_v2 --results-dir results/v2 --configs D_full A_baseline B_fusion C_gate --diagnose

`run_local.py` does every step in order and skips any step already done:

1. GPU check
2. tests
3. download CatCrops (once)
4. build the cache (once)
5. 1-epoch smoke run on 2,000 fields
6. runs A, B, C, D for seeds 0, 1, 2 (seed-major order, so the full A-D comparison for seed 0 exists first)
7. results table in `results/v2/`

Progress is written to `logs/run_local.log` and `logs/status.json`. If the computer restarts, run the same command
again: training resumes from the last saved epoch.

**On Colab:** `notebooks/colab_run.ipynb` runs the same steps with Google Drive for storage.

## 9. Results (v2 protocol, 3 seeds, 2023 test year)

These are the results to report. Every run follows the official CatCrops training setup, and run A is the faithful
baseline (section 10). Values are mean ± standard deviation over seeds 0, 1, 2, trained on a local RTX 3050 (12 runs,
17 h, finished 3 Nov 2026). The full tables are in `results/v2/` and the predictions in `runs_v2/`. The 2023 fields
were never used for training or for choosing epochs.

**Overall accuracy**

| Run | 1 Mar | 1 Jun | 1 Sep | 1 Dec |
| --- | --- | --- | --- | --- |
| A faithful baseline | 0.786 ± 0.016 | **0.831 ± 0.014** | **0.872 ± 0.011** | **0.882 ± 0.006** |
| B dual-branch + FiLM | 0.787 ± 0.003 | 0.823 ± 0.016 | 0.847 ± 0.016 | 0.848 ± 0.017 |
| C + rotation gate | 0.770 ± 0.027 | 0.815 ± 0.020 | 0.842 ± 0.019 | 0.845 ± 0.016 |
| D + CB focal loss | **0.791 ± 0.004** | 0.829 ± 0.008 | 0.849 ± 0.010 | 0.852 ± 0.012 |

**Macro-F1 (all 18 crops weighted equally)**

| Run | 1 Mar | 1 Jun | 1 Sep | 1 Dec |
| --- | --- | --- | --- | --- |
| A faithful baseline | 0.462 ± 0.002 | 0.501 ± 0.004 | **0.555 ± 0.008** | **0.576 ± 0.015** |
| B dual-branch + FiLM | 0.465 ± 0.001 | 0.503 ± 0.009 | 0.525 ± 0.007 | 0.529 ± 0.007 |
| C + rotation gate | 0.453 ± 0.013 | 0.499 ± 0.011 | 0.523 ± 0.020 | 0.527 ± 0.021 |
| D + CB focal loss | **0.467 ± 0.003** | **0.514 ± 0.004** | 0.541 ± 0.014 | 0.545 ± 0.015 |

**Accuracy on fields whose crop changed from 2022**

| Run | 1 Mar | 1 Jun | 1 Sep | 1 Dec |
| --- | --- | --- | --- | --- |
| A faithful baseline | 0.288 ± 0.051 | **0.442 ± 0.050** | **0.601 ± 0.043** | **0.648 ± 0.027** |
| B dual-branch + FiLM | 0.283 ± 0.009 | 0.409 ± 0.030 | 0.509 ± 0.027 | 0.518 ± 0.036 |
| C + rotation gate | 0.265 ± 0.053 | 0.407 ± 0.045 | 0.509 ± 0.053 | 0.529 ± 0.042 |
| D + CB focal loss | **0.292 ± 0.028** | 0.431 ± 0.026 | 0.542 ± 0.026 | 0.562 ± 0.023 |

**Reading the results**

* **The faithful baseline (A) reproduces the paper.** It reaches 0.872 / 0.882 accuracy on 1 Sep / 1 Dec (paper:
  0.88 / 0.88). On 1 Mar / 1 Jun it is about 2–3 points below the paper, which chose its best epoch on the 2023 data
  itself.
* **The full model (D) helps early in the season, mainly on rare crops.**
  * On 1 March it ties A on accuracy (0.791 vs 0.786), macro-F1 and changed-crop accuracy; the differences are
    within seed variation.
  * On 1 June its macro-F1 is 1.3 points higher (0.514 vs 0.501, with standard deviations of about 0.004). This is
    the clearest gain over the baseline.
* **From September on the baseline is clearly better**: 2–3 points of accuracy, 1–3 points of macro-F1, and 6–9
  points on changed-crop fields. The proposed models do not carry over from 2022 to 2023 as well. On the 2022
  validation split B and C score higher than A (mean validation score 0.72–0.73 vs 0.68), but on 2023 they score
  lower. The likely cause is the learned embeddings of municipality and irrigation system (and the positional
  encoding), which can fit 2022-specific patterns; the baseline only sees scaled codes.
* **Within the proposed models:**
  * the class-balanced focal loss is the component that helps (D vs C: +1.4 to +1.8 points macro-F1, +0.7 to
    +2.1 points accuracy);
  * the rotation gate on its own does not improve on the fixed previous-crop term (C ≈ B).
* **Targets from the implementation plan are not met.** Changed-crop accuracy early in the season is about 29%
  (target 75%). Rare-crop F1 is to be computed per class in the evaluation stage.
* **Suggested next experiment (optional, about one night):** give B–D the baseline's scaled encoding of the field
  facts instead of learned embeddings, and drop the positional encoding. This tests whether the year-transfer gap
  closes.

### Earlier v1 run (superseded)

The first ladder (30 epochs, AdamW, run A not yet faithful, D with β = 0.9999 / γ = 2) is kept in `runs/` and
`results/` for reference only:

**Overall accuracy**

| Run | 1 Mar | 1 Jun | 1 Sep | 1 Dec |
| --- | --- | --- | --- | --- |
| A baseline | 0.787 ± 0.007 | 0.815 ± 0.016 | 0.847 ± 0.017 | 0.853 ± 0.015 |
| B dual-branch + FiLM | **0.788 ± 0.011** | **0.823 ± 0.007** | 0.853 ± 0.002 | 0.855 ± 0.001 |
| C + rotation gate | 0.775 ± 0.027 | 0.820 ± 0.025 | **0.855 ± 0.011** | **0.859 ± 0.012** |
| D + CB focal loss | 0.732 ± 0.011 | 0.750 ± 0.015 | 0.769 ± 0.034 | 0.771 ± 0.035 |

**Macro-F1 (all 18 crops weighted equally)**

| Run | 1 Mar | 1 Jun | 1 Sep | 1 Dec |
| --- | --- | --- | --- | --- |
| A baseline | 0.448 ± 0.005 | **0.499 ± 0.011** | **0.542 ± 0.023** | **0.547 ± 0.022** |
| B dual-branch + FiLM | **0.454 ± 0.008** | 0.497 ± 0.004 | 0.527 ± 0.006 | 0.528 ± 0.004 |
| C + rotation gate | 0.447 ± 0.004 | 0.491 ± 0.008 | 0.532 ± 0.009 | 0.537 ± 0.010 |
| D + CB focal loss | 0.425 ± 0.007 | 0.441 ± 0.020 | 0.455 ± 0.035 | 0.456 ± 0.036 |

**Accuracy on fields whose crop changed from 2022**

| Run | 1 Mar | 1 Jun | 1 Sep | 1 Dec |
| --- | --- | --- | --- | --- |
| A baseline | **0.266 ± 0.008** | 0.383 ± 0.025 | 0.505 ± 0.045 | 0.524 ± 0.040 |
| B dual-branch + FiLM | 0.263 ± 0.034 | 0.401 ± 0.016 | 0.529 ± 0.020 | 0.537 ± 0.023 |
| C + rotation gate | 0.257 ± 0.076 | **0.421 ± 0.060** | **0.555 ± 0.033** | **0.567 ± 0.034** |
| D + CB focal loss | 0.127 ± 0.106 | 0.205 ± 0.173 | 0.282 ± 0.239 | 0.286 ± 0.243 |

## 10. Check against the baseline paper (done 2 Nov 2026)

Run A was compared with the paper (Gené-Mola et al., *Computers and Electronics in Agriculture*, 2026,
[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0168169926007337)) and its official code
([github.com/UEA-Irta/CatCrops](https://github.com/UEA-Irta/CatCrops): `catcrops/models/TransformerModel.py`,
`catcrops/transforms.py`, `processing/train.py`).

**What already matched**
* The same 18 classes, with the same field counts: 127,480 fields in 2022 and 137,055 in 2023.
* The same inputs: 10 bands, cloud probability, day of year, and the same ancillary data (previous crop,
  irrigation system, municipality, elevation, slope).
* The same Transformer: linear 12→64 → ReLU, 5 encoder blocks with 2 heads and feed-forward width 128,
  max-pool → ReLU → linear.
* The same evaluation dates and random-cutoff training.

**What differed in the v1 run A**

| | Official code | v1 run A |
| --- | --- | --- |
| Ancillary input | each value appended as one extra time step (scalar-scaled) | appended as extra columns on every step, with embeddings |
| History kept | the 365 days before the cutoff | the whole previous year plus the current year up to the cutoff |
| Time encoding | day-of-year channel only | day channel plus sinusoidal positional encoding |
| Dropout | 0.018 | 0.1 |
| Optimiser | Adam, weight decay 5e-8, constant learning rate 1e-3 | AdamW, weight decay 1e-4, learning rate halved on plateaus |
| Batch / epochs | 512 / up to 120 | 256 / 30 |

Reported accuracy on 2023 for the paper's best model (all ancillary data): 0.81 / 0.86 / 0.88 / 0.88 at
1 Mar / 1 Jun / 1 Sep / 1 Dec. Our v1 A reached 0.787 / 0.815 / 0.847 / 0.853. Two caveats:
* The paper's 2023 set is its *validation* set, and its best epoch appears to be chosen on it, so its numbers are
  optimistic.
* Our runs choose the epoch on a 2022 validation split, so 2023 stays unseen.

**v2 rerun (`configs/v2`, the `run_local.py` command in section 8, outputs in `runs_v2/` and `results/v2/`)**

* **Run A** is a faithful re-implementation of the official baseline (`models/catcrops_baseline.py`). Two small
  differences remain:
  * the same 365-day window is given as the list of observed dates with a padding mask, instead of a
    zero-filled daily grid;
  * the previous crop is given as its class (the code's `--pclassid` option) rather than the raw crop code,
    which is the same information the dual-branch models get.
* **Every run (A–D)** uses the official protocol: Adam, learning rate 1e-3, weight decay 5e-8, batch 512, the
  365-day window, and up to 120 epochs. Early stopping (patience 10) uses the mean 2022-validation macro-F1.
* **Run D** uses a gentler class-balanced focal loss: β = 0.999 and focal exponent 1. With the v1 setting
  (β = 0.9999, exponent 2), the rarest class (soybean, 41 training fields) weighed about 240× the most common
  one; with β = 0.999 the ratio is about 25×.
* **The collapsed v1 run D seed 2** was examined by `diagnose_collapse.py` (results in `logs/diagnose_D.json`).
  It had not died numerically: no dead units, and all 18 classes were still predicted. Instead it had learned a
  shortcut and stopped using the satellite series:
  * **its prediction for a field was the same on all four dates** (100% identical; the healthy runs change
    1–8% of their predictions from one date to the next);
  * **the FiLM layer had scaled the satellite features down to about 0.67 of their size** (1.00 in run C), so
    the previous-crop term (size 8.1) outweighed the satellite term (size 5.5);
  * **in the healthy run C** the satellite term (9.7) outweighs the previous crop (7.1).

  The extreme class weights made "repeat last year's crop" the cheapest early solution, and nothing pushed the
  model out of it. The gentler v2 loss reduces that pressure. In the first v2 epochs, D's validation score
  varies with the date again, and it is already above v1 D at epoch 2 (0.58 vs 0.52 at epoch 6).

The v1 results above are kept for reference. The paper should report the v2 results.

## 11. Open items

* Done (3 Nov 2026): v2 results are in section 9, `paper/results_table.tex` and `paper/methodology.tex`.
* Optional: test the year-transfer fix for B–D (section 9).
* Per-class F1, confusion matrices and significance tests (evaluation stage).
