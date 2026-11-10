# Enhanced Early Crop Type Classification via Dual-Branch Cross-Attention and Rotation-Aware Adaptive Gating

A research project focused on classifying crop types from CatCrops data earlier in the growing season keeping the 2026  (Gené-Mola et al., *Computers and Electronics in Agriculture*, 253, 112138, DOI: 10.1016/j.compag.2026.112138) as the baseline.

---

## Project Overview

The project investigates an early crop classification framework using multi-temporal Sentinel-2 observations.

The proposed architecture explores:

- Temporal feature extraction from satellite time-series observations.
- Static and temporal feature branches.
- Dual-branch feature fusion using FiLM and cross-attention.
- Rotation-aware adaptive gating.
- Class-balanced focal loss to address class imbalance.
- Early-season evaluation at predefined observation cutoffs.

The objective is to investigate whether these components can improve early crop-type classification performance compared with the reference baseline.

## Repository Structure

```text
Early-crop_classification/
│
├── configs/
│   └── default.yaml
│
├── data/
│   ├── download_catcrops.py
│   ├── raw/
│   └── processed/
│
├── methodology/
│   ├── models/
│   │   ├── baseline_transformer.py
│   │   ├── dual_branch_fusion.py
│   │   └── rotation_gate.py
│   │
│   ├── losses.py
│   └── train.py
│
├── src/
│   ├── dataset.py
│   ├── preprocess.py
│   ├── trainer.py
│   └── metrics.py
│
├── experiments/
│
├── tests/
│
├── docs/
│   ├── CONTRIBUTING.md
│   └── INTERFACE_CONTRACT.md
│
├── paper/
│
├── requirements.txt
├── pytest.ini
└── README.md
```

*Note: This is the logical project layout. Folders such as `data/raw/`, `data/processed/`, `experiments/`, and `paper/` may contain additional files as development progresses.*

---

## Repository Map and Ownership

Each team member is responsible for specific modules. Shared files should be coordinated before making changes.

| Path | Owner | Responsibility |
|---|---|---|
| `data/download_catcrops.py`, `src/dataset.py` | Ishu (Data) | CatCrops data acquisition, multi-year data loading, dataset preparation, random day-of-year cropping |
| `methodology/models/baseline_transformer.py`, `methodology/models/dual_branch_fusion.py` | Anshika (Model) | Baseline reproduction, temporal and static branches, FiLM and cross-attention fusion |
| `methodology/models/rotation_gate.py`, `methodology/losses.py` | Deepasha (Gate & Loss) | Rotation-aware divergence, adaptive gating, class-balanced focal loss |
| `src/trainer.py` | Deepasha (Initial Trainer Foundation), Divyanshi (Continuation) | Training configuration, reproducibility, DataLoader utilities, training and validation functions, early stopping, checkpointing and experiment initialization |
| `src/metrics.py`, `experiments/` | Divyanshi (Evaluation) | Evaluation metrics, ablation studies, experiment tracking and plots |
| `paper/`, `docs/` | Humaira (Paper Lead) | IEEE LaTeX manuscript, research notes, notation and project documentation |
| `configs/default.yaml`, `tests/` | Shared | Configuration, reproducibility settings, interface contracts and tests |

### Contribution Notes

- **Ishu (Data):** Responsible for CatCrops dataset acquisition and data preparation. Handles multi-year satellite data loading, dataset organization, preprocessing and the development of random day-of-year cropping to support early-season evaluation.

- **Anshika (Model):** Responsible for baseline transformer development and the dual-branch architecture. Works on temporal and static feature extraction, branch integration, and fusion mechanisms involving FiLM and cross-attention.

- **Deepasha (Gate & Loss):** Implemented the rotation-aware adaptive gating mechanism and class-balanced focal loss. Also established the foundational training utilities in `src/trainer.py`, including configuration validation, reproducibility, DataLoader utilities, training and validation functions, early stopping, checkpoint management and experiment initialization.

- **Divyanshi (Evaluation):** Responsible for continuing the trainer integration and developing the evaluation pipeline. Handles evaluation metrics, performance analysis, ablation studies, experiment tracking and visualization.

- **Humaira (Paper Lead):** Responsible for coordinating the research documentation and manuscript. Handles the IEEE LaTeX paper, literature references, methodology explanations, mathematical notation, research decisions and consolidation of experimental findings into the manuscript.

Contributions are collaborative, and module ownership indicates primary responsibility rather than exclusive involvement. Changes affecting shared interfaces or another member's module should be coordinated with the respective owner.

---

## Proposed Methodology

### 1. Temporal Branch

Processes multi-temporal Sentinel-2 observations to learn crop development patterns from satellite time-series data.

### 2. Static Branch

Processes static or categorical field-level information alongside temporal observations.

### 3. Dual-Branch Fusion

Investigates combining temporal and static representations through feature fusion approaches, including FiLM and cross-attention.

### 4. Rotation-Aware Adaptive Gating

Investigates an adaptive gating mechanism that accounts for crop rotation-related information and divergence between current and previous crop representations.

Implementation:
- `methodology/models/rotation_gate.py`

### 5. Class-Balanced Focal Loss

Investigates class-balanced focal loss to address class-frequency imbalance during model optimization.

Implementation:
- `methodology/losses.py`

---

## Training and Evaluation

### Trainer Foundation

The initial trainer utilities are implemented in:

`src/trainer.py`

The current implementation provides:

- Configuration loading and validation.
- Python, NumPy and PyTorch random seed initialization.
- Automatic CPU/CUDA device selection.
- DataLoader creation.
- Batch-to-device transfer.
- Training and validation epoch functions.
- Training history initialization.
- Early stopping utility.
- Model and optimizer checkpoint saving and loading.
- Experiment run-directory creation.
- Git commit hash retrieval for experiment traceability.

The current `train()` function initializes the experiment configuration and output directory. It is not yet a complete end-to-end training pipeline.

### Pending Integration

The following work remains:

- Dataset and train/validation split integration.
- Integration with the completed model implementations.
- Integration with the evaluation metrics module.
- Full epoch orchestration and validation-based checkpoint selection.
- Random day-of-year cropping.
- W&B or TensorBoard experiment tracking.
- Complete training and evaluation runs on the project dataset.

### Evaluation Metrics

The evaluation module is maintained in:

`src/metrics.py`

Planned evaluation includes:

- Accuracy.
- Macro F1-score.
- Weighted F1-score.
- Per-class F1-score and support.
- Confusion matrix.
- Rotated-field accuracy.
- Rare-class F1-score.

The precise metric implementation and integration are maintained by the evaluation owner.

### Early-Season Evaluation

The project uses predefined evaluation cutoffs to investigate classification performance at different stages of the growing season.

The current proposed development configuration includes:

```yaml
train:
  epochs: 30
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
  patience: 6
  num_workers: 0
  eval_cutoffs: [60, 152, 244, 335]
```

These are initial development defaults, not experimentally validated hyperparameters.

---

## Configuration

Project-level configuration is maintained in:

`configs/default.yaml`

The configuration includes:

- Random seed.
- Dataset and output paths.
- Training and testing year protocol.
- Evaluation dates.
- Model interface dimensions.
- Number of classes and model layers.
- Fusion and gating settings.
- Loss parameters.
- Training hyperparameters and evaluation cutoffs.

Changes to shared configuration should be coordinated with the team.

---

## Interface Contract

The draft model and tensor interface is documented in:

`docs/INTERFACE_CONTRACT.md`

The contract describes the expected shapes and types for:

- Temporal observations and masks.
- Static continuous and categorical features.
- Previous crop information.
- Temporal and static representations.
- Adaptive gate values.
- Fused representations.
- Classification logits.

The contract is still under development. Any changes to tensor dimensions, output structures or model interfaces should be communicated to the relevant team members.

---

## Testing

Tests are maintained in:

`tests/`

Run the test suite from the repository root:

```bash
pytest -q
```

The tests cover selected model interface and component behaviours.

Trainer utilities have also been manually checked using synthetic data, including training and validation steps, early stopping and checkpoint save/load operations.

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/HumairaNaaz/Early-crop_classification.git
cd Early-crop_classification
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Tests

```bash
pytest -q
```

### 5. Configure the Dataset

Follow the data preparation and access instructions in `data/download_catcrops.py` and coordinate dataset paths with the data owner.

The complete training workflow requires the dataset and the model and metrics integrations to be available.

---

## Development Guidelines

- Work on a separate branch for each module or contribution.
- Do not directly push changes to `main`.
- Submit changes through pull requests.
- Obtain at least one reviewer approval before merging.
- Avoid modifying another team member's owned modules without coordination.
- Do not commit raw datasets, model checkpoints, credentials or secrets.
- Communicate changes to tensor shapes and function interfaces.
- Record experiment configurations, random seeds and Git commit hashes.
- Keep experimental findings separate from hypotheses until validated.

Refer to `docs/CONTRIBUTING.md` for the contribution workflow.

---

## Research Integrity

All proposed architectural components and training strategies are being investigated.

The presence of an implementation does not establish its effectiveness. Performance comparisons, ablation findings and claims of improvement should only be reported after reproducible experiments and appropriate evaluation.

---

## Acknowledgements

This project is a collaborative minor research project involving contributions across data preparation, model development, adaptive gating, loss design, training, evaluation and research documentation.
