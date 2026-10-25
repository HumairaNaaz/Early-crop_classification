# Enhanced Early Crop Type Classification via Dual-Branch Cross-Attention and Rotation-Aware Adaptive Gating

Minor research project. Goal: classify crop types from Sentinel-2 time series **earlier in the season** than the 2026 baseline
(Gené-Mola et al., *Computers and Electronics in Agriculture* 253, 112138, DOI 10.1016/j.compag.2026.112138).

> Status: all components are **hypotheses being investigated**. No improvement is claimed until verified results exist.

## Repo map
| Path | Owner | Purpose |
|---|---|---|
| `data/download_catcrops.py`, `src/dataset.py` | Ishu (Data) | Get CatCrops, load multi-year data, random day-of-year crops |
| `src/models/baseline_transformer.py`, `dual_branch_fusion.py` | Anshika (Model) | Baseline reproduction; branches + FiLM/cross-attention |
| `src/models/rotation_gate.py`, `src/losses.py` | Deepasha (Gate & Loss) | Divergence, gate, CB focal loss |
| `src/trainer.py`, `src/metrics.py`, `experiments/` | Divyanshi (Eval) | Training, metrics, ablations, plots |
| `paper/`, `docs/` | Humaira (Paper Lead) | IEEE LaTeX manuscript, notes, notation, decisions |
| `configs/default.yaml`, `tests/` | Shared | Seeds, paths, tensor-shape contract |

## Quick start
```bash
git clone <REPO_URL> && cd catcrops-early-crop
python -m venv .venv && source .venv/bin/activate     # Python 3.10+
pip install -r requirements.txt
pytest -q          # skips until modules are implemented
```
Read first: `docs/CONTRIBUTING.md`, `docs/INTERFACE_CONTRACT.md`, `docs/TEAM_AND_OWNERSHIP.md`.
