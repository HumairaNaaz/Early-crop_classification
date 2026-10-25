# Team, code ownership and paper sections
> Source: team plan (5 members). Your original brief said 4 members - the plan lists **Ishu (Data Lead)** as a 5th. **Confirm.**

| Member | Role | Code | Lit-review theme | Paper sections |
|---|---|---|---|---|
| Ishu | Data Lead | data/*, src/dataset.py | Early-season classification & benchmarks | Experimental Setup |
| Anshika | Baseline & Fusion | baseline_transformer.py, dual_branch_fusion.py | Multimodal fusion | Problem Formulation (joint), Methodology: architecture |
| Deepasha | Rotation-Gate & Loss | rotation_gate.py, losses.py | Crop rotation & gating | Problem Formulation (joint), Methodology: gate & loss |
| Divyanshi | Training & Eval | trainer.py, metrics.py, experiments/* | Class imbalance & rare classes | Results, Discussion (joint) |
| Humaira (Paper Lead) | Paper + Coordination | paper/, docs/ | Crop classification from satellite time series | Introduction, Discussion (joint), Conclusion, Abstract, notation |

Writing order: Lit review + Problem Formulation (now) -> Methodology -> Setup/Results -> Discussion -> Conclusion -> Introduction -> Abstract.
