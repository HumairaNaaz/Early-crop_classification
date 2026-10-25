# How we use Git (simple rules)

1. **`main` is protected**: nobody pushes directly. Everything goes through a Pull Request (PR) with 1 reviewer.
2. **One branch per module**, named `<area>/<short-name>`:
   `data/dataset`, `model/baseline`, `model/fusion`, `gate/rotation-gate`, `loss/cb-focal`, `eval/trainer`, `eval/metrics`, `paper/intro`.
3. Workflow:
```bash
git checkout main && git pull
git checkout -b model/fusion
# ...work...
git add src/models/dual_branch_fusion.py tests/
git commit -m "fusion: add FiLM layer with shape test"
git push -u origin model/fusion        # then open a Pull Request on GitHub
```
4. Only edit files you own (see `docs/TEAM_AND_OWNERSHIP.md`). Need a change in someone else's file? Ask them or open an issue.
5. **Never commit**: raw data, `.h5/.parquet`, checkpoints, API keys. Share data via the team Drive with a checksum.
6. **Never change a tensor shape silently.** Update `docs/INTERFACE_CONTRACT.md` in the same PR and tell the group.
7. Every reported experiment must log: config file, seed, git commit hash. No hash, no number in the paper.
8. Commit messages: `area: what changed` (e.g. `loss: add class-balanced weights`).

## Setting up the repo (Humaira, once)
1. GitHub -> New repository (private), name e.g. `catcrops-early-crop`. Do not add a README (we have one).
2. In this folder: `git init -b main && git add . && git commit -m "init: project scaffold" && git remote add origin <URL> && git push -u origin main`
3. Settings -> Collaborators: add all teammates (Write access).
4. Settings -> Branches -> add rule for `main`: require a pull request before merging (1 approval).
5. Issues -> create one issue per checklist item in `docs/TEAMMATE_CHECKLIST.md`; assign owners.
