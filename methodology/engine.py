"""Training/evaluation loops and metrics shared by train.py and predict.py."""
from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

TENSOR_KEYS = ("x", "days", "mask", "y", "y_prev", "static_cont", "static_cat", "anc")


def to_device(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if k in TENSOR_KEYS else v) for k, v in batch.items()}


def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device, grad_clip: float = 1.0,
                    log_every: int = 200) -> dict:
    model.train()
    use_amp = device.type == "cuda"
    total, n, correct = 0.0, 0, 0
    t0 = time.time()
    for step, batch in enumerate(loader):
        batch = to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model(batch)
        loss = loss_fn(out["logits"].float(), batch["y"])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        bs = batch["y"].shape[0]
        total += loss.item() * bs
        n += bs
        correct += (out["logits"].argmax(-1) == batch["y"]).sum().item()
        if log_every and (step + 1) % log_every == 0:
            print(f"    step {step + 1}/{len(loader)}  loss {total / n:.4f}  {time.time() - t0:.0f}s", flush=True)
    return {"train_loss": total / max(n, 1), "train_acc": correct / max(n, 1)}


@torch.no_grad()
def predict(model, loader, device) -> dict:
    model.eval()
    use_amp = device.type == "cuda"
    keys = ("y", "y_prev", "index", "pred", "gate", "divergence")
    acc = {k: [] for k in keys}
    probs = []
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model(batch)
        p = torch.softmax(out["logits"].float(), dim=-1)
        probs.append(p.cpu().numpy())
        acc["pred"].append(p.argmax(-1).cpu().numpy())
        acc["y"].append(batch["y"].cpu().numpy())
        acc["y_prev"].append(batch["y_prev"].cpu().numpy())
        acc["index"].append(batch["index"].cpu().numpy())
        acc["gate"].append(out["gate"].float().cpu().numpy())
        acc["divergence"].append(out["divergence"].float().cpu().numpy())
    res = {k: np.concatenate(v) for k, v in acc.items()}
    res["probs"] = np.concatenate(probs)
    return res


def metrics(y_true, y_pred) -> dict:
    """Accuracy, macro-F1 and weighted-F1 over the classes present in y_true or y_pred
    (sklearn's default), so subsets such as rotation groups are not penalised for
    classes that never occur in them."""
    if len(y_true) == 0:
        return {"n": 0}
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
