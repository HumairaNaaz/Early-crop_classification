"""OWNER: Eval Lead (Divyanshi). Accuracy, macro/weighted F1, per-class F1 (+support), confusion
matrix, rotated-field accuracy, rare-class (<0.2% frequency) F1.

Pure numpy on purpose: the trainer, CI (which only installs torch/numpy/pyyaml/pytest) and the
experiment scripts can all import it without scikit-learn.

Conventions (all agreed with the team's data stage, see configs/evaluation_subsets.json):
  * labels are the MAPPED class ids 0..n_classes-1 (configs/class_mapping.json), not raw ids;
  * a field is "rotated" when its previous-year crop class is known and differs from the
    current class (same definition as methodology/data/cache.py rotation_group == "changed");
  * rare classes are those whose training-set frequency is below 0.2 % (threshold 0.002).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RARE_THRESHOLD = 0.002
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def load_rare_class_ids(config_dir: str | Path | None = None) -> list[int]:
    """Rare class ids in the MAPPED label space (0..17).

    configs/evaluation_subsets.json lists raw ids (4, 5, 6, 16, 20); class_mapping.json
    maps them to 0..17 (16 -> 14, 20 -> 17), which is what the models predict."""
    d = Path(config_dir) if config_dir else _CONFIG_DIR
    raw = json.loads((d / "evaluation_subsets.json").read_text())["rare_class_ids"]
    mapping = json.loads((d / "class_mapping.json").read_text())
    return sorted(int(mapping[str(r)]) for r in raw)


def rare_classes_from_counts(class_counts, threshold: float = RARE_THRESHOLD) -> list[int]:
    """Classes whose share of the (training) fields is below `threshold`."""
    counts = np.asarray(class_counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return []
    return [int(c) for c in np.flatnonzero((counts / total) < threshold)]


def rotation_mask_from_prev(y_true, y_prev) -> np.ndarray:
    """True where the previous crop is known (>= 0) and differs from the current one."""
    y_true = np.asarray(y_true)
    y_prev = np.asarray(y_prev)
    return (y_prev >= 0) & (y_prev != y_true)


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    """(n_classes, n_classes) counts; rows = true class, columns = predicted class."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length")
    if y_true.size and (y_true.min() < 0 or y_true.max() >= n_classes
                        or y_pred.min() < 0 or y_pred.max() >= n_classes):
        raise ValueError(f"labels must lie in 0..{n_classes - 1}")
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def _per_class_prf(cm: np.ndarray):
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1, support, predicted


def _summary(y_true, y_pred, n_classes: int) -> dict:
    """accuracy / macro-F1 / weighted-F1 for one set of fields.

    Macro-F1 averages over the classes present in y_true OR y_pred. This is sklearn's default
    and exactly what methodology/engine.py reports, so numbers from both code paths agree and
    subsets (e.g. rotated fields) are not penalised for classes that never occur in them."""
    n = int(len(y_true))
    if n == 0:
        return {"n": 0, "accuracy": None, "macro_f1": None, "weighted_f1": None}
    cm = confusion_matrix(y_true, y_pred, n_classes)
    _, _, f1, support, predicted = _per_class_prf(cm)
    present = (support > 0) | (predicted > 0)
    return {
        "n": n,
        "accuracy": float(np.trace(cm) / n),
        "macro_f1": float(f1[present].mean()),
        "weighted_f1": float((f1 * support).sum() / support.sum()),
    }


def compute_metrics(y_true, y_pred, y_prev=None, rotation_mask=None, n_classes: int = 18,
                    rare_class_ids=None, class_counts=None,
                    rare_threshold: float = RARE_THRESHOLD) -> dict:
    """All evaluation metrics for one set of predictions.

    y_true, y_pred : (N,) class ids 0..n_classes-1 (lists, numpy arrays or torch tensors)
    y_prev         : (N,) previous-year class id, -1 = unknown. Only used to derive
                     rotation_mask when that is not given.
    rotation_mask  : (N,) bool, True = crop changed since last year. Wins over y_prev.
    rare_class_ids : explicit rare classes (mapped ids). Otherwise derived from class_counts
                     (share < rare_threshold); otherwise the team's list from
                     configs/evaluation_subsets.json when n_classes == 18, else none.

    Returns a JSON-serialisable dict with keys: n, accuracy, macro_f1, weighted_f1,
    per_class_f1 / per_class_precision / per_class_recall / per_class_support (lists of
    length n_classes), confusion_matrix (n_classes x n_classes nested list, rows = true),
    rotated_n, rotated_accuracy, rotated_macro_f1, nonrotated_n, nonrotated_accuracy,
    rare_class_ids, rare_class_f1, rare_class_support.

    Edge cases: an empty input gives zeros/None instead of raising; a subset with no fields
    (no rotated fields, no rare-class fields) gives None for its metrics, never NaN; a class
    absent from y_true and y_pred gets F1 0 and support 0 in the per-class lists but is
    left out of the macro averages."""
    y_true = _as_int_array(y_true, "y_true")
    y_pred = _as_int_array(y_pred, "y_pred")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length")
    if y_prev is not None:
        y_prev = _as_int_array(y_prev, "y_prev")
        if y_prev.shape != y_true.shape:
            raise ValueError("y_prev must have the same length as y_true")

    if rotation_mask is None and y_prev is not None:
        rotation_mask = rotation_mask_from_prev(y_true, y_prev)
    rot = None if rotation_mask is None else _as_bool_array(rotation_mask)
    if rot is not None and rot.shape != y_true.shape:
        raise ValueError("rotation_mask must have the same length as y_true")

    cm = confusion_matrix(y_true, y_pred, n_classes)
    precision, recall, f1, support, _ = _per_class_prf(cm)
    out = {**_summary(y_true, y_pred, n_classes),
           "per_class_f1": f1.tolist(), "per_class_precision": precision.tolist(),
           "per_class_recall": recall.tolist(), "per_class_support": support.astype(int).tolist(),
           "confusion_matrix": cm.tolist()}

    if rot is None:
        out.update(rotated_n=None, rotated_accuracy=None, rotated_macro_f1=None,
                   nonrotated_n=None, nonrotated_accuracy=None)
    else:
        r = _summary(y_true[rot], y_pred[rot], n_classes)
        nr = _summary(y_true[~rot], y_pred[~rot], n_classes)
        out.update(rotated_n=r["n"], rotated_accuracy=r["accuracy"], rotated_macro_f1=r["macro_f1"],
                   nonrotated_n=nr["n"], nonrotated_accuracy=nr["accuracy"])

    if rare_class_ids is None:
        if class_counts is not None:
            rare_class_ids = rare_classes_from_counts(class_counts, rare_threshold)
        elif n_classes == 18:
            rare_class_ids = load_rare_class_ids()
        else:
            rare_class_ids = []
    rare = sorted({int(c) for c in rare_class_ids if 0 <= int(c) < n_classes})
    rare_present = [c for c in rare if support[c] > 0]
    out["rare_class_ids"] = rare
    out["rare_class_f1"] = float(f1[rare_present].mean()) if rare_present else None
    out["rare_class_support"] = int(support[rare].sum()) if rare else 0
    return out


def _as_int_array(x, name: str = "labels") -> np.ndarray:
    """1-D int64 array from a list / numpy array / torch tensor. Rejects non-integer values
    (e.g. 0.5) instead of silently truncating them."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x).reshape(-1)
    if arr.size and arr.dtype.kind == "f":
        if not np.all(np.isfinite(arr)) or not np.all(arr == np.round(arr)):
            raise ValueError(f"{name} must be integer class ids")
    elif arr.size and arr.dtype.kind not in "iub":
        raise ValueError(f"{name} must be integer class ids")
    return arr.astype(np.int64)


def _as_bool_array(x, name: str = "rotation_mask") -> np.ndarray:
    """1-D bool array from bools or 0/1 values; anything else is an error."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x).reshape(-1)
    if arr.dtype.kind == "b":
        return arr
    if arr.size and (arr.dtype.kind not in "iuf" or not np.all(np.isin(arr, (0, 1)))):
        raise ValueError(f"{name} must contain booleans (or 0/1)")
    return arr.astype(bool)
