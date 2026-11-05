"""OWNER: Eval Lead (Divyanshi). Accuracy, macro/weighted F1, per-class F1 (+support), confusion
matrix, rotated-field accuracy, rare-class (<0.2% frequency) F1.

Design notes
------------
* Inputs may be Python lists, NumPy arrays or torch tensors (``validate_one_epoch`` passes lists).
* Everything returned is a plain Python type (float / int / list / None), so the result can be
  written straight to JSON with ``json.dump`` and no custom encoder.
* ``macro_f1`` averages over classes that actually occur in ``y_true`` (support > 0). A class that
  is absent from the ground truth therefore cannot drag the score down just because the model
  never had a chance to predict it. ``weighted_f1`` weights each class by its support.
* Rare classes are those whose frequency is below ``rare_threshold`` (default 0.002 = 0.2 %).
  Frequency is measured on ``y_true`` unless ``class_frequencies`` is supplied.
* Rotation: a field is "rotated" when ``rotation_mask`` is True (the dataset's ``rotation_flag``).
  ``y_prev`` holds *raw* previous-year crop codes, while ``y_true`` holds *mapped* class ids
  (0..n_classes-1), so the two live in different code spaces and a rotation mask is NOT derived
  from them here. ``y_prev`` is only length-checked.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

RARE_CLASS_THRESHOLD = 0.002  # 0.2 % of the evaluated fields


def _to_numpy(values, name: str) -> np.ndarray:
    """Convert a list / ndarray / torch tensor to a 1-D NumPy array."""
    if hasattr(values, "detach"):  # torch.Tensor, without importing torch here
        values = values.detach().cpu().numpy()
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}.")
    return array


def _safe_float(value) -> float:
    return float(value)


def compute_metrics(
    y_true,
    y_pred,
    y_prev=None,
    n_classes: int = 18,
    rotation_mask=None,
    rare_threshold: float = RARE_CLASS_THRESHOLD,
    class_frequencies: Optional[Sequence[float]] = None,
) -> dict:
    """Compute the evaluation metrics for one evaluation date.

    Parameters
    ----------
    y_true, y_pred : 1-D integer class ids in ``[0, n_classes)``.
    y_prev : optional previous-year crop codes; only its length is checked (see module notes).
    n_classes : number of classes (18 for CatCrops).
    rotation_mask : optional 1-D boolean flags, True for rotated fields.
    rare_threshold : frequency below which a class counts as rare.
    class_frequencies : optional per-class frequencies (length ``n_classes``) to decide which
        classes are rare, e.g. computed on the training set. Defaults to frequencies in ``y_true``.

    Returns
    -------
    dict with keys: ``n_samples``, ``accuracy``, ``macro_f1``, ``weighted_f1``,
    ``per_class_f1`` (list, n_classes), ``per_class_support`` (list, n_classes),
    ``confusion_matrix`` (n_classes x n_classes list, rows = true, columns = predicted),
    ``rotated_accuracy``, ``rotated_support``, ``non_rotated_accuracy``, ``rare_f1``,
    ``rare_classes``. ``rotated_accuracy`` / ``non_rotated_accuracy`` / ``rare_f1`` are None when
    there is nothing to average over.
    """
    if n_classes <= 0:
        raise ValueError("n_classes must be positive.")

    y_true = _to_numpy(y_true, "y_true")
    y_pred = _to_numpy(y_pred, "y_pred")

    if len(y_true) == 0:
        raise ValueError("y_true is empty; cannot compute metrics.")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length ({len(y_true)} != {len(y_pred)})."
        )
    if not (np.issubdtype(y_true.dtype, np.integer) and np.issubdtype(y_pred.dtype, np.integer)):
        raise ValueError("y_true and y_pred must contain integer class ids.")

    for name, arr in (("y_true", y_true), ("y_pred", y_pred)):
        if arr.min() < 0 or arr.max() >= n_classes:
            raise ValueError(f"{name} contains class ids outside [0, {n_classes - 1}].")

    if y_prev is not None:
        y_prev = _to_numpy(y_prev, "y_prev")
        if len(y_prev) != len(y_true):
            raise ValueError("y_prev must have the same length as y_true.")

    n_samples = int(len(y_true))
    labels = np.arange(n_classes)

    # Per-class precision / recall / F1 / support, with every class present in the output.
    _, _, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    support = support.astype(int)

    correct = y_true == y_pred
    accuracy = correct.mean()

    present = support > 0
    macro_f1 = f1[present].mean()  # present is never all-False because y_true is non-empty
    weighted_f1 = float(np.sum(f1 * support) / support.sum())

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # --- Rotated fields -------------------------------------------------------------------
    rotated_accuracy = None
    non_rotated_accuracy = None
    rotated_support = 0
    if rotation_mask is not None:
        rotation_mask = _to_numpy(rotation_mask, "rotation_mask")
        if len(rotation_mask) != n_samples:
            raise ValueError("rotation_mask must have the same length as y_true.")
        if rotation_mask.dtype != bool:
            if not np.isin(rotation_mask, (0, 1)).all():
                raise ValueError("rotation_mask must contain only boolean / 0-1 values.")
            rotation_mask = rotation_mask.astype(bool)

        rotated_support = int(rotation_mask.sum())
        if rotated_support > 0:
            rotated_accuracy = _safe_float(correct[rotation_mask].mean())
        if rotated_support < n_samples:
            non_rotated_accuracy = _safe_float(correct[~rotation_mask].mean())

    # --- Rare classes -------------------------------------------------------------------
    if class_frequencies is None:
        frequencies = support / n_samples
    else:
        frequencies = np.asarray(class_frequencies, dtype=float)
        if frequencies.shape != (n_classes,):
            raise ValueError(f"class_frequencies must have length {n_classes}.")

    # A class must occur in y_true (support > 0) for its F1 to be meaningful.
    rare_mask = (frequencies < rare_threshold) & present
    rare_classes = [int(c) for c in labels[rare_mask]]
    rare_f1 = _safe_float(f1[rare_mask].mean()) if rare_mask.any() else None

    return {
        "n_samples": n_samples,
        "accuracy": _safe_float(accuracy),
        "macro_f1": _safe_float(macro_f1),
        "weighted_f1": _safe_float(weighted_f1),
        "per_class_f1": [float(v) for v in f1],
        "per_class_support": [int(v) for v in support],
        "confusion_matrix": cm.tolist(),
        "rotated_accuracy": rotated_accuracy,
        "rotated_support": rotated_support,
        "non_rotated_accuracy": non_rotated_accuracy,
        "rare_f1": rare_f1,
        "rare_classes": rare_classes,
    }