"""Tests for src/metrics.py (numpy only, so they run in CI without sklearn or torch)."""
import json

import numpy as np
import pytest

from src.metrics import (compute_metrics, confusion_matrix, load_rare_class_ids,
                         rare_classes_from_counts, rotation_mask_from_prev)

N = 18


def _data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, N, n)
    y_pred = np.where(rng.random(n) < 0.7, y_true, rng.integers(0, N, n))
    y_prev = np.where(rng.random(n) < 0.05, -1, np.where(rng.random(n) < 0.7, y_true, rng.integers(0, N, n)))
    return y_true, y_pred, y_prev


def test_keys_and_shapes():
    y, p, prev = _data()
    m = compute_metrics(y, p, y_prev=prev, n_classes=N)
    for k in ("n", "accuracy", "macro_f1", "weighted_f1", "rotated_accuracy", "rare_class_f1"):
        assert k in m
    assert len(m["per_class_f1"]) == len(m["per_class_support"]) == N
    cm = np.array(m["confusion_matrix"])
    assert cm.shape == (N, N) and cm.sum() == len(y)
    assert sum(m["per_class_support"]) == len(y)
    json.dumps(m)


def test_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    y, p, _ = _data()
    m = compute_metrics(y, p, n_classes=N)
    assert m["accuracy"] == pytest.approx(sk.accuracy_score(y, p))
    assert m["macro_f1"] == pytest.approx(sk.f1_score(y, p, average="macro", zero_division=0))
    assert m["weighted_f1"] == pytest.approx(sk.f1_score(y, p, average="weighted", zero_division=0))
    assert m["per_class_f1"] == pytest.approx(
        sk.f1_score(y, p, labels=list(range(N)), average=None, zero_division=0))
    assert np.array(m["confusion_matrix"]).tolist() == sk.confusion_matrix(y, p, labels=list(range(N))).tolist()


def test_rotation_mask_subset():
    y = np.array([0, 1, 2, 3])
    p = np.array([0, 1, 0, 0])
    mask = np.array([False, False, True, True])
    m = compute_metrics(y, p, rotation_mask=mask, n_classes=4)
    assert m["rotated_n"] == 2 and m["rotated_accuracy"] == 0.0
    assert m["nonrotated_n"] == 2 and m["nonrotated_accuracy"] == 1.0


def test_rotation_mask_from_prev_and_unknown():
    y = np.array([0, 1, 2, 3])
    prev = np.array([0, 2, -1, 1])
    assert rotation_mask_from_prev(y, prev).tolist() == [False, True, False, True]


def test_no_rotated_fields():
    y, p, _ = _data(100)
    m = compute_metrics(y, p, rotation_mask=np.zeros(100, dtype=bool), n_classes=N)
    assert m["rotated_n"] == 0 and m["rotated_accuracy"] is None and m["rotated_macro_f1"] is None
    assert m["nonrotated_accuracy"] == pytest.approx(m["accuracy"])


def test_missing_classes_and_empty_input():
    m = compute_metrics([0, 0, 1], [0, 1, 1], n_classes=N)
    assert m["per_class_support"][5] == 0 and m["per_class_f1"][5] == 0.0
    assert np.isfinite(m["macro_f1"]) and m["macro_f1"] == pytest.approx((2 / 3 + 2 / 3) / 2)
    e = compute_metrics([], [], n_classes=N)
    assert e["n"] == 0 and e["accuracy"] is None and e["rare_class_f1"] is None


def test_rare_class_f1():
    m = compute_metrics([0, 0, 1, 1, 2], [0, 0, 1, 0, 2], n_classes=3, rare_class_ids=[1, 2])
    assert m["rare_class_f1"] == pytest.approx(5 / 6) and m["rare_class_support"] == 3
    m = compute_metrics([0, 0], [0, 0], n_classes=3, rare_class_ids=[1])
    assert m["rare_class_f1"] is None


def test_rare_classes_from_counts_and_config():
    counts = [1000] * 17 + [1]
    assert rare_classes_from_counts(counts) == [17]
    ids = load_rare_class_ids()
    assert ids == [4, 5, 6, 14, 17]


def test_validation_errors():
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0], n_classes=N)
    with pytest.raises(ValueError):
        confusion_matrix([0, 99], [0, 1], N)
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0, 1], rotation_mask=[True], n_classes=N)


def test_rejects_non_integer_and_malformed_inputs():
    with pytest.raises(ValueError):
        compute_metrics([0.5, 1.0], [0, 1], n_classes=N)
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0, 1], y_prev=[3], n_classes=N)
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0, 1], rotation_mask=[0, 2])
    with pytest.raises(ValueError):
        compute_metrics([0, N], [0, 1], n_classes=N)
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0, -1], n_classes=N)


def test_rotation_mask_accepts_ints_and_all_or_nothing_rotated():
    y, p = [0, 1, 2, 3], [0, 1, 0, 3]
    as_bool = compute_metrics(y, p, rotation_mask=[False, False, True, False], n_classes=N)
    as_int = compute_metrics(y, p, rotation_mask=np.array([0, 0, 1, 0]), n_classes=N)
    assert as_bool["rotated_accuracy"] == as_int["rotated_accuracy"] == 0.0
    all_rot = compute_metrics(y, y, rotation_mask=[True] * 4, n_classes=N)
    assert all_rot["rotated_accuracy"] == 1.0 and all_rot["nonrotated_accuracy"] is None
    no_mask = compute_metrics(y, y, n_classes=N)
    assert no_mask["rotated_accuracy"] is None and no_mask["rotated_n"] is None


def test_perfect_predictions():
    y = np.arange(N).repeat(3)
    m = compute_metrics(y, y.copy(), n_classes=N)
    assert m["accuracy"] == m["macro_f1"] == m["weighted_f1"] == pytest.approx(1.0)
    assert np.array_equal(np.array(m["confusion_matrix"]), np.diag(np.full(N, 3)))


def test_accepts_the_call_validate_one_epoch_makes():
    """Same keyword call as src/trainer.py::validate_one_epoch (plain lists)."""
    m = compute_metrics(y_true=[0, 1, 2, 3], y_pred=[0, 1, 2, 0], y_prev=[-1, 5, 7, 3],
                        rotation_mask=[False, True, True, False], n_classes=18)
    assert m["accuracy"] == 0.75 and m["rotated_n"] == 2 and m["rotated_accuracy"] == 1.0


def test_accepts_torch_tensors():
    torch = pytest.importorskip("torch")
    m = compute_metrics(torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 0, 3]),
                        rotation_mask=torch.tensor([False, False, True, False]), n_classes=N)
    assert m["accuracy"] == pytest.approx(0.75) and m["rotated_accuracy"] == pytest.approx(0.0)
