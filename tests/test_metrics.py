"""Tests for src/metrics.py::compute_metrics."""
import json

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

from src.metrics import compute_metrics

N_CLASSES = 18

EXPECTED_KEYS = {
    "n_samples",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "per_class_f1",
    "per_class_support",
    "confusion_matrix",
    "rotated_accuracy",
    "rotated_support",
    "non_rotated_accuracy",
    "rare_f1",
    "rare_classes",
}


def _random_case(n=500, seed=0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, N_CLASSES, size=n)
    y_pred = np.where(rng.random(n) < 0.7, y_true, rng.integers(0, N_CLASSES, size=n))
    rotation = rng.random(n) < 0.3
    return y_true, y_pred, rotation


def test_output_keys_and_shapes():
    y_true, y_pred, rotation = _random_case()
    result = compute_metrics(y_true, y_pred, rotation_mask=rotation, n_classes=N_CLASSES)

    assert set(result) == EXPECTED_KEYS
    assert len(result["per_class_f1"]) == N_CLASSES
    assert len(result["per_class_support"]) == N_CLASSES
    assert np.asarray(result["confusion_matrix"]).shape == (N_CLASSES, N_CLASSES)
    assert sum(result["per_class_support"]) == result["n_samples"] == len(y_true)
    assert np.asarray(result["confusion_matrix"]).sum() == len(y_true)


def test_result_is_json_serialisable():
    y_true, y_pred, rotation = _random_case()
    result = compute_metrics(y_true, y_pred, rotation_mask=rotation)
    json.dumps(result)  # must not raise


def test_perfect_predictions():
    y_true = np.arange(N_CLASSES).repeat(3)
    result = compute_metrics(y_true, y_true.copy(), n_classes=N_CLASSES)

    assert result["accuracy"] == pytest.approx(1.0)
    assert result["macro_f1"] == pytest.approx(1.0)
    assert result["weighted_f1"] == pytest.approx(1.0)
    cm = np.asarray(result["confusion_matrix"])
    assert np.array_equal(cm, np.diag(np.full(N_CLASSES, 3)))


def test_matches_scikit_learn():
    y_true, y_pred, _ = _random_case(n=2000, seed=1)
    result = compute_metrics(y_true, y_pred, n_classes=N_CLASSES)

    assert result["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))
    assert result["macro_f1"] == pytest.approx(
        f1_score(y_true, y_pred, labels=np.unique(y_true), average="macro", zero_division=0)
    )
    assert result["weighted_f1"] == pytest.approx(
        f1_score(y_true, y_pred, labels=np.arange(N_CLASSES), average="weighted", zero_division=0)
    )


def test_macro_f1_ignores_classes_absent_from_ground_truth():
    # Only classes 0 and 1 occur in y_true; both are predicted perfectly.
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    result = compute_metrics(y_true, y_pred, n_classes=N_CLASSES)
    assert result["macro_f1"] == pytest.approx(1.0)
    assert result["per_class_support"][2:] == [0] * (N_CLASSES - 2)


def test_rotation_mask_accuracy():
    y_true = [0, 1, 2, 3, 4, 5]
    y_pred = [0, 1, 9, 9, 4, 5]
    rotation = [False, False, True, True, True, False]  # rotated fields: 2 wrong, 1 right

    result = compute_metrics(y_true, y_pred, rotation_mask=rotation, n_classes=N_CLASSES)

    assert result["rotated_support"] == 3
    assert result["rotated_accuracy"] == pytest.approx(1 / 3)
    assert result["non_rotated_accuracy"] == pytest.approx(1.0)


def test_rotation_mask_accepts_ints_and_arrays():
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 0, 3])
    as_bool = compute_metrics(y_true, y_pred, rotation_mask=[False, False, True, False])
    as_int = compute_metrics(y_true, y_pred, rotation_mask=np.array([0, 0, 1, 0]))
    assert as_bool["rotated_accuracy"] == as_int["rotated_accuracy"] == pytest.approx(0.0)


def test_rotation_mask_edge_cases():
    y_true = [0, 1, 2]
    y_pred = [0, 1, 2]

    none_rotated = compute_metrics(y_true, y_pred, rotation_mask=[False] * 3)
    assert none_rotated["rotated_accuracy"] is None
    assert none_rotated["rotated_support"] == 0
    assert none_rotated["non_rotated_accuracy"] == pytest.approx(1.0)

    all_rotated = compute_metrics(y_true, y_pred, rotation_mask=[True] * 3)
    assert all_rotated["rotated_accuracy"] == pytest.approx(1.0)
    assert all_rotated["non_rotated_accuracy"] is None

    no_mask = compute_metrics(y_true, y_pred)
    assert no_mask["rotated_accuracy"] is None
    assert no_mask["rotated_support"] == 0


def test_rotation_mask_validation():
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0, 1], rotation_mask=[True])  # wrong length
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0, 1], rotation_mask=[0, 2])  # not boolean-like


def test_rare_class_f1():
    # 2000 fields: class 0 = 1999 fields, class 1 = 1 field (0.05 % < 0.2 %, so rare).
    y_true = np.zeros(2000, dtype=int)
    y_true[0] = 1
    y_pred = y_true.copy()
    y_pred[0] = 0  # the single rare field is missed

    result = compute_metrics(y_true, y_pred, n_classes=N_CLASSES)

    assert result["rare_classes"] == [1]
    assert result["rare_f1"] == pytest.approx(0.0)


def test_rare_class_none_when_no_rare_classes():
    y_true = [0, 1, 0, 1]
    result = compute_metrics(y_true, y_true, n_classes=N_CLASSES)
    assert result["rare_classes"] == []
    assert result["rare_f1"] is None


def test_rare_class_uses_supplied_frequencies():
    y_true = [0, 1, 1, 1]
    y_pred = [0, 1, 1, 1]
    frequencies = np.full(N_CLASSES, 0.5)
    frequencies[0] = 0.001  # class 0 is rare in the *training* set
    result = compute_metrics(y_true, y_pred, n_classes=N_CLASSES, class_frequencies=frequencies)
    assert result["rare_classes"] == [0]
    assert result["rare_f1"] == pytest.approx(1.0)


def test_input_validation():
    with pytest.raises(ValueError):
        compute_metrics([], [], n_classes=N_CLASSES)
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0], n_classes=N_CLASSES)  # length mismatch
    with pytest.raises(ValueError):
        compute_metrics([0, N_CLASSES], [0, 1], n_classes=N_CLASSES)  # label out of range
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0, -1], n_classes=N_CLASSES)  # negative prediction
    with pytest.raises(ValueError):
        compute_metrics([0.5, 1.0], [0, 1], n_classes=N_CLASSES)  # not integer ids
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0, 1], y_prev=[3], n_classes=N_CLASSES)  # y_prev length


def test_accepts_the_arguments_validate_one_epoch_passes():
    """Mirrors the exact keyword call in src/trainer.py::validate_one_epoch (lists in)."""
    result = compute_metrics(
        y_true=[0, 1, 2, 3],
        y_pred=[0, 1, 2, 0],
        y_prev=[-1, 5, 7, 3],
        rotation_mask=[False, True, True, False],
        n_classes=18,
    )
    assert set(result) == EXPECTED_KEYS


def test_accepts_torch_tensors():
    torch = pytest.importorskip("torch")
    y_true = torch.tensor([0, 1, 2, 3])
    y_pred = torch.tensor([0, 1, 0, 3])
    rotation = torch.tensor([False, False, True, False])
    result = compute_metrics(y_true, y_pred, rotation_mask=rotation, n_classes=N_CLASSES)
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["rotated_accuracy"] == pytest.approx(0.0)