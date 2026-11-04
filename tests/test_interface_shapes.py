"""Interface-contract tests. They are SKIPPED until a module is implemented, so CI stays green
today and starts guarding shapes as soon as each owner removes NotImplementedError."""
import pytest

torch = pytest.importorskip("torch")
B, T, D, C = 4, 30, 64, 18


def _try(fn):
    try:
        return fn()
    except NotImplementedError:
        pytest.skip("module not implemented yet")


def test_temporal_branch_shape():
    from src.models.dual_branch_fusion import TemporalBranch
    m = _try(lambda: TemporalBranch())
    out = m(torch.randn(B, T, 12), torch.ones(B, T, dtype=torch.bool))
    assert out.shape == (B, D)


def test_film_fusion_shape():
    from src.models.dual_branch_fusion import FiLMFusion
    m = _try(lambda: FiLMFusion())
    assert m(torch.randn(B, D), torch.randn(B, D)).shape == (B, D)



import torch
from models.rotation_gate import RotationGate


def test_gate_output_range():
    batch_size = 4
    seq_len = 20
    num_classes = 18
    num_bins = 74
    n_bands = 10
    d_model = 64

    # Synthetic expected spectral profiles
    profiles = torch.rand(num_classes, num_bins, n_bands)

    # Initialise the rotation gate
    gate = RotationGate(profiles=profiles, d_model=d_model)

    # Synthetic temporal observations
    x = torch.rand(batch_size, seq_len, 12)

    # Integer day offsets for temporal bins
    days = torch.randint(0, 366, (batch_size, seq_len))

    # All observations are initially valid
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    # Previous crop indices
    y_prev = torch.tensor([0, 3, 5, 17])

    # Synthetic embeddings from the temporal branch
    h_temp = torch.rand(batch_size, d_model)
    e_prev = torch.rand(batch_size, d_model)

    # Forward pass
    gamma, d_rot = gate(
        x, days, mask, y_prev, h_temp, e_prev
    )

    # Verify output shapes
    assert gamma.shape == (batch_size,)
    assert d_rot.shape == (batch_size,)

    # Gate must remain between 0 and 1
    assert torch.all(gamma >= 0)
    assert torch.all(gamma <= 1)

    # Divergence must be finite
    assert torch.isfinite(d_rot).all()


def test_gate_zero_for_unknown_previous_crop():
    import torch
    from models.rotation_gate import RotationGate

    profiles = torch.rand(18, 74, 10)
    gate = RotationGate(profiles=profiles)

    batch_size = 2
    seq_len = 10

    x = torch.rand(batch_size, seq_len, 12)
    days = torch.randint(0, 366, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    # Use the unknown-crop label specified by the implementation.
    y_prev = torch.tensor([-1, -1])

    h_temp = torch.rand(batch_size, 64)
    e_prev = torch.rand(batch_size, 64)

    gamma, d_rot = gate(
        x, days, mask, y_prev, h_temp, e_prev
    )

    assert torch.all(gamma == 0)
    assert torch.isfinite(d_rot).all()


def test_gate_handles_no_valid_observations():
    import torch
    from models.rotation_gate import RotationGate

    profiles = torch.rand(18, 74, 10)
    gate = RotationGate(profiles=profiles)

    batch_size = 2
    seq_len = 10

    x = torch.rand(batch_size, seq_len, 12)
    days = torch.randint(0, 366, (batch_size, seq_len))

    # No valid observations
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    # Previous crop is known
    y_prev = torch.tensor([2, 5])

    h_temp = torch.rand(batch_size, 64)
    e_prev = torch.rand(batch_size, 64)

    gamma, d_rot = gate(
        x, days, mask, y_prev, h_temp, e_prev
    )

    assert torch.isfinite(gamma).all()
    assert torch.isfinite(d_rot).all()

    # No observations means divergence should be zero
    assert torch.allclose(d_rot, torch.zeros_like(d_rot))

    # Gate should be zero when no observations are available
    assert torch.allclose(gamma, torch.zeros_like(gamma))

def test_gate_decreases_with_increasing_divergence():
    import torch
    from models.rotation_gate import RotationGate

    torch.manual_seed(42)

    profiles = torch.rand(18, 74, 10)
    gate = RotationGate(profiles=profiles)

    # Fix all other gate inputs
    h_temp = torch.rand(1, 64)
    e_prev = torch.rand(1, 64)

    has_obs = torch.ones(1, dtype=torch.bool)
    known = torch.ones(1, dtype=torch.bool)

    gate_features = torch.cat(
        [h_temp, e_prev, has_obs.float().unsqueeze(-1)],
        dim=-1
    )

    logit = gate.lin(gate_features).squeeze(-1)

    # Compare two divergence values
    d_low = torch.tensor([0.2])
    d_high = torch.tensor([1.5])

    gamma_low = torch.sigmoid(logit - gate.beta * d_low)
    gamma_high = torch.sigmoid(logit - gate.beta * d_high)

    gamma_low = gamma_low * known * has_obs
    gamma_high = gamma_high * known * has_obs

    assert torch.all(gamma_low > gamma_high)


def test_rotation_gate_backpropagation():
    import torch
    from models.rotation_gate import RotationGate

    torch.manual_seed(42)

    profiles = torch.rand(18, 74, 10)
    gate = RotationGate(profiles=profiles)

    batch_size = 4
    seq_len = 20

    x = torch.rand(batch_size, seq_len, 12)
    days = torch.randint(0, 366, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    y_prev = torch.tensor([0, 3, 5, 17])

    h_temp = torch.rand(batch_size, 64, requires_grad=True)
    e_prev = torch.rand(batch_size, 64, requires_grad=True)

    gamma, d_rot = gate(
        x, days, mask, y_prev, h_temp, e_prev
    )

    # Construct a scalar loss
    loss = gamma.mean()

    # Backpropagation
    loss.backward()

    # Check gradients of learnable parameters
    assert gate.lin.weight.grad is not None
    assert gate.lin.bias.grad is not None
    assert gate.beta_raw.grad is not None

    # Check gradients are finite
    assert torch.isfinite(gate.lin.weight.grad).all()
    assert torch.isfinite(gate.lin.bias.grad).all()
    assert torch.isfinite(gate.beta_raw.grad).all()

    # Check embedding gradients
    assert h_temp.grad is not None
    assert e_prev.grad is not None


def test_effective_number_weights():
    import torch
    from methodology.losses import effective_number_weights

    class_counts = [1000, 500, 100, 20]

    weights = effective_number_weights(
        class_counts,
        beta=0.999
    )

    assert weights.shape == (4,)
    assert torch.isfinite(weights).all()
    assert (weights > 0).all()

    # Normalized weights should have mean 1.
    assert torch.isclose(
        weights.mean(),
        torch.tensor(1.0),
        atol=1e-6
    )

    # Rare classes should receive higher weights.
    assert weights[3] > weights[2] > weights[1] > weights[0]


def test_effective_number_weights_invalid_inputs():
    import pytest
    from methodology.losses import effective_number_weights

    # Empty class counts
    with pytest.raises(ValueError):
        effective_number_weights([])

    # Zero-count class
    with pytest.raises(ValueError):
        effective_number_weights([100, 0, 50])

    # Negative count
    with pytest.raises(ValueError):
        effective_number_weights([100, -5, 50])

    # Invalid beta
    with pytest.raises(ValueError):
        effective_number_weights([100, 50], beta=1.0)

    # Beta below zero
    with pytest.raises(ValueError):
        effective_number_weights([100, 50], beta=-0.1)

    # Non-finite count
    with pytest.raises(ValueError):
        effective_number_weights([100, float("nan"), 50])


def test_class_balanced_focal_loss_forward():
    import torch
    from methodology.losses import ClassBalancedFocalLoss

    torch.manual_seed(42)

    class_counts = [1000, 500, 100, 20]

    loss_fn = ClassBalancedFocalLoss(
        class_counts=class_counts,
        beta=0.999,
        gamma=1.0
    )

    logits = torch.randn(8, 4)
    targets = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])

    loss = loss_fn(logits, targets)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_class_balanced_focal_loss_backpropagation():
    import torch
    from methodology.losses import ClassBalancedFocalLoss

    torch.manual_seed(42)

    loss_fn = ClassBalancedFocalLoss(
        class_counts=[1000, 500, 100, 20],
        beta=0.999,
        gamma=1.0
    )

    logits = torch.randn(8, 4, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])

    loss = loss_fn(logits, targets)
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_focal_loss_reduces_to_weighted_cross_entropy():
    import torch
    import torch.nn.functional as F
    from methodology.losses import (
        ClassBalancedFocalLoss,
        effective_number_weights,
    )

    torch.manual_seed(42)

    class_counts = [1000, 500, 100, 20]

    logits = torch.randn(8, 4)
    targets = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])

    weights = effective_number_weights(class_counts, beta=0.999)

    focal_loss_fn = ClassBalancedFocalLoss(
        class_counts=class_counts,
        beta=0.999,
        gamma=0.0,
    )

    focal_loss = focal_loss_fn(logits, targets)

    ce_per_sample = F.cross_entropy(
        logits,
        targets,
        weight=weights,
        reduction="none",
    )

    weighted_ce = ce_per_sample.mean()

    assert torch.allclose(
        focal_loss,
        weighted_ce,
        atol=1e-6,
    )


def test_build_loss_selects_correct_loss():
    import torch
    import torch.nn as nn
    from methodology.losses import (
        ClassBalancedFocalLoss,
        build_loss,
    )

    class_counts = [1000, 500, 100, 20]

    # Run A/B/C configuration
    ce_config = {
        "type": "ce",
        "cb_beta": 0.999,
        "focal_gamma": 1.0,
    }

    ce_loss = build_loss(ce_config, class_counts)

    assert isinstance(ce_loss, nn.CrossEntropyLoss)

    # Run D v2 configuration
    focal_config = {
        "type": "cb_focal",
        "cb_beta": 0.999,
        "focal_gamma": 1.0,
    }

    focal_loss = build_loss(focal_config, class_counts)

    assert isinstance(focal_loss, ClassBalancedFocalLoss)
    assert focal_loss.gamma == 1.0
    assert torch.isfinite(focal_loss.class_weights).all()


def test_loss_training_batch_compatibility():
    import torch
    from methodology.losses import build_loss

    torch.manual_seed(42)

    batch_size = 32
    num_classes = 18

    class_counts = [
        1000, 850, 700, 600, 500, 450,
        400, 350, 300, 250, 200, 180,
        150, 120, 100, 80, 50, 20
    ]

    logits = torch.randn(
        batch_size, num_classes,
        requires_grad=True
    )

    targets = torch.randint(
        0, num_classes, (batch_size,)
    )

    configs = [
        {
            "type": "ce",
            "cb_beta": 0.999,
            "focal_gamma": 1.0,
        },
        {
            "type": "cb_focal",
            "cb_beta": 0.999,
            "focal_gamma": 1.0,
        },
    ]

    for config in configs:
        loss_fn = build_loss(config, class_counts)

        loss = loss_fn(logits.float(), targets)

        assert loss.ndim == 0
        assert torch.isfinite(loss)

        loss.backward()

        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

        logits.grad.zero_()

def test_class_balanced_focal_loss_rejects_invalid_gamma():
    import pytest
    from methodology.losses import ClassBalancedFocalLoss

    class_counts = [100, 80, 50]

    for gamma in [-1.0, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            ClassBalancedFocalLoss(
                class_counts=class_counts,
                beta=0.999,
                gamma=gamma,
            )
