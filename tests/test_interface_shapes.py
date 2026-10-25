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


def test_gate_output_range():
    from src.models.rotation_gate import RotationGate
    g = _try(lambda: RotationGate())
    out = g(torch.randn(B, D), torch.randn(B, D), torch.rand(B))
    assert out.shape == (B, 1) and (out >= 0).all() and (out <= 1).all()
