"""Model, dataset and loss tests (need torch). Run in Colab before training: pytest -q"""
from pathlib import Path

import numpy as np
import pytest
import torch

from config import load_config
from data.cache import YearCache, compute_stats, make_split
from data.dataset import CropDataset, collate
from losses import ClassBalancedFocalLoss, build_loss, class_balanced_weights
from models import build_model
from models.common import masked_max_pool

CFG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIGS = ["A_baseline", "B_fusion", "C_gate", "D_full", "D_full_prior_dropout", "D_full_cloud_weighted",
           "v2/A_baseline", "v2/B_fusion", "v2/C_gate", "v2/D_full"]


@pytest.fixture(scope="module")
def setup(fake_cache):
    cache = YearCache.load(fake_cache, 2022)
    tr, va = make_split(cache)
    stats = {agg: compute_stats(cache, tr, 18, agg) for agg in ("mean", "cloud_weighted")}
    return cache, tr, va, stats


def batch_for(setup, n=8, cutoff=152, multi_year=True, train=False):
    cache, tr, _, stats = setup
    ds = CropDataset(cache, tr[:n], stats["mean"], train=train, cutoff_doy=cutoff, multi_year=multi_year)
    return collate([ds[i] for i in range(len(ds))]), ds


# ---------------------------------------------------------------- dataset
def test_cutoff_never_leaks_future_observations(setup):
    for cutoff in (1, 60, 152, 244, 335):
        b, _ = batch_for(setup, cutoff=cutoff)
        cur = b["days"][b["mask"]]
        assert (cur < cutoff).all(), f"observation after cutoff doy {cutoff}"


def test_previous_year_kept_or_dropped(setup):
    b, _ = batch_for(setup, cutoff=60, multi_year=True)
    assert (b["days"][b["mask"]] < 0).any()
    b, _ = batch_for(setup, cutoff=60, multi_year=False)
    assert (b["days"][b["mask"]] >= 0).all()


def test_batch_shapes(setup):
    b, _ = batch_for(setup)
    B, T, C = b["x"].shape
    assert C == 12 and B == 8
    assert b["mask"].shape == (B, T) and b["days"].shape == (B, T)
    assert b["static_cont"].shape == (B, 2) and b["static_cat"].shape == (B, 2)
    assert b["static_cat"].dtype == torch.long
    assert b["mask"].any(dim=1).all(), "every sample needs at least one valid step"
    assert torch.isfinite(b["x"]).all()


def test_empty_series_gets_one_neutral_step(setup):
    b, _ = batch_for(setup, cutoff=1, multi_year=False)
    assert b["mask"].sum(dim=1).min() >= 1


def test_random_cutoff_in_training(setup):
    _, ds = batch_for(setup, n=40, train=True)
    cutoffs = {ds[i]["cutoff"] for i in range(40)}
    assert len(cutoffs) > 5 and min(cutoffs) >= 1 and max(cutoffs) <= 365


def test_prior_dropout_hides_previous_crop(setup):
    cache, tr, _, stats = setup
    ds = CropDataset(cache, tr, stats["mean"], train=True, prior_dropout=1.0)
    assert all(ds[i]["y_prev"] == -1 for i in range(20))


# ---------------------------------------------------------------- models
@pytest.mark.parametrize("name", CONFIGS)
def test_every_run_forward_backward(setup, name):
    cache, tr, _, stats = setup
    cfg = load_config(CFG_DIR / f"{name}.yaml")
    st = stats[cfg["data"]["aggregation"]]
    model = build_model(cfg["model"], st, 18)
    b, _ = batch_for(setup)
    out = model(b)
    assert out["logits"].shape == (8, 18)
    assert out["gate"].shape == (8,) and out["divergence"].shape == (8,)
    loss = build_loss(cfg["loss"], st["class_counts"])(out["logits"], b["y"])
    loss.backward()
    assert torch.isfinite(loss)
    unused = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not unused, f"parameters not used in the forward pass: {unused}"


def test_both_branches_receive_gradient(setup):
    cache, tr, _, stats = setup
    model = build_model(load_config(CFG_DIR / "D_full.yaml")["model"], stats["mean"], 18)
    b, _ = batch_for(setup)
    # FiLM starts as the identity (zero-initialised), so the static MLP receives gradient
    # only once FiLM's weights have moved: take one optimiser step before checking.
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    model(b)["logits"].sum().backward()
    opt.step()
    opt.zero_grad()
    model(b)["logits"].sum().backward()
    assert model.temporal.in_proj.weight.grad.abs().sum() > 0
    assert model.static.mlp[0].weight.grad.abs().sum() > 0
    assert model.static.mun_tok.weight.grad.abs().sum() > 0, "cross-attention must reach static tokens"
    assert model.gate.lin.weight.grad.abs().sum() > 0
    assert model.gate.beta_raw.grad is not None


def test_gate_zero_when_previous_crop_unknown(setup):
    cache, tr, _, stats = setup
    model = build_model(load_config(CFG_DIR / "C_gate.yaml")["model"], stats["mean"], 18).eval()
    b, _ = batch_for(setup)
    b["y_prev"][:4] = -1
    out = model(b)
    assert (out["gate"][:4] == 0).all()
    assert (out["gate"] >= 0).all() and (out["gate"] <= 1).all()


def test_gate_divergence_higher_for_wrong_previous_crop(setup):
    cache, tr, _, stats = setup
    model = build_model(load_config(CFG_DIR / "C_gate.yaml")["model"], stats["mean"], 18).eval()
    b, _ = batch_for(setup, n=32, cutoff=335)
    b["y_prev"] = b["y"].clone()                       # previous crop = what the field shows
    d_same, _, _ = model.gate.divergence(b["x"], b["days"], b["mask"], b["y_prev"])
    b["y_prev"] = (b["y"] + 9) % 18                     # previous crop = a different crop
    d_diff, _, _ = model.gate.divergence(b["x"], b["days"], b["mask"], b["y_prev"])
    assert d_diff.mean() > d_same.mean()


def test_masked_max_pool_ignores_padding():
    h = torch.randn(2, 5, 4)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    h2 = h.clone()
    h2[0, 2:] = 1e4
    assert torch.allclose(masked_max_pool(h, mask), masked_max_pool(h2, mask))


def test_padding_does_not_change_prediction(setup):
    cache, tr, _, stats = setup
    model = build_model(load_config(CFG_DIR / "D_full.yaml")["model"], stats["mean"], 18).eval()
    cache, tr, _, st = setup
    short = CropDataset(cache, tr[:2], st["mean"], train=False, cutoff_doy=152)
    long = CropDataset(cache, tr[:2], st["mean"], train=False, cutoff_doy=335)
    one = collate([short[0]])
    two = collate([short[0], long[1]])  # short[0] is padded in this batch
    assert two["x"].shape[1] > one["x"].shape[1]
    with torch.no_grad():
        a = model(one)["logits"][0]
        b = model(two)["logits"][0]
    assert torch.allclose(a, b, atol=1e-4)


def test_film_starts_as_identity(setup):
    cache, tr, _, stats = setup
    model = build_model(load_config(CFG_DIR / "B_fusion.yaml")["model"], stats["mean"], 18)
    h_t, h_s = torch.randn(3, 64), torch.randn(3, 64)
    assert torch.allclose(model.fusion(h_t, h_s), h_t)


# ---------------------------------------------------------------- loss
def test_cb_weights_favour_rare_classes():
    w = class_balanced_weights([50000, 1000, 46])
    assert w[2] > w[1] > w[0] and np.isclose(w.sum(), 3)


def test_focal_equals_ce_when_gamma0_and_uniform():
    logits, y = torch.randn(16, 5), torch.randint(0, 5, (16,))
    loss = ClassBalancedFocalLoss([10] * 5, beta=0.9999, gamma=0.0)(logits, y)
    assert torch.allclose(loss, torch.nn.functional.cross_entropy(logits, y), atol=1e-5)


# ---------------------------------------------------------------- v2 (faithful baseline protocol)
def test_window_keeps_only_365_days_before_cutoff(setup):
    cache, tr, _, stats = setup
    for cutoff in (1, 60, 200, 335):
        ds = CropDataset(cache, tr[:20], stats["mean"], train=False, cutoff_doy=cutoff, window_days=365)
        b = collate([ds[i] for i in range(len(ds))])
        d = b["days"][b["mask"]]
        assert (d < cutoff).all() and (d > cutoff - 366).all()
    # without the window, older previous-year observations are kept
    ds = CropDataset(cache, tr[:20], stats["mean"], train=False, cutoff_doy=335)
    b = collate([ds[i] for i in range(len(ds))])
    assert (b["days"][b["mask"]] <= 335 - 366).any()


def test_ancillary_values_scaled_like_catcrops(setup):
    cache, tr, _, stats = setup
    b, _ = batch_for(setup, n=40)
    anc = b["anc"]
    assert anc.shape == (40, 5)
    assert (anc >= -1).all() and (anc <= 1.5).all()
    known = b["y_prev"] >= 0
    assert torch.allclose(anc[known, 0], (b["y_prev"][known] + 1).float() / 18)
    assert (anc[~known, 0] == -1).all()


def test_catcrops_baseline_matches_official_structure(setup):
    cache, tr, _, stats = setup
    model = build_model(load_config(CFG_DIR / "v2" / "A_baseline.yaml")["model"], stats["mean"], 18)
    names = {n for n, _ in model.named_parameters()}
    assert "inlinear.weight" in names and "outlinear.weight" in names
    assert sum(p.numel() for p in model.parameters()) < 200_000
    enc = model.encoder
    assert len(enc.layers) == 5 and enc.norm is not None
    assert enc.layers[0].self_attn.num_heads == 2 and enc.layers[0].linear1.out_features == 128


def test_catcrops_baseline_uses_ancillary_and_ignores_padding(setup):
    cache, tr, _, stats = setup
    model = build_model(load_config(CFG_DIR / "v2" / "A_baseline.yaml")["model"], stats["mean"], 18).eval()
    b, _ = batch_for(setup, n=4)
    with torch.no_grad():
        a = model(b)["logits"]
        b2 = dict(b, anc=b["anc"].clone())
        b2["anc"][:, 0] = 0.9
        assert not torch.allclose(a, model(b2)["logits"]), "previous crop must change the prediction"
    short = CropDataset(cache, tr[:2], stats["mean"], train=False, cutoff_doy=152)
    long = CropDataset(cache, tr[:2], stats["mean"], train=False, cutoff_doy=335)
    with torch.no_grad():
        one = model(collate([short[0]]))["logits"][0]
        two = model(collate([short[0], long[1]]))["logits"][0]
    assert torch.allclose(one, two, atol=1e-4)
