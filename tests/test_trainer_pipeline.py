"""Synthetic smoke tests for src/trainer.py: fit loop, early stopping, checkpoints, batch keys.

Skipped when torch is missing. No real data and no other teammate's model is needed."""
import json

import pytest
import yaml

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402

from src import trainer  # noqa: E402

N_CLASSES, T, N_FIELDS = 5, 6, 64


class SyntheticDataset(Dataset):
    """Learnable toy task: the class is encoded in the mean of the temporal signal."""

    def __init__(self, n=N_FIELDS, seed=0, keys="contract"):
        g = torch.Generator().manual_seed(seed)
        self.label = torch.randint(0, N_CLASSES, (n,), generator=g)
        self.prev = torch.where(torch.rand(n, generator=g) < 0.1, torch.tensor(-1),
                                torch.where(torch.rand(n, generator=g) < 0.7, self.label,
                                            torch.randint(0, N_CLASSES, (n,), generator=g)))
        self.x = self.label[:, None, None].float() + 0.1 * torch.randn(n, T, 3, generator=g)
        self.keys = keys

    def __len__(self):
        return len(self.label)

    def __getitem__(self, i):
        item = {"temporal": self.x[i], "mask": torch.ones(T, dtype=torch.bool)}
        if self.keys == "contract":
            item.update(label=self.label[i], previous_crop=self.prev[i],
                        rotation_flag=(self.prev[i] >= 0) & (self.prev[i] != self.label[i]))
        else:
            item.update(y=self.label[i], y_prev=self.prev[i])
        return item


class TinyModel(nn.Module):
    def __init__(self, as_dict=True):
        super().__init__()
        self.fc = nn.Linear(3, N_CLASSES)
        self.as_dict = as_dict

    def forward(self, batch):
        logits = self.fc(batch["temporal"].mean(dim=1))
        return {"logits": logits} if self.as_dict else logits


def _loader(keys="contract", shuffle=False):
    return trainer.make_loader(SyntheticDataset(keys=keys), 16, shuffle)


@pytest.mark.parametrize("keys", ["contract", "methodology"])
@pytest.mark.parametrize("as_dict", [True, False])
def test_epoch_functions_accept_both_conventions(keys, as_dict):
    model = TinyModel(as_dict)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    crit = nn.CrossEntropyLoss()
    tr = trainer.train_one_epoch(model, _loader(keys, True), opt, crit, torch.device("cpu"))
    assert set(tr) == {"loss", "accuracy"}
    val = trainer.validate_one_epoch(model, _loader(keys), crit, torch.device("cpu"), n_classes=N_CLASSES)
    assert len(val["per_class_f1"]) == N_CLASSES
    assert "rotated_accuracy" in val and "rare_class_f1" in val


def test_normalize_batch_derives_rotation_flag():
    b = trainer.normalize_batch({"y": torch.tensor([0, 1, 2]), "y_prev": torch.tensor([0, 2, -1])})
    assert b["rotation_flag"].tolist() == [False, True, False]


def test_fit_writes_checkpoints_history_and_learns(tmp_path):
    torch.manual_seed(0)
    model = TinyModel()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-2)
    summary = trainer.fit(model, _loader(shuffle=True), {60: _loader(), 152: _loader()},
                          nn.CrossEntropyLoss(), opt, torch.device("cpu"), tmp_path,
                          epochs=8, patience=5, n_classes=N_CLASSES)
    assert (tmp_path / "best.pt").exists() and (tmp_path / "last.pt").exists()
    history = json.loads((tmp_path / "history.json").read_text())
    assert len(history["val_score"]) == summary["epochs_run"] <= 8
    assert set(history["val_macro_f1_by_cutoff"][0]) == {"60", "152"}
    assert summary["best_score"] == max(history["val_score"])


def test_early_stopping_and_checkpoint_roundtrip(tmp_path):
    stopper = trainer.EarlyStopping(patience=2)
    assert [stopper(s) for s in (0.5, 0.4, 0.4)] == [False, False, True]

    model = TinyModel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    trainer.save_checkpoint(model, opt, 3, 0.7, tmp_path / "c.pt")
    other = TinyModel()
    epoch, score = trainer.load_checkpoint(other, torch.optim.AdamW(other.parameters()), tmp_path / "c.pt",
                                           torch.device("cpu"))
    assert (epoch, score) == (3, 0.7)
    assert torch.equal(model.fc.weight, other.fc.weight)


def test_train_end_to_end_with_custom_builder(tmp_path):
    cfg = {"seed": 1, "paths": {"data_root": "x", "cache": "x", "outputs": str(tmp_path)},
           "protocol": {}, "interface": {"n_classes": N_CLASSES}, "model": {}, "loss": {}, "gate": {},
           "train": {"epochs": 3, "batch_size": 16, "learning_rate": 0.01, "weight_decay": 0.0,
                     "patience": 2, "num_workers": 0, "eval_cutoffs": [60, 152]}}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    def build(config, device):
        return TinyModel(), SyntheticDataset(), {60: SyntheticDataset(seed=1)}, nn.CrossEntropyLoss()

    summary = trainer.train(str(cfg_path), "smoke", build=build)
    run_dir = tmp_path / "smoke"
    assert summary["epochs_run"] >= 1
    info = json.loads((run_dir / "run_info.json").read_text())
    assert info["seed"] == 1 and "git_commit" in info
    assert (run_dir / "config.yaml").exists() and (run_dir / "best.pt").exists()


def test_train_without_builder_fails_clearly(tmp_path):
    cfg = {"seed": 1, "paths": {"outputs": str(tmp_path)}, "protocol": {}, "interface": {"n_classes": 5},
           "model": {}, "loss": {}, "gate": {},
           "train": {"epochs": 1, "batch_size": 4, "learning_rate": 0.01, "weight_decay": 0.0,
                     "patience": 1, "num_workers": 0, "eval_cutoffs": [60]}}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(NotImplementedError):
        trainer.train(str(p), "nobuild")
