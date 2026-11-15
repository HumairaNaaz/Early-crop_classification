"""OWNER: Eval Lead (Divyanshi). Training loop, early stopping, best/last checkpoints, fixed
seeds, experiment tracking (none / TensorBoard / W&B). Every run logs: config, seed, git commit.

The random day-of-year cutoff used during training (and the fixed 1 Mar / 1 Jun / 1 Sep / 1 Dec
cutoffs used for validation) is applied by the DATASET, not here: see
methodology/data/dataset.py (CropDataset: train=True draws a random cutoff per sample,
cutoff_doy=<day> fixes it). This module only needs batches that follow the contract keys
(see normalize_batch) and a model that returns logits or {"logits": ...}."""
from src.metrics import compute_metrics
import json
import random
import subprocess
import yaml
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_git_commit() -> str:
    """Return the current Git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()

    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"

def load_training_config(config_path: str) -> dict:
    """Load the YAML training configuration."""
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping.")

    return config

def validate_training_config(config: dict) -> None:
    """Validate required configuration sections and training parameters."""

    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping.")

    required_sections = [
        "paths",
        "protocol",
        "interface",
        "model",
        "loss",
        "gate",
        "train"
    ]

    for section in required_sections:
        if section not in config:
            raise KeyError(f"Missing configuration section: {section}")

    train_config = config["train"]

    if not isinstance(train_config, dict):
        raise ValueError("The train section must be a YAML mapping.")

    required_train_keys = [
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "patience",
        "num_workers",
        "eval_cutoffs"
    ]

    for key in required_train_keys:
        if key not in train_config:
            raise KeyError(f"Missing training parameter: {key}")

    integer_parameters = [
        "epochs",
        "batch_size",
        "patience",
        "num_workers"
    ]

    for key in integer_parameters:
        value = train_config[key]

        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{key} must be an integer.")

    numeric_parameters = [
        "learning_rate",
        "weight_decay"
    ]

    for key in numeric_parameters:
        value = train_config[key]

        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{key} must be numeric.")

        if not np.isfinite(value):
            raise ValueError(f"{key} must be finite.")

    if train_config["epochs"] <= 0:
        raise ValueError("epochs must be positive.")

    if train_config["batch_size"] <= 0:
        raise ValueError("batch_size must be positive.")

    if train_config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive.")

    if train_config["weight_decay"] < 0:
        raise ValueError("weight_decay cannot be negative.")

    if train_config["patience"] <= 0:
        raise ValueError("patience must be positive.")

    if train_config["num_workers"] < 0:
        raise ValueError("num_workers cannot be negative.")

    cutoffs = train_config["eval_cutoffs"]

    if not isinstance(cutoffs, list) or not cutoffs:
        raise ValueError("eval_cutoffs must be a non-empty list.")

    if any(
        not isinstance(c, int)
        or isinstance(c, bool)
        or not 1 <= c <= 366
        for c in cutoffs
    ):
        raise ValueError(
            "Each evaluation cutoff must be an integer between 1 and 366."
        )

    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError("Evaluation cutoffs must be unique.")

def get_device(device_name: str = None) -> torch.device:
    """Select the training device."""
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device_name)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    return device

def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int = 0):
    """Create a DataLoader for a dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

def create_history():
    """Initialize storage for epoch-wise training and validation metrics."""
    return {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": []
    }

def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move all tensor values in a batch to the selected device."""
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }

def normalize_batch(batch: dict) -> dict:
    """Accept both batch conventions used in the repo and return a dict with the trainer's keys.

    src/dataset.py (contract):   label, previous_crop, rotation_flag
    methodology/data/dataset.py: y, y_prev   (rotation derived: previous crop known and different)
    Original keys are kept, so models written against either convention still work."""
    b = dict(batch)
    if "label" not in b and "y" in b:
        b["label"] = b["y"]
    if "previous_crop" not in b and "y_prev" in b:
        b["previous_crop"] = b["y_prev"]
    if "rotation_flag" not in b and "previous_crop" in b and "label" in b:
        b["rotation_flag"] = (b["previous_crop"] >= 0) & (b["previous_crop"] != b["label"])
    return b


def get_logits(outputs) -> torch.Tensor:
    """The contract says the model returns {"logits": ..., "gate": ...}; also accept a bare tensor."""
    if isinstance(outputs, torch.Tensor):
        return outputs
    return outputs["logits"]


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train the model for one complete epoch."""
    model.train()

    total_loss = 0.0
    correct = 0
    total_samples = 0

    for batch in loader:
        batch = normalize_batch(move_batch_to_device(batch, device))

        labels = batch["label"]

        optimizer.zero_grad()

        outputs = model(batch)
        logits = get_logits(outputs)

        loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite training loss: check for fully masked time series or invalid inputs."
            )

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        predictions = torch.argmax(logits, dim=1)

        correct += (predictions == labels).sum().item()
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Training loader is empty.")

    average_loss = total_loss / total_samples
    accuracy = correct / total_samples

    return {
        "loss": average_loss,
        "accuracy": accuracy
    }

def validate_one_epoch(model, loader, criterion, device, n_classes: int = 18):
    """Evaluate the model on a validation dataset."""
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_predictions = []
    all_previous_crops = []
    all_rotation_flags = []

    with torch.no_grad():
        for batch in loader:
            batch = normalize_batch(move_batch_to_device(batch, device))

            labels = batch["label"]

            outputs = model(batch)
            logits = get_logits(outputs)

            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_previous_crops.extend(
                batch["previous_crop"].cpu().tolist()
            )
            all_rotation_flags.extend(
                batch["rotation_flag"].cpu().tolist()
            )

    if total_samples == 0:
        raise ValueError("Validation loader is empty.")

    metrics = compute_metrics(
        y_true=all_labels,
        y_pred=all_predictions,
        y_prev=all_previous_crops,
        rotation_mask=all_rotation_flags,
        n_classes=n_classes
    )

    return {
        "loss": total_loss / total_samples,
        **metrics
    }

class EarlyStopping:
    """Stop training when the monitored metric stops improving."""

    def __init__(self, patience: int):
        self.patience = patience
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0

        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

def save_checkpoint(model, optimizer, epoch, score, path):
    """Save the model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_score": score
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, path, device):
    """Load model and optimizer states from a checkpoint."""
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["epoch"], checkpoint["best_score"]

def create_run_directory(output_root: str, run_name: str) -> Path:
    """Create a dedicated directory for a training run."""
    if not run_name or Path(run_name).name != run_name:
        raise ValueError("run_name must be a non-empty directory name.")

    run_dir = Path(output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    return run_dir

class ExperimentLogger:
    """Scalar logging switch: config `logging: {backend: none | tensorboard | wandb}`.

    Default is "none" (history.json is always written). tensorboard / wandb are optional
    dependencies and only imported when selected."""

    def __init__(self, backend: str, run_dir, config: dict = None):
        self.backend = backend or "none"
        self._writer = None
        self._wandb = None
        if self.backend == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=str(Path(run_dir) / "tb"))
        elif self.backend == "wandb":
            import wandb
            self._wandb = wandb
            wandb.init(project="early-crop-classification", name=Path(run_dir).name,
                       dir=str(run_dir), config=config)
        elif self.backend != "none":
            raise ValueError("logging.backend must be none, tensorboard or wandb.")

    def log(self, metrics: dict, step: int) -> None:
        scalars = {k: v for k, v in metrics.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)}
        if self._writer is not None:
            for key, value in scalars.items():
                self._writer.add_scalar(key, value, step)
        if self._wandb is not None:
            self._wandb.log(scalars, step=step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
        if self._wandb is not None:
            self._wandb.finish()


def _as_loader_dict(val_loaders) -> dict:
    """One loader -> {"val": loader}; a dict {cutoff_day: loader} is kept as is."""
    if isinstance(val_loaders, dict):
        if not val_loaders:
            raise ValueError("val_loaders is empty.")
        return val_loaders
    return {"val": val_loaders}


def fit(model, train_loader, val_loaders, criterion, optimizer, device, run_dir,
        epochs: int, patience: int, n_classes: int = 18, scheduler=None, logger=None) -> dict:
    """Epoch loop with early stopping and best/last checkpoints.

    val_loaders is one DataLoader, or {cutoff_day: DataLoader} so the model is validated at
    each evaluation date. The monitored score is the mean validation macro-F1 over those
    loaders (same rule as methodology/train.py), not accuracy, so rare crops count.
    Validation data must NOT be the 2023 test year (see docs on the split).

    Writes run_dir/best.pt, last.pt, history.json and summary.json."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    loaders = _as_loader_dict(val_loaders)
    stopper = EarlyStopping(patience)
    history = create_history()
    history["val_score"] = []
    history["val_macro_f1_by_cutoff"] = []
    best_score, best_epoch = None, 0

    for epoch in range(1, epochs + 1):
        train_result = train_one_epoch(model, train_loader, optimizer, criterion, device)

        per_cutoff = {}
        val_loss, val_acc = [], []
        for cutoff, loader in loaders.items():
            result = validate_one_epoch(model, loader, criterion, device, n_classes=n_classes)
            per_cutoff[str(cutoff)] = result["macro_f1"]
            val_loss.append(result["loss"])
            val_acc.append(result["accuracy"])
        score = float(np.mean(list(per_cutoff.values())))

        history["train_loss"].append(train_result["loss"])
        history["train_accuracy"].append(train_result["accuracy"])
        history["val_loss"].append(float(np.mean(val_loss)))
        history["val_accuracy"].append(float(np.mean(val_acc)))
        history["val_macro_f1"].append(score)
        history["val_score"].append(score)
        history["val_macro_f1_by_cutoff"].append(per_cutoff)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(score)
            else:
                scheduler.step()

        improved = stopper.best_score is None or score > stopper.best_score
        if improved:
            best_score, best_epoch = score, epoch
            save_checkpoint(model, optimizer, epoch, score, run_dir / "best.pt")
        save_checkpoint(model, optimizer, epoch, score, run_dir / "last.pt")

        if logger is not None:
            logger.log({"train/loss": train_result["loss"],
                        "train/accuracy": train_result["accuracy"],
                        "val/loss": history["val_loss"][-1],
                        "val/accuracy": history["val_accuracy"][-1],
                        "val/macro_f1_mean": score,
                        "lr": optimizer.param_groups[0]["lr"]}, step=epoch)

        print(f"Epoch {epoch:03d} | train loss {train_result['loss']:.4f} | "
              f"val macro-F1 {score:.4f}{' *best*' if improved else ''}", flush=True)

        if stopper(score):
            print(f"Early stopping: no improvement for {stopper.patience} epochs.")
            break

    summary = {"best_epoch": best_epoch, "best_score": best_score,
               "epochs_run": len(history["train_loss"]), "stopped_early": stopper.early_stop}
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def default_build(config: dict, device):
    """Placeholder until the team agrees which model/dataset package is final
    (src/ stubs vs methodology/). Pass your own `build` to train()."""
    raise NotImplementedError(
        "No default builder: pass build=callable(config, device) -> "
        "(model, train_dataset, val_datasets, criterion) to train(). val_datasets is a "
        "dataset or {cutoff_day: dataset}."
    )


def train(config_path: str, run_name: str, build=None):
    """Full training run: config -> seed -> run dir -> build -> fit.

    `build(config, device)` must return (model, train_dataset, val_datasets, criterion);
    this keeps the trainer independent of whichever model/dataset implementation is final.
    Batches follow the contract keys (see normalize_batch) and models return
    {"logits": (B, n_classes), ...}."""
    config = load_training_config(config_path)
    validate_training_config(config)

    seed = config["seed"]
    set_seed(seed)

    device = get_device()
    commit = get_git_commit()

    print("Random seed:", seed)
    print("Training device:", device)
    print("Git commit:", commit)

    run_dir = create_run_directory(
        config["paths"]["outputs"],
        run_name
    )

    best_checkpoint_path = run_dir / "best.pt"
    last_checkpoint_path = run_dir / "last.pt"

    print("Run directory:", run_dir)
    print("Best checkpoint:", best_checkpoint_path)
    print("Latest checkpoint:", last_checkpoint_path)

    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (run_dir / "run_info.json").write_text(json.dumps(
        {"run_name": run_name, "seed": seed, "git_commit": commit, "device": str(device)}, indent=2))

    train_config = config["train"]
    n_classes = config["interface"]["n_classes"]

    model, train_dataset, val_datasets, criterion = (build or default_build)(config, device)
    model = model.to(device)
    if hasattr(criterion, "to"):
        criterion = criterion.to(device)

    train_loader = make_loader(train_dataset, train_config["batch_size"], True,
                               train_config["num_workers"])
    val_loaders = {
        key: make_loader(ds, train_config["batch_size"], False, train_config["num_workers"])
        for key, ds in _as_loader_dict(val_datasets).items()
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["learning_rate"],
                                  weight_decay=train_config["weight_decay"])
    scheduler = None
    if train_config.get("scheduler", "plateau") == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2)

    logger = ExperimentLogger(config.get("logging", {}).get("backend", "none"), run_dir, config)
    try:
        summary = fit(model, train_loader, val_loaders, criterion, optimizer, device, run_dir,
                      epochs=train_config["epochs"], patience=train_config["patience"],
                      n_classes=n_classes, scheduler=scheduler, logger=logger)
    finally:
        logger.close()

    print("Finished:", summary)
    return summary
