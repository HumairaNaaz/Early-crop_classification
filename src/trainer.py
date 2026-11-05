"""OWNER: Eval Lead (Divyanshi). Training loop, random day-of-year cropping, early stopping,
fixed seeds, experiment tracking (W&B or TensorBoard). Every run logs: config, seed, git commit."""
from src.metrics import compute_metrics
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

    except subprocess.CalledProcessError:
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

def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train the model for one complete epoch."""
    model.train()

    total_loss = 0.0
    correct = 0
    total_samples = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        labels = batch["label"]

        optimizer.zero_grad()

        outputs = model(batch)
        logits = outputs["logits"]

        loss = criterion(logits, labels)

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
            batch = move_batch_to_device(batch, device)

            labels = batch["label"]

            outputs = model(batch)
            logits = outputs["logits"]

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

def train(config_path: str, run_name: str):
    config = load_training_config(config_path)
    validate_training_config(config)

    seed = config["seed"]
    set_seed(seed)

    device = get_device()

    print("Random seed:", seed)
    print("Training device:", device)
    print("Git commit:", get_git_commit())

    run_dir = create_run_directory(
        config["paths"]["outputs"],
        run_name
    )

    best_checkpoint_path = run_dir / "best.pt"
    last_checkpoint_path = run_dir / "last.pt"

    print("Run directory:", run_dir)
    print("Best checkpoint:", best_checkpoint_path)
    print("Latest checkpoint:", last_checkpoint_path)
