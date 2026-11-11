"""
Reproducible patch-level training for ShuffleNetV2 and ResNet18.
"""

import argparse
import copy
import importlib.metadata
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.aggregate import validate_aggregation_mode, validate_tta_views
from src.data import resolve_image_path
from src.model import get_model, set_backbone_trainable
from src.split import leak_audit


class TimberPatchDataset(Dataset):
    """Extract 6x8 patches from each image for patch-level model training."""

    def __init__(
        self,
        df: pd.DataFrame,
        role_col: str,
        role: str,
        label_to_idx: Dict[str, int],
        grid_rows: int = 6,
        grid_cols: int = 8,
        transform: Optional[Any] = None,
        patch_cache_path: Optional[Path] = None,
    ) -> None:
        self.sub_df = df[df[role_col] == role].reset_index(drop=True)
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.transform = transform
        self.label_to_idx = label_to_idx
        self.patch_cache = (
            np.load(patch_cache_path, mmap_mode="r")
            if patch_cache_path is not None
            else None
        )
        self.cache_row_by_path: Dict[str, int] = {}
        if self.patch_cache is not None:
            if self.patch_cache.ndim != 5 or self.patch_cache.shape[1] != (
                grid_rows * grid_cols
            ):
                raise ValueError(f"Invalid patch-cache shape: {self.patch_cache.shape}")
            cache_manifest = pd.read_csv(
                patch_cache_path.with_suffix(".csv"),
                usecols=["path"],
            )
            self.cache_row_by_path = {
                path: index for index, path in enumerate(cache_manifest["path"])
            }
            missing_paths = set(self.sub_df["path"]) - set(self.cache_row_by_path)
            if missing_paths:
                raise ValueError(
                    f"Patch cache is missing {len(missing_paths)} dataset image paths."
                )
        self.patches: List[Tuple[str, int, int, int, int]] = []

        for image_idx, row in self.sub_df.iterrows():
            label_idx = self.label_to_idx[row["species"]]
            for row_idx in range(grid_rows):
                for col_idx in range(grid_cols):
                    self.patches.append(
                        (row["path"], row_idx, col_idx, label_idx, image_idx)
                    )

    def __len__(self) -> int:
        """Return the number of patches indexed by this dataset."""
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[Any, int, int]:
        """Load, crop, transform, and return one patch and its metadata."""
        image_path, row_idx, col_idx, label_idx, image_idx = self.patches[idx]
        if self.patch_cache is None:
            with Image.open(resolve_image_path(image_path)) as source:
                image = source.convert("RGB")
                width, height = image.size
                patch_width = width // self.grid_cols
                patch_height = height // self.grid_rows
                box = (
                    col_idx * patch_width,
                    row_idx * patch_height,
                    (col_idx + 1) * patch_width,
                    (row_idx + 1) * patch_height,
                )
                patch = image.crop(box)
        else:
            cache_row = self.cache_row_by_path[image_path]
            cache_patch = row_idx * self.grid_cols + col_idx
            patch = Image.fromarray(self.patch_cache[cache_row, cache_patch])

        if self.transform is not None:
            patch = self.transform(patch)
        return patch, label_idx, image_idx


def build_patch_cache(
    manifest: pd.DataFrame,
    cache_path: Path,
    grid_rows: int = 6,
    grid_cols: int = 8,
    image_size: int = 112,
) -> None:
    """Pre-crop and resize each image patch once into a memory-mapped cache."""
    required = {"path", "width", "height"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Patch caching requires columns: {sorted(missing)}")
    if cache_path.exists() or cache_path.with_suffix(".csv").exists():
        raise FileExistsError(f"Patch cache already exists: {cache_path}")
    if manifest["path"].duplicated().any():
        raise ValueError("Patch caching requires a unique manifest path per image.")
    if grid_rows < 1 or grid_cols < 1 or image_size < 1:
        raise ValueError("Grid dimensions and resized patch size must be positive.")
    if (manifest["width"] % grid_cols != 0).any() or (
        manifest["height"] % grid_rows != 0
    ).any():
        raise ValueError("Every image must divide evenly into the configured grid.")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    shape = (len(manifest), grid_rows * grid_cols, image_size, image_size, 3)
    cache = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=np.uint8,
        shape=shape,
    )
    resize = transforms.Resize((image_size, image_size))
    cache_manifest = manifest[["path"]].reset_index(drop=True)

    for image_index, row in enumerate(manifest.itertuples(index=False)):
        with Image.open(resolve_image_path(row.path)) as source:
            image = source.convert("RGB")
            patch_width = image.width // grid_cols
            patch_height = image.height // grid_rows
            for row_index in range(grid_rows):
                for col_index in range(grid_cols):
                    box = (
                        col_index * patch_width,
                        row_index * patch_height,
                        (col_index + 1) * patch_width,
                        (row_index + 1) * patch_height,
                    )
                    patch = resize(image.crop(box))
                    cache[image_index, row_index * grid_cols + col_index] = np.asarray(
                        patch,
                        dtype=np.uint8,
                    )
        if (image_index + 1) % 50 == 0 or image_index + 1 == len(manifest):
            print(
                f"Prepared cached patches for {image_index + 1}/{len(manifest)} images",
                flush=True,
            )

    cache.flush()
    del cache
    cache_manifest.to_csv(cache_path.with_suffix(".csv"), index=False)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load the project YAML configuration."""
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Training configuration is not a mapping: {config_path}")
    return config


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def initialize_worker(worker_id: int) -> None:
    """Seed Python and NumPy from the PyTorch DataLoader worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_transforms(
    img_size: int = 112,
    rotation_degrees: float = 10,
    horizontal_flip: bool = True,
    vertical_flip: bool = True,
    translation_fraction: float = 0.0,
    scale_min: float = 1.0,
    scale_max: float = 1.0,
    shear_degrees: float = 0.0,
    color_jitter: Optional[Dict[str, float]] = None,
    gaussian_blur_prob: float = 0.0,
    random_erasing_prob: float = 0.0,
) -> Tuple[Any, Any]:
    """Create augmentation and deterministic evaluation transforms.

    The default arguments reproduce the original notebook augmentation. The
    optimised configuration enables stronger, label-preserving augmentation for
    end-grain texture so the network cannot rely on specimen-specific lighting,
    background, or sensor cues.
    """
    if not 0 <= translation_fraction < 1:
        raise ValueError("Translation fraction must be in [0, 1).")
    if not 0 < scale_min <= scale_max:
        raise ValueError("Scale range must satisfy 0 < min <= max.")
    if not 0 <= gaussian_blur_prob <= 1 or not 0 <= random_erasing_prob <= 1:
        raise ValueError("Augmentation probabilities must be in [0, 1].")
    normalization = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_operations: List[Any] = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomAffine(
            degrees=rotation_degrees,
            translate=(translation_fraction, translation_fraction),
            scale=(scale_min, scale_max),
            shear=shear_degrees,
        ),
    ]
    if horizontal_flip:
        train_operations.append(transforms.RandomHorizontalFlip())
    if vertical_flip:
        train_operations.append(transforms.RandomVerticalFlip())
    jitter = color_jitter or {}
    if any(float(jitter.get(key, 0.0)) > 0 for key in (
        "brightness", "contrast", "saturation", "hue"
    )):
        train_operations.append(
            transforms.ColorJitter(
                brightness=float(jitter.get("brightness", 0.0)),
                contrast=float(jitter.get("contrast", 0.0)),
                saturation=float(jitter.get("saturation", 0.0)),
                hue=float(jitter.get("hue", 0.0)),
            )
        )
    if gaussian_blur_prob > 0:
        train_operations.append(
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))],
                p=gaussian_blur_prob,
            )
        )
    train_operations.extend([transforms.ToTensor(), normalization])
    if random_erasing_prob > 0:
        train_operations.append(
            transforms.RandomErasing(
                p=random_erasing_prob,
                scale=(0.02, 0.12),
                ratio=(0.3, 3.3),
                value="random",
            )
        )
    train_transform = transforms.Compose(train_operations)
    eval_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalization,
        ]
    )
    return train_transform, eval_transform


def compute_class_weights(
    df: pd.DataFrame,
    role_col: str,
    num_classes: int,
    label_to_idx: Dict[str, int],
    weighting: str = "image",
) -> torch.Tensor:
    """Calculate inverse-frequency class weights from the training split.

    weighting='image' (legacy) counts images. weighting='specimen' counts
    distinct specimens, which is the statistically honest unit because images
    from one specimen are not independent.
    """
    if weighting not in {"image", "specimen"}:
        raise ValueError("weighting must be 'image' or 'specimen'.")
    train_df = df[df[role_col] == "train"]
    counts = np.zeros(num_classes, dtype=np.float32)
    if weighting == "specimen":
        per_species = (
            train_df.groupby("species")["specimen_code"].nunique().to_dict()
        )
        for species, count in per_species.items():
            if species in label_to_idx:
                counts[label_to_idx[species]] = count
    else:
        for species, count in train_df["species"].value_counts().items():
            if species in label_to_idx:
                counts[label_to_idx[species]] = count
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def compute_sample_weights(
    df: pd.DataFrame,
    role_col: str,
    label_to_idx: Dict[str, int],
) -> torch.Tensor:
    """Per-image weights that give every specimen and species equal influence.

    Weight = 1 / (images in that specimen x specimens of that class). Each
    specimen therefore contributes the same total loss, and each species
    contributes the same total loss, regardless of how many photographs were
    taken of it. The tensor is aligned with the dataset's filtered image order.
    """
    train_df = df[df[role_col] == "train"]
    if train_df.empty:
        raise ValueError("Cannot compute sample weights for an empty training split.")
    images_per_specimen = train_df.groupby("specimen_code")["path"].transform(
        "size"
    ).astype(float)
    specimens_per_species = train_df.groupby("species")["specimen_code"].transform(
        "nunique"
    ).astype(float)
    weights = 1.0 / (images_per_specimen.to_numpy() * specimens_per_species.to_numpy())
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def build_data_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    """Create a seeded data loader for one role and fold."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=initialize_worker if num_workers else None,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[optim.Optimizer] = None,
    sample_weights: Optional[torch.Tensor] = None,
    grad_clip_norm: float = 0.0,
) -> Tuple[float, float]:
    """Run one training or evaluation epoch and return loss and accuracy."""
    training = optimizer is not None
    model.train(training)
    loss_total = 0.0
    correct_total = 0
    sample_total = 0

    with torch.set_grad_enabled(training):
        for inputs, labels, image_indices in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            if sample_weights is not None:
                per_sample = criterion(logits, labels)
                weights = sample_weights[image_indices].to(
                    device=device, dtype=per_sample.dtype
                )
                loss = (per_sample * weights).sum() / weights.sum().clamp_min(1e-12)
            else:
                loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), grad_clip_norm
                    )
                optimizer.step()

            count = len(labels)
            loss_total += loss.item() * count
            correct_total += int((logits.argmax(dim=1) == labels).sum().item())
            sample_total += count

    if sample_total == 0:
        raise ValueError("Cannot run an epoch with an empty data loader.")
    return loss_total / sample_total, correct_total / sample_total


def build_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str,
    total_epochs: int,
    warmup_epochs: int,
) -> Optional[Any]:
    """Build an optional warmup + cosine learning-rate scheduler."""
    if scheduler_name not in {"none", "cosine"}:
        raise ValueError("scheduler must be 'none' or 'cosine'.")
    if scheduler_name == "none":
        return None
    if total_epochs < 1:
        raise ValueError("total_epochs must be positive.")
    warmup_epochs = max(0, min(warmup_epochs, total_epochs - 1))

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    epochs_s1: int = 10,
    epochs_s2: int = 20,
    lr_s1: float = 1e-3,
    lr_s2: float = 1e-4,
    patience: int = 7,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    sample_weights: Optional[torch.Tensor] = None,
    backbone_lr_scale: float = 1.0,
    scheduler_name: str = "none",
    warmup_epochs: int = 0,
    grad_clip_norm: float = 0.0,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train one fold with a frozen-head stage and validation-based fine-tuning."""
    if epochs_s1 < 1 or epochs_s2 < 1 or patience < 1:
        raise ValueError("Epoch counts and early-stopping patience must be positive.")
    if not 0 < backbone_lr_scale <= 1:
        raise ValueError("backbone_lr_scale must be in (0, 1].")
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError("Each fold requires non-empty training and validation data.")

    model = model.to(device)
    train_criterion = nn.CrossEntropyLoss(
        weight=None if sample_weights is not None else class_weights.to(device),
        label_smoothing=label_smoothing,
        reduction="none" if sample_weights is not None else "mean",
    )
    val_criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=label_smoothing,
    )
    history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_stage2_epoch = 0

    set_backbone_trainable(model, trainable=False)
    optimizer = optim.AdamW(
        model.fc.parameters(), lr=lr_s1, weight_decay=weight_decay
    )
    for epoch in range(1, epochs_s1 + 1):
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            train_criterion,
            device,
            optimizer,
            sample_weights=sample_weights,
            grad_clip_norm=grad_clip_norm,
        )
        val_loss, val_accuracy = run_epoch(
            model, val_loader, val_criterion, device
        )
        history.append(
            {
                "stage": 1,
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    model.load_state_dict(best_weights)
    set_backbone_trainable(model, trainable=True)
    backbone_parameters = [
        parameter for name, parameter in model.named_parameters()
        if "fc" not in name
    ]
    head_parameters = [
        parameter for name, parameter in model.named_parameters()
        if "fc" in name
    ]
    optimizer = optim.AdamW(
        [
            {"params": backbone_parameters, "lr": lr_s2 * backbone_lr_scale},
            {"params": head_parameters, "lr": lr_s2},
        ],
        weight_decay=weight_decay,
    )
    scheduler = build_scheduler(
        optimizer,
        scheduler_name,
        total_epochs=epochs_s2,
        warmup_epochs=warmup_epochs,
    )
    epochs_without_improvement = 0

    for epoch in range(1, epochs_s2 + 1):
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            train_criterion,
            device,
            optimizer,
            sample_weights=sample_weights,
            grad_clip_norm=grad_clip_norm,
        )
        val_loss, val_accuracy = run_epoch(
            model, val_loader, val_criterion, device
        )
        history.append(
            {
                "stage": 2,
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
        if scheduler is not None:
            scheduler.step()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epochs_s1 + epoch
            best_stage2_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_weights)
    return model, {
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_stage2_epoch": best_stage2_epoch,
    }


def train_final_model(
    model: nn.Module,
    train_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    epochs_s1: int,
    epochs_s2: int,
    lr_s1: float,
    lr_s2: float,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    sample_weights: Optional[torch.Tensor] = None,
    backbone_lr_scale: float = 1.0,
    scheduler_name: str = "none",
    warmup_epochs: int = 0,
    grad_clip_norm: float = 0.0,
) -> Tuple[nn.Module, List[Dict[str, Any]]]:
    """Train the final website model for epoch counts selected by the folds."""
    if epochs_s1 < 1 or epochs_s2 < 1 or len(train_loader) == 0:
        raise ValueError("Final training requires positive epochs and non-empty data.")
    if not 0 < backbone_lr_scale <= 1:
        raise ValueError("backbone_lr_scale must be in (0, 1].")

    model = model.to(device)
    train_criterion = nn.CrossEntropyLoss(
        weight=None if sample_weights is not None else class_weights.to(device),
        label_smoothing=label_smoothing,
        reduction="none" if sample_weights is not None else "mean",
    )
    history: List[Dict[str, Any]] = []

    set_backbone_trainable(model, trainable=False)
    optimizer = optim.AdamW(
        model.fc.parameters(), lr=lr_s1, weight_decay=weight_decay
    )
    for epoch in range(1, epochs_s1 + 1):
        loss, accuracy = run_epoch(
            model,
            train_loader,
            train_criterion,
            device,
            optimizer,
            sample_weights=sample_weights,
            grad_clip_norm=grad_clip_norm,
        )
        history.append(
            {
                "stage": 1,
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": loss,
                "train_accuracy": accuracy,
            }
        )

    set_backbone_trainable(model, trainable=True)
    backbone_parameters = [
        parameter for name, parameter in model.named_parameters()
        if "fc" not in name
    ]
    head_parameters = [
        parameter for name, parameter in model.named_parameters()
        if "fc" in name
    ]
    optimizer = optim.AdamW(
        [
            {"params": backbone_parameters, "lr": lr_s2 * backbone_lr_scale},
            {"params": head_parameters, "lr": lr_s2},
        ],
        weight_decay=weight_decay,
    )
    scheduler = build_scheduler(
        optimizer,
        scheduler_name,
        total_epochs=epochs_s2,
        warmup_epochs=warmup_epochs,
    )
    for epoch in range(1, epochs_s2 + 1):
        loss, accuracy = run_epoch(
            model,
            train_loader,
            train_criterion,
            device,
            optimizer,
            sample_weights=sample_weights,
            grad_clip_norm=grad_clip_norm,
        )
        history.append(
            {
                "stage": 2,
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": loss,
                "train_accuracy": accuracy,
            }
        )
        if scheduler is not None:
            scheduler.step()
    return model, history


def export_final_model(
    model: nn.Module,
    class_names: List[str],
    artifacts_dir: Path,
    arch: str,
    img_size: int = 112,
    head_type: str = "legacy",
    hidden_dim: int = 256,
    aggregation: str = "majority",
    tta_views: Optional[List[str]] = None,
) -> None:
    """Save final PyTorch weights, ONNX graph, class order, and preprocessing."""
    if not class_names:
        raise ValueError("Class names cannot be empty.")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model.eval().to("cpu")

    torch.save(model.state_dict(), artifacts_dir / "model.pt")
    with (artifacts_dir / "class_names.json").open("w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)

    preprocess_config = {
        "model_version": f"p2-{arch}",
        "architecture": arch,
        "grid_rows": 6,
        "grid_cols": 8,
        "num_patches": 48,
        "patch_size": [img_size, img_size],
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "head_type": head_type,
        "hidden_dim": hidden_dim,
        "aggregation": aggregation,
        "tta_views": list(tta_views or []),
        "min_confidence_pct": None,
        "min_sharpness": None,
    }
    with (artifacts_dir / "preprocess_config.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(preprocess_config, file, indent=2)

    example_input = torch.randn(48, 3, img_size, img_size)
    torch.onnx.export(
        model,
        example_input,
        artifacts_dir / "model.onnx",
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=14,
        dynamo=False,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    with (artifacts_dir / "model_info.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "architecture": arch,
                "head_type": head_type,
                "hidden_dim": hidden_dim,
                "parameters": parameter_count,
                "pytorch_size_bytes": (artifacts_dir / "model.pt").stat().st_size,
                "onnx_size_bytes": (artifacts_dir / "model.onnx").stat().st_size,
                "num_classes": len(class_names),
            },
            file,
            indent=2,
        )


def load_class_names(species_csv: Path) -> List[str]:
    """Return species slugs in the user-approved CSV ID order."""
    species = pd.read_csv(species_csv).sort_values("id", kind="stable")
    if species["id"].duplicated().any() or species["slug"].duplicated().any():
        raise ValueError(f"Species IDs and slugs must be unique in {species_csv}.")
    return species["slug"].astype(str).tolist()


def validate_split_file(
    manifest: pd.DataFrame,
    splits: pd.DataFrame,
    class_names: List[str],
) -> None:
    """Check that the saved folds match the manifest and known class order."""
    required = {
        "path",
        "species",
        "specimen_idx",
        "specimen_code",
        "group_id",
        "sha256",
        "fold_1_role",
        "fold_2_role",
        "fold_3_role",
        "fold_1_eval_group",
        "fold_2_eval_group",
        "fold_3_eval_group",
    }
    missing = required - set(splits.columns)
    if missing:
        raise ValueError(f"Saved splits are missing columns: {sorted(missing)}")
    for column in (
        "path",
        "species",
        "specimen_idx",
        "specimen_code",
        "group_id",
        "sha256",
    ):
        if manifest[column].tolist() != splits[column].tolist():
            raise ValueError(f"Saved splits do not match manifest column {column}.")
    if set(splits["species"]) != set(class_names):
        raise ValueError("Saved splits and species CSV have different class sets.")
    for fold in (1, 2, 3):
        roles = splits[f"fold_{fold}_role"]
        if roles.isna().any() or not set(roles).issubset({"train", "val", "test"}):
            raise ValueError(f"Fold {fold} contains missing or invalid roles.")
        leak_audit(splits, roles, fold)


def run_training_pipeline(
    config_path: Path,
    output_dir: Path,
    num_workers: int = 2,
    cache_patches: bool = False,
    patch_cache_path: Optional[Path] = None,
    resume: bool = False,
) -> Dict[str, Any]:
    """Train and export both architectures for every fold and final use."""
    config = load_config(config_path)
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(
            f"Training output directory is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    data_config = config["data"]
    model_config = config["models"]
    training_config = config["training"]
    grid_config = config["grid"]
    patch_config = config["patch"]
    augmentation_config = config["augmentation"]
    inference_config = config.get("inference", {}) or {}
    head_type = str(model_config.get("head_type", "legacy"))
    hidden_dim = int(model_config.get("hidden_dim", 256))
    aggregation = validate_aggregation_mode(
        str(inference_config.get("aggregation", "majority"))
    )
    tta_views = list(inference_config.get("tta_views", []) or [])
    if tta_views:
        tta_views = list(validate_tta_views(tta_views))
    weight_decay = float(training_config.get("weight_decay", 0.0))
    label_smoothing = float(training_config.get("label_smoothing", 0.0))
    class_weighting = str(training_config.get("class_weighting", "image"))
    backbone_lr_scale = float(training_config.get("backbone_lr_scale", 1.0))
    scheduler_name = str(training_config.get("scheduler", "none"))
    warmup_epochs = int(training_config.get("warmup_epochs", 0))
    grad_clip_norm = float(training_config.get("grad_clip_norm", 0.0))
    species_csv = Path(data_config["species_csv"])
    class_names = load_class_names(species_csv)
    label_to_idx = {name: index for index, name in enumerate(class_names)}
    manifest = pd.read_csv(data_config["manifest_path"])
    splits = pd.read_csv(data_config["splits_path"])
    validate_split_file(manifest, splits, class_names)

    grid_rows = int(grid_config["rows"])
    grid_cols = int(grid_config["cols"])
    image_size = int(patch_config["resize_w"])
    if image_size != int(patch_config["resize_h"]):
        raise ValueError("The training pipeline requires square resized patches.")
    active_cache_path: Optional[Path] = None
    if cache_patches:
        active_cache_path = patch_cache_path or (output_dir / ".patch_cache.npy")
        if not active_cache_path.exists():
            build_patch_cache(
                manifest,
                active_cache_path,
                grid_rows=grid_rows,
                grid_cols=grid_cols,
                image_size=image_size,
            )
        cache_shape = (
            len(manifest),
            grid_rows * grid_cols,
            image_size,
            image_size,
            3,
        )
        if not active_cache_path.with_suffix(".csv").is_file():
            raise FileNotFoundError(
                f"Patch-cache manifest not found: "
                f"{active_cache_path.with_suffix('.csv')}"
            )
        cached_paths = pd.read_csv(
            active_cache_path.with_suffix(".csv"),
            usecols=["path"],
        )["path"]
        if cached_paths.tolist() != manifest["path"].tolist():
            raise ValueError("Patch cache does not match the current manifest order.")
        cache = np.load(active_cache_path, mmap_mode="r")
        if cache.shape != cache_shape or cache.dtype != np.uint8:
            raise ValueError(
                f"Patch cache has shape/dtype {cache.shape}/{cache.dtype}; "
                f"expected {cache_shape}/uint8."
            )
        del cache
    batch_size = int(training_config["batch_size"])
    train_transform, eval_transform = get_transforms(
        image_size,
        rotation_degrees=float(augmentation_config["rotation_degrees"]),
        horizontal_flip=bool(augmentation_config["horizontal_flip"]),
        vertical_flip=bool(augmentation_config["vertical_flip"]),
        translation_fraction=float(augmentation_config["translation_fraction"]),
        scale_min=float(augmentation_config.get("scale_min", 1.0)),
        scale_max=float(augmentation_config.get("scale_max", 1.0)),
        shear_degrees=float(augmentation_config.get("shear_degrees", 0.0)),
        color_jitter=augmentation_config.get("color_jitter"),
        gaussian_blur_prob=float(
            augmentation_config.get("gaussian_blur_prob", 0.0)
        ),
        random_erasing_prob=float(
            augmentation_config.get("random_erasing_prob", 0.0)
        ),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}; classes: {len(class_names)}", flush=True)

    architectures = [
        ("shufflenet_v2", str(model_config["primary"])),
        ("resnet18", str(model_config["comparison"])),
    ]
    summary: Dict[str, Any] = {
        "seed": seed,
        "device": str(device),
        "class_names": class_names,
        "runtime_versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision": importlib.metadata.version("torchvision"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "pillow": importlib.metadata.version("Pillow"),
        },
        "training_config": {
            "batch_size": batch_size,
            "epochs_stage1": int(training_config["epochs_stage1"]),
            "epochs_stage2_limit": int(training_config["epochs_stage2"]),
            "learning_rate_stage1": float(training_config["lr_stage1"]),
            "learning_rate_stage2": float(training_config["lr_stage2"]),
            "patience": int(training_config["patience"]),
            "weight_decay": weight_decay,
            "label_smoothing": label_smoothing,
            "class_weighting": class_weighting,
            "backbone_lr_scale": backbone_lr_scale,
            "scheduler": scheduler_name,
            "warmup_epochs": warmup_epochs,
            "grad_clip_norm": grad_clip_norm,
            "head_type": head_type,
            "hidden_dim": hidden_dim,
            "aggregation": aggregation,
            "tta_views": tta_views,
            "augmentation": augmentation_config,
            "patch_size": [image_size, image_size],
            "normalization_mean": patch_config["normalize_mean"],
            "normalization_std": patch_config["normalize_std"],
            "patch_cache_enabled": cache_patches,
        },
        "folds": {},
        "final": {},
    }
    stage1_epochs = int(training_config["epochs_stage1"])
    stage2_limit = int(training_config["epochs_stage2"])
    patience = int(training_config["patience"])
    learning_rate_s1 = float(training_config["lr_stage1"])
    learning_rate_s2 = float(training_config["lr_stage2"])

    for architecture_index, (architecture, model_name) in enumerate(architectures):
        fold_results = []
        for fold in (1, 2, 3):
            fold_seed = seed + architecture_index * 1000 + fold
            role_column = f"fold_{fold}_role"
            fold_dir = output_dir / "folds" / architecture / f"fold_{fold}"
            staging_dir = fold_dir.with_name(f"{fold_dir.name}.incomplete")
            if staging_dir.exists():
                raise FileExistsError(
                    f"Incomplete fold staging directory exists: {staging_dir}"
                )
            expected_fold_files = (
                fold_dir / "model.pt",
                fold_dir / "history.csv",
                fold_dir / "training_metadata.json",
            )
            if fold_dir.exists():
                if not resume:
                    raise FileExistsError(
                        f"Fold output already exists: {fold_dir}"
                    )
                if not all(path.is_file() for path in expected_fold_files):
                    raise FileExistsError(
                        f"Incomplete fold output cannot be resumed safely: {fold_dir}"
                    )
                with expected_fold_files[2].open(encoding="utf-8") as file:
                    fold_result = json.load(file)
                expected_identity = {
                    "architecture": architecture,
                    "fold": fold,
                    "seed": fold_seed,
                    "class_names": class_names,
                    "train_images": int((splits[role_column] == "train").sum()),
                    "validation_images": int((splits[role_column] == "val").sum()),
                    "test_images": int((splits[role_column] == "test").sum()),
                    "head_type": head_type,
                    "hidden_dim": hidden_dim,
                    "class_weighting": class_weighting,
                }
                for key, value in expected_identity.items():
                    if fold_result.get(key) != value:
                        raise ValueError(
                            f"Existing {architecture} fold {fold} metadata "
                            f"does not match current {key}."
                        )
                fold_results.append(fold_result)
                summary["folds"][f"{architecture}_fold_{fold}"] = fold_result
                print(
                    f"{architecture} fold {fold}: reusing completed checkpoint",
                    flush=True,
                )
                continue

            set_seed(fold_seed)
            train_dataset = TimberPatchDataset(
                splits,
                role_column,
                "train",
                label_to_idx,
                grid_rows,
                grid_cols,
                train_transform,
                active_cache_path,
            )
            val_dataset = TimberPatchDataset(
                splits,
                role_column,
                "val",
                label_to_idx,
                grid_rows,
                grid_cols,
                eval_transform,
                active_cache_path,
            )
            train_loader = build_data_loader(
                train_dataset,
                batch_size,
                True,
                num_workers,
                fold_seed,
            )
            val_loader = build_data_loader(
                val_dataset,
                batch_size,
                False,
                num_workers,
                fold_seed + 1,
            )
            class_weights = compute_class_weights(
                splits,
                role_column,
                len(class_names),
                label_to_idx,
                weighting=class_weighting,
            )
            sample_weights = (
                compute_sample_weights(splits, role_column, label_to_idx)
                if class_weighting == "specimen"
                else None
            )
            model = get_model(
                model_name,
                num_classes=len(class_names),
                pretrained=True,
                dropout=float(model_config["dropout"]),
                head_type=head_type,
                hidden_dim=hidden_dim,
            )
            model, result = train_fold(
                model,
                train_loader,
                val_loader,
                class_weights,
                device,
                epochs_s1=stage1_epochs,
                epochs_s2=stage2_limit,
                lr_s1=learning_rate_s1,
                lr_s2=learning_rate_s2,
                patience=patience,
                weight_decay=weight_decay,
                label_smoothing=label_smoothing,
                sample_weights=sample_weights,
                backbone_lr_scale=backbone_lr_scale,
                scheduler_name=scheduler_name,
                warmup_epochs=warmup_epochs,
                grad_clip_norm=grad_clip_norm,
            )
            fold_dir.parent.mkdir(parents=True, exist_ok=True)
            staging_dir.mkdir()
            torch.save(model.state_dict(), staging_dir / "model.pt")
            pd.DataFrame(result["history"]).to_csv(
                staging_dir / "history.csv", index=False
            )
            fold_result = {
                "architecture": architecture,
                "fold": fold,
                "seed": fold_seed,
                "train_images": int((splits[role_column] == "train").sum()),
                "validation_images": int((splits[role_column] == "val").sum()),
                "test_images": int((splits[role_column] == "test").sum()),
                "best_validation_loss": result["best_val_loss"],
                "best_epoch_total": result["best_epoch"],
                "best_stage2_epoch": result["best_stage2_epoch"],
                "head_type": head_type,
                "hidden_dim": hidden_dim,
                "class_weighting": class_weighting,
                "class_names": class_names,
            }
            with (staging_dir / "training_metadata.json").open(
                "w", encoding="utf-8"
            ) as file:
                json.dump(fold_result, file, indent=2)
            staging_dir.replace(fold_dir)
            fold_results.append(fold_result)
            summary["folds"][f"{architecture}_fold_{fold}"] = fold_result
            print(
                f"{architecture} fold {fold}: best val loss "
                f"{result['best_val_loss']:.4f}, stage 2 epoch "
                f"{result['best_stage2_epoch']}",
                flush=True,
            )
            with (output_dir / "training_summary.json").open(
                "w", encoding="utf-8"
            ) as file:
                json.dump(summary, file, indent=2)

        final_stage2_epochs = max(
            1,
            int(
                round(
                    statistics.median(
                        int(fold_result["best_stage2_epoch"])
                        for fold_result in fold_results
                    )
                )
            ),
        )
        final_dir = output_dir / "final" / architecture
        final_staging_dir = final_dir.with_name(f"{final_dir.name}.incomplete")
        if final_staging_dir.exists():
            raise FileExistsError(
                f"Incomplete final-model staging directory exists: "
                f"{final_staging_dir}"
            )
        if final_dir.exists():
            metadata_path = final_dir / "training_metadata.json"
            history_path = final_dir / "training_history.csv"
            model_path = final_dir / "model.pt"
            if not resume or not all(
                path.is_file() for path in (metadata_path, history_path, model_path)
            ):
                raise FileExistsError(
                    f"Final model output already exists or is incomplete: {final_dir}"
                )
            with metadata_path.open(encoding="utf-8") as file:
                final_result = json.load(file)
            expected_final_identity = {
                "architecture": architecture,
                "class_names": class_names,
                "stage1_epochs": stage1_epochs,
                "stage2_epochs": final_stage2_epochs,
                "head_type": head_type,
                "hidden_dim": hidden_dim,
                "class_weighting": class_weighting,
            }
            for key, value in expected_final_identity.items():
                if final_result.get(key) != value:
                    raise ValueError(
                        f"Existing {architecture} final metadata does not match {key}."
                    )
            summary["final"][architecture] = final_result
            with (output_dir / "training_summary.json").open(
                "w", encoding="utf-8"
            ) as file:
                json.dump(summary, file, indent=2)
            print(f"{architecture} final model: reusing completed checkpoint", flush=True)
            continue

        set_seed(seed + architecture_index * 1000 + 100)
        final_role = "_final_role"
        final_splits = splits.copy()
        final_splits[final_role] = "train"
        final_dataset = TimberPatchDataset(
            final_splits,
            final_role,
            "train",
            label_to_idx,
            grid_rows,
            grid_cols,
            train_transform,
            active_cache_path,
        )
        final_loader = build_data_loader(
            final_dataset,
            batch_size,
            True,
            num_workers,
            seed + architecture_index * 1000 + 100,
        )
        final_weights = compute_class_weights(
            final_splits,
            final_role,
            len(class_names),
            label_to_idx,
            weighting=class_weighting,
        )
        final_sample_weights = (
            compute_sample_weights(final_splits, final_role, label_to_idx)
            if class_weighting == "specimen"
            else None
        )
        final_model = get_model(
            model_name,
            num_classes=len(class_names),
            pretrained=True,
            dropout=float(model_config["dropout"]),
            head_type=head_type,
            hidden_dim=hidden_dim,
        )
        final_model, final_history = train_final_model(
            final_model,
            final_loader,
            final_weights,
            device,
            stage1_epochs,
            final_stage2_epochs,
            learning_rate_s1,
            learning_rate_s2,
            weight_decay=weight_decay,
            label_smoothing=label_smoothing,
            sample_weights=final_sample_weights,
            backbone_lr_scale=backbone_lr_scale,
            scheduler_name=scheduler_name,
            warmup_epochs=warmup_epochs,
            grad_clip_norm=grad_clip_norm,
        )
        export_final_model(
            final_model,
            class_names,
            final_staging_dir,
            arch=architecture,
            img_size=image_size,
            head_type=head_type,
            hidden_dim=hidden_dim,
            aggregation=aggregation,
            tta_views=tta_views,
        )
        pd.DataFrame(final_history).to_csv(
            final_staging_dir / "training_history.csv", index=False
        )
        final_result = {
            "architecture": architecture,
            "seed": seed + architecture_index * 1000 + 100,
            "train_images": len(final_splits),
            "stage1_epochs": stage1_epochs,
            "stage2_epochs": final_stage2_epochs,
            "stage2_epoch_source": "median best stage-2 epoch across three folds",
            "fold_stage2_best_epochs": [
                int(fold_result["best_stage2_epoch"])
                for fold_result in fold_results
            ],
            "head_type": head_type,
            "hidden_dim": hidden_dim,
            "class_weighting": class_weighting,
            "aggregation": aggregation,
            "tta_views": tta_views,
            "class_names": class_names,
        }
        with (final_staging_dir / "training_metadata.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(final_result, file, indent=2)
        final_staging_dir.replace(final_dir)
        summary["final"][architecture] = final_result
        print(
            f"{architecture} final model trained on {len(final_splits)} images "
            f"for {stage1_epochs}+{final_stage2_epochs} epochs.",
            flush=True,
        )
        with (output_dir / "training_summary.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(summary, file, indent=2)

    summary_path = output_dir / "training_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return summary


def main() -> None:
    """Parse command-line options and run the complete P2 training phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Project YAML configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/p2_training"),
        help="New, empty directory for fold and final model artifacts.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader worker processes.",
    )
    parser.add_argument(
        "--cache-patches",
        action="store_true",
        help="Precompute resized 6x8 patches once to a local memory-mapped cache.",
    )
    parser.add_argument(
        "--patch-cache-path",
        type=Path,
        default=None,
        help="Optional cache path, useful for keeping the large cache on local disk.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse complete fold/final outputs from a previous interrupted run.",
    )
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative.")
    run_training_pipeline(
        args.config,
        args.output_dir,
        args.num_workers,
        cache_patches=args.cache_patches,
        patch_cache_path=args.patch_cache_path,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
