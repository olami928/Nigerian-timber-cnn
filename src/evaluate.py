"""Evaluate trained fold models on the approved specimen-grouped test splits."""

import argparse
import hashlib
import json
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.aggregate import aggregate_probabilities, forward_probabilities
from src.model import get_model
from src.split import leak_audit
from src.train import load_class_names

matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def resolve_and_verify_images(
    manifest: pd.DataFrame,
    backup_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Map manifest paths to backup files and verify hash, readability, and grid."""
    if not backup_dir.is_dir():
        raise FileNotFoundError(f"Image backup directory does not exist: {backup_dir}")
    files_by_name: Dict[str, List[Path]] = defaultdict(list)
    for path in backup_dir.glob("*/*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            files_by_name[path.name].append(path)

    resolved_paths: List[str] = []
    mapping_rows: List[Dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        relative = Path(row.path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Manifest image path is unsafe: {row.path}")
        if not relative.parts or relative.parts[0] != "patch_ready_dataset":
            raise ValueError(f"Unexpected manifest image path: {row.path}")

        source = backup_dir / Path(*relative.parts[1:])
        mapping_method = "relative_path"
        if not source.is_file():
            matches = files_by_name.get(relative.name, [])
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Expected one backup image named {relative.name}, found {len(matches)}."
                )
            source = matches[0]
            mapping_method = "unique_filename"

        payload = source.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != row.sha256:
            raise ValueError(
                f"SHA-256 mismatch for {source}: expected {row.sha256}, found {digest}."
            )
        try:
            with Image.open(BytesIO(payload)) as image:
                image.load()
                width, height = image.size
        except (OSError, ValueError) as error:
            raise ValueError(f"Unreadable image {source}: {error}") from error
        if (width, height) != (int(row.width), int(row.height)):
            raise ValueError(
                f"Dimension mismatch for {source}: expected "
                f"{row.width}x{row.height}, found {width}x{height}."
            )
        if width % 8 or height % 6:
            raise ValueError(
                f"Image does not divide into the required 6x8 grid: {source} "
                f"({width}x{height})."
            )
        image_bgr = cv2.imdecode(
            np.frombuffer(payload, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image_bgr is None:
            raise ValueError(f"OpenCV could not decode image for sharpness: {source}")
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        resolved_paths.append(str(source))
        mapping_rows.append(
            {
                "manifest_path": str(row.path),
                "image_path": str(source),
                "mapping_method": mapping_method,
                "sha256": digest,
                "sharpness": sharpness,
            }
        )

    resolved_manifest = manifest.copy()
    resolved_manifest["image_path"] = resolved_paths
    return resolved_manifest, pd.DataFrame(mapping_rows)


class PatchEvaluationDataset(Dataset):
    """Lazily crop deterministic raster-order patches for model evaluation."""

    def __init__(
        self,
        images: pd.DataFrame,
        label_to_idx: Dict[str, int],
        image_size: int = 112,
        grid_rows: int = 6,
        grid_cols: int = 8,
    ) -> None:
        self.images = images.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.cached_image_path: str | None = None
        self.cached_image: Image.Image | None = None

    def __len__(self) -> int:
        """Return the number of indexed patches."""
        return len(self.images) * self.grid_rows * self.grid_cols

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        """Load and transform one patch; return its image and patch indices."""
        patches_per_image = self.grid_rows * self.grid_cols
        image_index, patch_index = divmod(index, patches_per_image)
        row_index, column_index = divmod(patch_index, self.grid_cols)
        image_row = self.images.iloc[image_index]
        image_path = str(image_row["image_path"])
        if self.cached_image_path != image_path:
            with Image.open(image_path) as source:
                self.cached_image = source.convert("RGB")
            self.cached_image_path = image_path
        if self.cached_image is None:
            raise RuntimeError(f"Image cache was not initialized for {image_path}.")
        patch_width = self.cached_image.width // self.grid_cols
        patch_height = self.cached_image.height // self.grid_rows
        box = (
            column_index * patch_width,
            row_index * patch_height,
            (column_index + 1) * patch_width,
            (row_index + 1) * patch_height,
        )
        patch = self.cached_image.crop(box)
        return (
            self.transform(patch),
            self.label_to_idx[image_row["species"]],
            image_index,
        )


def aggregate_patch_logits(
    logits: np.ndarray,
    class_names: Sequence[str],
    mode: str = "majority",
) -> List[Dict[str, Any]]:
    """Aggregate 48 patch predictions per image with the shared rules."""
    if logits.ndim != 3 or logits.shape[1] != 48:
        raise ValueError(f"Expected logits shaped [images, 48, classes], got {logits.shape}.")
    shifted = logits - logits.max(axis=2, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=2, keepdims=True)
    return [
        aggregate_probabilities(image_probabilities, class_names, mode=mode)
        for image_probabilities in probabilities
    ]


def predict_test_images(
    model: torch.nn.Module,
    images: pd.DataFrame,
    class_names: Sequence[str],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    tta_views: Sequence[str] = (),
    aggregation: str = "majority",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run patch inference and return image-level and patch-level predictions."""
    label_to_idx = {name: index for index, name in enumerate(class_names)}
    dataset = PatchEvaluationDataset(images, label_to_idx)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    all_probabilities: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_image_indices: List[np.ndarray] = []
    for inputs, labels, image_indices in loader:
        all_probabilities.append(
            forward_probabilities(model, inputs, device, tta_views=tta_views)
        )
        all_labels.append(labels.numpy())
        all_image_indices.append(image_indices.numpy())

    patch_probabilities = np.concatenate(all_probabilities, axis=0)
    patch_labels = np.concatenate(all_labels)
    patch_image_indices = np.concatenate(all_image_indices)
    expected_count = len(images) * 48
    if len(patch_probabilities) != expected_count:
        raise RuntimeError(
            f"Model returned {len(patch_probabilities)} patch predictions; "
            f"expected {expected_count}."
        )
    image_probabilities = patch_probabilities.reshape(
        len(images), 48, len(class_names)
    )
    aggregated = [
        aggregate_probabilities(probabilities, class_names, mode=aggregation)
        for probabilities in image_probabilities
    ]
    image_rows: List[Dict[str, Any]] = []
    patch_rows: List[Dict[str, Any]] = []
    for image_index, (image, result) in enumerate(
        zip(images.itertuples(index=False), aggregated)
    ):
        base = {
            "fold": int(image.fold),
            "eval_group": str(image.eval_group),
            "manifest_path": str(image.path),
            "specimen_code": str(image.specimen_code),
            "true_species": str(image.species),
            **result,
        }
        image_rows.append(base)
        start = image_index * 48
        end = start + 48
        per_patch_probabilities = patch_probabilities[start:end]
        per_patch_predictions = per_patch_probabilities.argmax(axis=1)
        for patch_index, predicted_index in enumerate(per_patch_predictions):
            patch_rows.append(
                {
                    "fold": int(image.fold),
                    "eval_group": str(image.eval_group),
                    "manifest_path": str(image.path),
                    "specimen_code": str(image.specimen_code),
                    "true_species": str(image.species),
                    "patch_index": patch_index,
                    "predicted_species": class_names[int(predicted_index)],
                }
            )
    if not np.all(patch_labels == np.repeat(
        images["species"].map(label_to_idx).to_numpy(), 48
    )):
        raise RuntimeError("Patch labels do not match their source image labels.")
    if not np.array_equal(
        patch_image_indices,
        np.repeat(np.arange(len(images)), 48),
    ):
        raise RuntimeError("Patch ordering does not match image ordering.")
    return image_rows, patch_rows


def calculate_metrics(
    frame: pd.DataFrame,
    true_column: str,
    predicted_column: str,
) -> Dict[str, float]:
    """Calculate classification metrics over classes represented in truth labels."""
    if frame.empty:
        raise ValueError("Cannot calculate metrics for an empty evaluation set.")
    y_true = frame[true_column].astype(str).to_numpy()
    y_pred = frame[predicted_column].astype(str).to_numpy()
    labels = sorted(set(y_true))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def bootstrap_specimen_intervals(
    frame: pd.DataFrame,
    true_column: str,
    predicted_column: str,
    samples: int,
    seed: int,
) -> Dict[str, Tuple[float, float]]:
    """Bootstrap metrics by resampling specimen clusters within each true class."""
    if samples < 1:
        raise ValueError("Bootstrap sample count must be positive.")
    if frame.empty:
        raise ValueError("Cannot bootstrap an empty evaluation set.")

    true_classes = sorted(frame[true_column].astype(str).unique())
    all_classes = sorted(
        set(true_classes) | set(frame[predicted_column].astype(str).unique())
    )
    class_indices = {species: index for index, species in enumerate(all_classes)}
    predicted_indices = {
        species: index for index, species in enumerate(all_classes)
    }
    specimen_predictions: Dict[str, np.ndarray] = {}
    specimen_species: Dict[str, str] = {}
    for (species, specimen), group in frame.groupby(
        [true_column, "specimen_code"], sort=False
    ):
        species = str(species)
        specimen = str(specimen)
        if specimen in specimen_species and specimen_species[specimen] != species:
            raise ValueError(f"Specimen {specimen} has more than one true species.")
        counts = np.zeros(len(all_classes), dtype=np.int64)
        prediction_counts = group[predicted_column].astype(str).value_counts()
        for prediction, count in prediction_counts.items():
            counts[predicted_indices[prediction]] = int(count)
        specimen_predictions[specimen] = counts
        specimen_species[specimen] = species

    specimens_by_class = {
        species: [
            specimen
            for specimen, specimen_species_name in specimen_species.items()
            if specimen_species_name == species
        ]
        for species in true_classes
    }
    rng = np.random.default_rng(seed)
    confusion_counts = np.zeros(
        (samples, len(all_classes), len(all_classes)), dtype=np.int64
    )
    for species, specimen_codes in specimens_by_class.items():
        if not specimen_codes:
            raise ValueError(f"No specimen groups found for {species}.")
        sampled_indices = rng.integers(
            0, len(specimen_codes), size=(samples, len(specimen_codes))
        )
        predictions = np.stack(
            [specimen_predictions[specimen] for specimen in specimen_codes]
        )
        confusion_counts[:, class_indices[species], :] = predictions[
            sampled_indices
        ].sum(axis=1)

    true_counts = confusion_counts.sum(axis=2)
    predicted_counts = confusion_counts.sum(axis=1)
    true_positives = np.diagonal(confusion_counts, axis1=1, axis2=2)
    totals = confusion_counts.sum(axis=(1, 2))
    accuracy = np.divide(
        true_positives.sum(axis=1),
        totals,
        out=np.zeros(samples, dtype=np.float64),
        where=totals != 0,
    )
    true_class_indices = np.array(
        [class_indices[species] for species in true_classes], dtype=np.intp
    )
    class_true_positives = true_positives[:, true_class_indices]
    class_true_counts = true_counts[:, true_class_indices]
    class_predicted_counts = predicted_counts[:, true_class_indices]
    precision = np.divide(
        class_true_positives,
        class_predicted_counts,
        out=np.zeros_like(class_true_positives, dtype=np.float64),
        where=class_predicted_counts != 0,
    )
    recall = np.divide(
        class_true_positives,
        class_true_counts,
        out=np.zeros_like(class_true_positives, dtype=np.float64),
        where=class_true_counts != 0,
    )
    f1 = np.divide(
        2 * class_true_positives,
        class_true_counts + class_predicted_counts,
        out=np.zeros_like(class_true_positives, dtype=np.float64),
        where=(class_true_counts + class_predicted_counts) != 0,
    )
    total_squared = totals**2
    numerator = (
        np.trace(confusion_counts, axis1=1, axis2=2) * totals
        - (true_counts * predicted_counts).sum(axis=1)
    )
    denominator = np.sqrt(
        (total_squared - (predicted_counts**2).sum(axis=1))
        * (total_squared - (true_counts**2).sum(axis=1))
    )
    mcc = np.divide(
        numerator,
        denominator,
        out=np.zeros(samples, dtype=np.float64),
        where=denominator != 0,
    )

    distributions = {
        "accuracy": accuracy,
        "precision_macro": precision.mean(axis=1),
        "recall_macro": recall.mean(axis=1),
        "f1_macro": f1.mean(axis=1),
        "mcc": mcc,
    }

    return {
        metric: (
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        )
        for metric, values in distributions.items()
    }


def metric_row(
    architecture: str,
    level: str,
    cohort: str,
    frame: pd.DataFrame,
    true_column: str,
    predicted_column: str,
    bootstrap_samples: int | None,
    seed: int,
) -> Dict[str, Any]:
    """Build one metric row with specimen-cluster bootstrap intervals."""
    metrics = calculate_metrics(frame, true_column, predicted_column)
    intervals = (
        bootstrap_specimen_intervals(
            frame,
            true_column,
            predicted_column,
            bootstrap_samples,
            seed,
        )
        if bootstrap_samples is not None
        else {}
    )
    row: Dict[str, Any] = {
        "architecture": architecture,
        "level": level,
        "cohort": cohort,
        "n_predictions": len(frame),
        "n_specimens": frame["specimen_code"].nunique(),
        **metrics,
    }
    for name, (low, high) in intervals.items():
        row[f"{name}_ci95_low"] = low
        row[f"{name}_ci95_high"] = high
    return row


def class_metric_rows(
    frame: pd.DataFrame,
    architecture: str,
    cohort: str,
    true_column: str,
    predicted_column: str,
) -> List[Dict[str, Any]]:
    """Return one precision, recall, and F1 row per class with test examples."""
    rows = []
    for species, group in frame.groupby(true_column, sort=True):
        y_true = frame[true_column].astype(str).to_numpy()
        y_pred = frame[predicted_column].astype(str).to_numpy()
        rows.append(
            {
                "architecture": architecture,
                "cohort": cohort,
                "species": species,
                "support": len(group),
                "specimens": group["specimen_code"].nunique(),
                "precision": float(
                    precision_score(
                        y_true, y_pred, labels=[species], average="macro", zero_division=0
                    )
                ),
                "recall": float(
                    recall_score(
                        y_true, y_pred, labels=[species], average="macro", zero_division=0
                    )
                ),
                "f1": float(
                    f1_score(
                        y_true, y_pred, labels=[species], average="macro", zero_division=0
                    )
                ),
            }
        )
    return rows


def save_confusion_outputs(
    frame: pd.DataFrame,
    class_names: Sequence[str],
    architecture: str,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """Save a labeled confusion matrix, grayscale plot, and top confused pairs."""
    y_true = frame["true_species"].astype(str)
    y_pred = frame["prediction"].astype(str)
    matrix = confusion_matrix(y_true, y_pred, labels=list(class_names))
    matrix_frame = pd.DataFrame(matrix, index=class_names, columns=class_names)
    matrix_frame.index.name = "true_species"
    matrix_frame.to_csv(output_dir / f"confusion_matrix_{architecture}.csv")

    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8})
    figure, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(matrix, cmap="Greys", interpolation="nearest")
    figure.colorbar(image, ax=axis, label="Image count")
    axis.set(
        title=f"{architecture.replace('_', ' ').title()} — independent-specimen tests",
        xlabel="Predicted species",
        ylabel="True species",
    )
    axis.set_xticks(np.arange(len(class_names)), labels=class_names, rotation=90)
    axis.set_yticks(np.arange(len(class_names)), labels=class_names)
    figure.tight_layout()
    figure.savefig(
        output_dir
        / f"Figure_{'4.1' if architecture == 'shufflenet_v2' else '4.2'}_confusion_{architecture}.png",
        dpi=300,
        facecolor="white",
    )
    plt.close(figure)

    pairs: List[Dict[str, Any]] = []
    for first in range(len(class_names)):
        for second in range(first + 1, len(class_names)):
            count = int(matrix[first, second] + matrix[second, first])
            if count:
                pairs.append(
                    {
                        "architecture": architecture,
                        "species_pair": f"{class_names[first]} / {class_names[second]}",
                        "errors_first_to_second": int(matrix[first, second]),
                        "errors_second_to_first": int(matrix[second, first]),
                        "combined_errors": count,
                    }
                )
    return sorted(pairs, key=lambda row: (-row["combined_errors"], row["species_pair"]))


def save_training_curves(training_dir: Path, output_dir: Path) -> None:
    """Plot all fold train and validation loss curves for both architectures."""
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10})
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, architecture in zip(axes, ("shufflenet_v2", "resnet18")):
        for fold in (1, 2, 3):
            history = pd.read_csv(
                training_dir / "folds" / architecture / f"fold_{fold}" / "history.csv"
            )
            history["total_epoch"] = np.where(
                history["stage"] == 1,
                history["epoch"],
                history["epoch"] + int(
                    history.loc[history["stage"] == 1, "epoch"].max()
                ),
            )
            axis.plot(
                history["total_epoch"],
                history["train_loss"],
                linestyle="--",
                alpha=0.65,
                label=f"Fold {fold} train",
            )
            axis.plot(
                history["total_epoch"],
                history["val_loss"],
                label=f"Fold {fold} validation",
            )
        axis.set(title=architecture.replace("_", " ").title(), xlabel="Epoch")
        axis.grid(color="0.85", linewidth=0.5)
        axis.legend(fontsize=7)
    axes[0].set_ylabel("Weighted cross-entropy loss")
    figure.tight_layout()
    figure.savefig(
        output_dir / "Figure_4.3_training_curves.png",
        dpi=300,
        facecolor="white",
    )
    plt.close(figure)


def write_confidence_curve(
    frames: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Report OOF patch-agreement thresholds, coverage, and accepted accuracy."""
    rows = []
    for architecture, frame in frames.items():
        for threshold in sorted(frame["confidence_pct"].unique()):
            accepted = frame[frame["confidence_pct"] >= threshold]
            rows.append(
                {
                    "architecture": architecture,
                    "minimum_patch_agreement_pct": float(threshold),
                    "accepted_images": len(accepted),
                    "total_independent_test_images": len(frame),
                    "accepted_share_pct": (
                        100.0 * len(accepted) / len(frame) if len(frame) else 0.0
                    ),
                    "accuracy_on_accepted": (
                        float((accepted["true_species"] == accepted["prediction"]).mean())
                        if len(accepted)
                        else None
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(
        output_dir / "confidence_threshold_curve.csv", index=False
    )


def write_sharpness_distribution(
    image_mapping: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, float]:
    """Report sharpness measured during source-image verification."""
    values = image_mapping["sharpness"].to_numpy(dtype=float)
    if len(values) != 900 or not np.isfinite(values).all():
        raise ValueError("Expected finite sharpness values for all 900 training images.")
    image_mapping[
        ["manifest_path", "image_path", "sharpness"]
    ].to_csv(output_dir / "training_image_sharpness.csv", index=False)
    percentiles = {
        str(percentile): float(np.percentile(values, percentile))
        for percentile in (1, 5, 10, 25, 50, 75, 90, 95, 99)
    }
    pd.DataFrame(
        [
            {"percentile": int(percentile), "sharpness": value}
            for percentile, value in percentiles.items()
        ]
    ).to_csv(output_dir / "sharpness_percentiles.csv", index=False)
    return percentiles


def load_and_validate_summary(
    summary_path: Path,
    species_csv: Path,
) -> Tuple[Dict[str, Any], List[str]]:
    """Validate training summary class order against the canonical species CSV."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    class_names = load_class_names(species_csv)
    if summary.get("class_names") != class_names:
        raise ValueError("Training summary class order differs from configs/species.csv.")
    if summary.get("device") not in {"cuda", "cpu"}:
        raise ValueError(
            f"Training summary has an unrecognised device: {summary.get('device')!r}."
        )
    for architecture in ("shufflenet_v2", "resnet18"):
        if architecture not in summary.get("final", {}):
            raise ValueError(f"Training summary has no final {architecture} result.")
        if summary["final"][architecture].get("class_names") != class_names:
            raise ValueError(f"Final {architecture} class order does not match config.")
        for fold in (1, 2, 3):
            key = f"{architecture}_fold_{fold}"
            if key not in summary.get("folds", {}):
                raise ValueError(f"Training summary is missing {key}.")
            if summary["folds"][key].get("class_names") != class_names:
                raise ValueError(f"{key} class order does not match config.")
    return summary, class_names


def evaluate(
    manifest_path: Path,
    splits_path: Path,
    species_csv: Path,
    backup_dir: Path,
    training_dir: Path,
    output_dir: Path,
    device_name: str = "auto",
    batch_size: int = 128,
    num_workers: int = 2,
    bootstrap_samples: int = 1000,
    resume: bool = False,
) -> Dict[str, Any]:
    """Run full image- and patch-level evaluation for approved test folds."""
    mapping_path = output_dir / "input_image_mapping.csv"
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(f"Evaluation output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, class_names = load_and_validate_summary(
        training_dir / "training_summary.json", species_csv
    )
    manifest = pd.read_csv(manifest_path)
    splits = pd.read_csv(splits_path)
    if len(manifest) != 900 or len(class_names) != 29:
        raise ValueError(
            f"Expected 900 manifest images and 29 classes; got {len(manifest)} and {len(class_names)}."
        )
    if manifest["path"].tolist() != splits["path"].tolist():
        raise ValueError("Manifest and saved split file paths or ordering do not match.")
    if manifest["sha256"].tolist() != splits["sha256"].tolist():
        raise ValueError("Manifest and saved split file hashes do not match.")
    for fold in (1, 2, 3):
        leak_audit(splits, splits[f"fold_{fold}_role"], fold)

    if resume and mapping_path.is_file():
        image_mapping = pd.read_csv(mapping_path)
        if (
            len(image_mapping) != len(manifest)
            or image_mapping["manifest_path"].tolist() != manifest["path"].tolist()
            or image_mapping["sha256"].tolist() != manifest["sha256"].tolist()
            or not image_mapping["image_path"].map(lambda path: Path(path).is_file()).all()
        ):
            raise ValueError(
                "Saved verified image mapping does not match the manifest or available files."
            )
        resolved_manifest = manifest.copy()
        resolved_manifest["image_path"] = image_mapping["image_path"]
        print(
            "Reusing the previously verified input mapping; images are not re-hashed.",
            flush=True,
        )
    else:
        resolved_manifest, image_mapping = resolve_and_verify_images(manifest, backup_dir)
        image_mapping.to_csv(mapping_path, index=False)
    target_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_name == "auto"
        else torch.device(device_name)
    )
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation was requested but CUDA is unavailable.")
    if target_device.type == "cpu":
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    image_prediction_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    patch_prediction_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    within_test_count_by_arch: Dict[str, int] = {}
    image_chunk_size = 32
    architecture_inference_config: Dict[str, Dict[str, Any]] = {}
    for architecture in ("shufflenet_v2", "resnet18"):
        final_preprocess_path = (
            training_dir / "final" / architecture / "preprocess_config.json"
        )
        final_preprocess = (
            json.loads(final_preprocess_path.read_text(encoding="utf-8"))
            if final_preprocess_path.is_file()
            else {}
        )
        head_type = str(final_preprocess.get("head_type", "legacy"))
        hidden_dim = int(final_preprocess.get("hidden_dim", 256))
        aggregation = str(final_preprocess.get("aggregation", "majority"))
        tta_views = tuple(final_preprocess.get("tta_views", []))
        architecture_inference_config[architecture] = {
            "head_type": head_type,
            "hidden_dim": hidden_dim,
            "aggregation": aggregation,
            "tta_views": list(tta_views),
        }
        for fold in (1, 2, 3):
            image_output_path = (
                output_dir / f"fold_image_predictions_{architecture}_fold_{fold}.csv"
            )
            patch_output_path = (
                output_dir / f"fold_patch_predictions_{architecture}_fold_{fold}.csv"
            )
            if resume and image_output_path.is_file() and patch_output_path.is_file():
                image_prediction_rows[architecture].extend(
                    pd.read_csv(image_output_path).to_dict(orient="records")
                )
                patch_prediction_rows[architecture].extend(
                    pd.read_csv(patch_output_path).to_dict(orient="records")
                )
                print(
                    f"Reusing saved {architecture} fold {fold} predictions.",
                    flush=True,
                )
                continue
            role_column = f"fold_{fold}_role"
            eval_column = f"fold_{fold}_eval_group"
            selected = splits.index[splits[role_column] == "test"]
            images = resolved_manifest.loc[selected].copy()
            images["fold"] = fold
            images["eval_group"] = splits.loc[selected, eval_column].fillna("").to_numpy()
            if len(images) != int(
                summary["folds"][f"{architecture}_fold_{fold}"]["test_images"]
            ):
                raise ValueError(f"{architecture} fold {fold} test image count mismatch.")
            if images["eval_group"].eq("").any():
                raise ValueError(f"Fold {fold} contains test images without evaluation type.")

            model_path = training_dir / "folds" / architecture / f"fold_{fold}" / "model.pt"
            metadata_path = model_path.with_name("training_metadata.json")
            if not model_path.is_file() or not metadata_path.is_file():
                raise FileNotFoundError(f"Missing fold checkpoint or metadata: {model_path}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected_metadata = summary["folds"][f"{architecture}_fold_{fold}"]
            for key in ("architecture", "fold", "class_names", "test_images"):
                if metadata.get(key) != expected_metadata.get(key):
                    raise ValueError(f"{model_path}: metadata mismatch for {key}.")
            for key, expected in (
                ("head_type", head_type),
                ("hidden_dim", hidden_dim),
            ):
                if key in metadata and metadata[key] != expected:
                    raise ValueError(
                        f"{model_path}: fold {key} does not match the final "
                        f"inference configuration."
                    )

            model = get_model(
                architecture,
                num_classes=len(class_names),
                pretrained=False,
                dropout=0.5,
                head_type=head_type,
                hidden_dim=hidden_dim,
            )
            state = torch.load(model_path, map_location=target_device, weights_only=True)
            model.load_state_dict(state, strict=True)
            model.to(target_device)
            image_rows: List[Dict[str, Any]] = []
            patch_rows: List[Dict[str, Any]] = []
            chunks = [
                (start, images.iloc[start : start + image_chunk_size].copy())
                for start in range(0, len(images), image_chunk_size)
            ]
            for chunk_number, (start, chunk_images) in enumerate(chunks, 1):
                image_chunk_path = output_dir / (
                    f"fold_image_predictions_{architecture}_fold_{fold}_"
                    f"chunk_{start:04d}.csv"
                )
                patch_chunk_path = output_dir / (
                    f"fold_patch_predictions_{architecture}_fold_{fold}_"
                    f"chunk_{start:04d}.csv"
                )
                if (
                    resume
                    and image_chunk_path.is_file()
                    and patch_chunk_path.is_file()
                ):
                    image_rows.extend(
                        pd.read_csv(image_chunk_path).to_dict(orient="records")
                    )
                    patch_rows.extend(
                        pd.read_csv(patch_chunk_path).to_dict(orient="records")
                    )
                else:
                    chunk_image_rows, chunk_patch_rows = predict_test_images(
                        model,
                        chunk_images,
                        class_names,
                        target_device,
                        batch_size,
                        num_workers,
                        tta_views=tta_views,
                        aggregation=aggregation,
                    )
                    for row in chunk_image_rows:
                        row["architecture"] = architecture
                    for row in chunk_patch_rows:
                        row["architecture"] = architecture
                    pd.DataFrame(chunk_image_rows).to_csv(
                        image_chunk_path, index=False
                    )
                    pd.DataFrame(chunk_patch_rows).to_csv(
                        patch_chunk_path, index=False
                    )
                    image_rows.extend(chunk_image_rows)
                    patch_rows.extend(chunk_patch_rows)
                print(
                    f"{architecture} fold {fold}: completed image chunk "
                    f"{chunk_number}/{len(chunks)} "
                    f"({min(start + image_chunk_size, len(images))}/{len(images)} images).",
                    flush=True,
                )
            pd.DataFrame(image_rows).to_csv(image_output_path, index=False)
            pd.DataFrame(patch_rows).to_csv(patch_output_path, index=False)
            image_prediction_rows[architecture].extend(image_rows)
            patch_prediction_rows[architecture].extend(patch_rows)
            print(
                f"Evaluated {architecture} fold {fold}: {len(images)} images "
                f"({int((images['eval_group'] == 'independent_specimen').sum())} "
                "independent-specimen test images).",
                flush=True,
            )
            del model
            if target_device.type == "cuda":
                torch.cuda.empty_cache()

    independent_by_arch: Dict[str, pd.DataFrame] = {}
    metric_rows: List[Dict[str, Any]] = []
    class_rows: List[Dict[str, Any]] = []
    confused_rows: List[Dict[str, Any]] = []
    for architecture in ("shufflenet_v2", "resnet18"):
        images = pd.DataFrame(image_prediction_rows[architecture])
        independent = images[images["eval_group"] == "independent_specimen"].copy()
        within = images[
            (images["eval_group"] == "within_specimen") & (images["fold"] == 1)
        ].copy()
        if independent.empty:
            raise ValueError(f"No independent-specimen test predictions for {architecture}.")
        if independent["manifest_path"].duplicated().any():
            raise ValueError(
                f"{architecture} has duplicate independent-specimen image predictions."
            )
        within_test_count_by_arch[architecture] = len(within)
        independent_by_arch[architecture] = independent

        metric_rows.append(
            metric_row(
                architecture, "image", "independent_specimen", independent,
                "true_species", "prediction", bootstrap_samples, 4200,
            )
        )
        class_rows.extend(
            class_metric_rows(
                independent, architecture, "independent_specimen",
                "true_species", "prediction",
            )
        )
        if not within.empty:
            metric_rows.append(
                metric_row(
                    architecture, "image", "within_specimen (optimistic)", within,
                    "true_species", "prediction", None, 4201,
                )
            )
            class_rows.extend(
                class_metric_rows(
                    within, architecture, "within_specimen (optimistic)",
                    "true_species", "prediction",
                )
            )

        independent_paths = set(independent["manifest_path"])
        patches = pd.DataFrame(patch_prediction_rows[architecture])
        independent_patches = patches[patches["manifest_path"].isin(independent_paths)]
        metric_rows.append(
            metric_row(
                architecture, "patch", "independent_specimen", independent_patches,
                "true_species", "predicted_species", bootstrap_samples, 4202,
            )
        )
        if not within.empty:
            within_paths = set(within["manifest_path"])
            within_patches = patches[patches["manifest_path"].isin(within_paths)]
            metric_rows.append(
                metric_row(
                    architecture, "patch", "within_specimen (optimistic)", within_patches,
                    "true_species", "predicted_species", None, 4203,
                )
            )
        confused_rows.extend(
            save_confusion_outputs(independent, class_names, architecture, output_dir)
        )

        images.to_csv(
            output_dir / f"all_test_predictions_{architecture}.csv", index=False
        )
        independent.to_csv(
            output_dir / f"independent_image_predictions_{architecture}.csv",
            index=False,
        )
        patches.to_csv(
            output_dir / f"all_patch_predictions_{architecture}.csv", index=False
        )

    all_images = pd.concat(
        [pd.DataFrame(image_prediction_rows[name]) for name in independent_by_arch],
        ignore_index=True,
    )
    all_patches = pd.concat(
        [pd.DataFrame(patch_prediction_rows[name]) for name in independent_by_arch],
        ignore_index=True,
    )
    all_images.to_csv(output_dir / "all_test_predictions.csv", index=False)
    all_patches.to_csv(output_dir / "all_patch_predictions.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(output_dir / "overall_metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(output_dir / "per_species_metrics.csv", index=False)
    pd.DataFrame(confused_rows).to_csv(output_dir / "most_confused_pairs.csv", index=False)
    write_confidence_curve(independent_by_arch, output_dir)
    sharpness = write_sharpness_distribution(image_mapping, output_dir)
    save_training_curves(training_dir, output_dir)

    comparison_rows = []
    for architecture in ("shufflenet_v2", "resnet18"):
        model_info_path = training_dir / "final" / architecture / "model_info.json"
        model_info = json.loads(model_info_path.read_text(encoding="utf-8"))
        comparison_rows.append(
            {
                "architecture": architecture,
                "parameters": int(model_info["parameters"]),
                "pytorch_size_mb": round(
                    int(model_info["pytorch_size_bytes"]) / (1024 * 1024), 2
                ),
                "onnx_size_mb": round(
                    int(model_info["onnx_size_bytes"]) / (1024 * 1024), 2
                ),
            }
        )
    pd.DataFrame(comparison_rows).to_csv(
        output_dir / "model_comparison.csv", index=False
    )

    for architecture, frame in independent_by_arch.items():
        if frame["manifest_path"].nunique() != len(frame):
            raise ValueError("Independent OOF images are not unique.")
    result_summary = {
        "evaluation_device": str(target_device),
        "inference_config": architecture_inference_config,
        "training_summary": str(training_dir / "training_summary.json"),
        "independent_test_images": {
            architecture: len(frame)
            for architecture, frame in independent_by_arch.items()
        },
        "within_specimen_test_images_fold_1_only": {
            architecture: count
            for architecture, count in within_test_count_by_arch.items()
        },
        "confidence_threshold_selected": None,
        "threshold_note": (
            "Threshold candidates are in confidence_threshold_curve.csv; no "
            "accepted-accuracy target was specified, so no threshold was selected."
        ),
        "sharpness_percentiles_final_model_training_images": sharpness,
        "confidence_intervals": "95% percentile bootstrap, resampling specimen codes within species",
        "macro_metrics": "Averaged over species represented in the corresponding test cohort.",
        "within_specimen_interval_note": (
            "Within-specimen results are reported separately without confidence "
            "intervals because each class contributes only one specimen."
        ),
    }
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(result_summary, indent=2), encoding="utf-8"
    )
    return result_summary


def main() -> None:
    """Parse command-line options and run Phase 3 evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained timber species fold models."
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits.csv"))
    parser.add_argument("--species-csv", type=Path, default=Path("configs/species.csv"))
    parser.add_argument(
        "--backup-dir", type=Path, default=Path("patch_ready_dataset_raw_backup")
    )
    parser.add_argument(
        "--training-dir", type=Path, default=Path("artifacts/p2_training")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/p3_evaluation")
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse a previously verified image mapping and completed fold predictions.",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("Batch size must be positive and num-workers non-negative.")
    result = evaluate(
        args.manifest,
        args.splits,
        args.species_csv,
        args.backup_dir,
        args.training_dir,
        args.output_dir,
        args.device,
        args.batch_size,
        args.num_workers,
        args.bootstrap_samples,
        args.resume,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
