"""Shared 48-patch inference for the Nigerian timber species classifier."""

import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.aggregate import aggregate_probabilities, forward_probabilities
from src.model import get_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "artifacts" / "p2_training" / "final" / "shufflenet_v2"
DEFAULT_SPECIES_INFO = PROJECT_ROOT / "configs" / "species_enduse.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifest.csv"


class TimberPredictor:
    """Load the primary trained model and predict species from a BGR image."""

    def __init__(
        self,
        model_dir: Path = DEFAULT_MODEL_DIR,
        species_info_path: Path = DEFAULT_SPECIES_INFO,
        manifest_path: Path = DEFAULT_MANIFEST,
        device: str | torch.device = "auto",
    ) -> None:
        """Validate model metadata and load model, preprocessing, and lookups."""
        self.model_dir = Path(model_dir)
        self.device = self._resolve_device(device)
        self.preprocess = self._load_json(self.model_dir / "preprocess_config.json")
        model_info = self._load_json(self.model_dir / "model_info.json")
        metadata = self._load_json(self.model_dir / "training_metadata.json")
        self.class_names = self._load_json(self.model_dir / "class_names.json")
        self._validate_model_metadata(model_info, metadata)

        with Path(species_info_path).open(encoding="utf-8") as file:
            species_document = json.load(file)
        self.species_lookup = self._load_species_lookup(species_document)
        self.specimen_counts = self._load_specimen_counts(Path(manifest_path))
        if set(self.class_names) != set(self.species_lookup):
            raise ValueError("Model classes do not match the documented species lookup.")
        if set(self.class_names) != set(self.specimen_counts):
            raise ValueError("Model classes do not match the training manifest.")

        self.grid_rows = int(self.preprocess["grid_rows"])
        self.grid_cols = int(self.preprocess["grid_cols"])
        patch_height, patch_width = self.preprocess["patch_size"]
        normalization = self.preprocess["normalization"]
        self.transform = transforms.Compose(
            [
                transforms.Resize((int(patch_height), int(patch_width))),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=normalization["mean"],
                    std=normalization["std"],
                ),
            ]
        )

        self.head_type = str(self.preprocess.get("head_type", "legacy"))
        self.hidden_dim = int(self.preprocess.get("hidden_dim", 256))
        self.aggregation = str(self.preprocess.get("aggregation", "majority"))
        self.tta_views = tuple(self.preprocess.get("tta_views", []))
        self.model = get_model(
            self.preprocess["architecture"],
            num_classes=len(self.class_names),
            pretrained=False,
            dropout=0.5,
            head_type=self.head_type,
            hidden_dim=self.hidden_dim,
        )
        weights_path = self.model_dir / "model.pt"
        state = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_device(device: str | torch.device) -> torch.device:
        """Resolve the requested device, preferring CPU only when asked."""
        if isinstance(device, torch.device):
            return device
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @staticmethod
    def _load_json(path: Path) -> Any:
        """Load a required JSON file with its path included in errors."""
        if not path.is_file():
            raise FileNotFoundError(f"Required inference metadata is missing: {path}")
        with path.open(encoding="utf-8") as file:
            return json.load(file)

    def _validate_model_metadata(
        self,
        model_info: Dict[str, Any],
        training_metadata: Dict[str, Any],
    ) -> None:
        """Reject exports whose architecture, class order, or patch setup differ."""
        architecture = self.preprocess.get("architecture")
        if architecture != "shufflenet_v2":
            raise ValueError("Website inference must use the primary ShuffleNetV2 model.")
        if (
            model_info.get("architecture") != architecture
            or model_info.get("num_classes") != len(self.class_names)
            or training_metadata.get("architecture") != architecture
            or training_metadata.get("class_names") != self.class_names
        ):
            raise ValueError("Model export metadata is inconsistent.")
        if (
            int(self.preprocess.get("grid_rows", 0)) != 6
            or int(self.preprocess.get("grid_cols", 0)) != 8
            or int(self.preprocess.get("num_patches", 0)) != 48
            or len(self.class_names) != 29
            or len(set(self.class_names)) != len(self.class_names)
        ):
            raise ValueError("Model export does not match the approved 6x8, 29-class setup.")
        if self.preprocess.get("model_version") is None:
            raise ValueError("Model export has no model version.")

    @staticmethod
    def _load_species_lookup(document: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Load only documented uses and public species names, excluding sources."""
        lookup: Dict[str, Dict[str, Any]] = {}
        for item in document.get("species", []):
            uses = item.get("uses", {})
            valid_uses = {
                category: status
                for category, status in uses.items()
                if status in {"primary", "secondary"}
            }
            lookup[item["slug"]] = {
                "local_name": item["local_name"],
                "botanical": item["botanical"],
                "english_name": item.get("english_name"),
                "wood_type": item["wood_type"],
                "documented_uses": valid_uses,
            }
        return lookup

    @staticmethod
    def _load_specimen_counts(manifest_path: Path) -> Dict[str, int]:
        """Count distinct training specimens per species from the canonical manifest."""
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Training manifest does not exist: {manifest_path}")
        import pandas as pd

        manifest = pd.read_csv(manifest_path, usecols=["species", "specimen_code"])
        counts = manifest.groupby("species")["specimen_code"].nunique().astype(int)
        if (counts < 1).any():
            raise ValueError("The training manifest contains a species with no specimens.")
        return counts.to_dict()

    def _make_patch_batch(self, image_rgb: Image.Image) -> torch.Tensor:
        """Cut raster-order grid patches and apply the notebook evaluation transform."""
        if image_rgb.width < self.grid_cols or image_rgb.height < self.grid_rows:
            raise ValueError("Image is too small to divide into the required 6x8 grid.")
        patch_width = image_rgb.width // self.grid_cols
        patch_height = image_rgb.height // self.grid_rows
        patches: List[torch.Tensor] = []
        for row in range(self.grid_rows):
            for column in range(self.grid_cols):
                box = (
                    column * patch_width,
                    row * patch_height,
                    (column + 1) * patch_width,
                    (row + 1) * patch_height,
                )
                patches.append(self.transform(image_rgb.crop(box)))
        return torch.stack(patches)

    def predict(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """Return patch-voted species, confidence, and documented lookup details."""
        if (
            not isinstance(image_bgr, np.ndarray)
            or image_bgr.ndim != 3
            or image_bgr.shape[2] != 3
            or image_bgr.dtype != np.uint8
        ):
            raise ValueError("Expected an 8-bit, three-channel BGR image.")
        if image_bgr.shape[1] < self.grid_cols or image_bgr.shape[0] < self.grid_rows:
            raise ValueError("Image is too small to divide into the required 6x8 grid.")

        min_sharpness = self.preprocess.get("min_sharpness")
        if min_sharpness is not None:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if sharpness < float(min_sharpness):
                return {"status": "retake", "reason": "blurry"}

        image_rgb = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        patches = self._make_patch_batch(image_rgb)
        probabilities = forward_probabilities(
            self.model,
            patches,
            self.device,
            tta_views=self.tta_views,
        )
        if probabilities.shape[1] != len(self.class_names):
            raise RuntimeError(
                f"Model returned probabilities with unexpected shape: "
                f"{probabilities.shape}."
            )
        result = aggregate_probabilities(
            probabilities,
            self.class_names,
            mode=self.aggregation,
        )
        winner_name = str(result["prediction"])
        confidence_pct = float(result["confidence_pct"])
        top3 = [
            {
                "species": item["species"],
                "vote_pct": item["vote_pct"],
                "prob_pct": item["probability_pct"],
            }
            for item in result["top3"]
        ]

        min_confidence = self.preprocess.get("min_confidence_pct")
        if min_confidence is not None and confidence_pct < float(min_confidence):
            return {
                "status": "retake",
                "reason": "low_confidence",
                "top3": top3,
            }

        return {
            "status": "ok",
            "species": winner_name,
            "confidence_pct": confidence_pct,
            "patch_agreement": result["patch_agreement"],
            "mean_probability_pct": result["mean_probability_pct"],
            "top3": top3,
            "limited_reference_data": self.specimen_counts[winner_name] < 3,
            "model_version": self.preprocess["model_version"],
            **self.species_lookup[winner_name],
        }


_default_predictor: TimberPredictor | None = None


def load_predictor(
    model_dir: Path = DEFAULT_MODEL_DIR,
    species_info_path: Path = DEFAULT_SPECIES_INFO,
    manifest_path: Path = DEFAULT_MANIFEST,
    device: str | torch.device = "auto",
) -> TimberPredictor:
    """Create an explicitly configured predictor instance."""
    return TimberPredictor(model_dir, species_info_path, manifest_path, device)


def predict(image_bgr: np.ndarray) -> Dict[str, Any]:
    """Run the lazily loaded primary-model predictor on an OpenCV BGR image."""
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = load_predictor()
    return _default_predictor.predict(image_bgr)
