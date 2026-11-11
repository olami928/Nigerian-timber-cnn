"""Tests for the shared primary-model inference path."""

import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.infer import (
    DEFAULT_MANIFEST,
    DEFAULT_MODEL_DIR,
    DEFAULT_SPECIES_INFO,
    TimberPredictor,
    load_predictor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_PATH = PROJECT_ROOT / "data" / "splits.csv"
MAPPING_PATH = PROJECT_ROOT / "artifacts" / "p3_evaluation" / "input_image_mapping.csv"
PARITY_INPUTS_AVAILABLE = all(
    path.is_file()
    for path in (
        DEFAULT_MODEL_DIR / "model.pt",
        DEFAULT_MODEL_DIR / "class_names.json",
        DEFAULT_MODEL_DIR / "preprocess_config.json",
        DEFAULT_MODEL_DIR / "model_info.json",
        DEFAULT_MODEL_DIR / "training_metadata.json",
        DEFAULT_SPECIES_INFO,
        DEFAULT_MANIFEST,
        SPLITS_PATH,
        MAPPING_PATH,
    )
)


class TestInference(unittest.TestCase):
    """Validate metadata lookup, image guards, and notebook prediction parity."""

    def test_species_lookup_keeps_only_valid_uses(self) -> None:
        lookup = TimberPredictor._load_species_lookup(
            {
                "species": [
                    {
                        "slug": "sample",
                        "local_name": "Sample",
                        "botanical": "Sample species",
                        "english_name": None,
                        "wood_type": "Hardwood",
                        "uses": {
                            "construction": "primary",
                            "furniture": "secondary",
                            "uncited": "unverified",
                        },
                        "sources": ["must not be returned"],
                    }
                ]
            }
        )
        self.assertEqual(
            lookup["sample"]["documented_uses"],
            {"construction": "primary", "furniture": "secondary"},
        )
        self.assertNotIn("sources", lookup["sample"])

    def test_predict_rejects_invalid_image_shape_and_dtype(self) -> None:
        predictor = TimberPredictor.__new__(TimberPredictor)
        predictor.grid_rows = 6
        predictor.grid_cols = 8
        predictor.preprocess = {"min_sharpness": 171.4}
        with self.assertRaisesRegex(ValueError, "8-bit, three-channel BGR"):
            predictor.predict(np.zeros((12, 16, 3), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "8-bit, three-channel BGR"):
            predictor.predict(np.zeros((12, 16), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "too small"):
            predictor.predict(np.zeros((5, 16, 3), dtype=np.uint8))
        self.assertEqual(
            predictor.predict(np.zeros((12, 16, 3), dtype=np.uint8)),
            {"status": "retake", "reason": "blurry"},
        )

    @unittest.skipUnless(
        PARITY_INPUTS_AVAILABLE,
        "P2 exports, verified images, or saved split artifacts are unavailable.",
    )
    def test_predict_matches_notebook_reference_on_20_held_out_images(self) -> None:
        """Compare model outputs against the original notebook's prediction steps."""
        predictor = load_predictor(device="cpu")
        splits = pd.read_csv(SPLITS_PATH)
        independent_test = pd.concat(
            [
                splits[
                    (splits[f"fold_{fold}_role"] == "test")
                    & (
                        splits[f"fold_{fold}_eval_group"]
                        == "independent_specimen"
                    )
                ]
                for fold in (1, 2, 3)
            ],
            ignore_index=True,
        )
        mapping = pd.read_csv(MAPPING_PATH).set_index("manifest_path")
        independent_test = independent_test[
            independent_test["path"].map(mapping["sharpness"]) >= 171.4
        ].sample(n=20, random_state=42)

        original_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            for row in independent_test.itertuples(index=False):
                image_path = Path(mapping.loc[row.path, "image_path"])
                image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                self.assertIsNotNone(image_bgr, f"Could not decode {image_path}")
                result = predictor.predict(image_bgr)
                expected = notebook_reference(image_path, predictor)

                self.assertEqual(result["species"], expected["species"])
                self.assertEqual(result["confidence_pct"], expected["confidence_pct"])
                self.assertEqual(result["patch_agreement"], expected["patch_agreement"])
                self.assertEqual(result["mean_probability_pct"], expected["mean_probability_pct"])
                self.assertEqual(result["top3"], expected["top3"])
        finally:
            torch.set_num_threads(original_threads)


def notebook_reference(
    image_path: Path,
    predictor: TimberPredictor,
) -> dict[str, object]:
    """Reproduce the original notebook's patch inference and vote aggregation."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    patch_width = image.width // 8
    patch_height = image.height // 6
    patches = []
    for row in range(6):
        for column in range(8):
            box = (
                column * patch_width,
                row * patch_height,
                (column + 1) * patch_width,
                (row + 1) * patch_height,
            )
            patches.append(predictor.transform(image.crop(box)))

    with torch.inference_mode():
        logits = predictor.model(torch.stack(patches).to(predictor.device)).cpu().numpy()
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    votes = probabilities.argmax(axis=1)
    counts = np.bincount(votes, minlength=len(predictor.class_names))
    mean_probabilities = probabilities.mean(axis=0)
    scores = counts + 1e-3 * mean_probabilities
    order = np.argsort(scores)[::-1]
    winner = int(order[0])
    return {
        "species": predictor.class_names[winner],
        "confidence_pct": round(100.0 * int(counts[winner]) / 48, 1),
        "patch_agreement": f"{int(counts[winner])}/48",
        "mean_probability_pct": round(100.0 * float(mean_probabilities[winner]), 1),
        "top3": [
            {
                "species": predictor.class_names[int(index)],
                "vote_pct": round(100.0 * int(counts[index]) / 48, 1),
                "prob_pct": round(100.0 * float(mean_probabilities[index]), 1),
            }
            for index in order[:3]
        ],
    }


if __name__ == "__main__":
    unittest.main()
