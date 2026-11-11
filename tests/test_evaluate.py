"""Tests for Phase 3 evaluation helpers."""

import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from PIL import Image

from src.evaluate import (
    PatchEvaluationDataset,
    aggregate_patch_logits,
    bootstrap_specimen_intervals,
    calculate_metrics,
    class_metric_rows,
    resolve_and_verify_images,
)


class TestEvaluationHelpers(unittest.TestCase):
    """Exercise deterministic aggregation, input verification, and bootstrap."""

    def test_patch_aggregation_uses_vote_agreement(self) -> None:
        logits = np.zeros((1, 48, 3), dtype=np.float32)
        logits[:, :, 1] = 2.0
        result = aggregate_patch_logits(logits, ["afara", "agboin", "apa"])[0]
        self.assertEqual(result["prediction"], "agboin")
        self.assertEqual(result["confidence_pct"], 100.0)
        self.assertEqual(result["patch_agreement"], "48/48")
        self.assertEqual(result["top3"][0]["species"], "agboin")

    def test_soft_vote_can_override_hard_majority_with_confidence(self) -> None:
        logits = np.zeros((1, 48, 3), dtype=np.float32)
        logits[0, :29, 1] = 1.0
        logits[0, 29:, 0] = 100.0
        majority = aggregate_patch_logits(
            logits, ["afara", "agboin", "apa"], mode="majority"
        )[0]
        soft = aggregate_patch_logits(
            logits, ["afara", "agboin", "apa"], mode="soft"
        )[0]
        self.assertEqual(majority["prediction"], "agboin")
        self.assertEqual(soft["prediction"], "afara")

    def test_backup_resolution_checks_hash_and_grid(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "backup" / "afara" / "image.png"
            image_path.parent.mkdir(parents=True)
            image = Image.fromarray(
                np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
            )
            image.save(image_path)
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            manifest = pd.DataFrame(
                [
                    {
                        "path": "patch_ready_dataset/afara/image.png",
                        "sha256": digest,
                        "width": 16,
                        "height": 12,
                    }
                ]
            )
            resolved, mapping = resolve_and_verify_images(
                manifest,
                root / "backup",
            )
            self.assertEqual(resolved.iloc[0]["image_path"], str(image_path))
            self.assertEqual(mapping.iloc[0]["mapping_method"], "relative_path")
            self.assertTrue(np.isfinite(mapping.iloc[0]["sharpness"]))

    def test_metrics_and_specimen_bootstrap_are_deterministic(self) -> None:
        frame = pd.DataFrame(
            {
                "true_species": ["a", "a", "b", "b"],
                "prediction": ["a", "b", "b", "a"],
                "specimen_code": ["a_1", "a_2", "b_1", "b_2"],
            }
        )
        metrics = calculate_metrics(frame, "true_species", "prediction")
        self.assertEqual(metrics["accuracy"], 0.5)
        per_class = class_metric_rows(
            frame, "test", "independent", "true_species", "prediction"
        )
        self.assertEqual(per_class[0]["precision"], 0.5)
        self.assertEqual(per_class[0]["recall"], 0.5)
        first = bootstrap_specimen_intervals(
            frame, "true_species", "prediction", samples=20, seed=7
        )
        second = bootstrap_specimen_intervals(
            frame, "true_species", "prediction", samples=20, seed=7
        )
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(metrics))

    def test_bootstrap_metrics_match_when_each_class_has_one_specimen(self) -> None:
        frame = pd.DataFrame(
            {
                "true_species": ["a", "a", "b", "b"],
                "prediction": ["a", "c", "b", "c"],
                "specimen_code": ["a_1", "a_1", "b_1", "b_1"],
            }
        )
        metrics = calculate_metrics(frame, "true_species", "prediction")
        intervals = bootstrap_specimen_intervals(
            frame, "true_species", "prediction", samples=20, seed=7
        )
        for metric, value in metrics.items():
            self.assertAlmostEqual(intervals[metric][0], value)
            self.assertAlmostEqual(intervals[metric][1], value)

    def test_patch_dataset_reuses_decoded_source_image(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "image.png"
            Image.fromarray(
                np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
            ).save(path)
            images = pd.DataFrame(
                [
                    {
                        "image_path": str(path),
                        "species": "afara",
                    }
                ]
            )
            dataset = PatchEvaluationDataset(
                images,
                {"afara": 0},
                image_size=8,
                grid_rows=2,
                grid_cols=2,
            )
            first = dataset[0][0]
            cached_image = dataset.cached_image
            second = dataset[1][0]
            self.assertIs(dataset.cached_image, cached_image)
            self.assertEqual(tuple(first.shape), (3, 8, 8))
            self.assertEqual(tuple(second.shape), (3, 8, 8))


if __name__ == "__main__":
    unittest.main()
