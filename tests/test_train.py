"""
Unit tests for dataset patch extraction, class weighting, and model export.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from torchvision import transforms

from src.train import (
    TimberPatchDataset,
    build_patch_cache,
    compute_class_weights,
    compute_sample_weights,
    export_final_model,
    get_transforms,
    load_class_names,
    run_training_pipeline,
    train_fold,
)
from src.model import get_model


class TestTrain(unittest.TestCase):

    def setUp(self):
        self.df_splits = pd.read_csv("data/splits.csv")
        self.df_species = pd.read_csv("configs/species.csv")
        self.species_list = (
            self.df_species.sort_values("id", kind="stable")["slug"].tolist()
        )
        self.label_to_idx = {sp: i for i, sp in enumerate(self.species_list)}

    def test_class_names_follow_species_csv_id_order(self):
        self.assertEqual(
            load_class_names(Path("configs/species.csv")),
            self.species_list,
        )

    def test_class_weights(self):
        w = compute_class_weights(self.df_splits, "fold_1_role", len(self.species_list), self.label_to_idx)
        self.assertEqual(len(w), 29)
        self.assertTrue((w > 0).all())

    def test_specimen_class_weights_use_specimens_not_images(self):
        frame = pd.DataFrame(
            {
                "role": ["train"] * 6,
                "species": ["a"] * 2 + ["b"] * 4,
                "specimen_code": ["a_1"] * 2 + ["b_1"] * 4,
                "path": [f"img_{index}.png" for index in range(6)],
            }
        )
        label_to_idx = {"a": 0, "b": 1}
        image_weights = compute_class_weights(
            frame, "role", 2, label_to_idx, weighting="image"
        )
        specimen_weights = compute_class_weights(
            frame, "role", 2, label_to_idx, weighting="specimen"
        )
        self.assertFalse(torch.allclose(image_weights, specimen_weights))
        self.assertTrue(torch.allclose(specimen_weights, torch.ones(2)))

    def test_sample_weights_give_each_species_equal_total_weight(self):
        frame = pd.DataFrame(
            {
                "role": ["train"] * 6,
                "species": ["a"] * 2 + ["b"] * 4,
                "specimen_code": ["a_1"] * 2 + ["b_1"] * 4,
                "path": [f"img_{index}.png" for index in range(6)],
            }
        )
        weights = compute_sample_weights(frame, "role", {"a": 0, "b": 1})
        self.assertEqual(tuple(weights.shape), (6,))
        self.assertTrue((weights > 0).all())
        total_a = float(weights[:2].sum())
        total_b = float(weights[2:].sum())
        self.assertAlmostEqual(total_a, total_b, places=5)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=5)

    def test_optimized_augmentation_adds_regularization_ops(self):
        train_transform, eval_transform = get_transforms(
            img_size=112,
            rotation_degrees=180,
            translation_fraction=0.08,
            scale_min=0.9,
            scale_max=1.1,
            shear_degrees=5,
            color_jitter={
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.2,
                "hue": 0.02,
            },
            gaussian_blur_prob=0.15,
            random_erasing_prob=0.2,
        )
        train_ops = str(train_transform)
        eval_ops = str(eval_transform)
        self.assertIn("ColorJitter", train_ops)
        self.assertIn("GaussianBlur", train_ops)
        self.assertIn("RandomErasing", train_ops)
        self.assertNotIn("RandomErasing", eval_ops)
        self.assertNotIn("ColorJitter", eval_ops)

    def test_patch_dataset_indexing(self):
        ds = TimberPatchDataset(self.df_splits, "fold_1_role", "train", self.label_to_idx)
        # Should have 48 patches per train image
        n_train_imgs = (self.df_splits["fold_1_role"] == "train").sum()
        self.assertEqual(len(ds), n_train_imgs * 48)

    def test_patch_cache_matches_direct_patch_preprocessing(self):
        image_array = np.arange(16 * 12 * 3, dtype=np.uint8).reshape(12, 16, 3)
        transform = transforms.Compose(
            [transforms.Resize((8, 8)), transforms.ToTensor()]
        )
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "source.png"
            cache_path = temp_path / "patch_cache.npy"
            Image.fromarray(image_array).save(image_path)
            manifest = pd.DataFrame(
                [
                    {
                        "path": str(image_path),
                        "width": 16,
                        "height": 12,
                        "species": self.species_list[0],
                        "role": "train",
                    }
                ]
            )
            build_patch_cache(
                manifest,
                cache_path,
                grid_rows=2,
                grid_cols=2,
                image_size=8,
            )
            direct = TimberPatchDataset(
                manifest,
                "role",
                "train",
                self.label_to_idx,
                grid_rows=2,
                grid_cols=2,
                transform=transform,
            )
            cached = TimberPatchDataset(
                manifest,
                "role",
                "train",
                self.label_to_idx,
                grid_rows=2,
                grid_cols=2,
                transform=transform,
                patch_cache_path=cache_path,
            )

            for index in range(len(direct)):
                torch.testing.assert_close(
                    direct[index][0],
                    cached[index][0],
                    rtol=0,
                    atol=0,
                )

    def test_pipeline_resume_reuses_completed_folds_and_final_models(self):
        project_root = Path.cwd()

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(1, 1)

        def fake_dataset(*args, **kwargs):
            del args, kwargs
            return TensorDataset(
                torch.zeros((2, 1)),
                torch.zeros(2, dtype=torch.long),
                torch.arange(2),
            )

        def fake_export(model, class_names, artifacts_dir, arch, img_size=112, **kwargs):
            del class_names, arch, img_size, kwargs
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), artifacts_dir / "model.pt")

        def fake_get_model(*args, **kwargs):
            del args, kwargs
            return TinyModel()

        def fake_train_fold(model, *args, **kwargs):
            del args, kwargs
            return model, {
                "history": [],
                "best_val_loss": 1.0,
                "best_epoch": 2,
                "best_stage2_epoch": 1,
            }

        def fake_train_final(model, *args, **kwargs):
            del args, kwargs
            return model, []

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = yaml.safe_load(
                (project_root / "configs/default.yaml").read_text(encoding="utf-8")
            )
            config["data"]["manifest_path"] = str(project_root / "data/manifest.csv")
            config["data"]["splits_path"] = str(project_root / "data/splits.csv")
            config["data"]["species_csv"] = str(project_root / "configs/species.csv")
            config_path = temp_path / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            output_path = temp_path / "output"

            with (
                patch("src.train.TimberPatchDataset", side_effect=fake_dataset),
                patch("src.train.get_model", side_effect=fake_get_model),
                patch(
                    "src.train.train_fold",
                    side_effect=fake_train_fold,
                ) as mocked_train_fold,
                patch(
                    "src.train.train_final_model",
                    side_effect=fake_train_final,
                ) as mocked_train_final,
                patch("src.train.export_final_model", side_effect=fake_export),
            ):
                run_training_pipeline(config_path, output_path, num_workers=0)
                self.assertEqual(mocked_train_fold.call_count, 6)
                self.assertEqual(mocked_train_final.call_count, 2)

                run_training_pipeline(
                    config_path,
                    output_path,
                    num_workers=0,
                    resume=True,
                )
                self.assertEqual(mocked_train_fold.call_count, 6)
                self.assertEqual(mocked_train_final.call_count, 2)

    def test_model_export(self):
        m = get_model("shufflenet_v2", num_classes=29, pretrained=False)
        with TemporaryDirectory() as temp_dir:
            test_art = Path(temp_dir)
            export_final_model(m, self.species_list, test_art, arch="shufflenet_v2")
            self.assertTrue((test_art / "model.pt").exists())
            self.assertTrue((test_art / "model.onnx").exists())
            self.assertTrue((test_art / "class_names.json").exists())
            self.assertTrue((test_art / "preprocess_config.json").exists())

    def test_fold_trainer_saves_epoch_history_and_best_weights(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Linear(2, 2)
                self.fc = nn.Linear(2, 2)

            def forward(self, inputs):
                return self.fc(self.backbone(inputs))

        inputs = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]]
        )
        labels = torch.tensor([0, 1, 0, 1])
        indices = torch.arange(len(labels))
        loader = DataLoader(TensorDataset(inputs, labels, indices), batch_size=2)
        model, result = train_fold(
            TinyModel(),
            loader,
            loader,
            torch.ones(2),
            torch.device("cpu"),
            epochs_s1=1,
            epochs_s2=2,
            patience=1,
        )
        self.assertEqual(len(result["history"]), 3)
        self.assertGreaterEqual(result["best_epoch"], 1)
        self.assertEqual(model.fc.out_features, 2)


if __name__ == "__main__":
    unittest.main()
