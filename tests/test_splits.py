"""
Unit and integration tests for data loading, manifest generation, and split leak audit.
Uses standard library unittest so it runs everywhere without extra dependencies.
"""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data import parse_folder, verify_grid_readiness
from src.split import assign_region_groups, leak_audit, roles_for_fold


class TestDataAndSplits(unittest.TestCase):

    def test_parse_folder(self):
        self.assertEqual(parse_folder("ayunre"), ("ayunre", 1))
        self.assertEqual(parse_folder("ayunre2"), ("ayunre", 2))
        self.assertEqual(parse_folder("ayunre3"), ("ayunre", 3))
        self.assertEqual(parse_folder("araba"), ("araba", 1))
        self.assertEqual(parse_folder("araba2"), ("araba", 2))
        self.assertEqual(parse_folder("teak"), ("teak", 1))
        self.assertEqual(parse_folder("teak2"), ("teak", 2))

    def test_manifest_exists_and_valid(self):
        manifest_path = Path("data/manifest.csv")
        self.assertTrue(manifest_path.exists(), "data/manifest.csv must exist")
        df_m = pd.read_csv(manifest_path)
        self.assertEqual(len(df_m), 900, "Clean transverse images must equal 900")
        self.assertEqual(df_m["species"].nunique(), 29, "Total species must be exactly 29")
        self.assertEqual(df_m["specimen_code"].nunique(), 45, "Total specimens must be exactly 45")
        
        expected_cols = [
            "path", "orig_name", "folder", "species", "specimen_idx",
            "specimen_code", "width", "height", "sha256", "group_id", "split_role"
        ]
        for col in expected_cols:
            self.assertIn(col, df_m.columns)

    def test_grid_readiness(self):
        manifest_path = Path("data/manifest.csv")
        self.assertTrue(manifest_path.exists())
        df_m = pd.read_csv(manifest_path)
        report = verify_grid_readiness(df_m)
        self.assertEqual(report["failing_count"], 0, "No image may fail 8x6 divisibility")
        self.assertEqual(report["passing_count"], 900)

    def test_splits_leak_audit(self):
        splits_path = Path("data/splits.csv")
        self.assertTrue(splits_path.exists(), "data/splits.csv must exist")
        df_s = pd.read_csv(splits_path)
        for fold in [1, 2, 3]:
            role_col = f"fold_{fold}_role"
            res = leak_audit(df_s, df_s[role_col], fold=fold)
            self.assertEqual(res["status"], "PASSED")

    def test_region_groups_use_embedding_clusters(self):
        df = pd.DataFrame(
            {
                "path": [f"specimen1/{i}.png" for i in range(4)]
                + [f"specimen2/{i}.png" for i in range(2)],
                "specimen_code": ["wood_1"] * 4 + ["wood_2"] * 2,
            }
        )
        embeddings = np.array(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            dtype=np.float32,
        )

        with patch("src.split.embed_with_resnet", return_value=embeddings) as embed:
            grouped = assign_region_groups(df, thr=0.25)

        embed.assert_called_once_with(
            [f"specimen1/{i}.png" for i in range(4)], device=None
        )
        self.assertEqual(grouped.iloc[:4]["group_id"].nunique(), 2)
        self.assertEqual(grouped.iloc[4:]["group_id"].unique().tolist(), ["wood_2_g0"])

    def test_fold_roles_keep_region_groups_together(self):
        df = pd.DataFrame(
            {
                "species": ["two_specimens"] * 8 + ["one_specimen"] * 4,
                "specimen_idx": [1] * 4 + [2] * 4 + [1] * 4,
                "specimen_code": ["two_1"] * 4
                + ["two_2"] * 4
                + ["one_1"] * 4,
                "group_id": [f"two_1_g{i}" for i in range(4)]
                + [f"two_2_g{i}" for i in range(4)]
                + [f"one_1_g{i}" for i in range(4)],
                "sha256": [f"hash_{i}" for i in range(12)],
            }
        )

        role, eval_group = roles_for_fold(df, fold=1)

        for _, group in df.assign(role=role).groupby("group_id"):
            self.assertEqual(group["role"].nunique(), 1)
        self.assertTrue((role.iloc[:4] == "test").all())
        self.assertTrue((eval_group.iloc[:4] == "independent_specimen").all())
        self.assertEqual(int((eval_group == "within_specimen").sum()), 1)
        self.assertEqual(
            leak_audit(df, role, fold=1)["status"],
            "PASSED",
        )


if __name__ == "__main__":
    unittest.main()
