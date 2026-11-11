"""Tests for manifest image path resolution."""

import tempfile
import unittest
from pathlib import Path

from src.data import resolve_image_path


class TestResolveImagePath(unittest.TestCase):
    def test_returns_expected_path_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "arere" / "image.png"
            image.parent.mkdir()
            image.write_bytes(b"expected")
            self.assertEqual(resolve_image_path(image), image)

    def test_falls_back_to_specimen_one_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            alias = root / "arere1" / "image.png"
            alias.parent.mkdir()
            alias.write_bytes(b"alias")
            expected = root / "arere" / "image.png"
            self.assertEqual(resolve_image_path(expected), alias)

    def test_does_not_alias_numbered_specimens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            alias = root / "arere21" / "image.png"
            alias.parent.mkdir()
            alias.write_bytes(b"alias")
            expected = root / "arere2" / "image.png"
            self.assertEqual(resolve_image_path(expected), expected)

    def test_returns_missing_path_when_no_alias_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            expected = Path(tmp) / "teak" / "image.png"
            self.assertEqual(resolve_image_path(expected), expected)


if __name__ == "__main__":
    unittest.main()
