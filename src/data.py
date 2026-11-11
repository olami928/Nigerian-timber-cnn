"""
Data loading, manifest creation, and integrity verification for Timber ID pipeline.
Transverse (end-grain) macroscopic images divided into 6x8 grid (48 patches).
"""

import os
import re
import struct
import hashlib
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import pandas as pd


def parse_folder(name: str) -> Tuple[str, int]:
    """
    Parses specimen folder name according to Section 7.1.
    e.g. 'ayunre' -> ('ayunre', 1), 'ayunre2' -> ('ayunre', 2).
    """
    m = re.fullmatch(r"\s*([A-Za-z_\- ]+?)\s*(\d*)\s*", name)
    if not m:
        return name.strip().lower(), 1
    slug = m.group(1).strip().lower().replace(" ", "_").replace("-", "_")
    idx = int(m.group(2)) if m.group(2) else 1
    return slug, idx


def resolve_image_path(path: Path | str) -> Path:
    """Resolve a manifest image path, allowing the specimen-1 '1' folder alias.

    Specimen 1 is normally written without a number (``arere/``), but some
    dataset copies use ``arere1/``. The expected on-disk path is returned when
    it exists; otherwise the numbered alias is used when it exists.
    """
    expected = Path(path)
    if expected.is_file():
        return expected
    parent = expected.parent
    if parent.name and not parent.name[-1].isdigit():
        alias = parent.with_name(f"{parent.name}1") / expected.name
        if alias.is_file():
            return alias
    return expected


def read_image_dimensions(path: Path) -> Tuple[int, int]:
    """Reads PNG or JPEG dimensions directly from header."""
    with open(path, "rb") as f:
        head = f.read(32)
        if head.startswith(b"\x89PNG\r\n\x1a\n"):
            return struct.unpack(">II", head[16:24])
        elif head.startswith(b"\xff\xd8"):
            # JPEG parsing fallback using Pillow if available
            try:
                from PIL import Image
                with Image.open(path) as im:
                    return im.size
            except ImportError:
                raise RuntimeError(f"Pillow required to read JPEG {path}")
        else:
            try:
                from PIL import Image
                with Image.open(path) as im:
                    return im.size
            except Exception as e:
                raise ValueError(f"Unreadable image header: {path} ({e})")


def build_manifest(data_dir: Path, species_csv_path: Path, output_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Builds data/manifest.csv from specimen folder structure.
    Strictly follows Section 7.3 and 7.4 of the brief.
    Excludes auxiliary thumbnail/result files.
    """
    df_species = pd.read_csv(species_csv_path)
    valid_species = set(df_species["slug"].str.lower())

    rows: List[Dict[str, Any]] = []
    corrupt_images = []
    
    for folder in sorted(p for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        slug, idx = parse_folder(folder.name)
        
        for img in sorted(folder.iterdir()):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            # Exclude auxiliary _result files
            if "_result" in img.name:
                continue

            try:
                w, h = read_image_dimensions(img)
                sha256 = hashlib.sha256(img.read_bytes()).hexdigest()
                rows.append({
                    "path": str(img),
                    "orig_name": img.name,
                    "folder": folder.name,
                    "species": slug,
                    "specimen_idx": idx,
                    "specimen_code": f"{slug}_{idx}",
                    "width": w,
                    "height": h,
                    "sha256": sha256,
                    "group_id": "",
                    "split_role": ""
                })
            except Exception as e:
                corrupt_images.append((str(img), str(e)))

    df = pd.DataFrame(rows)

    # 7.4 Integrity Checks
    # 1. Species match check
    unknown_species = set(df["species"]) - valid_species
    if unknown_species:
        raise ValueError(f"Unknown species found in dataset: {unknown_species}")

    # 2. Specimen indices 1, 2, 3 in order without gaps
    for sp, g in df.groupby("species"):
        indices = sorted(g["specimen_idx"].unique())
        expected = list(range(1, len(indices) + 1))
        if indices != expected or len(indices) > 3:
            raise ValueError(f"Species '{sp}' has invalid specimen indices: {indices} (expected {expected})")

    # 3. Exact duplicates
    dup_hashes = df[df.duplicated(subset=["sha256"], keep=False)]
    if not dup_hashes.empty:
        print(f"Warning: Found {len(dup_hashes)} exact duplicate image instances (same sha256).")

    # 4. Corrupt images
    if corrupt_images:
        print(f"Warning: {len(corrupt_images)} corrupt images found: {corrupt_images}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Columns strictly match Section 7.3:
        # path, orig_name, folder, species, specimen_idx, specimen_code, width, height, sha256, group_id, split_role
        cols = ["path", "orig_name", "folder", "species", "specimen_idx", "specimen_code",
                "width", "height", "sha256", "group_id", "split_role"]
        df[cols].to_csv(output_path, index=False)
        print(f"Manifest written to {output_path} ({len(df)} rows).")

    return df


def verify_grid_readiness(df: pd.DataFrame, grid_rows: int = 6, grid_cols: int = 8) -> Dict[str, Any]:
    """
    Verifies Section 7.2 checks:
    1. Check width is divisible by 8 and height by 6.
    2. Patch sizes (width/8 by height/6) and whether patches are square.
    3. Distribution of image sizes and failing count.
    """
    div_w = (df["width"] % grid_cols == 0)
    div_h = (df["height"] % grid_rows == 0)
    passes_both = div_w & div_h

    failing = df[~passes_both]
    patch_w = df["width"] // grid_cols
    patch_h = df["height"] // grid_rows
    is_square = (patch_w == patch_h)

    return {
        "total_images": len(df),
        "passing_count": int(passes_both.sum()),
        "failing_count": int((~passes_both).sum()),
        "failing_list": failing[["path", "width", "height"]].to_dict(orient="records"),
        "square_patches_count": int(is_square.sum()),
        "rectangular_patches_count": int((~is_square).sum()),
        "unique_image_sizes": df[["width", "height"]].drop_duplicates().values.tolist(),
        "unique_patch_sizes": pd.DataFrame({"pw": patch_w, "ph": patch_h}).drop_duplicates().values.tolist()
    }
