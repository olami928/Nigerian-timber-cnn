"""
Specimen-grouped cross-validation split planning and leak auditing.
"""

import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


def embed_with_resnet(paths: List[str], device: Optional[str] = None) -> np.ndarray:
    """Extract normalized ImageNet ResNet18 embeddings for the given images."""
    import torch
    from PIL import Image
    from torchvision.models import ResNet18_Weights, resnet18

    if not paths:
        return np.empty((0, 512), dtype=np.float32)

    weights = ResNet18_Weights.IMAGENET1K_V1
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(target_device)
    transform = weights.transforms()

    outputs = []
    with torch.no_grad():
        for start in range(0, len(paths), 64):
            batch = []
            for path in paths[start : start + 64]:
                with Image.open(path) as image:
                    batch.append(transform(image.convert("RGB")))
            outputs.append(model(torch.stack(batch).to(target_device)).cpu())

    features = torch.cat(outputs).numpy()
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, np.finfo(features.dtype).eps)


def assign_region_groups(
    df: pd.DataFrame,
    thr: float = 0.25,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """Cluster each specimen's images by normalized ResNet18 cosine distance."""
    required = {"path", "specimen_code"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Region grouping requires columns: {sorted(missing)}")
    if thr <= 0:
        raise ValueError("The region-group distance threshold must be positive.")

    result = df.copy()
    result["group_id"] = ""
    specimens = [
        group.sort_values("path", kind="stable")
        for _, group in result.groupby("specimen_code", sort=True)
    ]
    clusterable = [group for group in specimens if len(group) >= 3]
    paths = [path for group in clusterable for path in group["path"].tolist()]
    features = embed_with_resnet(paths, device=device)

    offset = 0
    for group in specimens:
        code = str(group["specimen_code"].iloc[0])
        if len(group) < 3:
            result.loc[group.index, "group_id"] = f"{code}_g0"
            continue

        group_features = features[offset : offset + len(group)]
        offset += len(group)
        labels = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=thr,
            metric="cosine",
            linkage="average",
        ).fit_predict(group_features)

        cluster_first_paths = {
            int(label): min(group.iloc[np.flatnonzero(labels == label)]["path"])
            for label in np.unique(labels)
        }
        ordered_labels = sorted(cluster_first_paths, key=cluster_first_paths.get)
        stable_labels = {label: index for index, label in enumerate(ordered_labels)}
        group_ids = [f"{code}_g{stable_labels[int(label)]}" for label in labels]
        result.loc[group.index, "group_id"] = group_ids

    return result


def roles_for_fold(
    df: pd.DataFrame,
    fold: int,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.Series, pd.Series]:
    """Assign deterministic train, validation, and test roles for one fold."""
    if fold not in (1, 2, 3):
        raise ValueError(f"Fold must be 1, 2, or 3; got {fold}.")
    if not 0 <= val_frac < 1:
        raise ValueError(f"Validation fraction must be in [0, 1); got {val_frac}.")
    if "group_id" not in df or df["group_id"].isna().any():
        raise ValueError("Assign region group IDs before planning data splits.")
    if df["group_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Assign region group IDs before planning data splits.")

    role = pd.Series("train", index=df.index, dtype="object")
    eval_group = pd.Series("", index=df.index, dtype="object")

    for species, group in df.groupby("species"):
        rng = np.random.default_rng(seed + zlib.crc32(species.encode()))
        specimen_count = group["specimen_idx"].nunique()
        if specimen_count >= 2:
            test_indices = group.index[group["specimen_idx"] == fold]
            role[test_indices] = "test"
            if len(test_indices):
                eval_group[test_indices] = "independent_specimen"
            pool = group.drop(test_indices)
        else:
            pool = group

        groups = np.array(sorted(pool["group_id"].unique()))
        rng.shuffle(groups)
        val_count = max(1, round(val_frac * len(groups))) if len(groups) >= 3 else 0
        role[pool.index[pool["group_id"].isin(groups[:val_count])]] = "val"

        if specimen_count == 1 and len(groups) >= 4:
            test_groups = groups[
                val_count : val_count + max(1, round(val_frac * len(groups)))
            ]
            test_indices = pool.index[pool["group_id"].isin(test_groups)]
            role[test_indices] = "test"
            eval_group[test_indices] = "within_specimen"

    return role, eval_group


def leak_audit(df: pd.DataFrame, role: pd.Series, fold: int) -> Dict[str, Any]:
    """Assert that specimen, region-group, and exact-image hashes do not leak."""
    if not role.index.equals(df.index):
        raise ValueError("Role assignments must use the same row index as the manifest.")

    data = df.assign(role=role)
    for species, group in data.groupby("species"):
        if group["specimen_idx"].nunique() >= 2:
            train_specimens = set(group[group["role"] == "train"]["specimen_code"])
            test_specimens = set(group[group["role"] == "test"]["specimen_code"])
            assert not (train_specimens & test_specimens), (
                f"{species}: specimen leak in train and test (fold {fold}): "
                f"{train_specimens & test_specimens}"
            )

    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        group_leaks = set(data[data["role"] == first]["group_id"]) & set(
            data[data["role"] == second]["group_id"]
        )
        assert not group_leaks, (
            f"group leak {first}/{second} in fold {fold}: {group_leaks}"
        )

        hash_leaks = set(data[data["role"] == first]["sha256"]) & set(
            data[data["role"] == second]["sha256"]
        )
        assert not hash_leaks, (
            f"duplicate image leak {first}/{second} in fold {fold}: {hash_leaks}"
        )

    return {
        "fold": fold,
        "status": "PASSED",
        "train_count": int((data["role"] == "train").sum()),
        "val_count": int((data["role"] == "val").sum()),
        "test_count": int((data["role"] == "test").sum()),
    }


def generate_and_save_splits(
    df_manifest: pd.DataFrame,
    output_path: Path,
    seed: int = 42,
    overwrite: bool = False,
    audit_output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Generate three fold assignments and save splits plus leak-audit results."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass overwrite=True only for an "
            "intentional regeneration."
        )
    audit_path = audit_output_path or output_path.with_name("leak_audit.csv")
    if audit_path.exists() and not overwrite:
        raise FileExistsError(
            f"{audit_path} already exists; pass overwrite=True only for an "
            "intentional regeneration."
        )
    if "group_id" not in df_manifest or df_manifest["group_id"].isna().any():
        raise ValueError("Assign and review region groups before generating splits.")
    if df_manifest["group_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Assign and review region groups before generating splits.")

    splits = df_manifest.copy()
    audit_results = []
    for fold in (1, 2, 3):
        roles, eval_group = roles_for_fold(splits, fold=fold, seed=seed)
        splits[f"fold_{fold}_role"] = roles
        splits[f"fold_{fold}_eval_group"] = eval_group
        audit_results.append(leak_audit(splits, roles, fold=fold))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(output_path, index=False)
    pd.DataFrame(audit_results).to_csv(audit_path, index=False)
    return splits
