"""Shared patch aggregation and optional test-time augmentation.

Both evaluation and serving use these functions so the reported metrics and the
website cannot drift apart. Defaults reproduce the original notebook path
(majority vote, no TTA). The optional soft-vote and TTA paths are enabled only
when the exported preprocessing configuration asks for them.
"""

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


ALLOWED_TTA_VIEWS = ("identity", "hflip", "vflip", "hvflip")
ALLOWED_AGGREGATION_MODES = ("majority", "soft")


def validate_tta_views(views: Iterable[str]) -> Tuple[str, ...]:
    """Return a validated tuple of TTA view names."""
    resolved = tuple(views)
    if not resolved:
        resolved = ("identity",)
    unknown = set(resolved) - set(ALLOWED_TTA_VIEWS)
    if unknown:
        raise ValueError(f"Unsupported TTA views: {sorted(unknown)}")
    if "identity" not in resolved:
        resolved = ("identity", *resolved)
    return resolved


def validate_aggregation_mode(mode: str) -> str:
    """Return a validated aggregation mode name."""
    if mode not in ALLOWED_AGGREGATION_MODES:
        raise ValueError(
            f"Unsupported aggregation mode {mode!r}; "
            f"must be one of {ALLOWED_AGGREGATION_MODES}."
        )
    return mode


def _apply_view(inputs: torch.Tensor, view: str) -> torch.Tensor:
    """Return one TTA view of a [N, C, H, W] patch batch."""
    if view == "identity":
        return inputs
    if view == "hflip":
        return torch.flip(inputs, dims=[3])
    if view == "vflip":
        return torch.flip(inputs, dims=[2])
    if view == "hvflip":
        return torch.flip(inputs, dims=[2, 3])
    raise ValueError(f"Unsupported TTA view: {view}")


def forward_probabilities(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    device: torch.device | str,
    tta_views: Iterable[str] = (),
) -> np.ndarray:
    """Run a patch batch through the model and average softmax over TTA views."""
    views = validate_tta_views(tta_views)
    model.eval()
    total: np.ndarray | None = None
    with torch.inference_mode():
        for view in views:
            logits = model(_apply_view(inputs, view).to(device, non_blocking=True))
            probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
            total = probabilities if total is None else total + probabilities
    if total is None:
        raise RuntimeError("No TTA views were evaluated.")
    return total / len(views)


def aggregate_probabilities(
    probabilities: np.ndarray,
    class_names: Sequence[str],
    mode: str = "majority",
) -> Dict[str, Any]:
    """Aggregate [patches, classes] probabilities into one image prediction."""
    if probabilities.ndim != 2 or probabilities.shape[1] != len(class_names):
        raise ValueError(
            "Expected probabilities shaped [patches, classes] matching class_names; "
            f"got {probabilities.shape} for {len(class_names)} classes."
        )
    if probabilities.shape[0] < 1:
        raise ValueError("At least one patch prediction is required.")
    mode = validate_aggregation_mode(mode)
    votes = probabilities.argmax(axis=1)
    counts = np.bincount(votes, minlength=len(class_names))
    mean_probabilities = probabilities.mean(axis=0)
    if mode == "soft":
        scores = mean_probabilities
    else:
        scores = counts + 1e-3 * mean_probabilities
    order = np.argsort(scores)[::-1][:3]
    winner = int(order[0])
    patch_count = len(votes)
    return {
        "prediction": class_names[winner],
        "vote_count": int(counts[winner]),
        "confidence_pct": round(100.0 * int(counts[winner]) / patch_count, 1),
        "patch_agreement": f"{int(counts[winner])}/{patch_count}",
        "mean_probability_pct": round(
            100.0 * float(mean_probabilities[winner]), 1
        ),
        "top3": [
            {
                "species": class_names[int(index)],
                "vote_pct": round(100.0 * int(counts[index]) / patch_count, 1),
                "probability_pct": round(
                    100.0 * float(mean_probabilities[index]), 1
                ),
            }
            for index in order
        ],
    }


def aggregate_logits(
    logits: np.ndarray,
    class_names: Sequence[str],
    mode: str = "majority",
) -> List[Dict[str, Any]]:
    """Aggregate [images, patches, classes] logits with the shared rules."""
    if logits.ndim != 3 or logits.shape[2] != len(class_names):
        raise ValueError(
            "Expected logits shaped [images, patches, classes] matching class_names; "
            f"got {logits.shape} for {len(class_names)} classes."
        )
    shifted = logits - logits.max(axis=2, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=2, keepdims=True)
    return [
        aggregate_probabilities(image_probabilities, class_names, mode=mode)
        for image_probabilities in probabilities
    ]
