"""
Model definitions for Nigerian Timber Species Identification.
Primary: ShuffleNetV2 (lightweight, efficient)
Comparison: ResNet18 (benchmark)
MobileNetV2 is strictly forbidden.
"""

import torch
import torch.nn as nn
from torchvision import models


HEAD_TYPES = ("legacy", "regularized", "linear")


def build_head(
    in_features: int,
    num_classes: int,
    dropout: float = 0.5,
    slope: float = 0.1,
    head_type: str = "legacy",
    hidden_dim: int = 256,
) -> nn.Sequential:
    """
    Build the classification head.

    legacy (default, backward compatible with existing checkpoints):
        Linear(in_features, 256) -> LeakyReLU(0.1) -> Dropout(0.5) -> Linear(256, C)
    regularized:
        Dropout -> Linear(in_features, hidden_dim) -> BatchNorm1d -> LeakyReLU
        -> Dropout -> Linear(hidden_dim, C)
    linear:
        Dropout -> Linear(in_features, C) (strongest parameter reduction)

    The new heads are opt-in so the original P2 exports still load.
    """
    if head_type not in HEAD_TYPES:
        raise ValueError(
            f"Unsupported head_type {head_type!r}; must be one of {HEAD_TYPES}."
        )
    if hidden_dim < 1:
        raise ValueError("hidden_dim must be positive.")

    if head_type == "legacy":
        return nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LeakyReLU(slope),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    if head_type == "regularized":
        return nn.Sequential(
            nn.Dropout(dropout * 0.5),
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(slope),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )


def get_model(
    arch: str = "shufflenet_v2",
    num_classes: int = 29,
    pretrained: bool = True,
    dropout: float = 0.5,
    head_type: str = "legacy",
    hidden_dim: int = 256,
) -> nn.Module:
    """
    Builds the model architecture.
    arch: 'shufflenet_v2' or 'resnet18'
    """
    arch_lower = arch.lower()
    assert "mobilenet" not in arch_lower, "MobileNetV2 is strictly forbidden in this project."

    if "shufflenet" in arch_lower:
        weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
        model = models.shufflenet_v2_x1_0(weights=weights)
        in_features = model.fc.in_features
        model.fc = build_head(
            in_features,
            num_classes,
            dropout=dropout,
            head_type=head_type,
            hidden_dim=hidden_dim,
        )
        model.arch_name = "shufflenet_v2"
        model.head_type = head_type
        model.hidden_dim = hidden_dim
        return model

    elif "resnet" in arch_lower:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = build_head(
            in_features,
            num_classes,
            dropout=dropout,
            head_type=head_type,
            hidden_dim=hidden_dim,
        )
        model.arch_name = "resnet18"
        model.head_type = head_type
        model.hidden_dim = hidden_dim
        return model

    else:
        raise ValueError(f"Unsupported architecture: {arch}. Must be 'shufflenet_v2' or 'resnet18'.")


def set_backbone_trainable(model: nn.Module, trainable: bool = False):
    """Freezes or unfreezes backbone parameters, keeping the classification head trainable."""
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = trainable
        else:
            param.requires_grad = True
