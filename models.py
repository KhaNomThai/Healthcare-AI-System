#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models.py -- Model Architectures

Provides build functions for three CNN architectures used in multi-model
comparison: ResNet50, EfficientNet-B3, and DenseNet121. All models share
a unified custom classifier head pattern.
"""

import torch
import torch.nn as nn
import torchvision.models as models

from config import NUM_CLASSES


# =============================================================================
# Internal Builders
# =============================================================================

def _build_resnet50(device: torch.device, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """
    Build ResNet50 with custom classifier head.
    Fine-tunes layer3 + layer4 + fc; earlier layers are frozen.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)

    # Freeze all layers except layer3, layer4 and fc
    for name, param in model.named_parameters():
        if not name.startswith(("layer3", "layer4", "fc")):
            param.requires_grad = False

    # Replace classifier head
    in_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )

    return model.to(device)


def _build_efficientnet_b3(device: torch.device, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """
    Build EfficientNet-B3 with custom classifier head.
    Fine-tunes the last three feature blocks + classifier; earlier blocks are frozen.
    """
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None)

    # Freeze all feature layers except features[6..8]
    for name, param in model.named_parameters():
        if not name.startswith(("features.6", "features.7", "features.8", "classifier")):
            param.requires_grad = False

    # Replace classifier head (EfficientNet-B3 in_features = 1536)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )

    return model.to(device)


def _build_densenet121(device: torch.device, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """
    Build DenseNet121 with custom classifier head.
    Fine-tunes denseblock3 + transition3 + denseblock4 + norm5 + classifier; earlier blocks are frozen.
    """
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)

    # Freeze all feature layers except the last dense blocks and the final norm
    for name, param in model.named_parameters():
        if not name.startswith(("features.denseblock3", "features.transition3", "features.denseblock4", "features.norm5", "classifier")):
            param.requires_grad = False

    # Replace classifier head (DenseNet121 in_features = 1024)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )

    return model.to(device)


# =============================================================================
# Unified Build Interface
# =============================================================================

# Dispatch table mapping model names to builder functions
_BUILDERS = {
    "ResNet50": _build_resnet50,
    "EfficientNet-B3": _build_efficientnet_b3,
    "DenseNet121": _build_densenet121,
}


def build_model(model_name: str, device: torch.device, pretrained: bool = True) -> nn.Module:
    """
    Build the specified model architecture.

    Args:
        model_name: One of "ResNet50", "EfficientNet-B3", "DenseNet121".
        device: Target compute device (cpu, cuda, mps).
        pretrained: Load ImageNet weights. Pass False when a trained
            checkpoint will be loaded right after (skips the download).

    Returns:
        The constructed model moved to the specified device.
    """
    if model_name not in _BUILDERS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(_BUILDERS.keys())}"
        )

    print(f"[INFO] Building {model_name} model with custom classifier head...")
    model = _BUILDERS[model_name](device, pretrained=pretrained)
    _print_param_count(model)
    return model


def _print_param_count(model: nn.Module) -> None:
    """Print total, trainable, and frozen parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"[INFO] Total parameters:     {total_params:,}")
    print(f"[INFO] Trainable parameters: {trainable_params:,}")
    print(f"[INFO] Frozen parameters:    {frozen_params:,}")
