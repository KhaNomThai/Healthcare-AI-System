#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data.py -- Data Preparation & DataLoaders

Handles ISIC 2019 dataset loading, image path resolution, stratified
splitting, class weight computation, data augmentation, and PyTorch
DataLoader creation.
"""

import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

from config import (
    ISIC_2019_GT_CSV,
    ISIC_2019_METADATA_CSV,
    ISIC_2019_IMAGE_DIR,
    ISIC_2019_LABEL_COLS,
    CLASS_CODES,
    CLASS_TO_IDX,
    CLASS_NAMES_EN,
    CANCER_INDICES,
    BENIGN_INDICES,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    RANDOM_SEED,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


# =============================================================================
# Image Augmentation Pipelines
# =============================================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# =============================================================================
# Image Path Resolution
# =============================================================================

def resolve_image_path(image_id: str, image_dir: Path) -> Optional[Path]:
    """
    Resolve image file path, handling _downsampled suffix.
    ISIC 2019 has some images with _downsampled.jpg suffix.
    Returns the path if found, None otherwise.
    """
    # Try standard filename first
    standard_path = image_dir / f"{image_id}.jpg"
    if standard_path.exists():
        return standard_path

    # Try _downsampled variant
    downsampled_path = image_dir / f"{image_id}_downsampled.jpg"
    if downsampled_path.exists():
        return downsampled_path

    return None


# =============================================================================
# Data Loading & Preparation
# =============================================================================

def load_and_prepare_data() -> pd.DataFrame:
    """
    Load ISIC 2019 ground truth CSV (one-hot encoded) and prepare
    a DataFrame with columns: [image_id, dx, label, group, image_path].
    `group` is the lesion_id (or the image id when unknown), used to keep
    all images of one lesion in the same split.
    """
    print("[INFO] Loading ISIC 2019 Training Ground Truth...")
    print(f"[INFO] CSV path: {ISIC_2019_GT_CSV}")
    print(f"[INFO] Image directory: {ISIC_2019_IMAGE_DIR}")

    if not ISIC_2019_GT_CSV.exists():
        print(f"[ERROR] Ground truth CSV not found: {ISIC_2019_GT_CSV}")
        sys.exit(1)

    if not ISIC_2019_METADATA_CSV.exists():
        print(f"[ERROR] Metadata CSV not found (needed for lesion_id): {ISIC_2019_METADATA_CSV}")
        sys.exit(1)

    if not ISIC_2019_IMAGE_DIR.exists():
        print(f"[ERROR] Image directory not found: {ISIC_2019_IMAGE_DIR}")
        sys.exit(1)

    # Read one-hot encoded ground truth
    gt_df = pd.read_csv(ISIC_2019_GT_CSV)
    print(f"[INFO] Loaded {len(gt_df)} entries from ground truth CSV")

    # Several images can show the same lesion; group them so no lesion spans two splits
    meta = pd.read_csv(ISIC_2019_METADATA_CSV, usecols=["image", "lesion_id"])
    gt_df = gt_df.merge(meta, on="image", how="left")
    gt_df["group"] = gt_df["lesion_id"].fillna(gt_df["image"])

    # Convert one-hot to single diagnosis label
    gt_df["dx"] = gt_df[ISIC_2019_LABEL_COLS].idxmax(axis=1)

    # Drop UNK rows (should be 0 in training set, but defensive)
    unk_mask = gt_df["dx"] == "UNK"
    if unk_mask.sum() > 0:
        print(f"[WARNING] Dropping {unk_mask.sum()} UNK rows")
        gt_df = gt_df[~unk_mask].reset_index(drop=True)

    # Map class codes to integer indices
    gt_df["label"] = gt_df["dx"].map(CLASS_TO_IDX)

    # Validate: all dx codes should be in our taxonomy
    unmapped = gt_df["label"].isna()
    if unmapped.sum() > 0:
        unknown_codes = gt_df.loc[unmapped, "dx"].unique()
        print(f"[WARNING] Unknown diagnosis codes: {unknown_codes}")
        gt_df = gt_df[~unmapped].reset_index(drop=True)

    gt_df["label"] = gt_df["label"].astype(int)

    # Resolve image paths (handle _downsampled suffix)
    print("[INFO] Resolving image file paths...")
    valid_rows = []
    missing_count = 0

    for _, row in tqdm(gt_df.iterrows(), total=len(gt_df), desc="Resolving images"):
        img_path = resolve_image_path(row["image"], ISIC_2019_IMAGE_DIR)
        if img_path is not None:
            valid_rows.append({
                "image_id": row["image"],
                "dx": row["dx"],
                "label": row["label"],
                "group": row["group"],
                "image_path": str(img_path),
            })
        else:
            missing_count += 1

    if missing_count > 0:
        print(f"[WARNING] {missing_count} images not found and excluded")

    df = pd.DataFrame(valid_rows)
    print(f"[INFO] Total valid samples: {len(df)}")

    # Print class distribution
    print()
    print("[INFO] Class Distribution:")
    print("-" * 60)
    for idx, code in enumerate(CLASS_CODES):
        count = (df["label"] == idx).sum()
        group = "Cancer/Pre-Cancer" if idx in CANCER_INDICES else "Benign"
        pct = 100.0 * count / len(df)
        print(f"  [{idx}] {code:5s} | {CLASS_NAMES_EN[idx]:45s} | {count:6d} ({pct:5.1f}%) | {group}")
    print("-" * 60)

    # Print group summary
    cancer_count = df["label"].isin(CANCER_INDICES).sum()
    benign_count = df["label"].isin(BENIGN_INDICES).sum()
    print(f"  Cancer / Pre-Cancer total: {cancer_count:6d} ({100.0 * cancer_count / len(df):.1f}%)")
    print(f"  Benign total:              {benign_count:6d} ({100.0 * benign_count / len(df):.1f}%)")
    print()

    return df


# =============================================================================
# Data Splitting
# =============================================================================

def _group_split(df: pd.DataFrame, holdout: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split off ~`holdout` of df, stratified by label, never splitting a lesion group."""
    sgkf = StratifiedGroupKFold(n_splits=round(1 / holdout), shuffle=True, random_state=RANDOM_SEED)
    keep_idx, hold_idx = next(sgkf.split(df, df["label"], groups=df["group"]))
    return df.iloc[keep_idx], df.iloc[hold_idx]


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Lesion-grouped, stratified split into train/val/test sets.
    All images of one lesion land in the same split (no leakage), and class
    distribution is kept close to the full set. Sizes are approximate because
    whole lesions move together (1/round(1/size) of the data).
    """
    print("[INFO] Splitting data (grouped by lesion, stratified by class label)...")

    train_val_df, test_df = _group_split(df, test_size)
    train_df, val_df = _group_split(train_val_df, val_size / (1 - test_size))

    # Leakage self-check: no lesion may appear in more than one split
    tr, va, te = (set(d["group"]) for d in (train_df, val_df, test_df))
    assert not (tr & va or tr & te or va & te), "Lesion leakage between splits"

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"[INFO] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


# =============================================================================
# Class Weights
# =============================================================================

def compute_class_weights_tensor(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    """Compute balanced class weights for handling class imbalance."""
    unique_classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=unique_classes, y=labels)
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"[INFO] Class weights computed for {len(unique_classes)} classes")
    for idx, code in enumerate(CLASS_CODES):
        if idx < len(weights):
            print(f"  [{idx}] {code:5s}: {weights[idx]:.4f}")
    return weight_tensor


# =============================================================================
# PyTorch Dataset
# =============================================================================

class SkinLesionDataset(Dataset):
    """PyTorch Dataset for skin lesion images from ISIC 2019."""

    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.dataframe.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        label = int(row["label"])
        if self.transform:
            image = self.transform(image)
        return image, label


# =============================================================================
# DataLoader Creation
# =============================================================================

def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoader instances for train, validation, and test sets."""
    train_dataset = SkinLesionDataset(train_df, transform=train_transform)
    val_dataset = SkinLesionDataset(val_df, transform=val_test_transform)
    test_dataset = SkinLesionDataset(test_df, transform=val_test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    print(f"[INFO] DataLoaders created (num_workers={NUM_WORKERS})")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    return train_loader, val_loader, test_loader
