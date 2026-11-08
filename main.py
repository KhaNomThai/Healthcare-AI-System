#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py -- Entry Point

CLI interface for the Skin Cancer Multi-class Classification & Screening
System. Supports training all or individual models, evaluation, model
comparison. The demo UI lives in web/ (see export_onnx.py).

Usage:
    python main.py                    # Train all models + evaluate
    python main.py --model ResNet50   # Train a specific model only
    python main.py --skip-train       # Skip training, evaluate existing checkpoints
"""

import json
import argparse

import torch

from config import (
    set_seed,
    get_device,
    print_banner,
    MODEL_REGISTRY,
    MODEL_NAMES,
)
from data import (
    load_and_prepare_data,
    split_data,
    create_dataloaders,
    compute_class_weights_tensor,
)
from models import build_model
from train import train_model
from evaluate import (
    evaluate_on_test_set,
    print_evaluation_report,
    plot_confusion_matrix,
    save_model_metrics,
    print_comparison_table,
)


def main():
    """Main entry point for the classification pipeline."""
    parser = argparse.ArgumentParser(
        description="Skin Cancer Multi-class Classification & Screening System"
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training, evaluate existing checkpoints on the test set",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=MODEL_NAMES,
        help="Train a specific model only (default: train all models)",
    )
    args = parser.parse_args()

    set_seed()
    print_banner()
    device = get_device()

    # --- Determine which models to train ---
    if args.model:
        models_to_train = [args.model]
    else:
        models_to_train = MODEL_NAMES.copy()

    # --- Load and prepare data (once for all models) ---
    df = load_and_prepare_data()
    train_df, val_df, test_df = split_data(df)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df
    )

    # --- Train and evaluate each model ---
    for model_name in models_to_train:
        print()
        print("#" * 80)
        print(f"#  MODEL: {model_name}")
        print("#" * 80)

        model = build_model(model_name, device)

        # Compute parameter counts for metrics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        if not args.skip_train:
            class_weights = compute_class_weights_tensor(
                train_df["label"].values, device
            )
            model, train_time = train_model(
                model, model_name, train_loader, val_loader, device, class_weights
            )
        else:
            checkpoint = MODEL_REGISTRY[model_name]["checkpoint"]
            if not checkpoint.exists():
                # Evaluating random weights would overwrite the real metrics JSON
                print(f"[WARNING] No checkpoint found for {model_name}, skipping")
                continue
            print(f"[INFO] Skipping training. Loading from: {checkpoint}")
            model.load_state_dict(
                torch.load(checkpoint, map_location=device, weights_only=True)
            )
            # Keep the recorded training time instead of resetting it to 0
            metrics_file = MODEL_REGISTRY[model_name]["metrics_file"]
            train_time = (
                json.loads(metrics_file.read_text()).get("training_time_seconds", 0.0)
                if metrics_file.exists() else 0.0
            )

        # Evaluate on test set
        print()
        print("=" * 80)
        print(f"  TEST SET EVALUATION: {model_name}")
        print("=" * 80)
        labels, preds, probs = evaluate_on_test_set(model, test_loader, device)
        print_evaluation_report(model_name, labels, preds, probs)
        plot_confusion_matrix(model_name, labels, preds)
        save_model_metrics(
            model_name, labels, preds, probs,
            training_time=train_time,
            total_params=total_params,
            trainable_params=trainable_params,
        )

    # --- Print comparison table (if multiple models) ---
    if len(models_to_train) > 1:
        print_comparison_table()


if __name__ == "__main__":
    main()
