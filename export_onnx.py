#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_onnx.py -- Build assets for the browser web app (web/)

Exports each trained checkpoint to ONNX, checks ONNX output matches PyTorch,
and writes everything the React app needs into web/public/:
    models/<key>.onnx, data.json (classes + metrics), cm/*.png, examples/*.jpg

Usage (re-run after retraining):
    python export_onnx.py
"""

import json
import shutil

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image

from config import (
    BASE_DIR, MODEL_REGISTRY, MODEL_NAMES, CLASS_CODES, CLASS_NAMES_EN, CLASS_NAMES_TH,
    CLASS_DESCRIPTIONS_EN, CLASS_DESCRIPTIONS_TH, CANCER_INDICES, IMAGE_SIZE,
    IMAGENET_MEAN, IMAGENET_STD,
)
from data import val_test_transform
from models import build_model

OUT = BASE_DIR / "web" / "public"
EXAMPLES = BASE_DIR / "examples"


def model_key(name: str) -> str:
    return name.lower().replace("-", "_")  # "EfficientNet-B3" -> "efficientnet_b3"


def export_one(name: str) -> dict:
    cfg = MODEL_REGISTRY[name]
    model = build_model(name, torch.device("cpu"), pretrained=False)
    model.load_state_dict(torch.load(cfg["checkpoint"], map_location="cpu", weights_only=True))
    model.eval()

    path = OUT / "models" / f"{model_key(name)}.onnx"
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    torch.onnx.export(model, dummy, str(path), input_names=["input"], output_names=["logits"],
                      opset_version=18)
    # Re-save as a single self-contained file (no external .data sidecar) for static hosting
    onnx.save_model(onnx.load(str(path)), str(path), save_as_external_data=False)
    for extra in path.parent.glob(f"{path.name}.data"):
        extra.unlink()

    # Verify ONNX == PyTorch on the sample images
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    max_diff = 0.0
    for img_path in sorted(EXAMPLES.glob("*.jpg")):
        x = val_test_transform(Image.open(img_path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            ref = model(x).numpy()
        got = sess.run(None, {"input": x.numpy()})[0]
        assert ref.argmax() == got.argmax(), f"{name}: prediction mismatch on {img_path.name}"
        max_diff = max(max_diff, float(np.abs(ref - got).max()))
    assert max_diff < 1e-3, f"{name}: ONNX output differs from PyTorch (max diff {max_diff})"

    size_mb = path.stat().st_size / 1e6
    print(f"[OK] {name}: {path.relative_to(BASE_DIR)} ({size_mb:.1f} MB, max diff {max_diff:.2e})")

    metrics = json.loads(cfg["metrics_file"].read_text()) if cfg["metrics_file"].exists() else None
    if cfg["cm_file"].exists():
        shutil.copy(cfg["cm_file"], OUT / "cm" / cfg["cm_file"].name)
    return {
        "name": name,
        "file": f"models/{path.name}",
        "sizeMB": round(size_mb, 1),
        "cm": f"cm/{cfg['cm_file'].name}" if cfg["cm_file"].exists() else None,
        "metrics": metrics,
    }


def main():
    shutil.rmtree(OUT / "examples", ignore_errors=True)  # drop samples that were removed from examples/
    for sub in ("models", "cm", "examples"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    models = [export_one(n) for n in MODEL_NAMES if MODEL_REGISTRY[n]["checkpoint"].exists()]
    assert models, "No trained checkpoints found"

    examples = []
    for img_path in sorted(EXAMPLES.glob("*.jpg")):
        shutil.copy(img_path, OUT / "examples" / img_path.name)
        examples.append({"file": f"examples/{img_path.name}", "label": img_path.stem.split("_")[1]})

    data = {
        "imageSize": IMAGE_SIZE,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "classes": [
            {
                "code": code,
                "nameEn": CLASS_NAMES_EN[i],
                "nameTh": CLASS_NAMES_TH[i],
                "descEn": CLASS_DESCRIPTIONS_EN[i],
                "descTh": CLASS_DESCRIPTIONS_TH[i],
                "cancer": i in CANCER_INDICES,
            }
            for i, code in enumerate(CLASS_CODES)
        ],
        "models": models,
        "examples": examples,
    }
    (OUT / "data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"[OK] Wrote {(OUT / 'data.json').relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
