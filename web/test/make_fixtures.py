"""Write resize_fixtures.json: random images + torchvision's expected tensors.

Run from the repo root after changing preprocessing:  python web/test/make_fixtures.py
"""
import json
from pathlib import Path

import numpy as np
import torchvision.transforms as T
from PIL import Image

MEAN, STD, SIZE = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 32
tf = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])  # same as data.val_test_transform

rng = np.random.default_rng(0)
cases = []
for w, h in [(75, 49), (20, 15), (32, 32)]:  # downscale, upscale, identity
    rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    rgba = np.concatenate([rgb, np.full((h, w, 1), 255, np.uint8)], axis=2)
    expected = tf(Image.fromarray(rgb)).numpy().ravel()
    cases.append({"w": w, "h": h, "rgba": rgba.ravel().tolist(), "expected": np.round(expected, 6).tolist()})

out = Path(__file__).with_name("resize_fixtures.json")
out.write_text(json.dumps({"size": SIZE, "mean": MEAN, "std": STD, "cases": cases}))
print(f"[OK] Wrote {out}")
