"""Self-check for the cancer threshold picker. Run: python test_evaluate.py"""
import numpy as np

from evaluate import _binary_screening, pick_cancer_threshold
from config import CANCER_INDICES

rng = np.random.default_rng(0)
n = 1000
labels = rng.integers(0, 8, n)
probs = rng.dirichlet(np.ones(8), n)
cancer = np.isin(labels, list(CANCER_INDICES))
probs[cancer, 0] += rng.random(cancer.sum()) * 2  # cancers lean towards the cancer group
probs /= probs.sum(axis=1, keepdims=True)

for target in (0.8, 0.9, 0.95):
    t = pick_cancer_threshold(labels, probs, target)
    bl, bp, _ = _binary_screening(labels, probs, t)
    sens = bp[bl == 1].mean()
    assert 0 < t <= 0.5, t
    assert sens >= target, (target, t, sens)
    print(f"[OK] target {target:.0%}: threshold {t:.4f}, sensitivity {sens:.1%}")

# A model that already catches every cancer at 0.5 keeps the 0.5 rule
easy = np.zeros((4, 8)); easy[:2, 0] = 1; easy[2:, 4] = 1
assert pick_cancer_threshold(np.array([0, 0, 4, 4]), easy) == 0.5
print("[OK] capped at 0.5")
