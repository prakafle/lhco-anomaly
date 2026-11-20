import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lhco.methods import classifier_scores, max_sic, sic_curve


def test_sic_perfect_and_random():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(1000), np.zeros(100000)]
    perfect = y + rng.normal(0, 0.01, len(y))
    assert max_sic(perfect, y) > 30
    rand = rng.random(len(y))
    assert max_sic(rand, y) < 5


def test_classifier_separates_shifted_gaussians():
    rng = np.random.default_rng(0)
    a = rng.normal(1, 1, (2000, 2)); b = rng.normal(0, 1, (2000, 2))
    Xt = np.vstack([rng.normal(1, 1, (500, 2)), rng.normal(0, 1, (500, 2))])
    s = classifier_scores(a, b, Xt, n_models=2)
    assert s[:500].mean() > s[500:].mean() + 0.1
