"""Tests for bias correction.  Run: python -m tests.test_bias_correction"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data import load_data, schema
from src.preprocessing import prepare_data
from src.evaluation.bias_correction import fit_bias_correction, correct_bag, age_bias


def _bag_and_age(n=1500, seed=42):
    d = prepare_data(load_data(source="synthetic", n=n, seed=seed), seed=seed)
    age_tr = d["train_df"][schema.AGE_COL].to_numpy(float)
    age_te = d["test_df"][schema.AGE_COL].to_numpy(float)
    return d["y_train"], age_tr, d["y_test"], age_te


def test_fit_returns_params():
    bag_tr, age_tr, _, _ = _bag_and_age()
    p = fit_bias_correction(bag_tr, age_tr)
    assert "slope" in p and "intercept" in p


def test_correction_zeroes_age_bias_in_sample():
    # by the OLS residual property, fitting and applying on the same data leaves
    # the corrected BAG uncorrelated with age.
    bag_tr, age_tr, _, _ = _bag_and_age()
    p = fit_bias_correction(bag_tr, age_tr)
    corr = age_bias(correct_bag(bag_tr, age_tr, p), age_tr)
    assert abs(corr) < 1e-3


def test_correction_reduces_age_bias_out_of_sample():
    bag_tr, age_tr, bag_te, age_te = _bag_and_age()
    p = fit_bias_correction(bag_tr, age_tr)
    before = abs(age_bias(bag_te, age_te))
    after = abs(age_bias(correct_bag(bag_te, age_te, p), age_te))
    assert after < before


def test_age_bias_range():
    bag_tr, age_tr, _, _ = _bag_and_age()
    r = age_bias(bag_tr, age_tr)
    assert -1.0 <= r <= 1.0


if __name__ == "__main__":
    for nm, fn in list(globals().items()):
        if nm.startswith("test_") and callable(fn):
            fn()
            print(f"PASSED {nm}")
