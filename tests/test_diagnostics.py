"""Tests for RQ1 diagnostics (ablation + residuals).  Run: python -m tests.test_diagnostics"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data import load_data
from src.evaluation.diagnostics import group_ablation, residual_analysis


def test_ablation_structure():
    abl = group_ablation(load_data(source="synthetic", n=800, seed=3), seed=3)
    assert "full" in abl
    for g in ("drop physical_function", "drop cardiovascular", "drop anthropometric"):
        assert g in abl
        assert "R2" in abl[g] and "delta_R2" in abl[g]


def test_physical_function_carries_most_signal():
    # grip + gait (physical function) hold the largest injected weights, so dropping
    # that group should hurt R2 more than dropping anthropometrics.
    abl = group_ablation(load_data(source="synthetic", n=1500, seed=42), seed=42)
    assert abl["drop physical_function"]["delta_R2"] > abl["drop anthropometric"]["delta_R2"]


def test_residual_keys_present():
    res = residual_analysis(load_data(source="synthetic", n=800, seed=3), seed=3)
    for k in ("corr_with_age", "corr_with_true_bag", "mean_residual_by_sex",
              "mean_abs_residual_by_age_band", "corr_with_biomarkers"):
        assert k in res


def test_residual_shows_regression_to_mean():
    # BAG estimators shrink toward the mean -> residual negatively correlates with
    # true BAG (over-estimate low, under-estimate high). Detecting this is the point.
    res = residual_analysis(load_data(source="synthetic", n=1500, seed=42), seed=42)
    assert res["corr_with_true_bag"]["r"] < 0


if __name__ == "__main__":
    for nm, fn in list(globals().items()):
        if nm.startswith("test_") and callable(fn):
            fn()
            print(f"PASSED {nm}")
