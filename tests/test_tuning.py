"""Tests for CV hyperparameter tuning.  Run: python -m tests.test_tuning"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data import load_data
from src.preprocessing import prepare_data
from src.models.tuning import tune_model, tune_all
from src.evaluation.metrics import bag_metrics


def _data(n=600, seed=3):
    return prepare_data(load_data(source="synthetic", n=n, seed=seed), seed=seed)


def test_tune_model_returns_expected_fields():
    d = _data()
    r = tune_model("ridge", d["X_train"], d["y_train"], cv=3)
    assert set(r) >= {"name", "best_params", "cv_r2", "estimator"}
    assert isinstance(r["cv_r2"], float)
    assert "alpha" in r["best_params"]


def test_tuned_estimator_predicts_on_test():
    d = _data()
    r = tune_model("ridge", d["X_train"], d["y_train"], cv=3)
    preds = r["estimator"].predict(d["X_test"])
    assert preds.shape[0] == d["X_test"].shape[0]
    assert np.isfinite(bag_metrics(d["y_test"], preds)["R2"])


def test_tune_all_sorted_by_cv_r2():
    d = _data()
    res = tune_all(d["X_train"], d["y_train"], models=["ridge", "elastic_net"], cv=3)
    assert set(res) == {"ridge", "elastic_net"}
    scores = [r["cv_r2"] for r in res.values()]
    assert scores == sorted(scores, reverse=True)   # best first


if __name__ == "__main__":
    for nm, fn in list(globals().items()):
        if nm.startswith("test_") and callable(fn):
            fn()
            print(f"PASSED {nm}")
