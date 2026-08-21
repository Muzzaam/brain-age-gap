"""Tests for the model families and BAG metrics.
Run: python -m tests.test_models  (or pytest -q)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data import load_data
from src.preprocessing import prepare_data
from src.models.estimators import build, PRINCIPAL_MODELS, MODEL_BUILDERS
from src.evaluation.metrics import bag_metrics, meets_threshold


def _data(n=1500, seed=7, signal=0.30):
    df = load_data(source="synthetic", n=n, seed=seed, target_signal_fraction=signal)
    return prepare_data(df, seed=seed)


def test_all_models_build_and_have_names():
    for name in PRINCIPAL_MODELS:
        m = build(name)
        assert m.name == name
        assert hasattr(m, "fit") and hasattr(m, "predict")


def test_each_model_fits_and_predicts_right_shape():
    d = _data()
    for name in PRINCIPAL_MODELS:
        preds = build(name).fit(d["X_train"], d["y_train"]).predict(d["X_test"])
        assert preds.shape[0] == d["X_test"].shape[0]
        assert np.isfinite(preds).all()


def test_metrics_keys_and_ranges():
    d = _data()
    preds = build("ridge").fit(d["X_train"], d["y_train"]).predict(d["X_test"])
    m = bag_metrics(d["y_test"], preds)
    assert set(m) == {"MAE", "RMSE", "R2"}
    assert m["MAE"] > 0 and m["RMSE"] >= m["MAE"]  # RMSE >= MAE always


def test_signal_recovered_above_threshold():
    # With a strong injected signal, the strong baselines must clear the bar.
    d = _data(signal=0.5)
    for name in ("ridge", "xgboost"):
        preds = build(name).fit(d["X_train"], d["y_train"]).predict(d["X_test"])
        r2 = bag_metrics(d["y_test"], preds)["R2"]
        assert meets_threshold(r2), f"{name} R2={r2:.3f} below threshold"


def test_registry_covers_principal_models():
    for name in PRINCIPAL_MODELS:
        assert name in MODEL_BUILDERS


if __name__ == "__main__":
    for nm, fn in list(globals().items()):
        if nm.startswith("test_") and callable(fn):
            fn()
            print(f"PASSED {nm}")
