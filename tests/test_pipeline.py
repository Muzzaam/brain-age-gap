"""Minimal sanity tests. Run: python -m pytest -q  (or python -m tests.test_pipeline)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data import load_data, schema
from src.preprocessing import prepare_data, exclude_participants


def test_schema_conformance():
    df = load_data(source="synthetic", n=300, seed=0)
    for col in schema.REQUIRED_COLUMNS:
        assert col in df.columns, f"missing {col}"


def test_exclusions_applied():
    df = load_data(source="synthetic", n=500, seed=1)
    kept = exclude_participants(df)
    # no prevalent-dementia participants survive
    assert (kept[schema.PREVALENT_DEMENTIA_COL] == 0).all()
    # nobody kept is missing an entire biomarker group
    for cols in schema.BIOMARKER_GROUPS.values():
        assert not kept[cols].isna().all(axis=1).any()


def test_no_nans_after_preprocessing():
    df = load_data(source="synthetic", n=600, seed=2)
    data = prepare_data(df, seed=2)
    for key in ("X_train", "X_val", "X_test"):
        assert not np.isnan(data[key]).any(), f"NaNs remain in {key}"


def test_signal_is_recoverable():
    # With injected signal fraction 0.5, a linear model should clear a low bar.
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    df = load_data(source="synthetic", n=2000, seed=3, target_signal_fraction=0.5)
    data = prepare_data(df, seed=3)
    pred = Ridge().fit(data["X_train"], data["y_train"]).predict(data["X_test"])
    assert r2_score(data["y_test"], pred) > 0.2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASSED {name}")
