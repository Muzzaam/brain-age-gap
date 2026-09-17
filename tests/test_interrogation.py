"""Tests for the interrogation layer.  Run: python -m tests.test_interrogation"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_data, schema
from src.preprocessing import prepare_data
from src.models import build
from src.evaluation.interpretation import shap_importances, shap_importance_table, subgroup_metrics
def _fit(n=800, seed=5):
    d = prepare_data(load_data(source="synthetic", n=n, seed=seed), seed=seed)
    return d, build("xgboost").fit(d["X_train"], d["y_train"])
def test_shap_importances_shape_and_sign():
    d, xm = _fit(); imp = shap_importances(xm, d["X_train"], d["X_test"], d["feature_names"])
    assert set(imp) == set(d["feature_names"])
    assert all(v >= 0 for v in imp.values()) and sum(imp.values()) > 0
def test_shap_recovers_injected_top_feature():
    d, xm = _fit(); imp = shap_importances(xm, d["X_train"], d["X_test"], d["feature_names"])
    assert "grip_strength" in sorted(imp, key=imp.get, reverse=True)[:2]
def test_shap_table_structure():
    d, _ = _fit()
    fitted = {m: build(m).fit(d["X_train"], d["y_train"]) for m in ("ridge", "xgboost")}
    table = shap_importance_table(fitted, d["X_train"], d["X_test"], d["feature_names"])
    assert list(table.columns) == ["ridge", "xgboost", "mean"]
    assert set(table.index) == set(d["feature_names"])
def test_subgroup_metrics_cover_groups():
    d, xm = _fit(); subs = subgroup_metrics(d["test_df"], d["y_test"], xm.predict(d["X_test"]))
    assert "sex=M" in subs and "sex=F" in subs
    assert any(k.startswith("site=") for k in subs) and any(k.startswith("age=") for k in subs)
    for m in subs.values(): assert set(m) == {"MAE","RMSE","R2","n"} and m["n"] >= 2
if __name__ == "__main__":
    for nm, fn in list(globals().items()):
        if nm.startswith("test_") and callable(fn): fn(); print(f"PASSED {nm}")
