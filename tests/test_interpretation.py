"""SHAP, subgroup and residual analyses.  Run: python -m tests.test_interpretation"""
import numpy as np

from tests.helpers import cohort, run_tests, COV
from src.data import schema
from src.models import make_pipeline
from src.evaluation.interpretation import shap_importances, subgroup_metrics
from src.evaluation.diagnostics import residual_analysis

COLS = COV + schema.all_biomarker_columns()


def _fit(name, n=1200, seed=2):
    c = cohort(n=n, seed=seed)
    tr, te = c.iloc[:900], c.iloc[900:]
    est = make_pipeline(name, COLS).fit(tr[COLS], tr[schema.TARGET_COL])
    return est, tr, te


def test_shap_every_family_covers_all_features():
    for m in ["ridge", "xgboost", "mlp"]:
        est, tr, te = _fit(m)
        imp = shap_importances(m, est, tr, te, COLS, max_explain=40, max_background=20)
        expected = set(COLS) - {schema.SITE_COL} | {f"site_{s}" for s in schema.SITES}
        assert set(imp) == expected, m
        assert all(v >= 0 for v in imp.values())


def test_shap_recovers_injected_top_features():
    # synthetic BAG weights grip strength and gait speed most heavily
    est, tr, te = _fit("ridge", n=1800)
    imp = shap_importances("ridge", est, tr, te, COLS)
    top3 = sorted(imp, key=imp.get, reverse=True)[:3]
    assert {"grip_strength", "gait_speed"} & set(top3), top3


def _pred_df():
    est, tr, te = _fit("ridge")
    return te.assign(y_true=te[schema.TARGET_COL], y_pred=est.predict(te[COLS])).reset_index(drop=True)


def test_subgroups_cover_groups_and_suppress_small_cells():
    pdf = _pred_df()
    rows = subgroup_metrics(pdf, extra_columns=["race"], min_cell=10)
    groups = {r["subgroup"] for r in rows}
    assert {"sex", "age_band", "site", "race"} <= groups
    small = subgroup_metrics(pdf.iloc[:15], min_cell=10)
    for r in small:
        if isinstance(r["n"], str):
            assert r["n"] == "<10" and np.isnan(r["R2"])


def test_residuals_show_regression_to_mean():
    rows = residual_analysis(_pred_df())
    r_true = next(r["value"] for r in rows if r["level"] == "true_bag")
    assert r_true < -0.5                       # classic brain-age regression to the mean
    assert any(r["quantity"] == "mean_residual_by_sex" for r in rows)


if __name__ == "__main__":
    run_tests(globals())
