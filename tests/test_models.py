"""Model pipelines and inner-loop tuning.  Run: python -m tests.test_models"""
import numpy as np
from sklearn.metrics import r2_score

from tests.helpers import cohort, run_tests, COV
from src.data import schema
from src.models import make_pipeline, param_grid, PRINCIPAL_MODELS, MODEL_SPECS, tune

COLS = COV + schema.all_biomarker_columns()


def test_registry_covers_principal_models():
    for m in PRINCIPAL_MODELS:
        assert m in MODEL_SPECS
        assert all(k.startswith("model__") for k in param_grid(m))
        assert all(len(v) == 1 for v in param_grid(m, quick=True).values())


def test_each_pipeline_fits_and_predicts_right_shape():
    c = cohort(n=400)
    tr, te = c.iloc[:300], c.iloc[300:]
    for m in PRINCIPAL_MODELS:
        pred = make_pipeline(m, COLS).fit(tr[COLS], tr[schema.TARGET_COL]).predict(te[COLS])
        assert pred.shape == (len(te),) and np.isfinite(pred).all(), m


def test_tuned_params_are_the_ones_used():
    c = cohort(n=400)
    res = tune("ridge", c, c[schema.TARGET_COL].to_numpy(), COLS, inner_folds=3, seed=0)
    assert res["estimator"].named_steps["model"].alpha == res["best_params"]["alpha"]
    assert np.isfinite(res["inner_cv_score"])


def test_preprocessing_fitted_on_training_rows_only():
    # the imputer inside the tuned pipeline must hold TRAINING medians, so the
    # held-out rows never influenced preprocessing
    c = cohort(n=500)
    tr = c.iloc[:350]
    res = tune("ridge", tr, tr[schema.TARGET_COL].to_numpy(), COLS, inner_folds=3, seed=0)
    imputer = res["estimator"].named_steps["prep"].named_transformers_["num"].named_steps["impute"]
    num_cols = [col for col in COLS if col != schema.SITE_COL]
    assert np.allclose(imputer.statistics_, tr[num_cols].median().to_numpy())


def test_signal_recovered_above_threshold():
    c = cohort(n=1500, seed=4)
    tr, te = c.iloc[:1200], c.iloc[1200:]
    pred = make_pipeline("ridge", COLS).fit(tr[COLS], tr[schema.TARGET_COL]).predict(te[COLS])
    assert r2_score(te[schema.TARGET_COL], pred) > 0.2


def test_mlp_handles_outcome_far_from_zero():
    # cognitive scores sit around 100; the MLP must still match a linear model
    c = cohort(n=1500, seed=6)
    c = c[c[schema.COGNITIVE_COL].notna()]
    tr, te = c.iloc[:1100], c.iloc[1100:]
    r2 = {}
    for m in ["ridge", "mlp"]:
        p = make_pipeline(m, COLS).fit(tr[COLS], tr[schema.COGNITIVE_COL]).predict(te[COLS])
        r2[m] = r2_score(te[schema.COGNITIVE_COL], p)
    assert r2["mlp"] > r2["ridge"] - 0.05, r2


if __name__ == "__main__":
    run_tests(globals())
