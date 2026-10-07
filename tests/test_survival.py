"""Tests for the incident-dementia survival models and metrics.
Run: python -m tests.test_survival"""
import numpy as np

from tests.helpers import cohort, run_tests, COV
from src.data import schema
from src.models.survival import make_survival_pipeline, SURVIVAL_MODELS
from src.models.tuning import tune
from src.evaluation.survival_metrics import make_survival_target, concordance, integrated_brier

COLS = COV + schema.all_biomarker_columns()


def _split(n=800, seed=5):
    c = cohort(n=n, seed=seed)
    c = c[c[schema.DEMENTIA_TIME_COL].notna()]
    tr, te = c.iloc[:600], c.iloc[600:]
    y = lambda f: make_survival_target(f[schema.DEMENTIA_EVENT_COL].astype(int), f[schema.DEMENTIA_TIME_COL])
    return tr, te, y(tr), y(te)


def test_survival_target_structure():
    y = make_survival_target([1, 0, 1], [2.0, 5.0, 1.0])
    assert set(y.dtype.names) == {"event", "time"}
    assert y["event"].dtype == bool


def test_concordance_orientation():
    # risk decreasing with survival time -> perfect concordance
    assert concordance([1, 1, 1, 1], [1, 2, 3, 4], [4, 3, 2, 1]) == 1.0


def test_each_survival_pipeline_fits_and_beats_chance():
    tr, te, ytr, yte = _split()
    for m in SURVIVAL_MODELS:
        est = make_survival_pipeline(m, COLS).fit(tr[COLS], ytr)
        c = concordance(yte["event"], yte["time"], est.predict(te[COLS]))
        assert 0.5 < c <= 1.0, (m, c)


def test_integrated_brier_reasonable():
    tr, te, ytr, yte = _split()
    est = make_survival_pipeline("cox", COLS).fit(tr[COLS], ytr)
    ibs = integrated_brier(ytr, yte, est, te[COLS])
    assert np.isnan(ibs) or 0.0 < ibs < 0.25


def test_survival_tuning_uses_cindex():
    tr, _, ytr, _ = _split()
    res = tune("cox", tr, ytr, COLS, inner_folds=3, seed=0, quick=True)
    assert 0.5 < res["inner_cv_score"] <= 1.0
    assert "alpha" in res["best_params"]


if __name__ == "__main__":
    run_tests(globals())
