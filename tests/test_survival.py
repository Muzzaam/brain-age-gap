"""Tests for the incident-dementia survival harness.  Run: python -m tests.test_survival"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data import load_data, schema
from src.preprocessing import prepare_data
from src.models.two_stage import estimate_bag, build_conditions
from src.models.survival import build_survival, SURVIVAL_MODELS
from src.evaluation.survival_metrics import (
    make_survival_target, concordance, integrated_brier, bootstrap_cindex_difference,
)
from scripts.run_rq2_survival import run


def _prep(n=800, seed=5):
    d = prepare_data(load_data(source="synthetic", n=n, seed=seed), seed=seed)
    ev = d["train_df"][schema.DEMENTIA_EVENT_COL].to_numpy()
    tt = d["train_df"][schema.DEMENTIA_TIME_COL].to_numpy()
    eve = d["test_df"][schema.DEMENTIA_EVENT_COL].to_numpy()
    tte = d["test_df"][schema.DEMENTIA_TIME_COL].to_numpy()
    return d, make_survival_target(ev, tt), make_survival_target(eve, tte), eve, tte


def test_survival_target_structure():
    y = make_survival_target([1, 0, 1], [2.0, 5.0, 1.0])
    assert set(y.dtype.names) == {"event", "time"}
    assert y["event"].dtype == bool


def test_concordance_orientation():
    # risk decreasing with survival time -> perfect concordance
    c = concordance([1, 1, 1, 1], [1, 2, 3, 4], [4, 3, 2, 1])
    assert c == 1.0


def test_cox_fits_and_cindex_in_range():
    d, y_tr, y_te, eve, tte = _prep()
    Xtr, Xte = build_conditions(d, estimate_bag(d))["biomarkers"]
    model = build_survival("cox").fit(Xtr, y_tr)
    c = concordance(eve, tte, model.predict_risk(Xte))
    assert 0.0 <= c <= 1.0 and c > 0.5   # biomarkers should beat chance


def test_integrated_brier_reasonable():
    d, y_tr, y_te, eve, tte = _prep()
    Xtr, Xte = build_conditions(d, estimate_bag(d))["biomarkers"]
    model = build_survival("cox").fit(Xtr, y_tr)
    ibs = integrated_brier(y_tr, y_te, model, Xte)
    assert np.isfinite(ibs) and 0.0 < ibs < 0.3


def test_bootstrap_keys():
    d, y_tr, y_te, eve, tte = _prep()
    conds = build_conditions(d, estimate_bag(d))
    m = build_survival("cox")
    r_base = m.fit(*[conds["biomarkers"][0], y_tr]).predict_risk(conds["biomarkers"][1])
    r_aug = build_survival("cox").fit(conds["biomarkers+BAG"][0], y_tr).predict_risk(conds["biomarkers+BAG"][1])
    res = bootstrap_cindex_difference(eve, tte, r_base, r_aug, n_boot=50)
    assert set(res) == {"delta_mean", "ci_low", "ci_high", "p_no_improve"}
    assert 0.0 <= res["p_no_improve"] <= 1.0


def test_run_end_to_end():
    report = run(n=600, seed=3, n_boot=40)
    for name in SURVIVAL_MODELS:
        assert name in report
        for cond in ("age_only", "biomarkers", "biomarkers+BAG", "biomarkers+placebo"):
            c = report[name]["metrics"][cond]["C_index"]
            assert np.isfinite(c) and 0.0 <= c <= 1.0


if __name__ == "__main__":
    for nm, fn in list(globals().items()):
        if nm.startswith("test_") and callable(fn):
            fn()
            print(f"PASSED {nm}")
