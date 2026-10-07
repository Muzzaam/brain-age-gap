"""RQ3 neural framings.  Run: python -m tests.test_neural"""
import numpy as np
import torch
from sklearn.base import clone
from sklearn.metrics import r2_score

from tests.helpers import cohort, quick_cfg, run_tests, COV
from src.data import schema, load_data
from src.experiment import run_experiment
from src.models.framings import RQ3_FRAMINGS, framing_grid, make_framing_pipeline
from src.models.neural import NeuralFraming, cox_loss, make_targets
from src.evaluation.survival_metrics import concordance

COLS = COV + schema.all_biomarker_columns()
FAST = {"max_epochs": 300, "patience": 20}


def _data(n=1200, seed=1):
    c = cohort(n=n, seed=seed)
    c = c[c[schema.COGNITIVE_COL].notna() & c[schema.DEMENTIA_TIME_COL].notna()]
    k = int(0.75 * len(c))
    return c.iloc[:k], c.iloc[k:]


def _y(task, f):
    return make_targets(task, f[schema.TARGET_COL], outcome=f[schema.COGNITIVE_COL],
                        event=f[schema.DEMENTIA_EVENT_COL], time=f[schema.DEMENTIA_TIME_COL])


def test_cox_loss_prefers_correct_ordering():
    time = torch.tensor([1.0, 2.0, 3.0, 4.0])
    event = torch.ones(4)
    good = cox_loss(torch.tensor([3.0, 2.0, 1.0, 0.0]), time, event)  # earliest failure = highest risk
    bad = cox_loss(torch.tensor([0.0, 1.0, 2.0, 3.0]), time, event)
    assert good < bad


def test_estimator_is_sklearn_clonable():
    est = NeuralFraming(framing="bottleneck", bag_weight=0.25)
    c = clone(est)
    assert c.get_params()["bag_weight"] == 0.25 and c.framing == "bottleneck"


def test_every_framing_fits_both_outcomes():
    tr, te = _data()
    for task in ["regression", "survival"]:
        for name in RQ3_FRAMINGS:
            if RQ3_FRAMINGS[name][1]:   # two_stage needs bag_hat; covered end to end below
                continue
            p = make_framing_pipeline(name, task, COLS, FAST).fit(tr[COLS], _y(task, tr))
            pred = p.predict(te[COLS])
            assert pred.shape == (len(te),) and np.isfinite(pred).all(), (task, name)


def test_baseline_matches_a_linear_model_on_cognition():
    tr, te = _data(n=1900)
    p = make_framing_pipeline("baseline", "regression", COLS, FAST).fit(tr[COLS], _y("regression", tr))
    assert r2_score(te[schema.COGNITIVE_COL], p.predict(te[COLS])) > 0.12


def test_bag_head_learns_bag_only_when_supervised():
    tr, te = _data(n=1900)
    r2 = {}
    for name in ["baseline", "multitask"]:
        p = make_framing_pipeline(name, "regression", COLS, FAST).fit(tr[COLS], _y("regression", tr))
        r2[name] = r2_score(te[schema.TARGET_COL], p[-1].predict_bag(p[:-1].transform(te[COLS])))
    assert r2["multitask"] > 0.15 and r2["baseline"] < 0.05, r2


def test_bottleneck_ablation_changes_predictions():
    tr, te = _data()
    p = make_framing_pipeline("bottleneck", "regression", COLS, FAST).fit(tr[COLS], _y("regression", tr))
    Xt = p[:-1].transform(te[COLS])
    assert not np.allclose(p[-1].predict(Xt), p[-1].predict(Xt, ablate=True))


def test_transfer_pretrains_then_finetunes():
    tr, te = _data(n=1900)
    p = make_framing_pipeline("transfer", "survival", COLS, FAST).fit(tr[COLS], _y("survival", tr))
    Xt = p[:-1].transform(te[COLS])
    assert r2_score(te[schema.TARGET_COL], p[-1].predict_bag(Xt)) > 0.1    # BAG pretraining took
    c = concordance(te[schema.DEMENTIA_EVENT_COL], te[schema.DEMENTIA_TIME_COL], p.predict(te[COLS]))
    assert c > 0.55                                                     # and the outcome head works


def test_grids_tune_bag_weight_only_where_it_matters():
    assert "model__bag_weight" in framing_grid("multitask")
    assert "model__bag_weight" in framing_grid("bottleneck")
    assert "model__bag_weight" not in framing_grid("baseline")


def test_rq3_end_to_end_tables():
    cfg = quick_cfg(bias_correction={"variants": ["uncorrected"]})
    df = load_data(source="synthetic", n=500, seed=0)
    res = run_experiment(df, cfg, parts=["rq3_cognitive", "rq3_dementia"], log=lambda *_: None)
    for part in ["rq3_cognitive", "rq3_dementia"]:
        framings = set(res[f"{part}_summary"]["framing"])
        assert set(RQ3_FRAMINGS) | {"bottleneck_ablated"} <= framings, framings
        comps = set(res[f"{part}_comparisons"]["comparison"])
        assert "multitask vs multitask_placebo" in comps and "bottleneck vs bottleneck_ablated" in comps


if __name__ == "__main__":
    run_tests(globals())
