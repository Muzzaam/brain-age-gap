"""RQ2 two-stage design: covariates in every condition, cross-fitting, placebo.
Run: python -m tests.test_two_stage"""
import numpy as np
from sklearn.metrics import r2_score

from tests.helpers import cohort, quick_cfg, run_tests, COV
from src.data import schema
from src.experiment import run_experiment
from src.data import load_data
from src.models import make_pipeline
from src.models.two_stage import (CONDITIONS, condition_columns, cross_fit_bag,
                                  add_bag_features, BAG_HAT_COL, PLACEBO_COL)

COLS = COV + schema.all_biomarker_columns()


def test_every_condition_includes_the_covariates():
    conds = condition_columns(COV)
    assert list(conds) == CONDITIONS
    for name, cols in conds.items():
        assert cols[:len(COV)] == COV, name
    assert BAG_HAT_COL in conds["covariates+biomarkers+BAG"]
    assert PLACEBO_COL in conds["covariates+biomarkers+placebo"]
    # BAG and placebo conditions have identical width: the placebo controls for the extra column
    assert len(conds["covariates+biomarkers+BAG"]) == len(conds["covariates+biomarkers+placebo"])


def test_placebo_has_same_values_but_no_link():
    c = cohort(n=400)
    tr, te = c.iloc[:300], c.iloc[300:]
    bag = tr[schema.TARGET_COL].to_numpy()
    tr2, _ = add_bag_features(tr, te, bag, te[schema.TARGET_COL].to_numpy(), seed=0)
    assert np.allclose(np.sort(tr2[PLACEBO_COL]), np.sort(tr2[BAG_HAT_COL]))
    assert abs(np.corrcoef(tr2[PLACEBO_COL], tr2[BAG_HAT_COL])[0, 1]) < 0.2


def _overfit_xgb():
    pipe = make_pipeline("xgboost", COLS)
    pipe.set_params(model__n_estimators=400, model__max_depth=6, model__learning_rate=0.1,
                    model__min_child_weight=1)
    return pipe


def test_cross_fitting_removes_in_sample_optimism():
    # In-sample stage-1 predictions on the training set look far better than
    # anything stage 2 will see at test time; cross-fitted ones look like test.
    c = cohort(n=1200, seed=1)
    tr, te = c.iloc[:900], c.iloc[900:]
    y_tr, y_te = tr[schema.TARGET_COL].to_numpy(), te[schema.TARGET_COL].to_numpy()
    est = _overfit_xgb().fit(tr[COLS], y_tr)
    in_sample = r2_score(y_tr, est.predict(tr[COLS]))
    oof, test_pred = cross_fit_bag(est, tr[COLS], y_tr, te[COLS], inner_folds=5, seed=0)
    oof_r2, test_r2 = r2_score(y_tr, oof), r2_score(y_te, test_pred)
    assert in_sample > oof_r2 + 0.3, (in_sample, oof_r2)
    assert abs(oof_r2 - test_r2) < 0.12, (oof_r2, test_r2)


def _age_only_world(seed=3):
    """Cognition depends on age only; BAG has a strong age bias. BAG has nothing to add."""
    df = load_data(source="synthetic", n=1200, seed=seed, bag_age_slope=-1.0)
    rng = np.random.default_rng(seed)
    df[schema.COGNITIVE_COL] = 100 - 1.5 * (df[schema.AGE_COL] - 76) + rng.normal(0, 8, len(df))
    return df


def test_leaving_age_out_of_stage_two_fakes_a_bag_benefit():
    # Why the fix matters: if stage 2 lacks age, estimated BAG (which absorbed
    # age from stage 1) looks useful even though cognition depends only on age.
    from src.preprocessing import prepare_cohort
    c = prepare_cohort(_age_only_world())[0]
    tr, te = c.iloc[:900], c.iloc[900:]
    y_tr = tr[schema.TARGET_COL].to_numpy()
    s1 = make_pipeline("ridge", COLS).fit(tr[COLS], y_tr)
    oof, bag_te = cross_fit_bag(s1, tr[COLS], y_tr, te[COLS], inner_folds=5, seed=0)
    tr2, te2 = add_bag_features(tr, te, oof, bag_te)
    bio = schema.all_biomarker_columns()

    def r2(cols):
        p = make_pipeline("ridge", cols).fit(tr2[cols], tr2[schema.COGNITIVE_COL]).predict(te2[cols])
        return r2_score(te2[schema.COGNITIVE_COL], p)
    assert r2(bio + [BAG_HAT_COL]) - r2(bio) > 0.02      # spurious "BAG helps"
    assert abs(r2(COV + bio + [BAG_HAT_COL]) - r2(COV + bio)) < 0.01   # gone once age is in


def test_new_design_does_not_find_bag_benefit_in_age_only_world():
    cfg = quick_cfg(cv={"outer_folds": 5, "repeats": 1, "inner_folds": 3},
                    models={"rq1": ["ridge"], "stage1": "ridge", "stage2_cognitive": ["ridge"]},
                    bias_correction={"variants": ["uncorrected"]})
    res = run_experiment(_age_only_world(), cfg, parts=["rq2_cognitive"], log=lambda *_: None)
    comp = res["rq2_cognitive_comparisons"]
    key = comp[(comp["comparison"] == "covariates+biomarkers+BAG vs covariates+biomarkers")
               & (comp["metric"] == "R2")].iloc[0]
    assert not key["significant"], key.to_dict()
    assert abs(key["improvement"]) < 0.01


if __name__ == "__main__":
    run_tests(globals())
