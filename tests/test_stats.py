"""Statistics over CV folds.  Run: python -m tests.test_stats"""
import numpy as np
import pandas as pd
from scipy import stats as st

from tests.helpers import run_tests
from src.evaluation.stats import corrected_ttest, summarize, compare, holm


def test_corrected_ttest_wider_than_naive():
    rng = np.random.default_rng(0)
    d = rng.normal(0.02, 0.03, 15)
    res = corrected_ttest(d, test_train_ratio=0.25)
    naive_p = st.ttest_1samp(d, 0).pvalue
    assert res["p"] > naive_p                                    # correction is more conservative
    assert res["ci_low"] < res["mean_diff"] < res["ci_high"]


def test_corrected_ttest_matches_formula():
    d = np.array([0.1, 0.2, 0.0, 0.15, 0.05])
    res = corrected_ttest(d, test_train_ratio=0.25)
    se = np.sqrt((1 / 5 + 0.25) * d.var(ddof=1))
    assert np.isclose(res["t"], d.mean() / se)


def test_zero_differences_give_p_one():
    assert corrected_ttest(np.zeros(10), 0.25)["p"] == 1.0


def test_compare_sign_follows_metric_direction():
    rows = []
    for f in range(5):
        rows += [{"repeat": 0, "fold": f, "cond": "a", "RMSE": 1.0, "R2": 0.3, "n_train": 80, "n_test": 20},
                 {"repeat": 0, "fold": f, "cond": "b", "RMSE": 1.2 + 0.01 * f, "R2": 0.2, "n_train": 80, "n_test": 20}]
    df = pd.DataFrame(rows)
    rmse = compare(df, [], "cond", "a", "b", "RMSE", higher_is_better=False)[0]
    r2 = compare(df, [], "cond", "a", "b", "R2", higher_is_better=True)[0]
    assert rmse["improvement"] > 0 and r2["improvement"] > 0        # "a" is better on both


def test_summarize_mean_sd():
    df = pd.DataFrame({"g": ["x"] * 4, "R2": [0.1, 0.2, 0.3, 0.4]})
    s = summarize(df, ["g"], ["R2"]).iloc[0]
    assert np.isclose(s["R2_mean"], 0.25) and np.isclose(s["R2_sd"], np.std([0.1, 0.2, 0.3, 0.4], ddof=1))
    assert s["n_folds"] == 4


def test_holm_adjustment():
    adj = holm([0.01, 0.04, 0.03, np.nan])
    assert np.allclose(adj[:3], [0.03, 0.06, 0.06]) and np.isnan(adj[3])
    assert (adj[:3] >= np.array([0.01, 0.04, 0.03])).all()


if __name__ == "__main__":
    run_tests(globals())
