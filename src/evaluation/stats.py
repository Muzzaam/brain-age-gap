"""
Statistics over cross-validation folds.

Every headline number is a mean +/- SD over the outer folds (proposal 3.6).
Differences between two models or conditions are tested with the corrected
resampled t-test (Nadeau & Bengio, 2003; Bouckaert & Frank, 2004). A plain
paired t-test over folds is badly over-confident because the folds share most
of their training data; the correction inflates the variance by n_test/n_train
to account for that overlap. It works the same way for every metric (R2,
RMSE, C-index, integrated Brier score), so all comparisons use one method.
"""

import numpy as np
import pandas as pd
from scipy import stats as st

FOLD_KEYS = ["repeat", "fold"]


def corrected_ttest(diffs, test_train_ratio, alpha=0.05):
    """
    Corrected resampled t-test on per-fold differences.

    diffs: one difference per (repeat, fold). test_train_ratio: n_test / n_train.
    Returns mean difference, (1 - alpha) CI, t, two-sided p, and the count.
    """
    d = np.asarray(diffs, float)
    d = d[np.isfinite(d)]
    j = len(d)
    out = {"mean_diff": float(d.mean()) if j else np.nan, "n_estimates": j}
    if j < 2:
        return {**out, "ci_low": np.nan, "ci_high": np.nan, "t": np.nan, "p": np.nan}
    se = np.sqrt((1.0 / j + test_train_ratio) * d.var(ddof=1))
    if se == 0:
        p = 1.0 if out["mean_diff"] == 0 else 0.0
        return {**out, "ci_low": out["mean_diff"], "ci_high": out["mean_diff"],
                "t": np.nan, "p": p}
    t = out["mean_diff"] / se
    half = st.t.ppf(1 - alpha / 2, j - 1) * se
    return {**out, "ci_low": out["mean_diff"] - half, "ci_high": out["mean_diff"] + half,
            "t": float(t), "p": float(2 * st.t.sf(abs(t), j - 1))}


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate)."""
    p = np.asarray(pvalues, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = ok.sum()
    if m == 0:
        return out
    idx = np.where(ok)[0][np.argsort(p[ok])]
    adj = np.maximum.accumulate((m - np.arange(m)) * p[idx])
    out[idx] = np.minimum(adj, 1.0)
    return out


def add_holm(comparisons, family=("variant",)):
    """Add p_holm, adjusting over all comparisons in each family (e.g. per variant)."""
    if comparisons.empty:
        return comparisons
    df = comparisons.copy()
    df["p_holm"] = np.nan
    for _, g in df.groupby(list(family)):
        df.loc[g.index, "p_holm"] = holm(g["p"])
    return df


def summarize(folds, by, metrics):
    """Mean and SD of each metric over folds, grouped by the `by` columns."""
    g = folds.groupby(by, sort=False)[metrics]
    mean = g.mean().add_suffix("_mean")
    sd = g.std(ddof=1).add_suffix("_sd")
    out = pd.concat([mean, sd], axis=1)
    out = out[[f"{m}_{s}" for m in metrics for s in ("mean", "sd")]]
    out["n_folds"] = g.size()
    return out.reset_index()


def compare(folds, by, column, a, b, metric, higher_is_better, alpha=0.05):
    """
    Per-fold comparison of level `a` vs level `b` of `column` on `metric`,
    within each group of `by`. `improvement` is positive when `a` is better
    than `b`, whatever the metric's direction.
    """
    rows = []
    groups = folds.groupby(by, sort=False) if by else [((), folds)]
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        pa = g[g[column] == a].set_index(FOLD_KEYS)[metric]
        pb = g[g[column] == b].set_index(FOLD_KEYS)[metric]
        both = pa.index.intersection(pb.index)
        if len(both) == 0:
            continue
        diff = (pa[both] - pb[both]) * (1 if higher_is_better else -1)
        ga = g[g[column] == a]
        ratio = float((ga["n_test"] / ga["n_train"]).mean())
        res = corrected_ttest(diff.to_numpy(), ratio, alpha)
        res["improvement"] = res.pop("mean_diff")
        rows.append({**dict(zip(by, key)), "comparison": f"{a} vs {b}", "metric": metric,
                     "higher_is_better": higher_is_better, **res,
                     "significant": bool(res["p"] < alpha) if np.isfinite(res["p"]) else False})
    return rows
