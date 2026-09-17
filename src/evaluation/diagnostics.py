"""
RQ1, part two: which biomarker groups carry the signal, and does the estimator
fail in systematic ways?

1. Biomarker-group ablation -- retrain the estimator with each group dropped in
   turn and measure the fall in test R2. A large drop means that group carries a
   lot of the recoverable ageing signal. This is the second half of RQ1 ("which
   biomarker groups carry the most ageing-related signal") and gives a second,
   independent view alongside SHAP: SHAP ranks individual features from one
   fitted model; ablation measures each group's marginal contribution by removing
   it. Where they agree, the finding is robust.

2. Residual analysis -- inspect the estimator's errors (predicted - true BAG) for
   structure. Random residuals are healthy; residuals that trend with age, differ
   by sex, or correlate with a biomarker mean the model is failing in an
   interpretable way (e.g. incomplete bias correction, or age-dependent signal
   the model missed).
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from ..data import schema
from ..preprocessing import AGE_BANDS
from ..models import build
from .metrics import bag_metrics


def _fit_eval_on_columns(df, columns, model_name, seed):
    """Prepare on a restricted feature set and return test metrics + predictions."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from ..preprocessing import exclude_participants, mask_implausible, make_splits

    d = mask_implausible(exclude_participants(df))
    tr, va, te = make_splits(d, seed=seed)
    imp = SimpleImputer(strategy="median").fit(tr[columns])
    sc = StandardScaler().fit(imp.transform(tr[columns]))

    def _t(frame):
        return sc.transform(imp.transform(frame[columns]))

    model = build(model_name).fit(_t(tr), tr[schema.TARGET_COL].to_numpy())
    preds = model.predict(_t(te))
    y_true = te[schema.TARGET_COL].to_numpy()
    return bag_metrics(y_true, preds), y_true, preds, te


def group_ablation(df, model_name="xgboost", seed=42):
    """
    Full-feature R2 vs R2 with each biomarker group dropped.

    Returns {"full": metrics, "drop <group>": {metrics, "delta_R2": ...}, ...}
    where delta_R2 = full_R2 - ablated_R2 (bigger = that group mattered more).
    """
    all_cols = schema.all_biomarker_columns()
    full_metrics, _, _, _ = _fit_eval_on_columns(df, all_cols, model_name, seed)
    result = {"full": full_metrics}

    for group, cols in schema.BIOMARKER_GROUPS.items():
        remaining = [c for c in all_cols if c not in cols]
        m, _, _, _ = _fit_eval_on_columns(df, remaining, model_name, seed)
        m = dict(m)
        m["delta_R2"] = full_metrics["R2"] - m["R2"]
        result[f"drop {group}"] = m
    return result


def residual_analysis(df, model_name="xgboost", seed=42):
    """
    Fit the full-feature estimator and analyse its residuals (pred - true BAG).

    Returns a dict with:
      * corr_with_age / corr_with_true : residual correlations (r, p)
      * mean_residual_by_sex           : mean residual per sex (should be ~0)
      * mean_abs_residual_by_age_band  : |residual| per age band
      * corr_with_biomarkers           : residual vs each biomarker (r, p)
    Systematic structure here flags interpretable failure modes.
    """
    all_cols = schema.all_biomarker_columns()
    _, y_true, preds, te = _fit_eval_on_columns(df, all_cols, model_name, seed)
    resid = preds - y_true  # positive = over-estimated BAG

    age = te[schema.AGE_COL].to_numpy(float)
    out = {}

    r, p = pearsonr(age, resid)
    out["corr_with_age"] = {"r": float(r), "p": float(p)}
    r, p = pearsonr(y_true, resid)
    out["corr_with_true_bag"] = {"r": float(r), "p": float(p)}

    sex = te[schema.SEX_COL].to_numpy()
    out["mean_residual_by_sex"] = {
        s: float(resid[sex == s].mean()) for s in np.unique(sex)
    }

    bands = pd.cut(te[schema.AGE_COL], bins=AGE_BANDS,
                   labels=["<=72", "72-77", "77-82", ">82"])
    out["mean_abs_residual_by_age_band"] = {
        str(b): float(np.abs(resid[bands == b]).mean())
        for b in bands.cat.categories if (bands == b).any()
    }

    bm = {}
    for col in all_cols:
        vals = te[col].to_numpy(float)
        ok = ~np.isnan(vals)
        if ok.sum() > 5:
            r, p = pearsonr(vals[ok], resid[ok])
            bm[col] = {"r": float(r), "p": float(p)}
    out["corr_with_biomarkers"] = bm
    return out
