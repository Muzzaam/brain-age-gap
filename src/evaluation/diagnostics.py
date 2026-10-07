"""
Residual analysis of the BAG estimator (RQ1 interrogation).

Inspects the out-of-fold errors (predicted - true BAG) for structure. Random
residuals are healthy; residuals that trend with age, differ by sex, or
correlate with a biomarker mean the model is failing in an interpretable way
(e.g. incomplete bias correction, or age-dependent signal the model missed).

The biomarker-group ablation (the other half of "which groups carry the
signal") is now part of the RQ1 feature-set comparison in src/experiment.py,
so it is tuned and cross-validated exactly like every other model.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from ..data import schema
from ..preprocessing import AGE_BANDS, AGE_LABELS


def residual_analysis(pred_df):
    """
    Long-format rows (quantity, level, value, p) describing residual structure.
    pred_df holds pooled out-of-fold predictions: y_true, y_pred, age, sex,
    biomarkers.
    """
    resid = (pred_df["y_pred"] - pred_df["y_true"]).to_numpy()  # positive = over-estimated BAG
    rows = []

    def _corr(label, x):
        x = np.asarray(x, float)
        ok = np.isfinite(x)
        if ok.sum() > 5 and np.std(x[ok]) > 0:
            r, p = pearsonr(x[ok], resid[ok])
            rows.append({"quantity": "corr_residual_with", "level": label,
                         "value": float(r), "p": float(p)})

    _corr("age", pred_df[schema.AGE_COL])
    _corr("true_bag", pred_df["y_true"])  # regression to the mean shows up here
    for col in schema.all_biomarker_columns():
        _corr(col, pred_df[col])

    for sex, g in pred_df.groupby(schema.SEX_COL):
        rows.append({"quantity": "mean_residual_by_sex", "level": str(sex),
                     "value": float(resid[g.index.to_numpy()].mean()), "p": np.nan})
    bands = pd.cut(pred_df[schema.AGE_COL], bins=AGE_BANDS, labels=AGE_LABELS)
    for band in AGE_LABELS:
        m = (bands == band).to_numpy()
        if m.any():
            rows.append({"quantity": "mean_abs_residual_by_age_band", "level": band,
                         "value": float(np.abs(resid[m]).mean()), "p": np.nan})
    return rows
