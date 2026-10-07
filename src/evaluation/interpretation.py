"""
Interrogation of the BAG estimator (RQ1 interpretability + honesty checks).

1. SHAP feature attribution -- which inputs drive the BAG estimate, per model
   family (TreeExplainer for XGBoost, LinearExplainer for linear models, the
   generic explainer for the MLP). Computed on each outer fold's held-out rows
   with that fold's fitted model, then averaged over folds, so attributions are
   never computed on data the model was trained on.
2. Subgroup performance breakdown -- BAG-recovery accuracy by sex, age band,
   site (and race / other columns if present), from the out-of-fold
   predictions, to surface failure modes aggregate metrics hide. Cells smaller
   than the minimum reportable size are suppressed.
"""

import numpy as np
import pandas as pd

from ..data import schema
from ..preprocessing import AGE_BANDS, AGE_LABELS
from .metrics import bag_metrics

LINEAR_FAMILIES = ("ridge", "elastic_net")
TREE_FAMILIES = ("xgboost",)


def shap_importances(name, pipeline, X_background, X_explain, columns,
                     max_explain=200, max_background=50, seed=0):
    """Mean |SHAP| per (preprocessed) input feature for a fitted pipeline."""
    import shap
    prep, est = pipeline.named_steps["prep"], pipeline.named_steps["model"]
    Xb = prep.transform(X_background[columns])
    Xe = prep.transform(X_explain[columns])
    names = list(prep.get_feature_names_out())
    rng = np.random.default_rng(seed)
    if name in TREE_FAMILIES:
        vals = shap.TreeExplainer(est).shap_values(Xe)
    elif name in LINEAR_FAMILIES:
        vals = shap.LinearExplainer(est, Xb).shap_values(Xe)
    else:
        bg = Xb[rng.choice(len(Xb), min(max_background, len(Xb)), replace=False)]
        Xe = Xe[rng.choice(len(Xe), min(max_explain, len(Xe)), replace=False)]
        vals = shap.Explainer(est.predict, bg)(Xe, silent=True).values
    imp = np.abs(np.asarray(vals)).mean(axis=0)
    return dict(zip(names, map(float, imp)))


def subgroup_metrics(pred_df, extra_columns=(), min_cell=10):
    """
    BAG-recovery metrics within subgroups, from a frame of pooled out-of-fold
    predictions with columns y_true, y_pred plus the participant attributes.
    Subgroups smaller than `min_cell` are listed but their metrics and exact n
    are suppressed.
    """
    df = pred_df.copy()
    df["age_band"] = pd.cut(df[schema.AGE_COL], bins=AGE_BANDS, labels=AGE_LABELS)
    columns = [schema.SEX_COL, "age_band", schema.SITE_COL] + [
        c for c in extra_columns if c in df.columns]
    rows = []
    for col in columns:
        for level, g in df.groupby(col, observed=True):
            row = {"subgroup": col, "level": str(level)}
            if len(g) >= max(min_cell, 2):
                row.update(n=int(len(g)), **bag_metrics(g["y_true"], g["y_pred"]))
            else:
                row.update(n=f"<{min_cell}", MAE=np.nan, RMSE=np.nan, R2=np.nan)
            rows.append(row)
    return rows
