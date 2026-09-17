"""
Interrogation of the BAG estimator (RQ1 interpretability + honesty checks).

1. SHAP feature attribution -- which biomarkers drive the BAG estimate, per
   model family (TreeExplainer for XGBoost, LinearExplainer for linear models,
   the generic explainer for the MLP). Computed per family so agreement (robust)
   and disagreement (reported) are both visible.
2. Subgroup performance breakdown -- BAG-recovery accuracy by sex, age band and
   site, to surface failure modes aggregate metrics hide.
"""

import numpy as np
import pandas as pd

from ..data import schema
from .metrics import bag_metrics

AGE_BANDS = [0, 72, 77, 82, 200]
AGE_LABELS = ["<=72", "72-77", "77-82", ">82"]


def shap_importances(model, X_background, X_explain, feature_names,
                     max_explain=200, max_background=50, seed=0):
    """Mean absolute SHAP value per feature for a fitted BAG model."""
    import shap
    name = getattr(model, "name", "")
    if name == "xgboost":
        vals = shap.TreeExplainer(model.estimator).shap_values(X_explain)
    elif name in ("ridge", "elastic_net"):
        vals = shap.LinearExplainer(model.estimator, X_background).shap_values(X_explain)
    else:
        rng = np.random.default_rng(seed)
        k = min(max_background, len(X_background))
        bg = X_background[rng.choice(len(X_background), k, replace=False)]
        Xe = X_explain[:max_explain]
        vals = shap.Explainer(model.predict, bg)(Xe, silent=True).values
    imp = np.abs(np.asarray(vals)).mean(axis=0)
    return {f: float(v) for f, v in zip(feature_names, imp)}


def shap_importance_table(models_fitted, X_background, X_explain, feature_names):
    """Feature x model table of SHAP importances, sorted by mean importance."""
    cols = {m: shap_importances(model, X_background, X_explain, feature_names)
            for m, model in models_fitted.items()}
    df = pd.DataFrame(cols)
    df["mean"] = df.mean(axis=1)
    return df.sort_values("mean", ascending=False)


def subgroup_metrics(test_df, y_true, y_pred, min_n=2):
    """BAG-recovery metrics within subgroups (sex, age band, site)."""
    df = test_df.copy()
    df["_yt"] = np.asarray(y_true, dtype=float)
    df["_yp"] = np.asarray(y_pred, dtype=float)
    out = {}

    def _add(prefix, key, g):
        if len(g) >= min_n:
            m = bag_metrics(g["_yt"], g["_yp"]); m["n"] = int(len(g))
            out[f"{prefix}={key}"] = m

    for sex, g in df.groupby(schema.SEX_COL):
        _add("sex", sex, g)
    bands = pd.cut(df[schema.AGE_COL], bins=AGE_BANDS, labels=AGE_LABELS)
    for band, g in df.groupby(bands, observed=True):
        _add("age", band, g)
    for site, g in df.groupby(schema.SITE_COL):
        _add("site", site, g)
    return out
