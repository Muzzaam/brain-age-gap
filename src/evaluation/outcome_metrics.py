"""
Metrics for the downstream cognitive outcome (RQ2, continuous outcome).

  * RMSE      -- prediction error, same units as the cognitive score.
  * R2        -- proportion of cognitive-score variance explained.
  * Pearson r -- correlation between predicted and observed scores.

Comparisons between conditions are made across CV folds with the corrected
resampled t-test (src/evaluation/stats.py).
"""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr


def cognitive_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    r = pearsonr(y_true, y_pred)[0] if np.std(y_pred) > 0 else np.nan
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "Pearson_r": float(r),
    }
