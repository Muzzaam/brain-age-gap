"""
Evaluation metrics for the BAG-estimation task (RQ1).

Reports the three metrics the proposal commits to:
  * MAE  — average absolute error, in years; robust to outliers.
  * RMSE — standard error metric in the brain-age literature; punishes big misses.
  * R2   — proportion of BAG variance explained; the headline recovery number.

Plus the pre-registered decision: R2 >= 0.20 on the held-out test set is the
threshold for "meaningful fidelity". Pre-registering it means we can't move the
goalposts after seeing the result.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BAG_R2_THRESHOLD = 0.20  # pre-registered "meaningful fidelity" bar


def bag_metrics(y_true, y_pred):
    """Return MAE, RMSE, R2 as a dict."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
        "R2": float(r2_score(y_true, y_pred)),
    }


def meets_threshold(r2, threshold=BAG_R2_THRESHOLD):
    """Does this R2 clear the pre-registered meaningful-fidelity bar?"""
    return r2 >= threshold

