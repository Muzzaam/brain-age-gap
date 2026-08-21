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


def format_metrics_table(results):
    """Pretty-print a name -> metrics dict as an aligned table.

    `results` is like {"xgboost": {"MAE":.., "RMSE":.., "R2":..}, ...}.
    """
    lines = [f"{'model':<14}{'MAE':>8}{'RMSE':>8}{'R2':>8}  {'>=0.20':>7}"]
    lines.append("-" * 47)
    for name, m in results.items():
        flag = "yes" if meets_threshold(m["R2"]) else "no"
        lines.append(f"{name:<14}{m['MAE']:>8.3f}{m['RMSE']:>8.3f}{m['R2']:>8.3f}  {flag:>7}")
    return "\n".join(lines)
