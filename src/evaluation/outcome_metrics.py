"""
Metrics for the downstream cognitive outcome (RQ2, continuous outcome).

  * RMSE      -- prediction error, same units as the cognitive score.
  * Pearson r -- correlation between predicted and observed scores.

Plus a paired test for the key RQ2 comparison: does adding BAG (condition 3)
significantly reduce per-participant error versus a baseline (condition 2)?
We use a Wilcoxon signed-rank test on the paired absolute errors, which makes
no normality assumption and respects that both models predict the SAME test
participants.
"""

import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, wilcoxon


def cognitive_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "Pearson_r": float(pearsonr(y_true, y_pred)[0]),
    }


def paired_error_test(y_true, pred_baseline, pred_augmented):
    """
    Compare per-participant absolute errors of two models on the same test set.

    Returns the RMSE of each, their difference (positive = augmented is better),
    and a Wilcoxon signed-rank p-value on the paired absolute errors.
    """
    y_true = np.asarray(y_true, float)
    err_base = np.abs(y_true - np.asarray(pred_baseline, float))
    err_aug = np.abs(y_true - np.asarray(pred_augmented, float))

    rmse_base = float(np.sqrt(np.mean(err_base ** 2)))
    rmse_aug = float(np.sqrt(np.mean(err_aug ** 2)))

    if np.allclose(err_base, err_aug):
        p = 1.0
    else:
        p = float(wilcoxon(err_base, err_aug).pvalue)

    return {
        "rmse_baseline": rmse_base,
        "rmse_augmented": rmse_aug,
        "rmse_improvement": rmse_base - rmse_aug,  # >0 means augmented helps
        "wilcoxon_p": p,
    }
