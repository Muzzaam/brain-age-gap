"""
Metrics for the incident-dementia outcome (RQ2, time-to-event).

  * C-index (Harrell)      -- discrimination: probability the model ranks a
                              higher-risk person as failing sooner. 0.5 = chance.
  * Integrated Brier score -- calibration + accuracy of the predicted survival
                              curves, averaged over time. Lower is better;
                              0.25 is uninformative.

Differences between conditions are tested across CV folds with the corrected
resampled t-test (src/evaluation/stats.py). DeLong's test, named in the
proposal, applies to binary AUCs, not to censored C-indices, so it is not used.
"""

import numpy as np
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored, integrated_brier_score


def make_survival_target(events, times):
    """Build the structured (event, time) array scikit-survival expects."""
    return Surv.from_arrays(event=np.asarray(events).astype(bool),
                            time=np.asarray(times, dtype=float))


def concordance(events_test, times_test, risk_scores):
    """Harrell's C-index. risk_scores: higher = higher predicted risk."""
    return float(concordance_index_censored(
        np.asarray(events_test).astype(bool),
        np.asarray(times_test, float),
        np.asarray(risk_scores, float),
    )[0])


def integrated_brier(y_train, y_test, model, X_test, n_times=10):
    """
    Integrated Brier score over a safe interior time grid.

    The grid must sit strictly inside the follow-up range of both train and test
    (the IPCW censoring weights are undefined outside it). Returns nan if a valid
    grid can't be built (e.g. too few events).
    """
    train_times = y_train["time"]
    test_times = y_test["time"]
    test_events = y_test["event"]
    if not test_events.any():
        return float("nan")

    lo = max(train_times.min(), test_times.min())
    hi = min(train_times.max(), test_times.max())
    grid = np.percentile(test_times[test_events], np.linspace(10, 90, n_times))
    grid = np.unique(grid[(grid > lo) & (grid < hi)])
    if grid.size < 2:
        return float("nan")

    surv_funcs = model.predict_survival_function(X_test)
    preds = np.asarray([[fn(t) for t in grid] for fn in surv_funcs])
    return float(integrated_brier_score(y_train, y_test, preds, grid))
