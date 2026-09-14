"""
Metrics for the incident-dementia outcome (RQ2, time-to-event).

  * C-index (Harrell)      -- discrimination: probability the model ranks a
                              higher-risk person as failing sooner. 0.5 = chance.
  * Integrated Brier score -- calibration + accuracy of the predicted survival
                              curves, averaged over time. Lower is better;
                              0.25 is uninformative.

C-index isn't a per-participant quantity, so the "does BAG help?" comparison
can't use a paired Wilcoxon like the cognitive outcome. Instead we bootstrap the
test set and look at the distribution of the C-index difference.
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


def bootstrap_cindex_difference(events, times, risk_base, risk_aug, n_boot=500, seed=42):
    """
    Bootstrap the C-index difference (augmented - baseline) on the test set.

    Returns the mean difference, a 95% CI, and p_no_improve = the fraction of
    resamples where BAG did NOT help (delta <= 0). Small p_no_improve + a CI
    above 0 means BAG genuinely improves discrimination.
    """
    rng = np.random.default_rng(seed)
    events = np.asarray(events).astype(bool)
    times = np.asarray(times, float)
    risk_base = np.asarray(risk_base, float)
    risk_aug = np.asarray(risk_aug, float)
    n = len(times)

    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if events[idx].sum() < 2:      # need comparable pairs
            continue
        try:
            cb = concordance_index_censored(events[idx], times[idx], risk_base[idx])[0]
            ca = concordance_index_censored(events[idx], times[idx], risk_aug[idx])[0]
        except Exception:              # degenerate resample
            continue
        deltas.append(ca - cb)

    deltas = np.asarray(deltas)
    if deltas.size == 0:
        return {"delta_mean": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "p_no_improve": float("nan")}
    return {
        "delta_mean": float(deltas.mean()),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "p_no_improve": float(np.mean(deltas <= 0)),
    }
