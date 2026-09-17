"""
Post-hoc bias correction for Brain Age Gap (the method the proposal cites,
Beheshti/de Lange/Cole).

The well-documented brain-age artifact is that BAG correlates with chronological
age: models over-estimate the young and under-estimate the old, so BAG drifts
with age rather than being a clean age-independent deviation. The standard fix
fits BAG ~ slope*age + intercept on a healthy reference set (here, the training
split) and subtracts that fitted line from all participants, leaving a BAG whose
linear age-dependence has been removed.

The catch the proposal raises: this choice can move downstream results, so it
must be reported with-and-without rather than applied silently. These functions
are the "with" half; the sensitivity runner reports both.
"""

import numpy as np
from scipy.stats import pearsonr


def fit_bias_correction(bag_reference, age_reference):
    """Fit BAG ~ slope*age + intercept on a reference set. Returns the params."""
    slope, intercept = np.polyfit(
        np.asarray(age_reference, dtype=float),
        np.asarray(bag_reference, dtype=float),
        1,
    )
    return {"slope": float(slope), "intercept": float(intercept)}


def correct_bag(bag, age, params):
    """Subtract the fitted age-dependent component from BAG."""
    bag = np.asarray(bag, dtype=float)
    age = np.asarray(age, dtype=float)
    return bag - (params["slope"] * age + params["intercept"])


def age_bias(bag, age):
    """Pearson correlation between BAG and chronological age (0 = unbiased)."""
    return float(pearsonr(np.asarray(bag, dtype=float), np.asarray(age, dtype=float))[0])
