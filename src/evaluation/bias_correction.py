"""
Post-hoc bias correction for Brain Age Gap (the linear age-residualisation
approach, Beheshti et al., 2019).

The well-documented brain-age artifact is that BAG correlates with chronological
age: models over-estimate the young and under-estimate the old, so BAG drifts
with age rather than being a clean age-independent deviation. The standard fix
fits BAG ~ slope*age + intercept on a healthy reference set and subtracts that
fitted line from all participants, leaving a BAG whose linear age-dependence
has been removed.

Leakage rule: the reference set is drawn from the TRAINING fold only (the
participants flagged in the reference column, e.g. cognitively normal; all
training participants if the column is absent). The fitted line is then
applied to train and test alike.

The catch the proposal raises: this choice can move downstream results, so it
is reported with-and-without rather than applied silently.
"""

import numpy as np
from scipy.stats import pearsonr

from ..data import schema


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


def reference_mask(train_df, reference_col=schema.REFERENCE_COL, min_n=30):
    """
    Which training rows form the reference set. Falls back to all training
    rows if the column is missing or flags fewer than `min_n` participants.
    Returns (boolean mask, description for the run log).
    """
    if reference_col and reference_col in train_df.columns:
        mask = (train_df[reference_col] == 1).to_numpy()
        if mask.sum() >= min_n:
            return mask, f"{reference_col} == 1"
    return np.ones(len(train_df), dtype=bool), "all training participants"


def fit_on_reference(train_df, reference_col=schema.REFERENCE_COL):
    """Fit the correction on the reference subset of a training frame."""
    mask, used = reference_mask(train_df, reference_col)
    params = fit_bias_correction(train_df[schema.TARGET_COL].to_numpy()[mask],
                                 train_df[schema.AGE_COL].to_numpy()[mask])
    params["reference"] = used
    params["n_reference"] = int(mask.sum())
    return params


def age_bias(bag, age):
    """Pearson correlation between BAG and chronological age (0 = unbiased)."""
    return float(pearsonr(np.asarray(bag, dtype=float), np.asarray(age, dtype=float))[0])
