"""
RQ2 - the two-stage utility framing.

Question: does an explicitly-estimated BAG, used as an intermediate feature,
help predict a downstream outcome BEYOND what age, sex, site and the raw
biomarkers already give?

Stage 1: a BAG estimator (biomarkers + covariates -> BAG), tuned on the outer
         training fold. Training-set BAG estimates are CROSS-FITTED (see
         cross_fit_bag) so they are as noisy as the test-set ones.
Stage 2: predict the outcome from one of four feature sets. The covariates
         (age, sex, site by default) are in ALL of them, so BAG can only help
         by carrying information beyond them, not by standing in for age:

  1. covariates                     -- how far do age/sex/site alone get?
  2. covariates+biomarkers          -- the practical baseline (no BAG concept)
  3. covariates+biomarkers+BAG      -- the test: add estimated BAG
  4. covariates+biomarkers+placebo  -- equivalent-capacity control: an extra
                                       column with BAG's exact distribution but
                                       its signal destroyed (shuffled). If (3)
                                       beats (4), the gain is BAG's information,
                                       not just "one more input column".

If condition 3 beats conditions 2 AND 4, estimated BAG carries genuine,
non-redundant downstream value. If not, BAG adds nothing the inputs didn't
already contain -- itself a valid, reportable finding.
"""

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict

from ..data import schema
from .tuning import inner_cv

BAG_HAT_COL = "bag_hat"
PLACEBO_COL = "bag_placebo"

CONDITIONS = [
    "covariates",
    "covariates+biomarkers",
    "covariates+biomarkers+BAG",
    "covariates+biomarkers+placebo",
]


def condition_columns(covariates):
    """{condition: input columns}. Every condition starts with the covariates."""
    bio = schema.all_biomarker_columns()
    return {
        "covariates": list(covariates),
        "covariates+biomarkers": list(covariates) + bio,
        "covariates+biomarkers+BAG": list(covariates) + bio + [BAG_HAT_COL],
        "covariates+biomarkers+placebo": list(covariates) + bio + [PLACEBO_COL],
    }


def cross_fit_bag(estimator, X_train, y_train, X_test, inner_folds=3, seed=42):
    """
    Stage-1 BAG estimates for the training and test rows.

    Training rows get OUT-OF-FOLD predictions: the training set is split into
    `inner_folds` parts, a copy of the estimator is fitted on all but one part
    and predicts the held-out part, rotating until every row is covered. No
    training row is ever predicted by a model that saw it, so its estimate is
    as noisy as a test row's. (Predicting training rows in-sample would hand
    stage 2 unrealistically accurate BAG values at training time, so it would
    learn to over-trust BAG and then meet much noisier values at test time.)

    Test rows are predicted by `estimator` already fitted on all training rows.
    """
    oof = cross_val_predict(clone(estimator), X_train, y_train,
                            cv=inner_cv(inner_folds, seed), n_jobs=-1)
    return np.asarray(oof, float), np.asarray(estimator.predict(X_test), float)


def add_bag_features(train_df, test_df, bag_train, bag_test, seed=42):
    """
    Return copies of the frames with estimated-BAG and placebo columns added.

    The placebo is BAG shuffled within each split: same values, same
    distribution, no link to the participant. Scaling happens inside each
    stage-2 pipeline (fitted on train only).
    """
    rng = np.random.default_rng(seed)
    train, test = train_df.copy(), test_df.copy()
    train[BAG_HAT_COL] = bag_train
    test[BAG_HAT_COL] = bag_test
    train[PLACEBO_COL] = rng.permutation(bag_train)
    test[PLACEBO_COL] = rng.permutation(bag_test)
    return train, test
