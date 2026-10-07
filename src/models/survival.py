"""
Survival models for the incident-dementia outcome (RQ2, time-to-event).

Two families, mirroring the cognitive-outcome pairing:
  * cox          -- Cox proportional hazards (the linear survival baseline)
  * survival_gbm -- gradient-boosted survival trees (the flexible analogue,
                    the survival counterpart of XGBoost)

Each is a Pipeline(preprocess -> estimator) like the regression families, and
tuned the same way. scikit-survival adds predict_survival_function to sklearn
Pipelines, which the integrated Brier score needs.

The ridge penalty (alpha) keeps the Cox model stable when the condition
includes BAG, which is correlated with the biomarkers it came from.
"""

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

from .estimators import make_pipeline as _make_pipeline, param_grid as _param_grid

SEED = 42

SURVIVAL_SPECS = {
    "cox": (
        lambda: CoxPHSurvivalAnalysis(),
        {"alpha": [0.1, 1.0, 10.0]},
    ),
    "survival_gbm": (
        lambda: GradientBoostingSurvivalAnalysis(subsample=0.8, learning_rate=0.05,
                                                 random_state=SEED),
        {"n_estimators": [100, 200], "max_depth": [2, 3]},
    ),
}

SURVIVAL_MODELS = list(SURVIVAL_SPECS)


def make_survival_pipeline(name, columns):
    return _make_pipeline(name, columns, specs=SURVIVAL_SPECS)


def survival_param_grid(name, quick=False):
    return _param_grid(name, specs=SURVIVAL_SPECS, quick=quick)
