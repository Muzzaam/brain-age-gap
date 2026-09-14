"""
Survival models for the incident-dementia outcome (RQ2, time-to-event).

Two families, mirroring the cognitive-outcome pairing:
  * cox          -- Cox proportional hazards (the linear survival baseline)
  * survival_gbm -- gradient-boosted survival trees (the flexible analogue,
                    the survival counterpart of XGBoost)

Both expose predict_risk (higher = higher dementia risk) and
predict_survival_function (needed for the integrated Brier score).

A small ridge penalty is added to the Cox model so it stays stable when the
condition includes BAG, which is correlated with the biomarkers it came from.
"""

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

SEED = 42


class SurvivalModel:
    def __init__(self, name, estimator):
        self.name = name
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def predict_risk(self, X):
        return self.estimator.predict(X)

    def predict_survival_function(self, X):
        return self.estimator.predict_survival_function(X)


def make_cox():
    return SurvivalModel("cox", CoxPHSurvivalAnalysis(alpha=0.1))


def make_survival_gbm():
    return SurvivalModel("survival_gbm", GradientBoostingSurvivalAnalysis(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=SEED,
    ))


SURVIVAL_BUILDERS = {"cox": make_cox, "survival_gbm": make_survival_gbm}
SURVIVAL_MODELS = ["cox", "survival_gbm"]


def build_survival(name):
    if name not in SURVIVAL_BUILDERS:
        raise KeyError(f"Unknown survival model {name!r}. Available: {list(SURVIVAL_BUILDERS)}")
    return SURVIVAL_BUILDERS[name]()
