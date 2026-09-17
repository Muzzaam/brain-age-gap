"""
Cross-validation hyperparameter tuning for the BAG estimators (RQ1).

The benchmark's fixed model settings were reasonable defaults, but comparing
model families on hand-picked settings risks measuring the settings, not the
models. This tunes each family on a shared footing: a small grid searched by
k-fold CV on the TRAINING split only (R2 scoring), so every family gets its best
honest shot and the no-leakage discipline holds -- the test set is untouched
until the final evaluation.

Grids are deliberately small (this is an Honours-scale search, not exhaustive):
enough to matter, cheap enough to run.
"""

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

SEED = 42
CV_FOLDS = 5

# (estimator factory, param grid) per family.
PARAM_GRIDS = {
    "ridge": (
        lambda: Ridge(),
        {"alpha": [0.1, 1.0, 10.0, 50.0]},
    ),
    "elastic_net": (
        lambda: ElasticNet(max_iter=5000, random_state=SEED),
        {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.2, 0.5, 0.8]},
    ),
    "xgboost": (
        lambda: xgb.XGBRegressor(random_state=SEED, n_jobs=-1),
        {"n_estimators": [200, 400], "max_depth": [2, 3, 4],
         "learning_rate": [0.03, 0.1], "subsample": [0.8]},
    ),
    "mlp": (
        lambda: MLPRegressor(max_iter=1000, early_stopping=True, random_state=SEED),
        {"hidden_layer_sizes": [(64,), (96, 64), (128, 64)], "alpha": [1e-4, 1e-3, 1e-2]},
    ),
}


def tune_model(name, X_train, y_train, cv=CV_FOLDS):
    """
    Grid-search one family on the training split. Returns best params, the mean
    CV R2 of the best setting, and the refit best estimator.
    """
    factory, grid = PARAM_GRIDS[name]
    search = GridSearchCV(
        factory(), grid, scoring="r2", cv=cv, n_jobs=-1, refit=True,
    )
    search.fit(X_train, y_train)
    return {
        "name": name,
        "best_params": search.best_params_,
        "cv_r2": float(search.best_score_),
        "estimator": search.best_estimator_,
    }


def tune_all(X_train, y_train, models=None, cv=CV_FOLDS):
    """Tune each family and return {name: result} sorted by CV R2 (best first)."""
    models = models or list(PARAM_GRIDS)
    results = {m: tune_model(m, X_train, y_train, cv=cv) for m in models}
    return dict(sorted(results.items(), key=lambda kv: kv[1]["cv_r2"], reverse=True))
