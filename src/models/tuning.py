"""
Inner-loop cross-validated hyperparameter tuning.

Called inside each OUTER fold with that fold's training data only, so the
outer test fold never influences which settings are chosen. This is the
"nested" part of nested cross-validation: the outer loop estimates how well
the whole procedure (tune, then fit) generalises; the inner loop does the
tuning.

Regression families are scored by R2; survival families by Harrell's C-index
(scikit-survival's default score); RQ3 neural framings by their own score
(R2 or C-index, depending on the outcome).
"""

from sklearn.model_selection import GridSearchCV, KFold

from .estimators import make_pipeline, param_grid
from .survival import SURVIVAL_SPECS, make_survival_pipeline, survival_param_grid
from .framings import make_framing_pipeline, framing_grid


def inner_cv(n_folds, seed):
    return KFold(n_splits=n_folds, shuffle=True, random_state=seed)


def _search(name, pipe, grid, X, y, columns, inner_folds, seed, scoring):
    search = GridSearchCV(pipe, grid, scoring=scoring, cv=inner_cv(inner_folds, seed),
                          n_jobs=-1, refit=True, error_score="raise")
    search.fit(X[columns], y)
    return {
        "name": name,
        "estimator": search.best_estimator_,
        "best_params": {k.split("__")[-1]: v for k, v in search.best_params_.items()},
        "inner_cv_score": float(search.best_score_),
    }


def tune(name, X, y, columns, inner_folds=3, seed=42, quick=False):
    """
    Grid-search one family on (X, y) with inner K-fold CV and refit the best.

    X is a DataFrame; only `columns` are used. Returns a dict with the fitted
    best pipeline, its params, and its mean inner-CV score.
    """
    if name in SURVIVAL_SPECS:
        return _search(name, make_survival_pipeline(name, columns), survival_param_grid(name, quick),
                       X, y, columns, inner_folds, seed, scoring=None)
    return _search(name, make_pipeline(name, columns), param_grid(name, quick=quick),
                   X, y, columns, inner_folds, seed, scoring="r2")


def tune_framing(name, task, X, y, columns, inner_folds=3, seed=42, quick=False, neural_cfg=None):
    """Tune one RQ3 framing (y packs outcome + BAG labels; see models.neural.make_targets)."""
    pipe = make_framing_pipeline(name, task, columns, neural_cfg, seed)
    return _search(name, pipe, framing_grid(name, quick), X, y, columns, inner_folds, seed,
                   scoring=None)
