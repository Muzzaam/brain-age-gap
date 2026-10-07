"""
Model families for the BAG-estimation task and the cognitive outcome, each
paired with its hyperparameter grid.

Every model is an sklearn Pipeline: per-fold preprocessing (impute, scale,
one-hot) followed by the estimator. Because preprocessing is inside the
pipeline, cross-validated tuning refits it on each inner training fold, so no
held-out row ever influences imputation or scaling.

The line-up mirrors the proposal:
  * ridge / elastic_net — regularised linear baselines (how much is linear?)
  * xgboost             — the STRONG baseline every neural approach must beat
  * mlp                 — a small neural baseline

All families are tuned on the same footing (src/models/tuning.py), so RQ1
comparisons reflect the models, not hand-picked settings.
"""

from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from ..preprocessing import make_preprocessor

SEED = 42

# name -> (estimator factory, param grid). Grids are deliberately small: an
# Honours-scale search, enough to matter, cheap enough to run inside nested CV.
# xgboost uses n_jobs=1 because the grid search itself runs in parallel.
MODEL_SPECS = {
    "ridge": (
        lambda: Ridge(),
        {"alpha": [0.1, 1.0, 10.0, 50.0]},
    ),
    "elastic_net": (
        lambda: ElasticNet(max_iter=5000, random_state=SEED),
        {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.2, 0.5, 0.8]},
    ),
    "xgboost": (
        lambda: xgb.XGBRegressor(subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                                 random_state=SEED, n_jobs=1),
        {"n_estimators": [200, 400], "max_depth": [2, 3, 4], "learning_rate": [0.03, 0.1]},
    ),
    # 2-3 hidden layers of width 64-128 (proposal 3.4.1). sklearn's MLP has no
    # dropout, so L2 (alpha) and early stopping do the regularising. The target
    # is standardised for the network and predictions mapped back: without this
    # the MLP fails on outcomes far from zero (e.g. a cognitive score of ~100).
    "mlp": (
        lambda: TransformedTargetRegressor(
            regressor=MLPRegressor(activation="relu", max_iter=1000, early_stopping=True,
                                   random_state=SEED),
            transformer=StandardScaler()),
        {"hidden_layer_sizes": [(64, 64), (128, 64), (128, 128, 64)],
         "alpha": [1e-3, 1e-2, 1e-1]},
    ),
}

# The principal families the proposal commits to benchmarking.
PRINCIPAL_MODELS = ["ridge", "elastic_net", "xgboost", "mlp"]


def param_grid(name, specs=MODEL_SPECS, quick=False):
    """Grid for a family, keyed for the pipeline ("model__alpha"). Quick = first value only."""
    factory, grid = specs[name]
    prefix = "model__regressor__" if isinstance(factory(), TransformedTargetRegressor) else "model__"
    return {f"{prefix}{k}": (v[:1] if quick else v) for k, v in grid.items()}


def make_pipeline(name, columns, specs=MODEL_SPECS):
    """Fresh unfitted Pipeline(preprocess -> estimator) for a family and input columns."""
    if name not in specs:
        raise KeyError(f"Unknown model {name!r}. Available: {list(specs)}")
    factory, _ = specs[name]
    return Pipeline([("prep", make_preprocessor(columns)), ("model", factory())])
