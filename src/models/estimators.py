"""
Model families for the BAG-estimation task, behind one common interface.

Every model here exposes the same sklearn-style `fit(X, y)` / `predict(X)` and
carries a `.name`, so the benchmark can treat them interchangeably and compare
them on identical splits. That uniformity is the whole point: it's what lets us
say "XGBoost beat the MLP on the same data" without special-casing each model.

The line-up mirrors the proposal:
  * ridge / elastic_net — regularised linear baselines (how much is linear?)
  * xgboost             — the STRONG baseline every neural approach must beat
  * mlp                 — a small neural baseline

XGBoost is the one to watch: on tabular data it's usually the hardest to beat,
so it's the reference the others are judged against.
"""

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

SEED = 42


class Model:
    """Thin wrapper giving every estimator a consistent name + fit/predict."""

    def __init__(self, name, estimator):
        self.name = name
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        return self.estimator.predict(X)


# --- builders: each returns a fresh, configured Model --------------------------
def make_ridge():
    return Model("ridge", Ridge(alpha=1.0))


def make_elastic_net():
    return Model("elastic_net",
                 ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=SEED))


def make_xgboost():
    return Model("xgboost", xgb.XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, reg_lambda=2.0,
        random_state=SEED, n_jobs=-1,
    ))


def make_mlp():
    # 2 hidden layers (96, 64), ReLU, L2-regularised, with early stopping.
    # A torch MLP (with dropout) replaces this later when we build the
    # multi-task / continual-learning framings that need differentiability.
    return Model("mlp", MLPRegressor(
        hidden_layer_sizes=(96, 64), activation="relu",
        alpha=1e-3, max_iter=1000, early_stopping=True, random_state=SEED,
    ))


# Registry: name -> builder. The benchmark iterates over this.
MODEL_BUILDERS = {
    "ridge": make_ridge,
    "elastic_net": make_elastic_net,
    "xgboost": make_xgboost,
    "mlp": make_mlp,
}

# The principal families the proposal commits to benchmarking.
PRINCIPAL_MODELS = ["ridge", "elastic_net", "xgboost", "mlp"]


def build(name):
    """Construct a fresh model by name."""
    if name not in MODEL_BUILDERS:
        raise KeyError(f"Unknown model {name!r}. Available: {list(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[name]()
