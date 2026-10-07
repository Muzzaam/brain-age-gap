"""
RQ3 framing registry: which network variant, which inputs, and which grid.

Every framing is a Pipeline(per-fold preprocessing -> NeuralFraming) tuned by
the same inner CV as the other model families. Inputs are covariates +
biomarkers, except two_stage, which also gets the cross-fitted BAG estimate.
"""

from sklearn.pipeline import Pipeline

from ..preprocessing import make_preprocessor
from .neural import NeuralFraming

# name -> (network framing, uses estimated BAG as an input, shuffle the BAG labels)
RQ3_FRAMINGS = {
    "baseline":          ("single", False, False),  # equivalent capacity, no BAG at all
    "two_stage":         ("single", True, False),   # BAG as a constructed input feature
    "multitask":         ("multitask", False, False),
    "multitask_placebo": ("multitask", False, True),  # auxiliary task on shuffled BAG
    "bottleneck":        ("bottleneck", False, False),
    "transfer":          ("transfer", False, False),
}

# Framings whose result does not depend on the BAG labels (shareable across
# bias-correction variants within a fold).
LABEL_FREE = {"baseline"}


def framing_grid(name, quick=False):
    kind = RQ3_FRAMINGS[name][0]
    grid = {"weight_decay": [1e-4, 1e-3]}
    if kind in ("multitask", "bottleneck"):
        grid["bag_weight"] = [0.25, 0.5, 0.75]  # proposal: start at 0.5, explore sensitivity
    return {f"model__{k}": (v[:1] if quick else v) for k, v in grid.items()}


def make_framing_pipeline(name, task, columns, neural_cfg=None, seed=42):
    kind = RQ3_FRAMINGS[name][0]
    params = dict(neural_cfg or {})
    if "hidden" in params:
        params["hidden"] = tuple(params["hidden"])
    est = NeuralFraming(task=task, framing=kind, random_state=seed, **params)
    return Pipeline([("prep", make_preprocessor(columns)), ("model", est)])
