"""
RQ2 - the two-stage utility framing.

Question: does an explicitly-estimated BAG, used as an intermediate feature,
help predict a downstream outcome BEYOND what the raw biomarkers already give?

Stage 1: fit a BAG estimator on biomarkers -> BAG (trained on TRAIN only, then
         frozen). This is exactly the RQ1 estimator.
Stage 2: predict the outcome from one of four feature sets, so we can isolate
         the contribution of BAG:

  1. age_only            -- does chronological age do all the work?
  2. biomarkers          -- the practical baseline (what you'd build with no BAG)
  3. biomarkers + BAG    -- the test: add estimated BAG as an extra feature
  4. biomarkers + placebo-- equivalent-capacity control: an extra column with the
                            SAME shape/distribution as BAG but its signal destroyed
                            (BAG values shuffled). If (3) beats (4), the gain is
                            BAG's information, not just "one more input feature".

If condition 3 beats condition 2 AND condition 4, estimated BAG carries genuine,
non-redundant downstream value. If not, BAG adds nothing the biomarkers didn't
already contain -- itself a valid, reportable finding.

Leakage discipline: the stage-1 estimator is fit on train only; age and BAG are
standardised using train statistics only; the placebo shuffles within each split.
"""

import numpy as np

from ..data import schema
from .estimators import build


def estimate_bag(data, model_name="xgboost"):
    """Stage 1: fit BAG estimator on train, return predicted BAG for each split."""
    model = build(model_name).fit(data["X_train"], data["y_train"])
    return {
        "train": np.asarray(model.predict(data["X_train"]), dtype=float),
        "val": np.asarray(model.predict(data["X_val"]), dtype=float),
        "test": np.asarray(model.predict(data["X_test"]), dtype=float),
        "model": model,
    }


def _standardize_on_train(train_vals, *other_vals):
    """Fit mean/std on train, apply to train and every other array (no leakage)."""
    mu = float(np.mean(train_vals))
    sd = float(np.std(train_vals)) or 1.0
    out = [(np.asarray(train_vals, float) - mu) / sd]
    out += [(np.asarray(v, float) - mu) / sd for v in other_vals]
    return out


def build_conditions(data, bag_hat, seed=42):
    """Return {condition_name: (X_train_cond, X_test_cond)} for the four conditions."""
    rng = np.random.default_rng(seed)

    X_bio_tr, X_bio_te = data["X_train"], data["X_test"]  # biomarkers, already scaled on train

    age_tr = data["train_df"][schema.AGE_COL].to_numpy(float)
    age_te = data["test_df"][schema.AGE_COL].to_numpy(float)
    age_tr_s, age_te_s = _standardize_on_train(age_tr, age_te)

    bag_tr_s, bag_te_s = _standardize_on_train(bag_hat["train"], bag_hat["test"])

    # placebo: same distribution as BAG, signal destroyed by shuffling within split
    placebo_tr = rng.permutation(bag_tr_s)
    placebo_te = rng.permutation(bag_te_s)

    return {
        "age_only": (age_tr_s.reshape(-1, 1), age_te_s.reshape(-1, 1)),
        "biomarkers": (X_bio_tr, X_bio_te),
        "biomarkers+BAG": (np.column_stack([X_bio_tr, bag_tr_s]),
                           np.column_stack([X_bio_te, bag_te_s])),
        "biomarkers+placebo": (np.column_stack([X_bio_tr, placebo_tr]),
                               np.column_stack([X_bio_te, placebo_te])),
    }


def outcome_target(data, column):
    """Return (y_train, y_test) for a named outcome column."""
    return (data["train_df"][column].to_numpy(float),
            data["test_df"][column].to_numpy(float))
