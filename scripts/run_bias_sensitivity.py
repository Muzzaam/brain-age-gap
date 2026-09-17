"""
Bias-correction sensitivity: does applying post-hoc bias correction to the
ground-truth BAG change the answer to "does estimated BAG help predict cognition"?

Runs the RQ2 cognitive comparison twice -- once on the raw BAG labels, once on
bias-corrected labels -- and reports both, plus whether the conclusion changes.
If the verdict is stable across the two, that's a robustness result; if it flips,
the methodological choice matters and both must be reported.

Run from the repo root:  python -m scripts.run_bias_sensitivity
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data import load_data, schema
from src.preprocessing import prepare_data
from src.models import build
from src.models.two_stage import estimate_bag, build_conditions, outcome_target
from src.evaluation.metrics import bag_metrics
from src.evaluation.outcome_metrics import cognitive_metrics, paired_error_test
from src.evaluation.bias_correction import fit_bias_correction, correct_bag, age_bias

STAGE2 = "ridge"


def _rq2_cognitive(data):
    """Run the four-condition cognitive comparison on the given data dict."""
    bag_hat = estimate_bag(data, model_name="xgboost")
    conds = build_conditions(data, bag_hat, seed=42)
    y_tr, y_te = outcome_target(data, schema.COGNITIVE_COL)
    preds, metrics = {}, {}
    for cond, (Xtr, Xte) in conds.items():
        p = build(STAGE2).fit(Xtr, y_tr).predict(Xte)
        preds[cond], metrics[cond] = p, cognitive_metrics(y_te, p)
    paired = paired_error_test(y_te, preds["biomarkers"], preds["biomarkers+BAG"])
    est_metrics = bag_metrics(data["y_test"], bag_hat["test"])
    return metrics, paired, est_metrics


def main(n=1500, seed=42):
    data = prepare_data(load_data(source="synthetic", n=n, seed=seed), seed=seed)
    age_tr = data["train_df"][schema.AGE_COL].to_numpy(float)
    age_te = data["test_df"][schema.AGE_COL].to_numpy(float)

    # --- fit correction on the TRAIN reference, apply to ground-truth BAG ---
    params = fit_bias_correction(data["y_train"], age_tr)
    corrected = dict(data)
    corrected["y_train"] = correct_bag(data["y_train"], age_tr, params)
    corrected["y_test"] = correct_bag(data["y_test"], age_te, params)
    corrected["y_val"] = correct_bag(
        data["y_val"], data["val_df"][schema.AGE_COL].to_numpy(float), params)

    print("Bias-correction sensitivity (ground-truth BAG)\n")
    print(f"age-bias of BAG  before: r={age_bias(data['y_test'], age_te):+.3f}"
          f"   after: r={age_bias(corrected['y_test'], age_te):+.3f}")
    print("(correction removes the linear age-dependence of BAG, as intended)\n")

    for label, dset in [("WITHOUT correction", data), ("WITH correction", corrected)]:
        metrics, paired, est = _rq2_cognitive(dset)
        print(f"### {label}  (estimator R2 on this BAG: {est['R2']:.3f})")
        print(f"{'condition':<22}{'RMSE':>8}{'Pearson_r':>11}")
        for cond, m in metrics.items():
            print(f"{cond:<22}{m['RMSE']:>8.3f}{m['Pearson_r']:>11.3f}")
        verdict = "HELPS" if (paired["rmse_improvement"] > 0 and paired["wilcoxon_p"] < 0.05) else "no help"
        print(f"  BAG vs biomarkers: improvement {paired['rmse_improvement']:+.3f}, "
              f"p={paired['wilcoxon_p']:.3f}  ->  {verdict}\n")

    print("Read it: if the verdict (HELPS / no help) is the same in both blocks, the")
    print("RQ2 conclusion is robust to the bias-correction choice. If it flips, the")
    print("choice is outcome-determining and both must be reported.")


if __name__ == "__main__":
    main()
