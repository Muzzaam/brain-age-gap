"""
RQ2 (dementia outcome): does estimated BAG improve prediction of INCIDENT
DEMENTIA, beyond the raw biomarkers?

Same four conditions as the cognitive-outcome harness (age-only, biomarkers,
biomarkers+BAG, biomarkers+placebo), but the outcome is time-to-event, so the
model is a survival model and the metrics are the C-index and integrated Brier
score. The key comparison (BAG vs biomarkers-only) is done by bootstrapping the
C-index difference.

Run from the repo root:  python -m scripts.run_rq2_survival
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_data, schema
from src.preprocessing import prepare_data
from src.models.two_stage import estimate_bag, build_conditions
from src.models.survival import build_survival, SURVIVAL_MODELS
from src.evaluation.survival_metrics import (
    make_survival_target, concordance, integrated_brier, bootstrap_cindex_difference,
)

BAG_ESTIMATOR = "xgboost"


def run(n=1500, seed=42, target_signal_fraction=0.30, n_boot=500):
    df = load_data(source="synthetic", n=n, seed=seed,
                   target_signal_fraction=target_signal_fraction)
    data = prepare_data(df, seed=seed)

    bag_hat = estimate_bag(data, model_name=BAG_ESTIMATOR)
    conditions = build_conditions(data, bag_hat, seed=seed)

    ev_tr = data["train_df"][schema.DEMENTIA_EVENT_COL].to_numpy()
    tt_tr = data["train_df"][schema.DEMENTIA_TIME_COL].to_numpy()
    ev_te = data["test_df"][schema.DEMENTIA_EVENT_COL].to_numpy()
    tt_te = data["test_df"][schema.DEMENTIA_TIME_COL].to_numpy()
    y_tr = make_survival_target(ev_tr, tt_tr)
    y_te = make_survival_target(ev_te, tt_te)

    report = {}
    for name in SURVIVAL_MODELS:
        metrics, risks = {}, {}
        for cond, (Xtr, Xte) in conditions.items():
            model = build_survival(name).fit(Xtr, y_tr)
            risk = model.predict_risk(Xte)
            risks[cond] = risk
            metrics[cond] = {
                "C_index": concordance(ev_te, tt_te, risk),
                "IBS": integrated_brier(y_tr, y_te, model, Xte),
            }
        boot = bootstrap_cindex_difference(
            ev_te, tt_te, risks["biomarkers"], risks["biomarkers+BAG"],
            n_boot=n_boot, seed=seed)
        report[name] = {"metrics": metrics, "boot": boot}
    return report


def main():
    report = run()
    print("RQ2 (dementia) - does estimated BAG help predict incident dementia? (synthetic)\n")
    for name, r in report.items():
        print(f"### survival model: {name}")
        print(f"{'condition':<22}{'C-index':>9}{'IBS':>8}")
        print("-" * 39)
        for cond, m in r["metrics"].items():
            print(f"{cond:<22}{m['C_index']:>9.3f}{m['IBS']:>8.3f}")
        b = r["boot"]
        print(f"\n  BAG vs biomarkers-only (C-index): delta {b['delta_mean']:+.4f} "
              f"[95% CI {b['ci_low']:+.4f}, {b['ci_high']:+.4f}], "
              f"P(no improvement)={b['p_no_improve']:.2f}\n")
    print("Reading it: higher C-index and lower IBS are better. A positive delta with")
    print("a CI above 0 means BAG improves discrimination. On synthetic data BAG is a")
    print("function of the biomarkers, so little/no gain is expected and honest.")


if __name__ == "__main__":
    main()
