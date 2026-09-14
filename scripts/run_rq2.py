"""
RQ2 benchmark: does estimated BAG improve prediction of the cognitive outcome?

For each stage-2 model family, trains the outcome model under the four
conditions and prints RMSE / Pearson r, then runs the key paired test:
biomarkers+BAG vs biomarkers-only, and vs the equivalent-capacity placebo.

Run from the repo root:  python -m scripts.run_rq2
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_data, schema
from src.preprocessing import prepare_data
from src.models import build
from src.models.two_stage import estimate_bag, build_conditions, outcome_target
from src.evaluation.outcome_metrics import cognitive_metrics, paired_error_test

STAGE2_MODELS = ["ridge", "xgboost"]
BAG_ESTIMATOR = "xgboost"


def run(n=1500, seed=42, target_signal_fraction=0.30):
    df = load_data(source="synthetic", n=n, seed=seed,
                   target_signal_fraction=target_signal_fraction)
    data = prepare_data(df, seed=seed)
    bag_hat = estimate_bag(data, model_name=BAG_ESTIMATOR)
    conditions = build_conditions(data, bag_hat, seed=seed)
    y_train, y_test = outcome_target(data, schema.COGNITIVE_COL)

    report = {}
    for stage2 in STAGE2_MODELS:
        preds, metrics = {}, {}
        for cond, (Xtr, Xte) in conditions.items():
            model = build(stage2).fit(Xtr, y_train)
            p = model.predict(Xte)
            preds[cond] = p
            metrics[cond] = cognitive_metrics(y_test, p)
        report[stage2] = {"metrics": metrics, "preds": preds}
    return report, y_test


def main():
    report, y_test = run()
    print("RQ2 - does estimated BAG help predict cognitive score? (synthetic data)\n")
    for stage2, r in report.items():
        print(f"### stage-2 model: {stage2}")
        print(f"{'condition':<22}{'RMSE':>8}{'Pearson_r':>11}")
        print("-" * 41)
        for cond, m in r["metrics"].items():
            print(f"{cond:<22}{m['RMSE']:>8.3f}{m['Pearson_r']:>11.3f}")
        vs_bio = paired_error_test(y_test, r["preds"]["biomarkers"], r["preds"]["biomarkers+BAG"])
        vs_plc = paired_error_test(y_test, r["preds"]["biomarkers+placebo"], r["preds"]["biomarkers+BAG"])
        print(f"\n  BAG vs biomarkers-only : RMSE {vs_bio['rmse_baseline']:.3f} -> "
              f"{vs_bio['rmse_augmented']:.3f} "
              f"(improvement {vs_bio['rmse_improvement']:+.3f}, Wilcoxon p={vs_bio['wilcoxon_p']:.3f})")
        print(f"  BAG vs placebo (capacity): improvement {vs_plc['rmse_improvement']:+.3f}, "
              f"p={vs_plc['wilcoxon_p']:.3f}\n")
    print("Reading it: a positive improvement with small p means BAG adds real signal.")
    print("On synthetic data BAG is largely a function of the biomarkers, so little or")
    print("no gain is expected and honest -- the same null RQ2 could produce on real data.")


if __name__ == "__main__":
    main()
