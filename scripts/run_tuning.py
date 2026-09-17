"""
Cross-validation tuning (RQ1): tune each model family on the training split,
then evaluate the tuned models once on the held-out test set.

Reports the best hyperparameters and CV R2 per family, then the final test
metrics, so you can see whether tuning changed the RQ1 ranking versus the
fixed-default benchmark.

Run from the repo root:  python -m scripts.run_tuning
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_data
from src.preprocessing import prepare_data
from src.models.tuning import tune_all
from src.evaluation.metrics import bag_metrics


def main(n=1500, seed=42):
    data = prepare_data(load_data(source="synthetic", n=n, seed=seed), seed=seed)

    print("Cross-validation hyperparameter tuning (train split only)\n")
    tuned = tune_all(data["X_train"], data["y_train"])

    # CV results
    print(f"{'model':<14}{'CV R2':>8}   best params")
    print("-" * 60)
    for name, r in tuned.items():
        print(f"{name:<14}{r['cv_r2']:>8.3f}   {r['best_params']}")

    # final held-out evaluation of the tuned estimators (test touched once)
    print(f"\nHeld-out test performance of tuned models")
    print(f"{'model':<14}{'MAE':>8}{'RMSE':>8}{'R2':>8}")
    print("-" * 38)
    rows = {}
    for name, r in tuned.items():
        preds = r["estimator"].predict(data["X_test"])
        rows[name] = bag_metrics(data["y_test"], preds)
    for name, m in sorted(rows.items(), key=lambda kv: kv[1]["R2"], reverse=True):
        print(f"{name:<14}{m['MAE']:>8.3f}{m['RMSE']:>8.3f}{m['R2']:>8.3f}")

    best = max(rows, key=lambda k: rows[k]["R2"])
    print(f"\nBest tuned model on test: {best} (R2 {rows[best]['R2']:.3f}).")
    print("Compare to the fixed-default benchmark (run_rq1): if the ranking is the")
    print("same, the RQ1 conclusion wasn't an artifact of the default settings.")


if __name__ == "__main__":
    main()
