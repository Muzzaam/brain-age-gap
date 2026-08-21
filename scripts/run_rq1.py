"""
RQ1 benchmark: how well can each model recover BAG from biomarkers?

Trains every principal model family on the same train split, evaluates on the
same held-out test split, and prints a comparison table with the pre-registered
R2 >= 0.20 check. XGBoost is the baseline the others are read against.

Run from the repo root:  python -m scripts.run_rq1
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_data
from src.preprocessing import prepare_data
from src.models.estimators import build, PRINCIPAL_MODELS
from src.evaluation.metrics import bag_metrics, format_metrics_table


def run_rq1(n=1500, seed=42, target_signal_fraction=0.30):
    # One dataset, one preprocessing, one set of splits — shared by every model
    # so differences are about the model, not the data handling.
    df = load_data(source="synthetic", n=n, seed=seed,
                   target_signal_fraction=target_signal_fraction)
    data = prepare_data(df, seed=seed)

    results = {}
    for name in PRINCIPAL_MODELS:
        model = build(name)
        model.fit(data["X_train"], data["y_train"])
        preds = model.predict(data["X_test"])
        results[name] = bag_metrics(data["y_test"], preds)

    return results, target_signal_fraction


def main():
    results, f = run_rq1()
    print("RQ1 — BAG recovery from biomarkers (synthetic data)")
    print(f"Injected signal fraction (recovery ceiling): ~{f:.2f}\n")
    print(format_metrics_table(results))

    best = max(results, key=lambda k: results[k]["R2"])
    xgb_r2 = results["xgboost"]["R2"]
    print(f"\nBest R2: {best} ({results[best]['R2']:.3f}).  XGBoost R2: {xgb_r2:.3f}.")
    print("On synthetic data the truth is linear, so ridge/XGBoost should sit near")
    print("the ceiling and close together; a model wildly beating it would signal overfitting.")


if __name__ == "__main__":
    main()
