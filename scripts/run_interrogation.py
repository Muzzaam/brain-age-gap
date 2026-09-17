"""SHAP importances across model families + subgroup breakdown.
Run from the repo root:  python -m scripts.run_interrogation"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_data
from src.preprocessing import prepare_data
from src.models import build
from src.evaluation.metrics import bag_metrics
from src.evaluation.interpretation import shap_importance_table, subgroup_metrics
SHAP_MODELS = ["ridge", "xgboost", "mlp"]
PRIMARY_MODEL = "xgboost"
def main(n=1500, seed=42):
    data = prepare_data(load_data(source="synthetic", n=n, seed=seed), seed=seed)
    fitted = {m: build(m).fit(data["X_train"], data["y_train"]) for m in SHAP_MODELS}
    print("SHAP feature importances (mean |SHAP|), by model family\n")
    table = shap_importance_table(fitted, data["X_train"], data["X_test"], data["feature_names"])
    print(table.round(3).to_string())
    print("\nAgreement check: matching top features across columns => robust ranking.\n")
    primary = fitted[PRIMARY_MODEL]; preds = primary.predict(data["X_test"])
    overall = bag_metrics(data["y_test"], preds)
    print(f"Subgroup BAG-recovery breakdown ({PRIMARY_MODEL})")
    print(f"overall: MAE {overall['MAE']:.3f}  RMSE {overall['RMSE']:.3f}  R2 {overall['R2']:.3f}  n {len(data['y_test'])}\n")
    subs = subgroup_metrics(data["test_df"], data["y_test"], preds)
    print(f"{'subgroup':<16}{'MAE':>8}{'RMSE':>8}{'R2':>8}{'n':>6}"); print("-"*46)
    for label, m in subs.items():
        print(f"{label:<16}{m['MAE']:>8.3f}{m['RMSE']:>8.3f}{m['R2']:>8.3f}{m['n']:>6}")
    print("\nA subgroup R2 far below the overall figure is a hidden failure mode.")
if __name__ == "__main__":
    main()
