"""
RQ1 part two: biomarker-group ablation and residual analysis of the BAG estimator.

Run from the repo root:  python -m scripts.run_diagnostics
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_data
from src.evaluation.diagnostics import group_ablation, residual_analysis

MODEL = "xgboost"


def main(n=1500, seed=42):
    df = load_data(source="synthetic", n=n, seed=seed)

    # --- biomarker-group ablation ---
    print(f"Biomarker-group ablation ({MODEL}) - drop each group, watch R2 fall\n")
    abl = group_ablation(df, model_name=MODEL, seed=seed)
    full = abl["full"]
    print(f"{'configuration':<26}{'R2':>7}{'delta_R2':>10}")
    print("-" * 43)
    print(f"{'full (all biomarkers)':<26}{full['R2']:>7.3f}{'-':>10}")
    dropped = {k: v for k, v in abl.items() if k != "full"}
    for name, m in sorted(dropped.items(), key=lambda kv: kv[1]["delta_R2"], reverse=True):
        print(f"{name:<26}{m['R2']:>7.3f}{m['delta_R2']:>10.3f}")
    print("\nLargest delta_R2 = the group carrying the most recoverable ageing signal.")
    print("Compare this ranking to the SHAP table - agreement across the two is the")
    print("robust answer to RQ1's 'which biomarkers matter'.\n")

    # --- residual analysis ---
    print(f"Residual analysis ({MODEL}) - is the error structured or random?\n")
    res = residual_analysis(df, model_name=MODEL, seed=seed)
    ca = res["corr_with_age"]
    ct = res["corr_with_true_bag"]
    print(f"residual vs age        : r={ca['r']:+.3f} (p={ca['p']:.3f})")
    print(f"residual vs true BAG   : r={ct['r']:+.3f} (p={ct['p']:.3f})")
    print("mean residual by sex   :", {k: round(v, 3) for k, v in res["mean_residual_by_sex"].items()})
    print("mean |residual| by age :", {k: round(v, 2) for k, v in res["mean_abs_residual_by_age_band"].items()})
    strongest = sorted(res["corr_with_biomarkers"].items(), key=lambda kv: abs(kv[1]["r"]), reverse=True)[:3]
    print("strongest residual-biomarker correlations:",
          {k: round(v["r"], 3) for k, v in strongest})
    print("\nA strong residual-vs-true-BAG correlation is the classic brain-age")
    print("regression-to-the-mean effect (young over-estimated, old under-estimated),")
    print("which is exactly what bias correction targets. Age/sex structure would flag")
    print("that the correction is incomplete.")


if __name__ == "__main__":
    main()
