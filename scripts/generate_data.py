"""
Generate a synthetic ARIC-shaped cohort, save it, and run a quick sanity check.

Run from the repo root:  python -m scripts.generate_data
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_data, schema
from src.preprocessing import prepare_data


def main():
    # 1. Load via the same entry point we'll use for real data later.
    df = load_data(source="synthetic", n=1500, seed=42, target_signal_fraction=0.30)

    out_dir = Path(__file__).resolve().parents[1] / "data" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cohort.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} participants -> {out_path}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}\n")

    # 2. Quick data-quality readout (the messiness we injected on purpose).
    biomarkers = schema.all_biomarker_columns()
    miss = df[biomarkers].isna().mean().mul(100).round(1)
    print("Missingness by biomarker (%):")
    print(miss.to_string(), "\n")

    print(f"Prevalent dementia (excluded): {int(df[schema.PREVALENT_DEMENTIA_COL].sum())}")
    print(f"Incident dementia events: {int(df[schema.DEMENTIA_EVENT_COL].sum())} "
          f"({df[schema.DEMENTIA_EVENT_COL].mean():.1%})")
    print(f"BAG: mean={df[schema.TARGET_COL].mean():.2f}  sd={df[schema.TARGET_COL].std():.2f}\n")

    # 3. Preprocess and confirm the injected BAG signal is recoverable.
    data = prepare_data(df, seed=42)
    print(f"Split sizes -> train {len(data['y_train'])}, "
          f"val {len(data['y_val'])}, test {len(data['y_test'])}")

    # A plain linear fit should already see part of the injected signal; this is
    # just a smoke test that the pipeline is wired correctly, not a real model.
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score, mean_absolute_error

    model = Ridge(alpha=1.0).fit(data["X_train"], data["y_train"])
    pred = model.predict(data["X_test"])
    print(f"\nSanity check (Ridge on synthetic BAG):")
    print(f"  test R^2 = {r2_score(data['y_test'], pred):.3f} "
          f"(injected signal fraction was 0.30)")
    print(f"  test MAE = {mean_absolute_error(data['y_test'], pred):.2f} years")


if __name__ == "__main__":
    main()
