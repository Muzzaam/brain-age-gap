"""
Generate a synthetic ARIC-shaped cohort, save it, and run a quick sanity check.

Run from the code folder:  python -m scripts.generate_data
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data import load_data, schema
from src.preprocessing import prepare_cohort


def main():
    cfg = load_config()
    # Same entry point the real data will use later.
    df = load_data(source="synthetic", **cfg["data"]["synthetic"])

    out_dir = Path(__file__).resolve().parents[1] / "data" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cohort.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} participants -> {out_path}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}\n")

    miss = df[schema.all_biomarker_columns() + [schema.TARGET_COL, schema.COGNITIVE_COL]] \
        .isna().mean().mul(100).round(1)
    print("Missingness (%):")
    print(miss.to_string(), "\n")
    print(f"BAG: mean={df[schema.TARGET_COL].mean():.2f}  sd={df[schema.TARGET_COL].std():.2f}\n")

    _, flow = prepare_cohort(df)
    print("Participant flow:")
    for step, n in flow:
        print(f"  {step:<62}{n}")


if __name__ == "__main__":
    main()
