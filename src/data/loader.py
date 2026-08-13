"""
Data loading — the single swap point for the project.

Everything downstream calls `load_data(...)`. Today it returns synthetic data;
when the real ARIC-NCS file arrives you fill in `_load_aric` and switch
`source="aric"`. No model, metric, or plot code should need to change.
"""

import pandas as pd

from . import schema
from .synthetic import generate_synthetic_cohort


def load_data(source="synthetic", path=None, **kwargs):
    """
    Load a cohort dataframe conforming to schema.REQUIRED_COLUMNS.

    Parameters
    ----------
    source : {"synthetic", "aric"}
        "synthetic" generates a mock ARIC-shaped cohort (kwargs forwarded to
        generate_synthetic_cohort). "aric" loads the real dataset from `path`.
    path : str, optional
        Path to the real ARIC-NCS file (required when source="aric").
    """
    if source == "synthetic":
        df = generate_synthetic_cohort(**kwargs)
    elif source == "aric":
        if path is None:
            raise ValueError("path is required when source='aric'.")
        df = _load_aric(path)
    else:
        raise ValueError(f"Unknown source: {source!r}")

    validate_schema(df)
    return df


# Fill this in when the real data lands. The keys are the ARIC column names as
# they actually appear in the delivered file; the values are our schema names.
# Left empty on purpose — populate it once you can see the real columns.
ARIC_COLUMN_MAP = {
    # "GRIPSTR": "grip_strength",
    # "SYSBP":   "sbp",
    # ...
}


def _load_aric(path):
    """
    Load and normalise the real ARIC-NCS dataset onto our schema.

    TODO (when data arrives):
      1. Read the file (pd.read_csv / pd.read_sas / etc. depending on format).
      2. Rename columns via ARIC_COLUMN_MAP so they match schema names.
      3. Derive any missing schema columns (e.g. compute BAG once BrainageR has
         produced brain_age: df['bag'] = df['brain_age'] - df['age']).
      4. Return df[schema.REQUIRED_COLUMNS].
    """
    raise NotImplementedError(
        "Real ARIC loading not implemented yet. Map the delivered columns onto "
        "schema.REQUIRED_COLUMNS here once the dataset is available."
    )


def validate_schema(df):
    """Raise if required columns are missing; warn-return the extras."""
    missing = [c for c in schema.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataframe is missing required columns: {missing}. "
            f"Every loader must return all of schema.REQUIRED_COLUMNS."
        )
    extra = [c for c in df.columns if c not in schema.REQUIRED_COLUMNS
             and c != schema.BRAIN_AGE_COL]
    return extra
