"""
Data loading — the single swap point for the project.

Everything downstream calls `load_data(...)` (or `load_from_config(cfg)`, which
reads the source and paths from config.yaml). Today it returns synthetic data;
when the real ARIC-NCS files arrive you fill in ARIC_COLUMN_MAP / ARIC_VALUE_MAPS
and set `data.source: aric` in config.yaml. No model, metric, or plot code
should need to change.

The real data arrives as two files joined on participant id:
  1. the ARIC-NCS tabular extract (demographics, biomarkers, outcomes), and
  2. the brain-age CSV written by the imaging step (participant_id, brain_age).
"""

from pathlib import Path

import pandas as pd

from . import schema
from .synthetic import generate_synthetic_cohort


def load_data(source="synthetic", path=None, brain_age_path=None, **kwargs):
    """
    Load a cohort dataframe conforming to schema.REQUIRED_COLUMNS.

    Parameters
    ----------
    source : {"synthetic", "aric"}
        "synthetic" generates a mock ARIC-shaped cohort (kwargs forwarded to
        generate_synthetic_cohort). "aric" loads the real dataset.
    path : str, optional
        Path to the ARIC-NCS tabular file (required when source="aric").
    brain_age_path : str, optional
        Path to the brain-age CSV from the imaging step. If omitted, the
        tabular file must already contain brain_age or bag.
    """
    if source == "synthetic":
        df = generate_synthetic_cohort(**kwargs)
    elif source == "aric":
        if path is None:
            raise ValueError("path is required when source='aric'.")
        df = _load_aric(path, brain_age_path)
    else:
        raise ValueError(f"Unknown source: {source!r}")

    validate_schema(df)
    return df


def load_from_config(cfg):
    """Load whichever source config.yaml names, with its settings."""
    from ..config import resolve_path

    source = cfg["data"]["source"]
    if source == "synthetic":
        return load_data("synthetic", **cfg["data"]["synthetic"])
    aric = cfg["data"]["aric"]
    tab = resolve_path(aric["tabular_path"])
    if not tab.exists():
        raise FileNotFoundError(
            f"ARIC tabular file not found at {tab}. Set data.aric.tabular_path "
            f"in config.yaml.")
    ba = aric.get("brain_age_path")
    ba = resolve_path(ba) if ba else None
    if ba is not None and not ba.exists():
        raise FileNotFoundError(
            f"Brain-age file not found at {ba}. Run the imaging step first, or "
            f"set data.aric.brain_age_path to null if the tabular file has brain_age.")
    return load_data("aric", path=str(tab), brain_age_path=str(ba) if ba else None)


# Fill this in when the real data lands. The keys are the ARIC column names as
# they actually appear in the delivered file; the values are our schema names.
# Left empty on purpose — populate it once you can see the real columns.
ARIC_COLUMN_MAP = {
    # "GRIPSTR": "grip_strength",
    # "SYSBP":   "sbp",
    # ...
}

# Recode categorical values onto the schema's coding, e.g. sex "1"/"2" -> "M"/"F".
ARIC_VALUE_MAPS = {
    # "sex": {1: "M", 2: "F"},
    # "site": {"F": "Forsyth", "J": "Jackson", "M": "Minneapolis", "W": "Washington"},
}


def _read_any(path):
    """Read a tabular file by extension (ARIC extracts are often SAS or Stata)."""
    ext = Path(path).suffix.lower()
    readers = {".csv": pd.read_csv, ".sas7bdat": pd.read_sas, ".xpt": pd.read_sas,
               ".dta": pd.read_stata, ".sav": pd.read_spss, ".parquet": pd.read_parquet}
    if ext not in readers:
        raise ValueError(f"Don't know how to read {ext!r} files ({path}).")
    return readers[ext](path)


def _load_aric(path, brain_age_path=None):
    """
    Load and normalise the real ARIC-NCS dataset onto our schema.

    Steps that need the real file before they can be finished are marked TODO;
    everything else is ready.
    """
    df = _read_any(path).rename(columns=ARIC_COLUMN_MAP)
    for col, mapping in ARIC_VALUE_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # TODO (when data arrives): derive columns ARIC doesn't store directly, e.g.
    #   * dementia_time = years from the visit 5 date to diagnosis/censoring date
    #   * prevalent_dementia from the visit 5 adjudicated diagnosis
    #   * healthy_reference (e.g. adjudicated cognitively normal at visit 5)

    if brain_age_path is not None:
        ba = pd.read_csv(brain_age_path)
        if "status" in ba.columns:          # the imaging batch runners write a status column
            ba = ba[ba["status"] == "ok"]
        ba = ba[[schema.ID_COL, schema.BRAIN_AGE_COL]]
        if ba[schema.ID_COL].duplicated().any():
            raise ValueError("Brain-age file has duplicate participant ids.")
        df[schema.ID_COL] = df[schema.ID_COL].astype(str)
        ba[schema.ID_COL] = ba[schema.ID_COL].astype(str)
        # left join: participants without a usable scan keep NaN brain_age and
        # are counted (then excluded) in the participant flow, not silently lost
        df = df.drop(columns=[schema.BRAIN_AGE_COL], errors="ignore").merge(
            ba, on=schema.ID_COL, how="left")

    if schema.TARGET_COL not in df.columns and schema.BRAIN_AGE_COL in df.columns:
        df[schema.TARGET_COL] = df[schema.BRAIN_AGE_COL] - df[schema.AGE_COL]
    return df


def validate_schema(df):
    """Raise if required columns are missing; return the list of extra columns."""
    missing = [c for c in schema.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataframe is missing required columns: {missing}. "
            f"Every loader must return all of schema.REQUIRED_COLUMNS "
            f"(values may be missing; columns may not)."
        )
    if df[schema.ID_COL].duplicated().any():
        raise ValueError("Duplicate participant ids in the cohort dataframe.")
    known = set(schema.REQUIRED_COLUMNS) | {schema.BRAIN_AGE_COL, schema.REFERENCE_COL,
                                            schema.RACE_COL}
    return [c for c in df.columns if c not in known]
