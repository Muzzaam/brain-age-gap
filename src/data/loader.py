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


def load_data(source="synthetic", path=None, brain_age_path=None, scan_id_map_path=None, **kwargs):
    """
    Load a cohort dataframe conforming to schema.REQUIRED_COLUMNS.

    Parameters
    ----------
    source : {"synthetic", "aric"}
        "synthetic" generates a mock ARIC-shaped cohort (kwargs forwarded to
        generate_synthetic_cohort). "aric" loads the real dataset.
    path : str or list of str, optional
        Path to the ARIC-NCS tabular file, or several files joined on
        participant id (required when source="aric").
    brain_age_path : str, optional
        Path to the brain-age CSV from the imaging step. If omitted, the
        tabular file must already contain brain_age or bag.
    scan_id_map_path : str, optional
        CSV with columns scan_id, participant_id, for when scan filenames carry
        an imaging id rather than the ARIC participant id.
    """
    if source == "synthetic":
        df = generate_synthetic_cohort(**kwargs)
    elif source == "aric":
        if path is None:
            raise ValueError("path is required when source='aric'.")
        df = _load_aric(path, brain_age_path, scan_id_map_path)
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
    paths = aric["tabular_path"]
    paths = [paths] if isinstance(paths, str) else list(paths)
    tabs = [resolve_path(p) for p in paths]
    for tab in tabs:
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
    mp = aric.get("scan_id_map_path")
    mp = resolve_path(mp) if mp else None
    if mp is not None and not mp.exists():
        raise FileNotFoundError(f"Scan id map not found at {mp} (data.aric.scan_id_map_path).")
    return load_data("aric", path=[str(t) for t in tabs], brain_age_path=str(ba) if ba else None,
                     scan_id_map_path=str(mp) if mp else None)


# Fill this in when the real data lands. The keys are the ARIC column names as
# they actually appear in the delivered file(s); the values are our schema names.
# The participant id column must map to "participant_id" in EVERY file.
# Left empty on purpose — populate it once you can see the real columns.
ARIC_COLUMN_MAP = {
    # "ID_C":    "participant_id",
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


def _read_tabular(paths):
    """
    Read one or several tabular files, rename columns via ARIC_COLUMN_MAP, and
    join them on participant id. The first file is the base (one row per
    participant, e.g. the visit 5 file); later files are left-joined onto it.
    A column present in more than one file is kept from the first file only,
    and noted in df.attrs["merge_notes"] for the preflight to show.
    """
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    notes, df = [], None
    for p in paths:
        part = _read_any(p).rename(columns=ARIC_COLUMN_MAP)
        if schema.ID_COL not in part.columns:
            raise ValueError(
                f"{Path(p).name} has no participant id column after renaming. Map its id "
                f"column to {schema.ID_COL!r} in ARIC_COLUMN_MAP. Its columns: {list(part.columns)[:30]}")
        part[schema.ID_COL] = part[schema.ID_COL].astype(str).str.strip()
        if part[schema.ID_COL].duplicated().any():
            raise ValueError(f"{Path(p).name} has more than one row per participant; "
                             f"reduce it to one row each (e.g. the visit 5 record) before joining.")
        if df is None:
            df = part
            continue
        dup = [c for c in part.columns if c in df.columns and c != schema.ID_COL]
        if dup:
            notes.append(f"{Path(p).name}: kept {dup} from the earlier file")
        df = df.merge(part.drop(columns=dup), on=schema.ID_COL, how="left")
    df.attrs["merge_notes"] = notes
    return df


def _load_aric(path, brain_age_path=None, scan_id_map_path=None):
    """
    Load and normalise the real ARIC-NCS dataset onto our schema.

    Steps that need the real file before they can be finished are marked TODO;
    everything else is ready.
    """
    df = _read_tabular(path)
    for col, mapping in ARIC_VALUE_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # TODO (when data arrives): derive columns ARIC doesn't store directly, e.g.
    #   * dementia_time = years from the visit 5 date to diagnosis/censoring date
    #   * prevalent_dementia from the visit 5 adjudicated diagnosis
    #   * healthy_reference (e.g. adjudicated cognitively normal at visit 5)

    if brain_age_path is not None:
        id_map = pd.read_csv(scan_id_map_path) if scan_id_map_path else None
        df = link_brain_age(df, pd.read_csv(brain_age_path), id_map)

    if schema.TARGET_COL not in df.columns and schema.BRAIN_AGE_COL in df.columns:
        df[schema.TARGET_COL] = df[schema.BRAIN_AGE_COL] - df[schema.AGE_COL]
    return df


def link_brain_age(df, ba, id_map=None):
    """
    Join the imaging step's brain ages onto the tabular data by participant id.

    `ba` is the runner's CSV (participant_id, brain_age, status). If scan
    filenames carried an imaging id, `id_map` (scan_id, participant_id)
    translates it first. Participants without a usable scan keep NaN
    brain_age and are counted, then excluded, in the participant flow.
    The counts of the match are stored in df.attrs["linkage"] for preflight.
    """
    ba = ba.copy()
    ba[schema.ID_COL] = ba[schema.ID_COL].astype(str).str.strip()
    report = {"brain_age_rows": len(ba)}
    if "status" in ba.columns:  # the imaging batch runners write a status column
        report["scans_failed"] = int((ba["status"] != "ok").sum())
        ba = ba[ba["status"] == "ok"]
    if id_map is not None:
        id_map = id_map.astype(str).apply(lambda s: s.str.strip())
        if id_map["scan_id"].duplicated().any():
            raise ValueError("Scan id map has duplicate scan_id values.")
        lookup = dict(zip(id_map["scan_id"], id_map[schema.ID_COL]))
        report["scans_not_in_id_map"] = int((~ba[schema.ID_COL].isin(lookup)).sum())
        ba[schema.ID_COL] = ba[schema.ID_COL].map(lookup)
        ba = ba[ba[schema.ID_COL].notna()]
    if ba[schema.ID_COL].duplicated().any():
        raise ValueError("More than one usable brain age for some participants; keep one scan each.")

    attrs = dict(df.attrs)  # e.g. merge notes from reading several files; merge() can drop them
    df = df.copy()
    df[schema.ID_COL] = df[schema.ID_COL].astype(str).str.strip()
    known = set(df[schema.ID_COL])
    report["scans_usable"] = len(ba)
    report["scans_matched_to_participant"] = int(ba[schema.ID_COL].isin(known).sum())
    report["scans_with_no_participant"] = len(ba) - report["scans_matched_to_participant"]
    df = df.drop(columns=[schema.BRAIN_AGE_COL], errors="ignore").merge(
        ba[[schema.ID_COL, schema.BRAIN_AGE_COL]], on=schema.ID_COL, how="left")
    report["participants_with_brain_age"] = int(df[schema.BRAIN_AGE_COL].notna().sum())
    df.attrs = {**attrs, "linkage": report}
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
