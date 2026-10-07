"""
Preprocessing, implemented as the proposal specifies.

Order matters for avoiding leakage:
  1. Mask biologically implausible biomarker values as missing (fixed ranges,
     so no information comes from the data).
  2. Exclude participants: prevalent dementia, missing age/sex/BAG, or missing
     an entire biomarker group. Each step is counted for the participant-flow
     table the report needs.
  3. Split into folds, stratified by age band x sex (x dementia event).
  4. Imputation and scaling live INSIDE each model's sklearn Pipeline
     (make_preprocessor), so they are fitted on the training part of every
     fold, including the inner tuning folds, and never see held-out rows.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import schema

AGE_BANDS = [0, 72, 77, 82, 200]  # (-,72], (72,77], (77,82], (82,+)
AGE_LABELS = ["<=72", "72-77", "77-82", ">82"]


def mask_implausible(df):
    """Set biomarker values outside plausible ranges to NaN (so they get imputed).

    Uses the fixed ranges in schema.PLAUSIBLE_RANGES, so this introduces no
    leakage from the data. Rather than let a mis-recorded SBP of 900 distort
    standardisation, treat it as missing.
    """
    df = df.copy()
    for col in schema.all_biomarker_columns():
        if col in schema.PLAUSIBLE_RANGES:
            lo, hi = schema.PLAUSIBLE_RANGES[col]
            bad = (df[col] < lo) | (df[col] > hi)
            df.loc[bad, col] = np.nan
    return df


def prepare_cohort(df):
    """
    Apply masking and exclusions; return (analysis cohort, participant flow).

    Masking runs BEFORE the missing-group exclusion, so someone whose whole
    group was implausible is excluded rather than having the group imputed.
    The flow is a list of (step, n remaining) rows for the report's flow diagram.
    Outcome-specific missingness (cognitive score, dementia follow-up) is NOT
    excluded here; each RQ2 analysis drops its own missing outcomes.
    """
    flow = [("ARIC-NCS participants with data", len(df))]
    n_cells = int(df[schema.all_biomarker_columns()].notna().sum().sum())
    df = mask_implausible(df)
    n_masked = n_cells - int(df[schema.all_biomarker_columns()].notna().sum().sum())

    df = df[df[schema.PREVALENT_DEMENTIA_COL] != 1]
    flow.append(("after excluding prevalent dementia", len(df)))

    df = df[df[schema.AGE_COL].notna() & df[schema.SEX_COL].isin(["M", "F"])]
    flow.append(("after excluding missing age or sex", len(df)))

    df = df[df[schema.TARGET_COL].notna()]
    flow.append(("after excluding missing brain age (failed scan / QC)", len(df)))

    keep = pd.Series(True, index=df.index)
    for cols in schema.BIOMARKER_GROUPS.values():
        keep &= ~df[cols].isna().all(axis=1)
    df = df[keep].copy()
    flow.append(("after excluding participants missing a whole biomarker group", len(df)))

    df[schema.SEX_MALE_COL] = (df[schema.SEX_COL] == "M").astype(float)
    df = df.reset_index(drop=True)

    flow.append(("analysis cohort with cognitive score", int(df[schema.COGNITIVE_COL].notna().sum())))
    surv_ok = df[schema.DEMENTIA_TIME_COL].notna() & df[schema.DEMENTIA_EVENT_COL].notna()
    flow.append(("analysis cohort with dementia follow-up", int(surv_ok.sum())))
    flow.append(("incident dementia events", int(df.loc[surv_ok, schema.DEMENTIA_EVENT_COL].sum())))
    flow.append(("biomarker values masked as implausible (cells)", n_masked))
    return df, flow


def exclude_participants(df):
    """Masking + exclusions only (the cohort from prepare_cohort, without the flow)."""
    return prepare_cohort(df)[0]


def covariate_columns(cfg):
    """Model input columns for the covariates named in config.yaml."""
    names = cfg["features"]["covariates"]
    unknown = [n for n in names if n not in schema.COVARIATE_COLUMNS]
    if unknown:
        raise ValueError(f"Unknown covariates {unknown}; options: {list(schema.COVARIATE_COLUMNS)}")
    return [schema.COVARIATE_COLUMNS[n] for n in names]


# --- fold construction -------------------------------------------------------
def stratify_key(df, n_splits, include_event=True):
    """
    Stratum label per participant: age band x sex (x incident dementia).

    Strata too small to appear in every fold are pooled into one "rare"
    stratum (and into the largest stratum if even that is too small), so
    StratifiedKFold never fails on real data with sparse cells.
    """
    band = pd.cut(df[schema.AGE_COL], bins=AGE_BANDS, labels=False)
    key = band.astype(str) + "_" + df[schema.SEX_COL].astype(str)
    if include_event:
        key = key + "_" + df[schema.DEMENTIA_EVENT_COL].fillna(-1).astype(int).astype(str)
    counts = key.value_counts()
    rare = key.map(counts) < n_splits
    if rare.any():
        key = key.where(~rare, "rare")
        if (key == "rare").sum() < n_splits:
            key = key.replace("rare", key[key != "rare"].value_counts().idxmax())
    return key.to_numpy()


def outer_folds(df, n_splits=5, repeats=1, seed=42):
    """Yield (repeat, fold, train_idx, test_idx) for repeated stratified K-fold."""
    strat = stratify_key(df, n_splits)
    for r in range(repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for k, (tr, te) in enumerate(skf.split(np.zeros(len(df)), strat)):
            yield r, k, tr, te


# --- per-fold preprocessing (lives inside each model pipeline) ---------------
def make_preprocessor(columns):
    """
    Median-impute + standardise numeric columns; mode-impute + one-hot encode
    categorical ones (site). Fitted on whatever data the pipeline is fitted on,
    which is always a training split.
    """
    cat = [c for c in columns if c in schema.CATEGORICAL_COLUMNS]
    num = [c for c in columns if c not in cat]
    transformers = []
    if num:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num))
    if cat:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)
