"""
Preprocessing, implemented exactly as the proposal specifies.

Order matters for avoiding leakage:
  1. Exclude participants (prevalent dementia; missing an entire biomarker group).
  2. Split into train / val / test, stratified by age band x sex.
  3. Fit imputer + scaler on TRAIN ONLY, then apply to val and test.

Returns a dict of ready-to-model arrays plus the fitted transformers, so the
same objects can later be applied to the real held-out data untouched.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from .data import schema

AGE_BANDS = [0, 72, 77, 82, 200]  # (-,72], (72,77], (77,82], (82,+)


def mask_implausible(df):
    """Set biomarker values outside plausible ranges to NaN (so they get imputed).

    Uses the fixed ranges in schema.PLAUSIBLE_RANGES, so this introduces no
    leakage from the data. This is the proposal's outlier-flagging step: rather
    than let a mis-recorded SBP of 900 distort standardisation, treat it as
    missing and impute it.
    """
    df = df.copy()
    for col in schema.all_biomarker_columns():
        if col in schema.PLAUSIBLE_RANGES:
            lo, hi = schema.PLAUSIBLE_RANGES[col]
            bad = (df[col] < lo) | (df[col] > hi)
            df.loc[bad, col] = np.nan
    return df


def exclude_participants(df):
    """Drop prevalent dementia and anyone missing an entire biomarker group."""
    df = df[df[schema.PREVALENT_DEMENTIA_COL] == 0].copy()

    keep = pd.Series(True, index=df.index)
    for cols in schema.BIOMARKER_GROUPS.values():
        group_all_missing = df[cols].isna().all(axis=1)
        keep &= ~group_all_missing
    return df[keep].copy()


def _stratify_key(df):
    band = pd.cut(df[schema.AGE_COL], bins=AGE_BANDS, labels=False)
    return band.astype(str) + "_" + df[schema.SEX_COL].astype(str)


def make_splits(df, test_size=0.2, val_size=0.2, seed=42):
    """Stratified train/val/test split. val_size is a fraction of the non-test data."""
    strat = _stratify_key(df)
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=strat, random_state=seed
    )
    strat_tv = _stratify_key(train_val)
    train, val = train_test_split(
        train_val, test_size=val_size, stratify=strat_tv, random_state=seed
    )
    return train, val, test


def build_feature_frame(df, include_bag=False):
    """Assemble the model input matrix: biomarkers (+ optionally BAG)."""
    cols = schema.all_biomarker_columns()
    if include_bag:
        cols = cols + [schema.TARGET_COL]
    return df[cols].copy()


def prepare_data(df, target=schema.TARGET_COL, test_size=0.2, val_size=0.2, seed=42):
    """
    Full preprocessing for the BAG-estimation task (RQ1).

    Returns a dict with X_train/val/test (numpy), y_train/val/test, the fitted
    imputer and scaler, the feature names, and the raw split frames (for later
    outcome modelling and subgroup analysis).
    """
    df = exclude_participants(df)
    df = mask_implausible(df)
    train, val, test = make_splits(df, test_size, val_size, seed)

    feature_cols = schema.all_biomarker_columns()

    imputer = SimpleImputer(strategy="median").fit(train[feature_cols])
    scaler = StandardScaler().fit(imputer.transform(train[feature_cols]))

    def _transform(frame):
        return scaler.transform(imputer.transform(frame[feature_cols]))

    return {
        "X_train": _transform(train), "y_train": train[target].to_numpy(),
        "X_val": _transform(val),     "y_val": val[target].to_numpy(),
        "X_test": _transform(test),   "y_test": test[target].to_numpy(),
        "feature_names": feature_cols,
        "imputer": imputer, "scaler": scaler,
        "train_df": train, "val_df": val, "test_df": test,
    }
