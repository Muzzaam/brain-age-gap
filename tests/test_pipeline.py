"""Data contract, loader, exclusions and fold construction.
Run: python -m tests.test_pipeline"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from tests.helpers import cohort, run_tests
from src.data import load_data, schema, loader
from src.preprocessing import prepare_cohort, outer_folds, stratify_key, make_preprocessor


def test_schema_conformance():
    df = load_data(source="synthetic", n=300, seed=0)
    for col in schema.REQUIRED_COLUMNS:
        assert col in df.columns, f"missing {col}"


def test_exclusions_applied_and_counted():
    df = load_data(source="synthetic", n=600, seed=1, bag_missing_rate=0.05)
    kept, flow = prepare_cohort(df)
    assert (kept[schema.PREVALENT_DEMENTIA_COL] == 0).all()
    assert kept[schema.TARGET_COL].notna().all()          # failed scans excluded
    for cols in schema.BIOMARKER_GROUPS.values():
        assert not kept[cols].isna().all(axis=1).any()
    steps = dict(flow)
    assert steps["ARIC-NCS participants with data"] == 600
    assert steps["after excluding participants missing a whole biomarker group"] == len(kept)


def test_masking_happens_before_group_exclusion():
    # someone whose ENTIRE cardiovascular group is implausible must be excluded,
    # not kept with the whole group imputed
    df = load_data(source="synthetic", n=200, seed=2, cell_missing_rate=0, group_missing_rate=0,
                   outlier_rate=0, prevalent_dementia_rate=0)
    pid = df[schema.ID_COL].iloc[0]
    df.loc[df.index[0], ["sbp", "dbp", "resting_hr"]] = [900, 900, 900]
    kept, _ = prepare_cohort(df)
    assert pid not in set(kept[schema.ID_COL])


def test_outcome_missingness_not_excluded_globally():
    c = cohort(outcome_missing_rate=0.1)
    assert c[schema.COGNITIVE_COL].isna().any()           # handled per analysis, not dropped here


def test_outer_folds_disjoint_and_complete():
    c = cohort(n=500)
    seen = {}
    for r, k, tr, te in outer_folds(c, n_splits=5, repeats=2, seed=0):
        assert not set(tr) & set(te)
        assert len(tr) + len(te) == len(c)
        seen.setdefault(r, []).extend(te)
    for r, te_all in seen.items():
        assert sorted(te_all) == list(range(len(c)))      # each participant tested once per repeat


def test_stratification_survives_tiny_strata():
    # 49 people in one stratum and a lone participant in another: StratifiedKFold
    # would refuse this unless the lone stratum is pooled
    c = pd.DataFrame({schema.AGE_COL: [60.0] * 49 + [90.0], schema.SEX_COL: ["M"] * 50,
                      schema.DEMENTIA_EVENT_COL: [0] * 50})
    key = stratify_key(c, n_splits=5)
    assert pd.Series(key).value_counts().min() >= 5
    list(outer_folds(c, n_splits=5))                      # must not raise


def test_preprocessor_handles_site_and_missing():
    c = cohort(n=300)
    cols = ["age", "sex_male", "site"] + schema.all_biomarker_columns()
    X = make_preprocessor(cols).fit_transform(c[cols])
    assert not np.isnan(X).any()
    assert X.shape[1] == len(cols) - 1 + len(schema.SITES)  # site one-hot encoded


def test_aric_loader_joins_brain_age():
    df = load_data(source="synthetic", n=50, seed=3)
    tab = df.drop(columns=[schema.TARGET_COL, schema.BRAIN_AGE_COL]).rename(columns={"sbp": "SYSBP"})
    ba = pd.DataFrame({schema.ID_COL: df[schema.ID_COL], schema.BRAIN_AGE_COL: df[schema.AGE_COL] + 2.0,
                       "status": ["ok"] * 49 + ["failed"]})
    old = dict(loader.ARIC_COLUMN_MAP)
    loader.ARIC_COLUMN_MAP["SYSBP"] = "sbp"
    try:
        with tempfile.TemporaryDirectory() as d:
            tab.to_csv(Path(d) / "t.csv", index=False)
            ba.to_csv(Path(d) / "b.csv", index=False)
            out = load_data(source="aric", path=str(Path(d) / "t.csv"), brain_age_path=str(Path(d) / "b.csv"))
    finally:
        loader.ARIC_COLUMN_MAP.clear()
        loader.ARIC_COLUMN_MAP.update(old)
    assert "sbp" in out.columns
    assert np.allclose(out[schema.TARGET_COL].iloc[:49], 2.0)
    assert np.isnan(out[schema.TARGET_COL].iloc[49])      # failed scan -> missing BAG, not dropped


def test_linkage_with_scan_id_map_and_report():
    df = load_data(source="synthetic", n=40, seed=4).drop(columns=[schema.BRAIN_AGE_COL])
    ids = df[schema.ID_COL].tolist()
    ba = pd.DataFrame({schema.ID_COL: [f"IMG{i}" for i in range(42)],     # imaging ids, 2 orphans
                       schema.BRAIN_AGE_COL: 75.0, "status": ["ok"] * 41 + ["failed: X"]})
    id_map = pd.DataFrame({"scan_id": [f"IMG{i}" for i in range(40)], schema.ID_COL: ids})
    out = loader.link_brain_age(df, ba, id_map)
    rep = out.attrs["linkage"]
    assert rep["scans_failed"] == 1 and rep["scans_not_in_id_map"] == 1
    assert rep["participants_with_brain_age"] == 40
    assert out.set_index(schema.ID_COL).loc[ids[3], schema.BRAIN_AGE_COL] == 75.0


def test_linkage_rejects_two_scans_for_one_participant():
    df = load_data(source="synthetic", n=10, seed=4)
    pid = df[schema.ID_COL].iloc[0]
    ba = pd.DataFrame({schema.ID_COL: [pid, pid], schema.BRAIN_AGE_COL: [70.0, 71.0]})
    try:
        loader.link_brain_age(df, ba)
    except ValueError:
        return
    raise AssertionError("expected ValueError for duplicate scans")


if __name__ == "__main__":
    run_tests(globals())
