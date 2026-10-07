"""End-to-end: run_all and preflight on synthetic data, real-data error paths,
and the no-participant-data guarantee on saved outputs.
Run: python -m tests.test_run_all"""
import io
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from tests.helpers import run_tests, quick_cfg
from src.data import schema, load_data
from scripts import run_all, preflight


def _quiet(fn, *a):
    with redirect_stdout(io.StringIO()):
        return fn(*a)


def test_run_all_quick_writes_aggregate_outputs_only():
    with tempfile.TemporaryDirectory() as d:
        run_dir = _quiet(run_all.main, ["--quick", "--results-dir", d, "--tag", "t"])
        names = {p.name for p in Path(run_dir).iterdir()}
        for f in ["participant_flow.csv", "rq1_summary.csv", "rq1_comparisons.csv", "rq1_shap.csv",
                  "rq1_subgroups.csv", "rq2_stage1.csv", "rq2_cognitive_comparisons.csv",
                  "rq2_dementia_comparisons.csv", "bias_correction.csv", "summary.txt",
                  "run_info.json", "run.log"]:
            assert f in names, f
        n_cohort = None
        for p in Path(run_dir).glob("*.csv"):
            t = pd.read_csv(p)
            assert schema.ID_COL not in t.columns, p.name
            if p.name == "participant_flow.csv":
                n_cohort = int(t["n"].iloc[0])
        for p in Path(run_dir).glob("*.csv"):  # no table is one-row-per-participant
            assert len(pd.read_csv(p)) != n_cohort, p.name
        s = pd.read_csv(Path(run_dir) / "rq1_summary.csv")
        assert {"covariates", "biomarkers", "covariates+biomarkers"} <= set(s["feature_set"])
        assert set(s["variant"]) == {"uncorrected", "corrected"}
        assert "SYNTHETIC DATA" in (Path(run_dir) / "summary.txt").read_text(encoding="utf-8")


def test_only_flag_limits_parts():
    with tempfile.TemporaryDirectory() as d:
        run_dir = _quiet(run_all.main, ["--quick", "--only", "rq1", "--results-dir", d])
        names = {p.name for p in Path(run_dir).iterdir()}
        assert "rq1_summary.csv" in names and "rq2_cognitive_summary.csv" not in names


def test_aric_source_without_file_fails_clearly():
    with tempfile.TemporaryDirectory() as d:
        try:
            _quiet(run_all.main, ["--quick", "--source", "aric", "--results-dir", d])
        except FileNotFoundError as e:
            assert "tabular_path" in str(e)
        else:
            raise AssertionError("expected FileNotFoundError")


def test_preflight_passes_on_synthetic():
    df = load_data(source="synthetic", n=800, seed=0, bag_missing_rate=0.02)
    lines, table, n_err = preflight.check(df, quick_cfg())
    assert n_err == 0, lines
    assert not table.empty


def test_preflight_catches_bad_codings_and_missing_columns():
    df = load_data(source="synthetic", n=300, seed=0)
    bad = df.assign(sex=df["sex"].map({"M": 1, "F": 2}))
    _, _, n_err = preflight.check(bad, quick_cfg())
    assert n_err >= 1
    lines, _, n_err = preflight.check(df.drop(columns=["grip_strength"]), quick_cfg())
    assert n_err >= 1 and any("grip_strength" in l for l in lines)


def test_preflight_flags_scrambled_scan_linkage():
    # if scans are attached to the wrong people, brain age stops tracking age
    df = load_data(source="synthetic", n=800, seed=0)
    ba = df[schema.BRAIN_AGE_COL].sample(frac=1, random_state=1).to_numpy()
    bad = df.assign(**{schema.BRAIN_AGE_COL: ba, schema.TARGET_COL: ba - df[schema.AGE_COL]})
    lines, _, n_err = preflight.check(bad, quick_cfg())
    assert n_err >= 1 and any("wrong participants" in l for l in lines), lines


if __name__ == "__main__":
    run_tests(globals())
