"""
Preflight check: the FIRST thing to run on the real ARIC data.

    python -m scripts.preflight                  # uses data.source from config.yaml
    python -m scripts.preflight --source aric

Loads the data exactly as the analysis will, then reports (aggregate numbers
only) everything that could silently break or bias the analysis: missing or
unmapped columns, unexpected codings, missingness, implausible values, the
BAG distribution and its age bias, event counts, and the participant flow.
Writes preflight.txt and preflight_columns.csv to results/<timestamp>_preflight/.
Fix every ERROR before running scripts.run_all; read every WARNING.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.data import schema, validate_schema
from src.data import loader
from src.preprocessing import prepare_cohort
from src.reporting import new_run_dir, suppress_small


def _load_unvalidated(cfg):
    if cfg["data"]["source"] == "synthetic":
        return loader.generate_synthetic_cohort(**cfg["data"]["synthetic"])
    a = cfg["data"]["aric"]
    ba = resolve_path(a["brain_age_path"]) if a.get("brain_age_path") else None
    mp = resolve_path(a["scan_id_map_path"]) if a.get("scan_id_map_path") else None
    paths = a["tabular_path"]
    paths = [paths] if isinstance(paths, str) else list(paths)
    return loader._load_aric([str(resolve_path(p)) for p in paths], str(ba) if ba else None,
                             str(mp) if mp else None)


def check(df, cfg):
    """Return (report lines, per-column table, n_errors)."""
    L, errors = [], 0
    min_cell = cfg["reporting"].get("min_cell_size", 10)
    K = cfg["cv"]["outer_folds"]

    def err(msg):
        nonlocal errors
        errors += 1
        L.append(f"ERROR   {msg}")

    def warn(msg):
        L.append(f"WARNING {msg}")

    L.append(f"Rows: {len(df)}   Columns: {len(df.columns)}")
    missing = [c for c in schema.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        err(f"required columns missing: {missing}")
        L.append("        available columns: " + ", ".join(map(str, df.columns)))
        L.append("        -> add mappings to ARIC_COLUMN_MAP in src/data/loader.py")
        return L, pd.DataFrame(), errors
    try:
        extra = validate_schema(df)
        if extra:
            L.append(f"Extra columns (ignored by the models): {extra}")
    except ValueError as e:
        err(str(e))

    # codings
    sex_vals = set(df[schema.SEX_COL].dropna().unique())
    if not sex_vals <= {"M", "F"}:
        err(f"sex must be coded 'M'/'F'; found {sorted(map(str, sex_vals))} -> ARIC_VALUE_MAPS")
    site_vals = set(df[schema.SITE_COL].dropna().unique())
    if not site_vals <= set(schema.SITES):
        warn(f"unexpected site values {sorted(map(str, site_vals - set(schema.SITES)))}")
    for col in [schema.AGE_COL, schema.TARGET_COL, schema.COGNITIVE_COL,
                schema.DEMENTIA_TIME_COL] + schema.all_biomarker_columns():
        if not pd.api.types.is_numeric_dtype(df[col]):
            err(f"{col} is not numeric (dtype {df[col].dtype})")
    ev = set(df[schema.DEMENTIA_EVENT_COL].dropna().unique())
    if not ev <= {0, 1}:
        err(f"dementia_event must be 0/1; found {sorted(map(str, ev))}")
    pv = set(df[schema.PREVALENT_DEMENTIA_COL].dropna().unique())
    if not pv <= {0, 1}:
        err(f"prevalent_dementia must be 0/1; found {sorted(map(str, pv))}")

    # per-column table: missingness and implausible values
    rows = []
    for col in schema.REQUIRED_COLUMNS[1:]:
        s = df[col]
        row = {"column": col, "pct_missing": round(100 * s.isna().mean(), 1)}
        if pd.api.types.is_numeric_dtype(s):
            # percentiles, not min/max: an extreme is a single participant's value
            row.update(mean=s.mean(), sd=s.std(), p01=s.quantile(0.01), p99=s.quantile(0.99))
        if col in schema.PLAUSIBLE_RANGES and pd.api.types.is_numeric_dtype(s):
            lo, hi = schema.PLAUSIBLE_RANGES[col]
            row["n_implausible"] = int(((s < lo) | (s > hi)).sum())
        rows.append(row)
        if col in schema.all_biomarker_columns() and row["pct_missing"] > 30:
            warn(f"{col} is {row['pct_missing']}% missing")
    table = pd.DataFrame(rows)
    if "n_implausible" in table:
        table = suppress_small(table.rename(columns={"n_implausible": "n"}), "n", min_cell) \
            .rename(columns={"n": "n_implausible"})

    for note in df.attrs.get("merge_notes", []):
        warn(f"duplicate column across files: {note}")

    # scan-to-participant linkage
    link = df.attrs.get("linkage")
    if link:
        L.append("Scan linkage: " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in link.items()))
        if link.get("scans_with_no_participant", 0) or link.get("scans_not_in_id_map", 0):
            warn("some usable scans did not match any participant; check the id format "
                 "(--id-regex in the imaging runner, or data.aric.scan_id_map_path)")
        if link["scans_usable"] and link["scans_matched_to_participant"] < 0.9 * link["scans_usable"]:
            err("fewer than 90% of usable scans matched a participant id")
    if schema.BRAIN_AGE_COL in df.columns:
        ok = df[schema.BRAIN_AGE_COL].notna() & df[schema.AGE_COL].notna()
        if ok.sum() > 10:
            r_ba = np.corrcoef(df.loc[ok, schema.BRAIN_AGE_COL], df.loc[ok, schema.AGE_COL])[0, 1]
            L.append(f"Predicted brain age vs chronological age: r = {r_ba:+.3f} (n = {int(ok.sum())})")
            # A brain-age model always tracks real age somewhat; a correlation
            # near zero means the scans are attached to the wrong people.
            if r_ba < 0.1:
                err("predicted brain age barely correlates with age: scans are probably "
                    "linked to the wrong participants")
            elif r_ba < 0.3:
                warn("predicted brain age correlates weakly with age; spot-check the scan linkage")

    # BAG sanity
    bag, age = df[schema.TARGET_COL], df[schema.AGE_COL]
    ok = bag.notna() & age.notna()
    if ok.sum() > 10:
        r = np.corrcoef(bag[ok], age[ok])[0, 1]
        L.append(f"BAG: mean {bag.mean():+.2f}, SD {bag.std():.2f} years; correlation with age r = {r:+.3f}")
        if abs(bag.mean()) > 5:
            warn("BAG mean is far from 0: the brain-age model has a large offset in this cohort "
                 "(common when its training ages differ). Bias correction matters; report both variants.")
        if not 1 < bag.std() < 20:
            warn("BAG SD looks implausible; check brain_age units and the participant-id join.")
        if abs(r) > 0.3:
            warn("BAG is strongly age-dependent; expect a large 'covariates' R2 in RQ1 on uncorrected BAG.")
    if age.notna().any() and (age.min() < 40 or age.max() > 100):
        warn("some ages are outside 40-100; check units / which visit's age was used.")
    # ARIC data hold each variable once per visit. Visit 5 participants were about
    # 67 to 90, so a much younger median means an earlier visit's column was mapped
    # (and the biomarkers may come from the wrong visit too).
    if age.notna().any() and age.median() < 65:
        warn(f"median age is {age.median():.0f}, but visit 5 participants were about 67 to 90: "
             "check that age and the biomarkers are mapped from the visit 5 columns")

    # healthy reference
    ref = cfg["bias_correction"].get("reference_column")
    if ref and ref in df.columns:
        L.append(f"Bias-correction reference set: {int((df[ref] == 1).sum())} participants flagged in '{ref}'")
    else:
        warn(f"no '{ref}' column: bias correction will be fitted on all training participants")

    # participant flow and event counts
    cohort, flow = prepare_cohort(df)
    L += ["", "Participant flow:"] + [
        f"  {s:<62}{n}" for s, n in suppress_small(pd.DataFrame(flow, columns=["s", "n"]),
                                                    "n", min_cell).itertuples(index=False)]
    surv = cohort[schema.DEMENTIA_TIME_COL].notna() & cohort[schema.DEMENTIA_EVENT_COL].notna()
    events = int(cohort.loc[surv, schema.DEMENTIA_EVENT_COL].sum())
    if surv.any():
        L.append(f"Follow-up: median {cohort.loc[surv, schema.DEMENTIA_TIME_COL].median():.1f}, "
                 f"99th percentile {cohort.loc[surv, schema.DEMENTIA_TIME_COL].quantile(0.99):.1f} years")
    if events < 50:
        warn(f"only {events} incident dementia events: the survival analysis will be underpowered "
             f"(~{events / K:.0f} per test fold). Report it with its uncertainty, as the proposal planned.")
    if len(cohort) < 200:
        err(f"analysis cohort has only {len(cohort)} participants after exclusions")

    L += ["", f"{errors} error(s). " + ("Fix them before running scripts.run_all." if errors
                                         else "OK to run scripts.run_all.")]
    return L, table, errors


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", choices=["synthetic", "aric"])
    args = ap.parse_args(argv)
    cfg = load_config(args.config, overrides={"data": {"source": args.source}} if args.source else None)

    df = _load_unvalidated(cfg)
    lines, table, n_err = check(df, cfg)
    out = new_run_dir(cfg, "preflight")
    text = "\n".join([f"Preflight for source '{cfg['data']['source']}'", ""] + lines)
    (out / "preflight.txt").write_text(text, encoding="utf-8")
    if not table.empty:
        table.to_csv(out / "preflight_columns.csv", index=False, float_format="%.3f")
        text += "\n\nPer-column summary:\n" + table.to_string(index=False)
    print(text)
    print(f"\nSaved to {out}")
    return n_err


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
