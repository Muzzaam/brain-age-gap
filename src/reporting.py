"""
Saving results: aggregate tables only.

Each run writes to its own folder, results/<timestamp>_<source>/, containing
CSV tables, a human-readable summary.txt, run_info.json (config, package
versions, git commit, timing) and the console log. These files contain no
participant-level rows, so they are the part of a run on the real ARIC data
that can be carried away from the secure machine (check the data-use agreement
first). Figures can then be made from the CSVs anywhere.

Safety checks: writing refuses any table that has a participant id column,
and counts below the minimum cell size are suppressed in the flow table.
"""

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, resolve_path
from .data import schema


def new_run_dir(cfg, tag=None):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}_{tag or cfg['data']['source']}"
    d = resolve_path(cfg["reporting"]["results_dir"]) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def suppress_small(df, column="n", min_cell=10):
    """Replace counts in (0, min_cell) with '<min_cell' (disclosure control)."""
    df = df.copy()
    small = pd.to_numeric(df[column], errors="coerce").between(1, min_cell - 1)
    df[column] = df[column].astype(object)
    df.loc[small, column] = f"<{min_cell}"
    return df


def write_tables(results, run_dir, cfg):
    min_cell = cfg["reporting"].get("min_cell_size", 10)
    written = []
    for name, obj in results.items():
        if name.startswith("_") or not isinstance(obj, pd.DataFrame):
            continue
        if schema.ID_COL in obj.columns:
            raise RuntimeError(f"Refusing to write {name}: it contains participant ids.")
        if name == "participant_flow":
            obj = suppress_small(obj, "n", min_cell)
        path = run_dir / f"{name}.csv"
        obj.to_csv(path, index=False, float_format="%.4f")
        written.append(path.name)
    return written


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def _versions():
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "pandas", "sklearn", "xgboost", "shap", "sksurv", "scipy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def write_run_info(run_dir, cfg, meta):
    info = {"finished": datetime.now().isoformat(timespec="seconds"), "git_commit": _git_commit(),
            "versions": _versions(), "meta": meta, "config": cfg}
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2, default=str))


# --- human-readable summary ---------------------------------------------------
def _fmt(m, s):
    return f"{m:.3f} +/- {s:.3f}" if np.isfinite(s) else f"{m:.3f}"


def summary_text(results, cfg):
    thr = cfg["evaluation"]["bag_r2_threshold"]
    L = []
    if cfg["data"]["source"] == "synthetic":
        L += ["SYNTHETIC DATA: these numbers validate the code, not the biology.", ""]
    L += ["PARTICIPANT FLOW"] + [f"  {s:<62}{n}" for s, n in
                                 suppress_small(results["participant_flow"], "n",
                                                cfg["reporting"].get("min_cell_size", 10)).itertuples(index=False)]

    if "rq1_summary" in results:
        s = results["rq1_summary"]
        for variant, sv in s.groupby("variant", sort=False):
            L += ["", f"RQ1  BAG recovery, mean +/- SD over outer folds  [{variant} BAG]",
                  f"  {'feature set':<44}{'model':<13}{'R2':>16}{'MAE':>16}  >= {thr}"]
            for row in sv.itertuples(index=False):
                L.append(f"  {row.feature_set:<44}{row.model:<13}{_fmt(row.R2_mean, row.R2_sd):>16}"
                         f"{_fmt(row.MAE_mean, row.MAE_sd):>16}  {'yes' if row.meets_R2_threshold else 'no'}")
        c = results["rq1_comparisons"]
        L += ["", "RQ1  incremental R2 over covariates, and R2 lost when each group is dropped"]
        for row in c.itertuples(index=False):
            L.append(f"  [{row.variant}] {row.model:<12}{row.comparison:<72}"
                     f"{row.improvement:+.3f} [{row.ci_low:+.3f}, {row.ci_high:+.3f}] p={row.p:.3f}")
    if "rq1_shap" in results:
        L += ["", "RQ1  top 5 SHAP features (mean |SHAP| over folds)"]
        for (variant, m), g in results["rq1_shap"].groupby(["variant", "model"], sort=False):
            top = ", ".join(f"{f} {v:.2f}" for f, v in zip(g["feature"][:5], g["mean"][:5]))
            L.append(f"  [{variant}] {m:<12}{top}")
    if "bias_correction" in results:
        b = results["bias_correction"]
        L += ["", f"BIAS CORRECTION  fitted on: {b['reference'].iloc[0]} "
                  f"(n~{b['n_reference'].mean():.0f} per fold); test-fold BAG-age correlation "
                  f"{b['test_age_bias_before'].mean():+.3f} -> {b['test_age_bias_after'].mean():+.3f}"]
        if b["test_age_bias_after"].abs().mean() >= b["test_age_bias_before"].abs().mean():
            L.append("  WARNING: correction did not reduce BAG's age dependence. The reference set "
                     "may be selected on something related to BAG and age; check its definition.")
    if "rq2_stage1" in results:
        s1 = results["rq2_stage1"]
        L += ["", "RQ2  stage-1 BAG estimator (cross-fitted train R2 should be close to test R2)"]
        for variant, g in s1.groupby("variant", sort=False):
            L.append(f"  [{variant}] models chosen: {g['model'].value_counts().to_dict()}  "
                     f"train OOF R2 {g['train_oof_R2'].mean():.3f}, test R2 {g['test_R2'].mean():.3f}")
    for part, label, metrics in (("rq2_cognitive", "cognitive score", ["RMSE", "R2"]),
                                 ("rq2_dementia", "incident dementia", ["C_index", "IBS"])):
        if f"{part}_summary" not in results:
            continue
        s = results[f"{part}_summary"]
        for (variant, m), g in s.groupby(["variant", "model"], sort=False):
            L += ["", f"RQ2  {label}  [{variant} BAG, stage 2 = {m}]",
                  f"  {'condition':<34}" + "".join(f"{x:>18}" for x in metrics)]
            for row in g.itertuples(index=False):
                L.append(f"  {row.condition:<34}" + "".join(
                    f"{_fmt(getattr(row, x + '_mean'), getattr(row, x + '_sd')):>18}" for x in metrics))
            c = results[f"{part}_comparisons"]
            c = c[(c["variant"] == variant) & (c["model"] == m)]
            for row in c.itertuples(index=False):
                L.append(f"    {row.comparison:<62}{row.metric:<8} improvement {row.improvement:+.4f} "
                         f"[{row.ci_low:+.4f}, {row.ci_high:+.4f}] p={row.p:.3f} p_holm={row.p_holm:.3f}"
                         f"{'  *' if row.significant else ''}")
    for part, label, metrics in (("rq3_cognitive", "cognitive score", ["RMSE", "R2"]),
                                 ("rq3_dementia", "incident dementia", ["C_index", "IBS"])):
        if f"{part}_summary" not in results:
            continue
        s = results[f"{part}_summary"]
        for variant, g in s.groupby("variant", sort=False):
            extra = ["bag_head_R2"] if "bag_head_R2_mean" in g else []
            L += ["", f"RQ3  {label}, neural framings  [{variant} BAG]",
                  f"  {'framing':<22}" + "".join(f"{x:>18}" for x in metrics + extra)]
            for row in g.itertuples(index=False):
                L.append(f"  {row.framing:<22}" + "".join(
                    f"{_fmt(getattr(row, x + '_mean'), getattr(row, x + '_sd')):>18}"
                    if np.isfinite(getattr(row, x + "_mean")) else f"{'-':>18}"
                    for x in metrics + extra))
            c = results[f"{part}_comparisons"]
            for row in c[c["variant"] == variant].itertuples(index=False):
                L.append(f"    {row.comparison:<40}{row.metric:<8} improvement {row.improvement:+.4f} "
                         f"[{row.ci_low:+.4f}, {row.ci_high:+.4f}] p={row.p:.3f} p_holm={row.p_holm:.3f}"
                         f"{'  *' if row.significant else ''}")
    L += ["", "Improvement > 0 means the first condition is better. * = p < "
              f"{cfg['evaluation']['alpha']} (corrected resampled t-test, unadjusted). p_holm is "
              "Holm-adjusted over all comparisons for that outcome and BAG variant."]
    return "\n".join(L)
