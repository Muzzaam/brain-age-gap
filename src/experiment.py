"""
The experiment engine: one nested cross-validation loop for every analysis.

Outer loop  repeated stratified K-fold over the analysis cohort. Each outer
            test fold is predicted exactly once per repeat by models that never
            saw it; the spread over folds gives the reported mean +/- SD.
Inner loop  inside each outer training fold: hyperparameter tuning (GridSearchCV)
            and stage-1 cross-fitting. The outer test fold is never touched.

Within each outer fold, for each bias-correction variant of the BAG labels:

  RQ1   every model family x every feature set:
          covariates                       (age/sex/site only: how much of BAG is
                                            just the brain-age model's age bias?)
          biomarkers                       (biomarkers alone)
          covariates+biomarkers            (primary; incremental R2 over covariates)
          covariates+biomarkers-<group>    (biomarker-group ablation)
        plus SHAP (repeat 0), and out-of-fold predictions for the subgroup and
        residual analyses (repeat 0, where every participant is predicted once).
  RQ2   stage 1 = the chosen BAG estimator, cross-fitted; stage 2 = every
        outcome model x the four conditions, for the cognitive score and for
        incident dementia.
  RQ3   the neural framings (src/models/framings.py) on both outcomes: one
        network, BAG entering as nothing / an input / an auxiliary head / a
        bottleneck / a pretraining task, plus the bottleneck ablation.

All analyses share the same folds, so comparisons between them are paired.
Only aggregate results leave this module; participant-level predictions are
kept in memory and summarised.
"""

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from .data import schema
from .preprocessing import prepare_cohort, covariate_columns, outer_folds
from .models.tuning import tune, tune_framing
from .models.framings import RQ3_FRAMINGS, LABEL_FREE
from .models.neural import make_targets
from .models.two_stage import (CONDITIONS, condition_columns, cross_fit_bag,
                               add_bag_features)
from .evaluation.metrics import bag_metrics
from .evaluation.outcome_metrics import cognitive_metrics
from .evaluation.survival_metrics import make_survival_target, concordance, integrated_brier
from .evaluation.bias_correction import fit_on_reference, correct_bag, age_bias
from .evaluation.interpretation import shap_importances, subgroup_metrics
from .evaluation.diagnostics import residual_analysis
from .evaluation.stats import summarize, compare, add_holm

PRIMARY_SET = "covariates+biomarkers"
ALL_PARTS = ("rq1", "rq2_cognitive", "rq2_dementia", "rq3_cognitive", "rq3_dementia")
RQ2_PARTS = ("rq2_cognitive", "rq2_dementia", "rq3_cognitive", "rq3_dementia")  # need stage 1


def rq1_feature_sets(covariates):
    bio = schema.all_biomarker_columns()
    sets = {
        "covariates": list(covariates),
        "biomarkers": bio,
        PRIMARY_SET: list(covariates) + bio,
    }
    for group, cols in schema.BIOMARKER_GROUPS.items():
        sets[f"{PRIMARY_SET}-{group}"] = list(covariates) + [c for c in bio if c not in cols]
    return sets


def bag_labels(train, test, variant, reference_col):
    """BAG labels for one fold: raw, or bias-corrected with a line fit on train only."""
    y_tr = train[schema.TARGET_COL].to_numpy(float)
    y_te = test[schema.TARGET_COL].to_numpy(float)
    if variant == "uncorrected":
        return y_tr, y_te, None
    if variant != "corrected":
        raise ValueError(f"Unknown bias-correction variant {variant!r}")
    params = fit_on_reference(train, reference_col)
    return (correct_bag(y_tr, train[schema.AGE_COL], params),
            correct_bag(y_te, test[schema.AGE_COL], params), params)


def run_experiment(df_raw, cfg, parts=ALL_PARTS, log=print):
    """Run the requested analyses; return a dict of aggregate result tables."""
    t0 = time.time()
    parts = tuple(parts)
    cohort, flow = prepare_cohort(df_raw)
    cov = covariate_columns(cfg)
    cv, mcfg = cfg["cv"], cfg["models"]
    K, R, inner, seed = cv["outer_folds"], cv["repeats"], cv["inner_folds"], cv["seed"]
    quick = cfg.get("quick", False)
    variants = cfg["bias_correction"]["variants"]
    ref_col = cfg["bias_correction"].get("reference_column")
    rep = cfg["reporting"]
    do_rq1 = "rq1" in parts
    do_rq2 = any(p in parts for p in RQ2_PARTS)

    sets = rq1_feature_sets(cov)
    if not do_rq1:  # RQ2 still needs the primary set for stage 1
        sets = {PRIMARY_SET: sets[PRIMARY_SET]}
    conds = condition_columns(cov)

    log(f"Analysis cohort: {len(cohort)} participants. Nested CV: {R} x {K}-fold outer, "
        f"{inner}-fold inner. Variants: {variants}. Parts: {list(parts)}.")

    rows = {k: [] for k in ("rq1", "shap", "bias", "stage1", "rq2_cognitive", "rq2_dementia",
                            "rq3_cognitive", "rq3_dementia")}
    n = len(cohort)
    oof_true = {v: np.full(n, np.nan) for v in variants}
    oof_pred = {}

    for r, k, tr, te in outer_folds(cohort, K, R, seed):
        tf = time.time()
        train, test = cohort.iloc[tr], cohort.iloc[te]
        fseed = seed + 1000 * r + k
        base = {"repeat": r, "fold": k}
        cache = {}  # stage-2 results for conditions without BAG are identical across variants

        for variant in variants:
            y_tr, y_te, bc = bag_labels(train, test, variant, ref_col)
            vb = {"variant": variant, **base}
            if bc is not None:
                rows["bias"].append({**vb, "slope": bc["slope"], "intercept": bc["intercept"],
                                     "reference": bc["reference"], "n_reference": bc["n_reference"],
                                     "test_age_bias_before": age_bias(test[schema.TARGET_COL], test[schema.AGE_COL]),
                                     "test_age_bias_after": age_bias(y_te, test[schema.AGE_COL])})
            if r == 0:
                oof_true[variant][te] = y_te

            # ---------------- RQ1 ----------------
            primary_fits = {}
            for fs, cols in sets.items():
                for m in mcfg["rq1"]:
                    res = tune(m, train, y_tr, cols, inner, fseed, quick)
                    pred = res["estimator"].predict(test[cols])
                    if do_rq1:
                        rows["rq1"].append({**vb, "feature_set": fs, "model": m,
                                            "n_train": len(tr), "n_test": len(te),
                                            **bag_metrics(y_te, pred),
                                            "inner_cv_R2": res["inner_cv_score"],
                                            "best_params": json.dumps(res["best_params"], default=str)})
                    if fs != PRIMARY_SET:
                        continue
                    primary_fits[m] = res
                    if do_rq1 and r == 0:
                        oof_pred.setdefault((variant, m), np.full(n, np.nan))[te] = pred
                        if m in mcfg.get("shap", []):
                            imp = shap_importances(m, res["estimator"], train, test, cols,
                                                   max_explain=rep.get("shap_max_explain", 200),
                                                   seed=fseed)
                            rows["shap"] += [{**vb, "model": m, "feature": f, "mean_abs_shap": v}
                                             for f, v in imp.items()]

            if not do_rq2:
                continue

            # ---------------- RQ2 stage 1 ----------------
            s1 = mcfg.get("stage1", "auto")
            if s1 == "auto":
                s1 = max(primary_fits, key=lambda m: primary_fits[m]["inner_cv_score"])
            s1_fit = primary_fits.get(s1) or tune(s1, train, y_tr, sets[PRIMARY_SET], inner, fseed, quick)
            cols1 = sets[PRIMARY_SET]
            bag_tr, bag_te = cross_fit_bag(s1_fit["estimator"], train[cols1], y_tr, test[cols1],
                                           inner, fseed)
            rows["stage1"].append({**vb, "model": s1, "train_oof_R2": float(r2_score(y_tr, bag_tr)),
                                   "test_R2": float(r2_score(y_te, bag_te))})
            tr2, te2 = add_bag_features(train, test, bag_tr, bag_te, fseed)

            # ---------------- RQ2 cognitive ----------------
            if "rq2_cognitive" in parts:
                mtr = tr2[schema.COGNITIVE_COL].notna().to_numpy()
                mte = te2[schema.COGNITIVE_COL].notna().to_numpy()
                ytr = tr2[schema.COGNITIVE_COL].to_numpy(float)[mtr]
                yte = te2[schema.COGNITIVE_COL].to_numpy(float)[mte]
                for m in mcfg["stage2_cognitive"]:
                    for c in CONDITIONS:
                        key = ("cog", m, c)
                        if "BAG" not in c and "placebo" not in c and key in cache:
                            out = cache[key]
                        else:
                            res = tune(m, tr2[mtr], ytr, conds[c], inner, fseed, quick)
                            out = {**cognitive_metrics(yte, res["estimator"].predict(te2[mte][conds[c]])),
                                   "best_params": json.dumps(res["best_params"], default=str)}
                            cache[key] = out
                        rows["rq2_cognitive"].append({**vb, "model": m, "condition": c,
                                                      "n_train": int(mtr.sum()), "n_test": int(mte.sum()),
                                                      **out})

            # ---------------- RQ2 dementia ----------------
            if "rq2_dementia" in parts:
                def _surv(frame):
                    ok = (frame[schema.DEMENTIA_TIME_COL].notna()
                          & frame[schema.DEMENTIA_EVENT_COL].notna()).to_numpy()
                    f = frame[ok]
                    return ok, make_survival_target(f[schema.DEMENTIA_EVENT_COL].astype(int),
                                                    f[schema.DEMENTIA_TIME_COL])
                mtr, ytr = _surv(tr2)
                mte, yte = _surv(te2)
                for m in mcfg["stage2_dementia"]:
                    for c in CONDITIONS:
                        key = ("dem", m, c)
                        if "BAG" not in c and "placebo" not in c and key in cache:
                            out = cache[key]
                        else:
                            res = tune(m, tr2[mtr], ytr, conds[c], inner, fseed, quick)
                            Xte = te2[mte][conds[c]]
                            out = {"C_index": concordance(yte["event"], yte["time"],
                                                          res["estimator"].predict(Xte)),
                                   "IBS": integrated_brier(ytr, yte, res["estimator"], Xte),
                                   "best_params": json.dumps(res["best_params"], default=str)}
                            cache[key] = out
                        rows["rq2_dementia"].append({**vb, "model": m, "condition": c,
                                                     "n_train": int(mtr.sum()), "n_test": int(mte.sum()),
                                                     "n_events_test": int(yte["event"].sum()), **out})

            # ---------------- RQ3 framings ----------------
            for task, part in (("regression", "rq3_cognitive"), ("survival", "rq3_dementia")):
                if part in parts:
                    rows[part] += _rq3_fold(task, tr2, te2, y_tr, y_te, cov, cfg, fseed, cache, vb)

        log(f"  repeat {r + 1}/{R}, fold {k + 1}/{K} done in {time.time() - tf:.0f}s")

    results = _assemble(rows, flow, cohort, oof_true, oof_pred, cfg, parts)
    results["_meta"] = {"n_cohort": len(cohort), "seconds": round(time.time() - t0, 1),
                        "parts": list(parts)}
    return results


def _rq3_fold(task, tr2, te2, bag_tr, bag_te, cov, cfg, fseed, cache, vb):
    """All RQ3 framings for one outcome in one outer fold and BAG variant."""
    from .models.two_stage import BAG_HAT_COL
    inner, quick = cfg["cv"]["inner_folds"], cfg.get("quick", False)
    neural_cfg = cfg.get("neural", {})
    base_cols = list(cov) + schema.all_biomarker_columns()
    if task == "regression":
        mtr = tr2[schema.COGNITIVE_COL].notna().to_numpy()
        mte = te2[schema.COGNITIVE_COL].notna().to_numpy()
        ytr = make_targets(task, bag_tr[mtr], outcome=tr2[schema.COGNITIVE_COL][mtr])
    else:
        ok = lambda f: (f[schema.DEMENTIA_TIME_COL].notna() & f[schema.DEMENTIA_EVENT_COL].notna()).to_numpy()
        mtr, mte = ok(tr2), ok(te2)
        ytr = make_targets(task, bag_tr[mtr], event=tr2[schema.DEMENTIA_EVENT_COL][mtr],
                           time=tr2[schema.DEMENTIA_TIME_COL][mtr])
        ys_tr = make_survival_target(tr2[schema.DEMENTIA_EVENT_COL][mtr].astype(int),
                                     tr2[schema.DEMENTIA_TIME_COL][mtr])
        ys_te = make_survival_target(te2[schema.DEMENTIA_EVENT_COL][mte].astype(int),
                                     te2[schema.DEMENTIA_TIME_COL][mte])
    Xtr, Xte = tr2[mtr], te2[mte]

    def _metrics(pred):
        if task == "regression":
            return cognitive_metrics(Xte[schema.COGNITIVE_COL].to_numpy(float), pred)
        return {"C_index": concordance(ys_te["event"], ys_te["time"], pred)}

    rows = []
    for name in cfg["models"].get("rq3_framings", list(RQ3_FRAMINGS)):
        kind, uses_bag_hat, placebo = RQ3_FRAMINGS[name]
        cols = base_cols + [BAG_HAT_COL] if uses_bag_hat else base_cols
        key = ("rq3", task, name)
        if name in LABEL_FREE and key in cache:
            outs = cache[key]
        else:
            y = ytr.copy()
            if placebo:  # auxiliary task with BAG's distribution but no link to the person
                rng = np.random.default_rng(fseed)
                if task == "regression":
                    y[:, 1] = rng.permutation(y[:, 1])
                else:
                    y["bag"] = rng.permutation(y["bag"])
            res = tune_framing(name, task, Xtr, y, cols, inner, fseed, quick, neural_cfg)
            pipe = res["estimator"]
            out = _metrics(pipe.predict(Xte[cols]))
            if task == "survival":
                out["IBS"] = integrated_brier(ys_tr, ys_te, pipe, Xte[cols])
            Xt = pipe[:-1].transform(Xte[cols])
            if kind != "single" and not placebo:  # how well did the network learn BAG?
                out["bag_head_R2"] = float(r2_score(bag_te[mte], pipe[-1].predict_bag(Xt)))
            out["best_params"] = json.dumps(res["best_params"], default=str)
            outs = [(name, out)]
            if kind == "bottleneck":  # same fitted model, BAG scalar zeroed at test time
                abl = _metrics(pipe[-1].predict(Xt, ablate=True))
                outs.append(("bottleneck_ablated", {**abl, "best_params": out["best_params"]}))
            cache[key] = outs
        for fname, out in outs:
            rows.append({**vb, "framing": fname, "n_train": int(mtr.sum()), "n_test": int(mte.sum()), **out})
    return rows


def _assemble(rows, flow, cohort, oof_true, oof_pred, cfg, parts):
    """Turn per-fold rows into the summary and comparison tables."""
    alpha = cfg["evaluation"]["alpha"]
    thr = cfg["evaluation"]["bag_r2_threshold"]
    rep = cfg["reporting"]
    out = {"participant_flow": pd.DataFrame(flow, columns=["step", "n"])}

    if rows["bias"]:
        out["bias_correction"] = pd.DataFrame(rows["bias"])

    if "rq1" in parts:
        f = pd.DataFrame(rows["rq1"])
        out["rq1_folds"] = f
        s = summarize(f, ["variant", "feature_set", "model"], ["MAE", "RMSE", "R2"])
        s["meets_R2_threshold"] = s["R2_mean"] >= thr
        out["rq1_summary"] = s
        comps = []
        for g in schema.BIOMARKER_GROUPS:
            comps += compare(f, ["variant", "model"], "feature_set", PRIMARY_SET,
                             f"{PRIMARY_SET}-{g}", "R2", True, alpha)
        comps += compare(f, ["variant", "model"], "feature_set", PRIMARY_SET,
                         "covariates", "R2", True, alpha)
        out["rq1_comparisons"] = add_holm(pd.DataFrame(comps))

        if rows["shap"]:
            sh = pd.DataFrame(rows["shap"])
            out["rq1_shap"] = (sh.groupby(["variant", "model", "feature"])["mean_abs_shap"]
                               .agg(["mean", "std"]).reset_index()
                               .sort_values(["variant", "model", "mean"], ascending=[True, True, False]))

        sub, res = [], []
        attrs = cohort.drop(columns=[schema.ID_COL])
        for (variant, m), pred in oof_pred.items():
            pdf = attrs.assign(y_true=oof_true[variant], y_pred=pred).dropna(subset=["y_true", "y_pred"])
            pdf = pdf.reset_index(drop=True)
            tag = {"variant": variant, "model": m}
            sub += [{**tag, **row} for row in subgroup_metrics(
                pdf, rep.get("extra_subgroup_columns", []), rep.get("min_cell_size", 10))]
            res += [{**tag, **row} for row in residual_analysis(pdf)]
        if sub:
            out["rq1_subgroups"] = pd.DataFrame(sub)
            out["rq1_residuals"] = pd.DataFrame(res)

    if rows["stage1"]:
        s1 = pd.DataFrame(rows["stage1"])
        out["rq2_stage1"] = s1

    for part, metrics in (("rq2_cognitive", {"RMSE": False, "R2": True, "Pearson_r": True}),
                          ("rq2_dementia", {"C_index": True, "IBS": False})):
        if part not in parts or not rows[part]:
            continue
        f = pd.DataFrame(rows[part])
        out[f"{part}_folds"] = f
        out[f"{part}_summary"] = summarize(f, ["variant", "model", "condition"], list(metrics))
        comps = []
        for a, b in (("covariates+biomarkers+BAG", "covariates+biomarkers"),
                     ("covariates+biomarkers+BAG", "covariates+biomarkers+placebo"),
                     ("covariates+biomarkers", "covariates")):
            for metric, higher in metrics.items():
                if metric == "Pearson_r":
                    continue
                comps += compare(f, ["variant", "model"], "condition", a, b, metric, higher, alpha)
        out[f"{part}_comparisons"] = add_holm(pd.DataFrame(comps))

    for part, metrics in (("rq3_cognitive", {"RMSE": False, "R2": True}),
                          ("rq3_dementia", {"C_index": True, "IBS": False})):
        if part not in parts or not rows[part]:
            continue
        f = pd.DataFrame(rows[part])
        out[f"{part}_folds"] = f
        extra = [m for m in ("Pearson_r", "bag_head_R2") if m in f.columns]
        out[f"{part}_summary"] = summarize(f, ["variant", "framing"], list(metrics) + extra)
        present = set(f["framing"])
        pairs = [(n, "baseline") for n in RQ3_FRAMINGS if n != "baseline"]
        pairs += [("multitask", "multitask_placebo"), ("bottleneck", "bottleneck_ablated")]
        comps = []
        for a, b in pairs:
            if a in present and b in present:
                for metric, higher in metrics.items():
                    has = lambda lvl: f.loc[f["framing"] == lvl, metric].notna().any()
                    if metric in f.columns and has(a) and has(b):
                        comps += compare(f, ["variant"], "framing", a, b, metric, higher, alpha)
        out[f"{part}_comparisons"] = add_holm(pd.DataFrame(comps))
    return out
