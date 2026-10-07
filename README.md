# Brain Age Gap from Non-Invasive Biomarkers

Honours pipeline for estimating MRI-derived Brain Age Gap (BAG) from non-invasive
clinical biomarkers, and testing whether the estimate carries downstream value for
cognitive-risk modelling (ARIC-NCS cohort).

The real dataset is not available yet. The repo is built against a **synthetic,
ARIC-shaped** dataset so the whole pipeline can be developed and tested now.
Synthetic results validate the code, never the biology.

## Run it

Run everything from this `code` folder. On Windows, call the venv's Python
directly (`.venv\Scripts\python.exe -m ...`).

```bash
pip install -r requirements.txt
python -m tests                      # full test suite (~1 minute)
python -m scripts.run_all --quick    # smoke test of every analysis (~30 s)
python -m scripts.run_all            # the real thing (see run time below)
python -m scripts.run_all --only rq1 # a subset: rq1, rq2_cognitive, rq2_dementia,
                                     #           rq3_cognitive, rq3_dementia
```

Each run writes `results/<timestamp>_<source>/`: CSV tables, `summary.txt`,
`run_info.json` (config, package versions, git commit) and `run.log`. These hold
**aggregate results only** (no participant ids, small counts suppressed), so they
are what you carry away from the secure machine, subject to the data-use agreement.

## When the real data arrives

1. Put the ARIC-NCS extract and the brain-age CSV (`participant_id, brain_age`,
   from the imaging step) in `data/raw/` (git-ignored; never commit patient data),
   and point `data.aric` in `config.yaml` at them.
2. Fill in `ARIC_COLUMN_MAP` / `ARIC_VALUE_MAPS` and the TODOs in `src/data/loader.py`.
3. Set `data.source: aric` in `config.yaml`.
4. `python -m scripts.preflight` and fix every ERROR it reports.
5. `python -m scripts.run_all`.

## Design

**One swap point.** Data enters only through `load_data()` / `load_from_config()`.
Column names live once in `src/data/schema.py`; nothing else hard-codes them.

**Covariates everywhere.** Age, sex and site (config `features.covariates`) are
inputs to every model. RQ1 reports the R2 of covariates alone and the incremental
R2 the biomarkers add, so the brain-age model's own age bias can't masquerade as
biomarker signal. In RQ2 every condition includes them, so estimated BAG can't
look useful merely by standing in for age.

**Nested cross-validation** (`src/experiment.py`). Outer: repeated stratified
K-fold (age band x sex x dementia event); every number is a mean +/- SD over outer
folds. Inner: hyperparameter tuning for every model family and stage-1
cross-fitting. Imputation, scaling and one-hot encoding live inside each model's
sklearn Pipeline, so they are refitted on every training split, inner ones included.

**Cross-fitted stage 1.** Training-set BAG estimates for stage 2 are out-of-fold,
so they are as noisy as test-set estimates (`src/models/two_stage.py`).

**Four RQ2 conditions:** covariates; covariates + biomarkers; + estimated BAG;
+ placebo (BAG shuffled: same distribution, signal destroyed).

**RQ3 framings** (`src/models/neural.py`, `framings.py`). One PyTorch network and
training routine for every framing, so only the role of BAG differs: `baseline`
(no BAG, equivalent capacity), `two_stage` (estimated BAG as an input),
`multitask` (auxiliary BAG head, loss weight tuned), `multitask_placebo`
(auxiliary head on shuffled BAG), `bottleneck` (the BAG scalar feeds the outcome
head; `bottleneck_ablated` zeroes it at test time), `transfer` (pretrain on BAG,
fine-tune on the outcome). Both outcomes: MSE for cognition, Cox partial
likelihood for dementia.

**Bias correction is a sensitivity analysis.** Every analysis runs on uncorrected
and corrected BAG (linear age-residualisation fitted on the training fold's
healthy-reference subset) and both are reported.

**Statistics.** Differences between conditions use the corrected resampled t-test
(Nadeau & Bengio, 2003) on per-fold differences, with mean improvement and 95% CI.

## Layout

```
config.yaml           every setting: data source, covariates, CV, models, reporting
src/
  config.py           loads config.yaml (+ --quick overrides)
  data/
    schema.py         the data contract: columns, biomarker groups, plausible ranges
    synthetic.py      synthetic ARIC-shaped cohort with known injected signal and realistic mess
    loader.py         load_data(): synthetic <-> real ARIC swap point; joins brain-age CSV
  preprocessing.py    masking, exclusions + participant flow, folds, per-fold preprocessor
  models/
    estimators.py     ridge, elastic net, XGBoost, MLP pipelines + tuning grids
    survival.py       Cox and gradient-boosted survival pipelines + grids
    tuning.py         inner-CV grid search
    two_stage.py      RQ2 conditions, stage-1 cross-fitting, placebo
    neural.py         RQ3 PyTorch network + sklearn-compatible estimator, Cox loss
    framings.py       RQ3 framing registry and tuning grids
  evaluation/         metrics, stats (corrected t-test), bias correction, SHAP/subgroups, residuals
  experiment.py       the nested-CV engine running RQ1, RQ2 and RQ3
  reporting.py        writes aggregate tables + summary
scripts/
  run_all.py          run everything, save results
  preflight.py        data checks to run first on the real data
  generate_data.py    save + describe the synthetic cohort
tests/                python -m tests runs all suites
imaging/              brain-age labelling runners (run in WSL)
docs/                 notes, e.g. aric_dataset_notes.md
```
