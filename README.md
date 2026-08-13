# Brain Age Gap from Non-Invasive Biomarkers

Honours pipeline for estimating MRI-derived Brain Age Gap (BAG) from non-invasive
clinical biomarkers, and testing whether the estimate carries downstream value for
cognitive-risk modelling (ARIC-NCS cohort).

The real dataset is not available yet. This repo is built against a **synthetic,
ARIC-shaped** dataset so the whole pipeline can be developed and tested now. When
the real data lands, only one function changes — everything downstream runs
unchanged.

## The idea: one swap point

All data enters through `src/data/load_data()`. Today it returns synthetic data;
later you set `source="aric"` and fill in `loader._load_aric`. The column names,
groups, and target are fixed once in `src/data/schema.py` (the "data contract"),
and every other module imports from there. No model or metric code hard-codes a
column name, so swapping the source can't silently break anything downstream.

## Layout

```
src/
  data/
    schema.py       # THE DATA CONTRACT: column names, biomarker groups, ranges
    synthetic.py    # synthetic ARIC-shaped generator (known injected signal)
    loader.py       # load_data(): synthetic <-> real ARIC swap point
  preprocessing.py  # exclusions, train-only impute/scale, stratified splits
  models/           # (next) regularised linear, XGBoost, small MLP
  evaluation/       # (next) MAE/RMSE/R2, C-index/Brier, SHAP, subgroups
scripts/
  generate_data.py  # generate + save + sanity-check the synthetic cohort
tests/
  test_pipeline.py  # schema conformance, no-leakage, signal recoverable
config.yaml         # central knobs (data source, split, thresholds)
```

## Data contract (what the real ARIC loader must produce)

- Demographics: `age`, `sex`, `site`
- Physical function: `grip_strength`, `gait_speed`, `fev1`, `fvc`
- Cardiovascular: `sbp`, `dbp`, `resting_hr`
- Anthropometric: `bmi`, `waist_circumference`, `weight`
- Target: `bag` (= `brain_age` − `age`, where `brain_age` comes from the
  brain-age tool, e.g. BrainageR, run on the MRI scans)
- Outcomes: `cognitive_score`, `dementia_time`, `dementia_event`
- Exclusion flag: `prevalent_dementia`

## Synthetic generator

`synthetic.py` builds correlated biomarkers from a latent ageing factor, then
constructs ground-truth BAG from those biomarkers plus noise. `target_signal_fraction`
sets the fraction of BAG variance explained by biomarkers — roughly the **upper
bound** on the R² a perfect estimator could reach on clean data. Because you
injected the signal, the RQ1 feasibility experiment is self-validating: a correct
pipeline should recover close to that bound, minus what missingness/outliers cost.

Cognitive score and incident dementia are generated downstream of BAG and age, so
the RQ2 question ("does BAG add value beyond raw biomarkers?") has real structure
to detect.

## Run it

```bash
pip install -r requirements.txt
python -m scripts.generate_data      # writes data/synthetic/cohort.csv + sanity check
python -m pytest -q                  # or: python -m tests.test_pipeline
```

## When the real data arrives

1. Drop the file in `data/raw/` (git-ignored — never commit patient data).
2. Fill in `ARIC_COLUMN_MAP` and `_load_aric` in `src/data/loader.py`.
3. Set `source: aric` in `config.yaml`.
4. Run BrainageR (or the chosen tool) on the MRI scans to get `brain_age`, then
   `bag = brain_age - age`.

## Next steps

- `src/models/`: XGBoost (the baseline to beat), ridge/elastic net, small MLP.
- `src/evaluation/`: BAG metrics + pre-registered R² ≥ 0.20 check, then the
  four-condition RQ2 comparison, then SHAP / subgroup / bias-correction analyses.
