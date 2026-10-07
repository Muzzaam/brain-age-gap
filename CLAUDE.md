# Brain Age Gap pipeline: notes for Claude

Honours project (Wits): estimate MRI-derived Brain Age Gap (BAG) from routine
biomarkers in ARIC-NCS, and test whether the estimate helps predict cognition
and incident dementia. `README.md` describes the design; `docs/` holds the
run guide and the ARIC notes.

## Data protection (overrides everything)

- ARIC data is controlled access. **Never open, print or summarise files in
  `data/raw/`, `dbn_out/`, `nifti_out/`, or any file with participant rows.**
  `.claude/settings.json` denies reading those folders; do not work around it
  (no `cat`, `head`, pandas prints of rows, etc.).
- Work from what the user gives you: column names, the ARIC data dictionary,
  error messages, and `preflight` output (aggregate only).
- Never hard-code participant ids or data values in the code.

## Adapting to how the real data is stored

All data-specific code lives in `src/data/loader.py` (`ARIC_COLUMN_MAP`,
`ARIC_VALUE_MAPS`, the derived-column block in `_load_aric`) and the `data:`
section of `config.yaml`. Downstream code must not change when the data source
changes. See `docs/ADAPTING_TO_THE_REAL_DATA.md` for the common cases.

## Rules that keep the results valid (do not break)

1. Data enters only through `load_data()` / `load_from_config()`; column names come from `src/data/schema.py`.
2. No leakage: preprocessing lives inside sklearn Pipelines; tuning and stage-1 cross-fitting happen inside each outer training fold; bias correction is fitted on training folds only.
3. Age, sex and site are covariates in every model and every RQ2 condition.
4. Saved outputs are aggregate only (`src/reporting.py` refuses participant ids; small counts suppressed).
5. Synthetic data validates code, never biology.

## Working here

- Run from this `code` folder. Windows: `.venv\Scripts\python.exe -m tests`
  (Linux: `.venv/bin/python -m tests`). Run the full suite after any change; it
  uses synthetic data only.
- On real data: `python -m scripts.preflight --source aric`, then
  `python -m scripts.run_all --source aric` (`--quick` for a smoke test).
- Explain changes in plain language; the user wants to understand the work.
- Match the surrounding code style. Commit in small, clearly described steps.
