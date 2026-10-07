# Adapting the code to how the real data is stored

The real ARIC files will not look exactly like the synthetic data. This guide
shows where to make changes, covers the likely cases, and explains how to get
help from Claude (Claude Code or claude.ai) **without sharing participant data**.

## The one rule

**Data-specific changes go in two places only:**

- `src/data/loader.py`: `ARIC_COLUMN_MAP`, `ARIC_VALUE_MAPS`, and the
  derived-column block in `_load_aric` (marked TODO)
- the `data:` section of `config.yaml`

Everything downstream (models, statistics, reporting) reads the standard
column names from `src/data/schema.py` and must not change. If a fix seems to
need edits anywhere else, stop and ask why. The one exception is case 9 below.

After every change:

```
python -m tests                         # still passes (uses synthetic data, proves nothing broke)
python -m scripts.preflight --source aric   # re-check the real data
```

## Common cases

**1. Column names differ** (they will). Map each ARIC name to ours:

```python
ARIC_COLUMN_MAP = {
    "ID_C": "participant_id",     # the id column, in every file
    "GRIP_MAX": "grip_strength",
    "SBP_AVG": "sbp",
    ...
}
```

The full list of names to map to is `schema.REQUIRED_COLUMNS` (preflight prints
any that are missing).

**1b. Variables exist once per visit** (as in UK Biobank's "Instance 0/1/2/3" columns).
Map the **visit 5** version of every variable: age, biomarkers and covariates all
come from the same visit as the MRI. The preflight warns if the median age looks
like an earlier visit.

**2. The data comes as several files** (likely: ARIC delivers one file per form
or visit). List them in `config.yaml`; they are joined on participant id. The
first file must have one row per participant:

```yaml
tabular_path: [data/raw/v5_exam.sas7bdat, data/raw/ncs_cognition.sas7bdat, data/raw/dementia.sas7bdat]
```

If a file has several rows per participant (one per visit), it needs filtering
to the visit 5 row first. Ask Claude to add that filter in `_read_tabular`.

**3. Values are coded differently.** Recode to ours:

```python
ARIC_VALUE_MAPS = {
    "sex": {1: "M", 2: "F"},
    "site": {"F": "Forsyth", "J": "Jackson", "M": "Minneapolis", "W": "Washington"},
}
```

**4. A column has to be computed.** Add lines in the TODO block of
`_load_aric` (after renaming, before the brain-age join). Typical examples,
with made-up ARIC names:

```python
df["gait_speed"] = 4 / df["WALK4M_SEC"]                      # stored as a 4 m walk time
df["grip_strength"] = df[["GRIP_T1", "GRIP_T2"]].max(axis=1)  # best of two trials
df["prevalent_dementia"] = (df["V5_COGDX"] == 3).astype(int)  # adjudicated status at visit 5
v5 = pd.to_datetime(df["V5_DATE"])
end = pd.to_datetime(df["DEM_DATE"]).fillna(pd.to_datetime(df["CENSOR_DATE"]))
df["dementia_time"] = (end - v5).dt.days / 365.25
df["dementia_event"] = df["DEM_DATE"].notna().astype(int)
df["healthy_reference"] = (df["V5_COGDX"] == 1).astype(int)   # cognitively normal
```

**5. Units differ** (for example FEV1 in mL, not litres). Convert in the same
block: `df["fev1"] = df["fev1"] / 1000`. The preflight's implausible-value
counts will flag a unit problem (almost every value out of range).

**6. Scan filenames are not ARIC ids.** Either pull the id out of the name with
`--id-regex` in the imaging runner, or give a mapping file
(`scan_id, participant_id`) in `data.aric.scan_id_map_path`. See
`docs/RUNNING_AT_WITS.md`, step 4.

**7. Scans are DICOM, not NIfTI.** Convert with `dcm2niix` (run guide, step 4a).

**8. The cognitive score is on a different scale** (for example a z-score).
Nothing to change: map it to `cognitive_score` and the models handle any scale.

**9. A biomarker does not exist in ARIC** (or is unusable). This is the one
change outside the loader: remove it from `BIOMARKER_GROUPS` in
`src/data/schema.py` (and its entry in `PLAUSIBLE_RANGES`). Note it for the
Methods.

## Getting help from Claude

### What may and may not be shared

| Fine to share | Never share |
|---|---|
| Column names and descriptions (the ARIC data dictionary is public) | Any row of data, even one |
| Error messages and tracebacks (check they show no data values) | Participant ids |
| `preflight.txt` and `summary.txt` (aggregate only) | Screenshots of the data |
| The code files | `brain_age.csv`, the scans, the extract |

### Option A: Claude Code on the secure machine

If Claude Code can be installed there, open it in the `code` folder. It reads
`CLAUDE.md` for the project context, and `.claude/settings.json` stops it
reading the raw-data folders. That block is a safeguard, not a guarantee, so
still never ask it to "look at the data". Give it column names and preflight
output instead.

Changes it makes are code only, so they can be committed and pushed (check no
participant id ever ends up in the code).

### Option B: claude.ai in the browser

The website cannot see the machine, so paste the context in. Copy this
template, fill in the bracketed parts, and attach the files:

```text
I'm running my Honours brain-age pipeline on real ARIC-NCS data on a secure
machine. All data-specific code is in src/data/loader.py (ARIC_COLUMN_MAP,
ARIC_VALUE_MAPS, and the derived-column block in _load_aric) plus the data:
section of config.yaml. Downstream code must not change. The column names the
pipeline needs are in src/data/schema.py (REQUIRED_COLUMNS).

Attached: loader.py, schema.py, config.yaml, and preflight.txt.

The ARIC files are: [file names and formats]
Their relevant columns are (from the data dictionary, no data values):
[column name: description] ...

Problem: [what preflight or run_all says, or what you need to derive]

Please give me the exact edits to loader.py / config.yaml, and explain them.
```

Then apply the edits, rerun `python -m tests` and the preflight, and repeat
until the preflight shows 0 errors.

## Keep a record

Write down every data decision and change (date, what, why) as you go. It
becomes part of the Methods (Sections 2.3 to 2.5), and examiners will ask.
