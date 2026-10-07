# Running the pipeline on the secure machine

Step-by-step instructions for running everything where the ARIC data must stay
(a Wits machine or server). Follow the steps in order: each one has a check that
must pass before the next. Expect one day for setup and imaging, then about
30 minutes per analysis run.

**The rule that overrides everything:** participant-level data (scans, the ARIC
extract, `brain_age.csv`, anything with `participant_id` in it) never leaves the
secure machine. Only the `results/<timestamp>_aric/` folder from step 9 comes
home, and only if the data-use agreement allows it.

---

## 0. Before you go (questions for Devon / IT)

- [ ] Which machine: Windows or Linux? How many CPU cores and how much RAM? A GPU?
- [ ] Can you install Python packages there (pip)? Is there internet access?
- [ ] Where are the scans and the ARIC extract stored, and in what format
      (DICOM folders or NIfTI files; CSV, SAS or Stata)?
- [ ] Are scan files named by ARIC participant id, or by a separate imaging id?
      If separate, where is the file that maps one to the other?
- [ ] What does the data-use agreement say about copying aggregate results off the machine?

## 1. Get the code onto the machine

With internet: `git clone https://github.com/Muzzaam/brain-age-gap.git`

Without internet: on your own laptop, download the repository as a zip from
GitHub (code only, no data) and copy it across.

All commands below are run from the `code` folder.

## 2. Set up the two Python environments

The analysis and the imaging step use separate environments because TensorFlow
(for DeepBrainNet) and the analysis libraries conflict.

**Analysis environment** (Python 3.11):

```
python -m venv .venv
# Windows:  .venv\Scripts\python.exe -m pip install -r requirements.txt
# Linux:    .venv/bin/python -m pip install -r requirements.txt
```

PyTorch: if the plain install pulls a large GPU build, use
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

**Imaging environment** (DeepBrainNet):

```
python -m venv dbn-venv
dbn-venv/bin/python -m pip install antspyx antspynet tensorflow "tf-keras~=2.16.0" "numpy<2"
export TF_USE_LEGACY_KERAS=1        # Windows PowerShell: $env:TF_USE_LEGACY_KERAS = "1"
```

DeepBrainNet downloads its weights and a template the first time it runs
(into the Keras cache, normally `~/.keras/ANTsXNet`). **No internet on the
secure machine?** Run one scan on your own laptop first, then copy that cache
folder to the same place on the secure machine.

> Below, `python` means the analysis environment's Python and `dbn-python` the
> imaging environment's. On Windows that is `.venv\Scripts\python.exe`.

**Check:** `python -m tests` prints `ALL PASSED`. This uses no real data and
proves the environment works.

## 3. Put the data in place

```
code/data/raw/
  aric_ncs.csv          the ARIC extract (any of csv, sas7bdat, dta, sav, parquet)
  brain_age.csv         written in step 4
  scan_id_map.csv       only if scan names are not ARIC ids (columns: scan_id, participant_id)
```

`data/raw/` is git-ignored, so nothing there can be committed by accident.
Point `config.yaml` at the files (`data.aric.tabular_path`, `brain_age_path`,
`scan_id_map_path`) if you use other names or locations.

## 4. Imaging: brain age from the T1 scans

**4a. Get one T1 NIfTI per participant.** If the scans are DICOM folders,
convert them with [dcm2niix](https://github.com/rordenlab/dcm2niix):

```
dcm2niix -z y -f %i_%p -o nifti_out/ dicom_in/
```

(`%i` puts the scanner's patient id in the filename and `%p` the protocol name).
Keep only the T1 MPRAGE series, one per participant. If a participant has two
T1s (a rescan), keep the one the ARIC imaging documentation marks as usable.

**4b. Test on one scan:**

```
dbn-python imaging/run_deepbrainnet.py --input nifti_out/<one scan>.nii.gz
```

**Check:** it prints a predicted age somewhere in the 50 to 100 range for a 70 to 90 year old.

**4c. Run the batch:**

```
dbn-python imaging/run_deepbrainnet.py --input nifti_out/ --output-dir dbn_out/
```

If filenames hold the id inside other text (for example `scan_A123456_T1.nii.gz`),
add `--id-regex "(A\d{6})"` with a pattern matching your ids. The runner refuses
to start if two files map to the same id.

- Time the first 10 scans and multiply up to estimate the total. Let it run unattended.
- It is **resumable**: if it stops (reboot, logout), run the same command again
  and it continues where it left off.
- When it finishes, rerun the same command once to retry any failures.

**4d.** Copy `dbn_out/deepbrainnet_results.csv` to `data/raw/brain_age.csv`.

**Check:** open the CSV. Most rows should say `ok`. Note how many failed (they
count as "no usable brain age" in the participant flow).

## 5. Map the ARIC columns onto the pipeline

Open the ARIC data dictionary next to `src/data/loader.py` and fill in the items
below. `docs/ADAPTING_TO_THE_REAL_DATA.md` has worked examples for each case
(several files, coded values, dates, units) and a template for asking Claude.


1. `ARIC_COLUMN_MAP`: each ARIC variable name to our name (`grip_strength`,
   `sbp`, `cognitive_score` and so on; the full list is in `src/data/schema.py`).
2. `ARIC_VALUE_MAPS`: recodings, for example sex `1`/`2` to `M`/`F`, centre codes to site names.
3. The TODO block in `_load_aric`: derive columns ARIC does not store directly
   (`dementia_time` from dates, `prevalent_dementia`, `healthy_reference`,
   gait speed from walk time).

Make the decisions listed in `docs/aric_dataset_notes.md` (spirometry
missingness, grip trials, reference group, dementia certainty levels) and
**write each one down with the date**. They go straight into the Methods.

## 6. Preflight: check the data before analysing it

```
python -m scripts.preflight --source aric
```

It loads the data exactly as the analysis will and reports problems. Fix every
`ERROR`, read every `WARNING`, and rerun until it says `0 error(s)`.

What it checks, and what to do:

| Check | Looks like | If it fails |
|---|---|---|
| Required columns present | `required columns missing: [...]` | add the mapping in `ARIC_COLUMN_MAP` |
| Codings | `sex must be coded 'M'/'F'` | add a recoding in `ARIC_VALUE_MAPS` |
| Scan linkage | `Scan linkage: scans usable 1950, scans matched to participant 1948, ...` | unmatched scans: fix `--id-regex` or the id map |
| Brain age tracks age | `Predicted brain age vs chronological age: r = +0.45` | r below 0.1 means scans are attached to the **wrong people**: stop and fix the linkage |
| BAG distribution | `BAG: mean +3.10, SD 6.20 years` | a mean far from 0 is common (model offset) and is what bias correction handles; an SD under 1 or over 20 means a units or join problem |
| Missingness | `fev1 is 41% missing` | make the spirometry decision (step 5) |
| Events | `only 38 incident dementia events` | the survival analysis will be underpowered; report it as planned |
| Participant flow | counts after each exclusion | this becomes Figure 1 of the report |

**Extra linkage spot-check (do this once):** pick five participants, find their
scan files by hand, and confirm the ids in `brain_age.csv` and the ARIC extract
refer to the same people (same sex and a sensible age where visible in the
scan header). A wrong join silently ruins every result; the correlation check
catches a total scramble, not a partial one.

## 7. Smoke test on the real data (minutes)

```
python -m scripts.run_all --source aric --quick
```

Tiny settings, so it finishes in a few minutes. It proves the real data flows
through every analysis. **The numbers mean nothing**; do not report them.

## 8. The real run (about 30 minutes)

Set `data.source: aric` in `config.yaml` (or keep passing `--source aric`), then:

```
python -m scripts.run_all --source aric
```

It prints progress per fold. Do not close the session while it runs: on Linux
use `nohup python -m scripts.run_all --source aric &` or `tmux`; on Windows,
leave the terminal open and stop the machine from sleeping.

To rerun one part after a fix: `--only rq1` (or `rq2_cognitive`,
`rq2_dementia`, `rq3_cognitive`, `rq3_dementia`).

## 9. What comes home

Everything you need is in `results/<timestamp>_aric/`:

| File | What it is |
|---|---|
| `summary.txt` | readable overview of every result |
| `participant_flow.csv` | counts for the flow diagram |
| `rq1_*.csv`, `rq2_*.csv`, `rq3_*.csv` | per-fold results, summaries, comparisons, SHAP, subgroups |
| `bias_correction.csv` | the bias-correction fit per fold |
| `run_info.json` | config, package versions, git commit (for reproducibility) |
| `run.log` | the console output |

These files contain no participant ids or rows, and counts under 10 are shown as
`<10`. Before copying them:

- [ ] Open `summary.txt` and check that it looks sensible.
- [ ] Confirm the data-use agreement allows taking aggregate results off the machine.
- [ ] Copy **only** this folder. Never `data/raw/`, `dbn_out/` or `nifti_out/`.

Figures for the report are then made from these CSVs on your own laptop.

## Troubleshooting

| Problem | Fix |
|---|---|
| `python -m tests` fails after install | check the Python version (3.11) and reinstall from `requirements.txt` |
| DeepBrainNet: `TF_USE_LEGACY_KERAS` / Keras errors | set the environment variable (step 2) in the same terminal |
| DeepBrainNet tries to download and fails | copy the Keras cache from your laptop (step 2) |
| `ValueError: More than one usable brain age` | two scans per participant: keep one (step 4a) |
| `FileNotFoundError: ARIC tabular file not found` | fix the paths in `config.yaml` |
| A model crashes mid-run | rerun with `--only` for the parts that did not finish; send the error text (no data) |
| Run is much slower than 30 minutes | fewer cores than expected; it will finish, just later |

## One-page checklist

1. [ ] Environment set up; `python -m tests` passes
2. [ ] Data in `data/raw/`; paths in `config.yaml`
3. [ ] DeepBrainNet works on one scan; batch finished; failures retried
4. [ ] `brain_age.csv` in place; id format handled
5. [ ] Column mapping and derived columns done; decisions written down
6. [ ] `preflight` shows 0 errors; brain age correlates with age; five-participant spot-check done
7. [ ] `run_all --quick` completes
8. [ ] `run_all` completes
9. [ ] Results folder checked and copied out (aggregate only)
