# Imaging stage — producing brain-age labels

This folder is the MRI → brain-age step. It's separate from the modelling
pipeline (`src/`) on purpose: it runs in a different environment and hands off
just one thing — a CSV of `participant_id, brain_age`. From there the pipeline
computes `bag = brain_age - age` and never touches the imaging stack again.

Two independent runners, one per tool:

| Script | Tool | Environment | Uses GPU? |
|---|---|---|---|
| `run_brainager_batch.py` | brainageR | WSL (needs SPM/MATLAB/R/FSL) | no (CPU-bound) |
| `run_deepbrainnet.py` | DeepBrainNet (ANTsPyNet) | plain Python | yes (your 3080) |

Both write the **same** output schema, so either can feed the pipeline, and you
can run both and compare them (the labelling-model robustness check).

## brainageR runner

Run this inside WSL, where brainageR already works. `brainageR` must be on your
PATH and configured. It processes scans in parallel and is resumable — if the
batch is interrupted, just run the exact same command again and it continues.

```bash
python run_brainager_batch.py \
    --input-dir /mnt/d/aric/t1 \
    --output-dir ./brainager_out \
    --workers 4
```

`--workers`: your Ryzen 5 3600 is 6c/12t, but each brainageR (SPM) process is
RAM-heavy. Start at 4, watch memory, go to 6 only with headroom. Too many is
slower, not faster.

Output: `brainager_out/brainager_results.csv` with columns
`participant_id, brain_age, status, seconds`. Per-scan raw output lands in
`brainager_out/perscan/`.

> One thing to confirm against your brainageR version: the column name it uses
> for predicted age. The script tries the common names and falls back to the
> first numeric value; if a scan comes back `done_unparsed`, open its file in
> `perscan/` and add the real column name to `AGE_COLUMN_CANDIDATES` at the top
> of the script.

## DeepBrainNet runner

Pure Python — no MATLAB/R. Install once (the first run also downloads the
pretrained weights, so needs internet):

```bash
pip install antspyx antspynet tensorflow
```

Verify on a single scan first:

```bash
python run_deepbrainnet.py --input /path/to/one_T1.nii.gz
```

If that prints a sensible age, batch a folder (also resumable):

```bash
python run_deepbrainnet.py --input /mnt/d/aric/t1 --output-dir ./dbn_out
```

Output: `dbn_out/deepbrainnet_results.csv`, same schema as above.

## Handing off to the pipeline

Both runners give you `participant_id, brain_age`. In the pipeline's real-data
loader you merge that onto the biomarker table by `participant_id` and compute
`bag = brain_age - age`. That `bag` column is what the synthetic generator has
been standing in for all along.

## Before a big local run

ARIC is controlled-access data. Confirm with your supervisor that the data-use
terms permit the raw scans on your own machine before copying ~2,000 scans
locally — this may be constrained, and is worth settling first. Also budget disk:
SPM leaves large intermediates, so the full set can be tens to a couple hundred
GB.
