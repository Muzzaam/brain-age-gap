"""
Run brainageR over a folder of T1 scans, in parallel, on your own machine.

brainageR processes one scan per invocation and its heavy step (SPM
segmentation/normalisation) is CPU-bound, so the way to go faster is to run
several scans at once across your cores. This script does that, and is built
for a long unattended batch:

  * parallel across N workers (subprocess per scan)
  * RESUMABLE: re-running skips scans already done, so an interrupted batch
    (reboot, crash, closing the lid) just picks up where it left off
  * every scan's status is logged; failures are collected for easy retry
  * results are aggregated into one master CSV keyed by participant_id

Run this INSIDE WSL, where your brainageR install works.

Example
-------
    python run_brainager_batch.py \
        --input-dir /mnt/d/aric/t1 \
        --output-dir ./brainager_out \
        --workers 4

Notes on --workers: your Ryzen 5 3600 has 6 cores / 12 threads, but each
brainageR (SPM/MATLAB) process is memory-hungry. Start at 4 and watch RAM;
bump to 6 only if you have headroom. Too many workers will thrash and be
SLOWER than fewer.
"""

import argparse
import csv
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# brainageR writes a small CSV; we try these columns for the predicted age.
# Confirm the exact name against your brainageR version's output and add it
# here if it differs — everything else keeps working regardless.
AGE_COLUMN_CANDIDATES = [
    "brain.predicted.age", "brain_predicted_age", "predicted_age",
    "brainageR", "age.prediction", "pred", "brainpad",
]

_manifest_lock = threading.Lock()


def discover_scans(input_dir, pattern):
    scans = sorted(Path(input_dir).glob(pattern))
    return [s for s in scans if s.is_file()]


def subject_id(scan_path):
    """Filename with .nii / .nii.gz stripped -> participant id."""
    name = scan_path.name
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return scan_path.stem


def load_done(manifest_path):
    """Set of participant ids already completed successfully."""
    done = set()
    if manifest_path.exists():
        with open(manifest_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "ok":
                    done.add(row["participant_id"])
    return done


def parse_predicted_age(csv_path):
    """Pull the predicted brain age out of brainageR's per-scan output CSV."""
    try:
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None
        row = rows[0]
        for col in AGE_COLUMN_CANDIDATES:
            if col in row and row[col] not in ("", None):
                return float(row[col])
        # column name unknown: grab the first numeric value we can find
        for v in row.values():
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    except (OSError, ValueError):
        return None
    return None


def run_one(scan, cmd_template, perscan_dir):
    """Run brainageR on a single scan. Returns a result record dict."""
    pid = subject_id(scan)
    out_csv = perscan_dir / f"{pid}.csv"
    cmd = [
        part.format(input=str(scan), output=str(out_csv))
        for part in cmd_template
    ]
    start = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        return {"participant_id": pid, "brain_age": "", "status": "cmd_not_found",
                "seconds": 0, "detail": cmd[0]}
    dur = round(time.time() - start, 1)

    if proc.returncode != 0:
        return {"participant_id": pid, "brain_age": "", "status": "failed",
                "seconds": dur, "detail": proc.stderr.strip().splitlines()[-1:] or ""}

    age = parse_predicted_age(out_csv)
    if age is None:
        return {"participant_id": pid, "brain_age": "", "status": "done_unparsed",
                "seconds": dur, "detail": str(out_csv)}
    return {"participant_id": pid, "brain_age": age, "status": "ok",
            "seconds": dur, "detail": ""}


def append_manifest(manifest_path, record, write_header):
    with _manifest_lock:
        with open(manifest_path, "a", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["participant_id", "brain_age", "status", "seconds", "detail"]
            )
            if write_header:
                w.writeheader()
            w.writerow(record)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, help="folder of T1 .nii/.nii.gz scans")
    ap.add_argument("--pattern", default="*.nii*", help="glob for scans (default *.nii*)")
    ap.add_argument("--output-dir", required=True, help="where results are written")
    ap.add_argument("--workers", type=int, default=4, help="parallel scans (default 4)")
    ap.add_argument(
        "--brainager-cmd", default="brainageR -f {input} -o {output}",
        help="command template; {input} and {output} are filled per scan",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    perscan_dir = out_dir / "perscan"
    perscan_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "brainager_results.csv"

    scans = discover_scans(args.input_dir, args.pattern)
    if not scans:
        sys.exit(f"No scans matching {args.pattern!r} in {args.input_dir}")

    done = load_done(manifest)
    todo = [s for s in scans if subject_id(s) not in done]
    cmd_template = args.brainager_cmd.split()

    print(f"Found {len(scans)} scans; {len(done)} already done; {len(todo)} to process.")
    print(f"Running {args.workers} at a time. Writing to {manifest}\n")

    header_needed = not manifest.exists()
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_one, s, cmd_template, perscan_dir): s for s in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            append_manifest(manifest, rec, header_needed)
            header_needed = False
            if rec["status"] == "ok":
                ok += 1
            else:
                fail += 1
            print(f"[{i}/{len(todo)}] {rec['participant_id']}: {rec['status']}"
                  + (f"  age={rec['brain_age']}" if rec["status"] == "ok" else "")
                  + f"  ({rec['seconds']}s)")

    print(f"\nDone this run: {ok} ok, {fail} failed/unparsed.")
    print(f"Master results: {manifest}")
    if fail:
        print("Re-run the same command to retry only the failed/unfinished scans.")


if __name__ == "__main__":
    main()
