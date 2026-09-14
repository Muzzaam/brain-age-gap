"""
Estimate brain age with DeepBrainNet (via ANTsPyNet) — the Python-native route.

Unlike brainageR, this is pure Python (TensorFlow + ANTs), needs no MATLAB/R,
and uses your GPU automatically if TensorFlow sees one (your RTX 3080 will
accelerate it). ANTsPyNet's brain_age() does its own preprocessing internally
(N4 bias correction, brain extraction, affine registration to MNI), so you can
feed it a raw T1.

Two modes, chosen by what you point --input at:
  * a single .nii/.nii.gz  -> prints the predicted age (quick verification)
  * a folder of scans      -> resumable batch, aggregated to a master CSV

The master CSV uses the SAME schema as run_brainager_batch.py
(participant_id, brain_age, status), so both tools feed the pipeline
identically: bag = brain_age - age.

Examples
--------
    # verify it works on one scan
    python run_deepbrainnet.py --input /path/to/one_T1.nii.gz

    # batch a whole folder
    python run_deepbrainnet.py --input /mnt/d/aric/t1 --output-dir ./dbn_out
"""

import argparse
import csv
import sys
import time
from pathlib import Path


def subject_id(scan_path):
    name = Path(scan_path).name
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return Path(scan_path).stem


def load_done(manifest_path):
    done = set()
    if manifest_path.exists():
        with open(manifest_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "ok":
                    done.add(row["participant_id"])
    return done


def predict_one(scan_path, ants, antspynet, verbose=False):
    """Return predicted brain age (float) for a single T1 scan."""
    t1 = ants.image_read(str(scan_path))
    result = antspynet.brain_age(t1, do_preprocessing=True, verbose=verbose)
    return float(result["predicted_age"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="a single T1 scan, or a folder of them")
    ap.add_argument("--pattern", default="*.nii*", help="glob when --input is a folder")
    ap.add_argument("--output-dir", default="./dbn_out", help="where batch results go")
    args = ap.parse_args()

    # Imported here so --help works without the heavy deps installed.
    try:
        import ants
        import antspynet
    except ImportError:
        sys.exit(
            "Missing deps. Install with:\n"
            "    pip install antspyx antspynet tensorflow\n"
            "(first run also downloads the pretrained weights — needs internet)."
        )

    in_path = Path(args.input)

    # --- single-scan mode: quick verification -------------------------------
    if in_path.is_file():
        print(f"Predicting brain age for {in_path.name} ...")
        start = time.time()
        age = predict_one(in_path, ants, antspynet, verbose=True)
        print(f"\nPredicted brain age: {age:.2f} years  ({time.time()-start:.0f}s)")
        print("If that looks sensible, the DeepBrainNet route works.")
        return

    # --- batch mode ---------------------------------------------------------
    if not in_path.is_dir():
        sys.exit(f"--input is neither a file nor a folder: {in_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "deepbrainnet_results.csv"

    scans = sorted(p for p in in_path.glob(args.pattern) if p.is_file())
    if not scans:
        sys.exit(f"No scans matching {args.pattern!r} in {in_path}")

    done = load_done(manifest)
    todo = [s for s in scans if subject_id(s) not in done]
    print(f"Found {len(scans)} scans; {len(done)} already done; {len(todo)} to process.\n")

    write_header = not manifest.exists()
    ok = fail = 0
    with open(manifest, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["participant_id", "brain_age", "status", "seconds"])
        if write_header:
            w.writeheader()
        for i, scan in enumerate(todo, 1):
            pid = subject_id(scan)
            start = time.time()
            try:
                age = predict_one(scan, ants, antspynet)
                rec = {"participant_id": pid, "brain_age": round(age, 2),
                       "status": "ok", "seconds": round(time.time() - start, 1)}
                ok += 1
            except Exception as e:  # noqa: BLE001 - log and continue the batch
                rec = {"participant_id": pid, "brain_age": "",
                       "status": f"failed: {type(e).__name__}",
                       "seconds": round(time.time() - start, 1)}
                fail += 1
            w.writerow(rec)
            f.flush()  # persist after every scan so a crash loses nothing
            print(f"[{i}/{len(todo)}] {pid}: {rec['status']}"
                  + (f"  age={rec['brain_age']}" if rec["status"] == "ok" else "")
                  + f"  ({rec['seconds']}s)")

    print(f"\nDone this run: {ok} ok, {fail} failed.")
    print(f"Master results: {manifest}")
    if fail:
        print("Re-run the same command to retry only the unfinished scans.")


if __name__ == "__main__":
    main()
