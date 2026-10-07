"""
Run every analysis end to end and save the aggregate results.

    python -m scripts.run_all                 # everything, settings from config.yaml
    python -m scripts.run_all --quick         # seconds-long smoke test (tiny grids, 2 folds)
    python -m scripts.run_all --only rq1      # just RQ1 (also: rq2_cognitive, rq2_dementia)
    python -m scripts.run_all --source aric   # override data.source from config.yaml

Results go to results/<timestamp>_<source>/ (see src/reporting.py). On the real
data, run scripts.preflight first.
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data import load_from_config
from src.experiment import run_experiment, ALL_PARTS
from src.reporting import new_run_dir, write_tables, write_run_info, summary_text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument("--source", choices=["synthetic", "aric"], help="override data.source")
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test settings (not for results)")
    ap.add_argument("--only", nargs="+", choices=ALL_PARTS, help="run a subset of analyses")
    ap.add_argument("--tag", help="suffix for the results folder name")
    ap.add_argument("--results-dir", help="override reporting.results_dir")
    args = ap.parse_args(argv)

    overrides = {}
    if args.source:
        overrides["data"] = {"source": args.source}
    if args.results_dir:
        overrides["reporting"] = {"results_dir": args.results_dir}
    cfg = load_config(args.config, quick=args.quick, overrides=overrides or None)
    tag = args.tag or (cfg["data"]["source"] + ("_quick" if args.quick else ""))
    run_dir = new_run_dir(cfg, tag)
    log_file = open(run_dir / "run.log", "w", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    # convergence chatter from inner-loop fits would bury the progress log
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*converge.*")

    log(f"Results folder: {run_dir}")
    t0 = time.time()
    df = load_from_config(cfg)
    log(f"Loaded {len(df)} participants from source '{cfg['data']['source']}'.")
    results = run_experiment(df, cfg, parts=args.only or ALL_PARTS, log=log)

    files = write_tables(results, run_dir, cfg)
    write_run_info(run_dir, cfg, results["_meta"])
    text = summary_text(results, cfg)
    (run_dir / "summary.txt").write_text(text, encoding="utf-8")
    log("")
    log(text)
    log("")
    log(f"Done in {time.time() - t0:.0f}s. Wrote: {', '.join(files)}, summary.txt, run_info.json, run.log")
    log_file.close()
    return run_dir


if __name__ == "__main__":
    main()
