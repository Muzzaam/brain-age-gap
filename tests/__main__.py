"""Run every test suite:  python -m tests"""
import importlib
import sys
import time
import traceback
import warnings

SUITES = ["test_pipeline", "test_models", "test_stats", "test_bias_correction", "test_survival",
          "test_interpretation", "test_two_stage", "test_run_all"]

warnings.filterwarnings("ignore")
failed = 0
t0 = time.time()
for suite in SUITES:
    mod = importlib.import_module(f"tests.{suite}")
    for name, fn in list(vars(mod).items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASSED {suite}.{name}")
            except Exception:
                failed += 1
                print(f"FAILED {suite}.{name}")
                traceback.print_exc()
print(f"\n{'ALL PASSED' if not failed else f'{failed} FAILED'} in {time.time() - t0:.0f}s")
sys.exit(1 if failed else 0)
