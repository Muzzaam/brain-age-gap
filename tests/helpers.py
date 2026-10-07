"""Shared fixtures for the test suites."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data import load_data
from src.preprocessing import prepare_cohort

COV = ["age", "sex_male", "site"]


def cohort(n=800, seed=0, **kw):
    """Analysis-ready synthetic cohort (masked, excluded, sex_male added)."""
    kw.setdefault("bag_missing_rate", 0.02)
    kw.setdefault("outcome_missing_rate", 0.03)
    return prepare_cohort(load_data(source="synthetic", n=n, seed=seed, **kw))[0]


def quick_cfg(**overrides):
    return load_config(quick=True, overrides=overrides or None)


def run_tests(namespace):
    """Run every test_* function in a module namespace (no pytest needed)."""
    for name, fn in list(namespace.items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASSED {name}")
