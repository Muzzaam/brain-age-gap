"""
Load config.yaml. Every script goes through `load_config`, so the YAML file is
the one place where the data source, CV settings and model line-up are set.
"""

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "config.yaml"

# Settings for smoke tests and `run_all --quick`: tiny grids, few folds.
# Checks the machinery runs end to end in seconds; never use for results.
QUICK_OVERRIDES = {
    "data": {"synthetic": {"n": 600}},
    "cv": {"outer_folds": 2, "repeats": 1, "inner_folds": 2},
    "models": {"rq1": ["ridge", "xgboost"], "stage2_cognitive": ["ridge", "xgboost"],
               "stage2_dementia": ["cox"], "shap": ["ridge", "xgboost"]},
    "quick": True,
}


def _merge(base, override):
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path=None, quick=False, overrides=None):
    """Read config.yaml, optionally applying quick-mode and extra overrides."""
    with open(path or DEFAULT_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("quick", False)
    if quick:
        cfg = _merge(cfg, QUICK_OVERRIDES)
    if overrides:
        cfg = _merge(cfg, overrides)
    return cfg


def resolve_path(p):
    """Paths in the config are relative to the code folder."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
