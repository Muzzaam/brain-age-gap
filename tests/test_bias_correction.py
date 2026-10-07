"""Tests for bias correction.  Run: python -m tests.test_bias_correction"""
import numpy as np

from tests.helpers import cohort, run_tests
from src.data import schema
from src.evaluation.bias_correction import (fit_bias_correction, correct_bag, age_bias,
                                            reference_mask, fit_on_reference)
from src.experiment import bag_labels


def _split(n=1500, seed=42, **kw):
    c = cohort(n=n, seed=seed, **kw)
    return c.iloc[: int(0.8 * len(c))], c.iloc[int(0.8 * len(c)):]


def test_fit_returns_params():
    tr, _ = _split()
    p = fit_bias_correction(tr[schema.TARGET_COL], tr[schema.AGE_COL])
    assert "slope" in p and "intercept" in p


def test_correction_zeroes_age_bias_in_sample():
    # OLS residual property: fitting and applying on the same data leaves the
    # corrected BAG uncorrelated with age
    tr, _ = _split()
    p = fit_bias_correction(tr[schema.TARGET_COL], tr[schema.AGE_COL])
    assert abs(age_bias(correct_bag(tr[schema.TARGET_COL], tr[schema.AGE_COL], p), tr[schema.AGE_COL])) < 1e-3


def test_correction_reduces_age_bias_out_of_sample():
    tr, te = _split(bag_age_slope=-0.8)
    y_tr, y_te, params = bag_labels(tr, te, "corrected", schema.REFERENCE_COL)
    before = abs(age_bias(te[schema.TARGET_COL], te[schema.AGE_COL]))
    after = abs(age_bias(y_te, te[schema.AGE_COL]))
    assert after < before / 2, (before, after)


def test_reference_subset_used_when_present():
    tr, _ = _split()
    mask, used = reference_mask(tr)
    assert used == f"{schema.REFERENCE_COL} == 1"
    assert mask.sum() == (tr[schema.REFERENCE_COL] == 1).sum() < len(tr)
    assert fit_on_reference(tr)["n_reference"] == mask.sum()


def test_reference_falls_back_to_all_training_rows():
    tr, _ = _split()
    mask, used = reference_mask(tr.drop(columns=[schema.REFERENCE_COL]))
    assert used == "all training participants" and mask.all()


def test_uncorrected_labels_are_untouched():
    tr, te = _split()
    y_tr, y_te, params = bag_labels(tr, te, "uncorrected", schema.REFERENCE_COL)
    assert params is None and np.array_equal(y_te, te[schema.TARGET_COL].to_numpy())


def test_age_bias_range():
    tr, _ = _split()
    assert -1.0 <= age_bias(tr[schema.TARGET_COL], tr[schema.AGE_COL]) <= 1.0


if __name__ == "__main__":
    run_tests(globals())
