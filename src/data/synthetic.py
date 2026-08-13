"""
Synthetic ARIC-shaped cohort generator.

Purpose: let us build and test the entire pipeline before the real ARIC-NCS
data exists. The output dataframe conforms exactly to src/data/schema.py, so
code written against this generator will run unchanged on the real data.

Design notes
------------
1. A latent "systemic ageing" factor z drives realistic correlations between
   biomarkers (e.g. grip strength falls while systolic BP rises).
2. Ground-truth BAG is built from a weighted combination of the *true*
   biomarker values plus independent noise. The parameter `target_signal_fraction`
   sets the fraction of BAG variance that is explained by biomarkers, i.e. an
   approximate UPPER BOUND on the R^2 a perfect estimator could reach on clean
   data. Missingness and outliers then degrade recoverability below that bound.
   This makes the RQ1 feasibility experiment self-validating: you injected the
   signal, so you know roughly what a correct pipeline should recover.
3. Cognitive score and incident dementia are generated downstream of BAG and
   age, so RQ2 ("does BAG add value beyond raw biomarkers?") has real structure
   to detect — the ageing signal genuinely flows through BAG into the outcomes.

Everything is seeded for reproducibility.
"""

import numpy as np
import pandas as pd

from . import schema


def _expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def _z(x):
    x = np.asarray(x, dtype=float)
    return (x - np.nanmean(x)) / np.nanstd(x)


def generate_synthetic_cohort(
    n=1500,
    seed=42,
    target_signal_fraction=0.30,   # ~upper bound on recoverable BAG R^2 on clean data
    bag_sd=7.5,                     # spread of ground-truth BAG in years
    cell_missing_rate=0.05,         # per-cell MCAR missingness on biomarkers
    group_missing_rate=0.02,        # fraction of participants missing an entire group
    outlier_rate=0.003,             # fraction of biomarker cells set to implausible values
    prevalent_dementia_rate=0.04,   # baseline fraction with dementia already present
    max_followup_years=10.0,
):
    """Return a DataFrame conforming to schema.REQUIRED_COLUMNS."""
    rng = np.random.default_rng(seed)

    # --- demographics --------------------------------------------------------
    # ARIC-NCS MRI subcohort skews older; centre age at 76.
    age = np.clip(rng.normal(76, 5, n), 67, 92)
    age_c = age - 76.0
    sex = rng.choice(["M", "F"], size=n)
    is_male = (sex == "M").astype(float)
    site = rng.choice(schema.SITES, size=n)

    # --- latent systemic-ageing factor (beyond chronological age) ------------
    z = rng.normal(0, 1, n)

    # --- biomarkers: functions of age, sex, latent ageing, plus noise --------
    grip = 30 + 8 * is_male - 0.35 * age_c - 3.0 * z + rng.normal(0, 4, n)
    gait = 1.05 - 0.010 * age_c - 0.06 * z + rng.normal(0, 0.12, n)
    fev1 = 2.4 + 0.6 * is_male - 0.025 * age_c - 0.15 * z + rng.normal(0, 0.30, n)
    fvc = 3.1 + 0.7 * is_male - 0.030 * age_c - 0.18 * z + rng.normal(0, 0.35, n)

    sbp = 128 + 0.45 * age_c + 6.0 * z + rng.normal(0, 12, n)
    dbp = 74 + 0.10 * age_c + 2.5 * z + rng.normal(0, 8, n)
    resting_hr = 67 + 0.05 * age_c + 1.5 * z + rng.normal(0, 9, n)

    bmi = 28 + 0.5 * z + rng.normal(0, 3.5, n)
    height = np.where(is_male == 1, 1.75, 1.62) + rng.normal(0, 0.06, n)
    weight = bmi * height ** 2 + rng.normal(0, 3, n)
    waist = 95 + 2.2 * (bmi - 28) + 6 * is_male + rng.normal(0, 6, n)

    # --- ground-truth BAG ----------------------------------------------------
    # Lower grip/gait/fev1 and higher SBP/waist -> older-appearing brain.
    signal = (
        0.9 * _z(-grip)
        + 0.7 * _z(-gait)
        + 0.6 * _z(sbp)
        + 0.5 * _z(-fev1)
        + 0.3 * _z(-fvc)
        + 0.2 * _z(waist)
    )
    signal = _z(signal)  # unit variance
    f = float(target_signal_fraction)
    eps = rng.normal(0, 1, n)
    bag_std = np.sqrt(f) * signal + np.sqrt(1 - f) * eps
    bag = bag_sd * bag_std                      # mean ~0, sd ~bag_sd (years)
    brain_age = age + bag

    # --- cognitive score (RQ2 outcome A): worse with higher BAG and age ------
    cog_latent = -0.55 * _z(bag) - 0.30 * _z(age) + 0.75 * rng.normal(0, 1, n)
    cognitive_score = 100 + 15 * _z(cog_latent)  # test-like scale (mean 100, sd 15)

    # --- incident dementia (RQ2 outcome B): survival with BAG/age hazard -----
    linpred = 0.60 * _z(bag) + 0.50 * _z(age)
    rate = 0.03 * np.exp(linpred)                # per-year hazard
    t_event = rng.exponential(1.0 / rate)
    t_censor = rng.uniform(2.0, max_followup_years, n)
    dementia_time = np.minimum(t_event, t_censor)
    dementia_event = (t_event <= t_censor).astype(int)

    # --- prevalent dementia at baseline (to be excluded downstream) ----------
    p_prev = _expit(-3.2 + 0.5 * _z(bag) + 0.4 * _z(age))
    prevalent = rng.binomial(1, np.clip(p_prev, 0, 0.5))
    # scale roughly to the requested prevalence
    if prevalent.mean() > 0:
        keep = rng.random(n) < (prevalent_dementia_rate / max(prevalent.mean(), 1e-6))
        prevalent = (prevalent & keep).astype(int)

    df = pd.DataFrame(
        {
            schema.ID_COL: [f"SYN{ i:05d}" for i in range(n)],
            schema.AGE_COL: np.round(age, 1),
            schema.SEX_COL: sex,
            schema.SITE_COL: site,
            "grip_strength": np.round(grip, 1),
            "gait_speed": np.round(gait, 3),
            "fev1": np.round(fev1, 2),
            "fvc": np.round(fvc, 2),
            "sbp": np.round(sbp, 0),
            "dbp": np.round(dbp, 0),
            "resting_hr": np.round(resting_hr, 0),
            "bmi": np.round(bmi, 1),
            "waist_circumference": np.round(waist, 1),
            "weight": np.round(weight, 1),
            schema.TARGET_COL: np.round(bag, 2),
            schema.BRAIN_AGE_COL: np.round(brain_age, 2),
            schema.COGNITIVE_COL: np.round(cognitive_score, 1),
            schema.DEMENTIA_TIME_COL: np.round(dementia_time, 2),
            schema.DEMENTIA_EVENT_COL: dementia_event,
            schema.PREVALENT_DEMENTIA_COL: prevalent,
        }
    )

    # --- inject observation messiness on the biomarker columns only ----------
    biomarker_cols = schema.all_biomarker_columns()
    df = _inject_outliers(df, biomarker_cols, outlier_rate, rng)
    df = _inject_group_missing(df, group_missing_rate, rng)
    df = _inject_cell_missing(df, biomarker_cols, cell_missing_rate, rng)
    return df


def _inject_cell_missing(df, cols, rate, rng):
    if rate <= 0:
        return df
    for col in cols:
        mask = rng.random(len(df)) < rate
        df.loc[mask, col] = np.nan
    return df


def _inject_group_missing(df, rate, rng):
    if rate <= 0:
        return df
    groups = list(schema.BIOMARKER_GROUPS.values())
    for i in range(len(df)):
        if rng.random() < rate:
            grp = groups[rng.integers(len(groups))]
            df.loc[df.index[i], grp] = np.nan
    return df


def _inject_outliers(df, cols, rate, rng):
    if rate <= 0:
        return df
    for col in cols:
        mask = rng.random(len(df)) < rate
        if mask.any():
            # push to an implausible multiple of the column's own scale
            df.loc[mask, col] = df[col].mean() * rng.uniform(4, 8, mask.sum())
    return df


if __name__ == "__main__":
    d = generate_synthetic_cohort()
    print(d.head())
    print(d.describe(include="all").T)
