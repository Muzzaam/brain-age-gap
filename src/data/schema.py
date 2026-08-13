"""
Data contract for the Brain Age Gap project.

This module is the single source of truth for what a valid cohort dataframe
looks like: column names, how biomarkers group together, and plausible ranges
for outlier checking. Everything downstream (synthetic generator, loader,
preprocessing, models) imports from here rather than hard-coding strings.

When the real ARIC-NCS data arrives, the goal is to map its columns onto the
names defined here (see loader._load_aric). Nothing else should need to change.
"""

# --- identifiers and demographics -------------------------------------------
ID_COL = "participant_id"
AGE_COL = "age"            # chronological age at the imaging visit (years)
SEX_COL = "sex"            # "M" / "F"
SITE_COL = "site"          # ARIC field centre (scanner-heterogeneity covariate)

SITES = ["Forsyth", "Jackson", "Minneapolis", "Washington"]

# --- non-invasive biomarker groups (the RQ1 ablation groups) ----------------
BIOMARKER_GROUPS = {
    "physical_function": ["grip_strength", "gait_speed", "fev1", "fvc"],
    "cardiovascular":    ["sbp", "dbp", "resting_hr"],
    "anthropometric":    ["bmi", "waist_circumference", "weight"],
}

# --- target and downstream outcomes -----------------------------------------
TARGET_COL = "bag"                       # MRI-derived Brain Age Gap (years) — ground truth
BRAIN_AGE_COL = "brain_age"              # predicted biological brain age (years) = age + bag
COGNITIVE_COL = "cognitive_score"        # continuous cognitive battery score (RQ2 outcome A)
DEMENTIA_TIME_COL = "dementia_time"      # time-to-event / censoring from imaging visit (years)
DEMENTIA_EVENT_COL = "dementia_event"    # 1 = incident dementia, 0 = censored
PREVALENT_DEMENTIA_COL = "prevalent_dementia"  # 1 = dementia already present at imaging visit (excluded)


# --- convenience accessors ---------------------------------------------------
def all_biomarker_columns():
    """Flat list of every biomarker column, in group order."""
    cols = []
    for group_cols in BIOMARKER_GROUPS.values():
        cols.extend(group_cols)
    return cols


def group_of(column):
    """Return the biomarker group a column belongs to, or None."""
    for group, cols in BIOMARKER_GROUPS.items():
        if column in cols:
            return group
    return None


# Columns every valid cohort dataframe must contain.
REQUIRED_COLUMNS = (
    [ID_COL, AGE_COL, SEX_COL, SITE_COL]
    + all_biomarker_columns()
    + [TARGET_COL, COGNITIVE_COL, DEMENTIA_TIME_COL, DEMENTIA_EVENT_COL,
       PREVALENT_DEMENTIA_COL]
)

# Biologically plausible ranges, used to flag (not silently fix) outliers.
# Values outside these bounds are suspicious, not necessarily impossible.
PLAUSIBLE_RANGES = {
    "age": (40, 100),
    "grip_strength": (5, 70),        # kg
    "gait_speed": (0.2, 2.0),        # m/s
    "fev1": (0.5, 5.0),              # litres
    "fvc": (0.8, 6.5),               # litres
    "sbp": (80, 220),                # mmHg
    "dbp": (40, 130),                # mmHg
    "resting_hr": (35, 130),         # bpm
    "bmi": (14, 55),                 # kg/m^2
    "waist_circumference": (55, 160),  # cm
    "weight": (35, 180),             # kg
}
