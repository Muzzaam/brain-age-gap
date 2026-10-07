# ARIC-NCS: what we expect the data to look like

A light first pass from public sources (October 2026), written before we have
the data or its documentation. Treat every point as a guess to check against
the real data dictionary. Each point is marked **found** (a public source says
so) or **expected** (standard for ARIC but not checked).

## The study

- ARIC enrolled 15,792 middle-aged adults in 1987 to 1989. **found**
- Ages 45 to 64 at enrolment, at four US field centres: Forsyth County NC,
  Jackson MS, Minneapolis MN, Washington County MD; the Jackson centre enrolled
  Black participants only. **expected** (well known, not checked this pass)
- The ARIC Neurocognitive Study (ARIC-NCS) ran alongside visit 5 (2011 to 2013):
  6,538 participants examined, ages roughly 70 to 89. **found**
- Later visits: visit 6 (2016 to 2017), visit 7 (2018 to 2019), visit 8 onward. **found**

## Who got an MRI (important)

- About **1,978** visit 5 participants had a **3T brain MRI with a T1 MPRAGE**
  sequence (the kind of scan DeepBrainNet takes). **found**
- The MRI group was **not a random sample**. Invited were: (1) everyone with
  signs of cognitive impairment at visit 5, (2) everyone with an earlier ARIC
  brain MRI (2004 to 2006), and (3) a random sample of cognitively normal
  participants. People with MRI contraindications were excluded, and people
  examined at home or in care facilities were never scanned. **found**
- ARIC provides **sampling weights** to weight the MRI group back to the full
  visit 5 sample. **found**

## Variables we need

| Our column | What ARIC seems to have | Status |
|---|---|---|
| `grip_strength` | handheld dynamometer, 2 trials after a practice trial | found |
| `gait_speed` | usual-pace 4 m walk, part of the SPPB (may be stored as a *time*) | found |
| `fev1`, `fvc` | spirometry at visit 5, but one paper had it for only ~3,850 of 6,538 | found |
| chair stands | also part of the SPPB (proposal mentions it; not in our schema) | found |
| `sbp`, `dbp`, `resting_hr`, `bmi`, `waist_circumference`, `weight` | standard exam measures | expected |
| `cognitive_score` | global cognitive **factor score** (z-score scale), plus memory, executive and language domain scores | found |
| `prevalent_dementia` | expert-panel adjudicated status at visit 5 (normal / MCI / dementia) | found |
| `dementia_event`, `dementia_time` | incident dementia from later visits, phone interviews (TICS), informants, and hospital / death-certificate codes; ARIC grades these by level of certainty | found (details unclear) |
| `site`, `race`, `sex`, `age` | standard | expected |

## Getting the data

- Route 1: **BioLINCC** (the NHLBI repository). Free, no ARIC approval needed,
  but extreme values may be removed and restricted data left out. **found**
- Route 2: the **ARIC coordinating centre**. Needs an approved ARIC manuscript
  proposal and a signed Data and Materials Distribution Agreement (DMDA); a fee
  applies. The DMDA sets the confidentiality rules, which will decide where
  the data may be stored and processed. **found**
- Not found: whether raw MRI scans come through either route, or only
  derived measures.

## How this lines up with the code

**Already matches**
- The four sites in `schema.SITES`, and site as a covariate (scanner differences).
- Age range: the synthetic cohort is centred at 76, inside ARIC's 70 to 89.
- The T1 MPRAGE scans suit DeepBrainNet.
- Prevalent dementia exclusion; incident dementia as time to event.
- The code no longer assumes any particular scale for the cognitive score, so a
  z-scored factor score works as is.

**Needs a decision once we see the data**
1. **Spirometry missingness.** If FEV1/FVC is missing for a large share of the
   MRI group, median imputation fills a lot of values. Options: drop lung
   function from the physical-function group, keep only complete cases, or add
   missingness indicators. The preflight warns above 30% missing.
2. **Gait stored as time.** Convert with `gait_speed = 4 / time` in the loader.
3. **Grip strength.** Two trials: use the maximum (common) or the mean. Decide
   and state it in Methods.
4. **Healthy reference set for bias correction.** A natural choice is visit 5
   adjudicated cognitively normal. Warning: a reference set chosen on cognition
   can distort the correction (we saw this on synthetic data), so check the
   before/after age-bias line in the run summary.
5. **Sampling weights.** Our analyses are unweighted, so results describe the
   MRI group (which has more cognitive impairment than ARIC overall), not the
   whole cohort. Either state that as a limitation, or later pass the weights
   to the models as `sample_weight`.
6. **Dementia timing.** `dementia_time` must be computed from dates (visit 5
   date to diagnosis, death or last contact). Decide which certainty levels of
   dementia diagnosis count as events.
7. **Death before dementia.** In a cohort this age many people die first. We
   treat death as censoring (standard Cox); this belongs in the limitations.

## Questions for Devon / Samuel

- Which route (BioLINCC or the coordinating centre), and does it include raw T1 scans?
- What does the DMDA say about where the data can be stored and processed?
- Is a visit 5 "cognitively normal" flag available for the reference set?
- Are the MRI sampling weights included in our extract?

## Sources

- [ARIC-NCS overview, visit 5 manual 16](https://www2.cscc.unc.edu/aric9/sites/default/files/public/visitdocuments/v5/16%20NCS%20Overview.pdf)
- [ARIC-NCS MRI procedures, visit 5 manual 13](https://www5.cscc.unc.edu/aric9/sites/default/files/public/visitdocuments/v5/13%20NCS_MRI.pdf)
- [MRI selection criteria (Walker et al. supplement)](https://cdn-links.lww.com/permalink/aln/c/aln_2020_02_19_walker_aln-d-19-00914r2_sdc1.pdf)
- [ARIC-NCS MRI sampling and weights (PMC8359773)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8359773)
- [Visit 5 physical function (PMC6075198)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6075198)
- [Visit 5 spirometry (PMC9707834)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9707834)
- [Cognitive factor scores (ARIC MP1858)](https://aric.cscc.unc.edu/aric9/system/files/MP1858.pdf)
- [Dementia ascertainment (ARIC MP4311)](https://aric.cscc.unc.edu/aric9/system/files/MP4311.pdf)
- [Dementia status variables, visit 8](https://www2.cscc.unc.edu/aric9/sites/default/files/public/visitdocuments/v8/STATUS81_np.pdf)
- [Requesting ARIC data](https://aric.cscc.unc.edu/aric9/researchers/Obtain_Submit_Data)
- [ARIC data sharing policy](https://www5.cscc.unc.edu/aric9/sites/default/files/publications/policies/ARIC%20Data%20Sharing%20Policy_20251124.pdf)
