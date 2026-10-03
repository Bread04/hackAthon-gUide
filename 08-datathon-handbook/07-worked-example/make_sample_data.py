"""Generate sample-readmissions.csv: a small, fully synthetic hospital-readmission dataset.

No real patients. Built so the runbook's lessons show up in the data:
- patients have several admissions (so a random split leaks; group by patient_id),
- about 12% positives (so PR-AUC matters more than accuracy),
- missing values that carry signal,
- one planted leak: `followup_call_outcome` is only known AFTER the outcome.

Run: python make_sample_data.py   (deterministic, seed 7)
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n_patients = 900
patients = pd.DataFrame({
    "patient_id": [f"P{i:04d}" for i in range(n_patients)],
    "age": rng.integers(18, 95, n_patients),
    "chronic_conditions": rng.poisson(1.8, n_patients),
    "frailty": rng.normal(0, 1, n_patients),  # hidden patient-level risk
})
visits = patients.loc[patients.index.repeat(rng.integers(1, 6, n_patients))].reset_index(drop=True)
m = len(visits)
visits["admission_id"] = [f"A{i:05d}" for i in range(m)]
visits["admit_date"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, m), unit="D")
visits["length_of_stay"] = np.clip(rng.gamma(2.0, 2.5, m), 1, 30).round(1)
visits["n_medications"] = rng.poisson(5 + visits["chronic_conditions"], m)
visits["prior_admissions_12m"] = rng.poisson(0.6 + 0.4 * visits["chronic_conditions"], m)
visits["discharge_to"] = rng.choice(["home", "home_care", "nursing_facility"], m, p=[0.65, 0.2, 0.15])
lab = rng.normal(1.0 + 0.15 * visits["chronic_conditions"], 0.3, m)
lab_missing = rng.random(m) < 0.25 + 0.15 * (visits["discharge_to"] == "home")
visits["creatinine"] = np.where(lab_missing, np.nan, lab.round(2))

logit = (-5.3 + 0.02 * (visits["age"] - 60) + 0.35 * visits["chronic_conditions"]
         + 0.08 * visits["length_of_stay"] + 0.45 * visits["prior_admissions_12m"]
         + 0.6 * (visits["discharge_to"] == "nursing_facility") + 2.2 * visits["frailty"]
         + 0.5 * np.nan_to_num(visits["creatinine"] - 1.2) - 0.4 * lab_missing)
visits["readmitted_30d"] = (rng.random(m) < 1 / (1 + np.exp(-logit))).astype(int)
# Planted leak: recorded by the follow-up team after the 30-day window closes.
visits["followup_call_outcome"] = np.where(
    visits["readmitted_30d"] == 1,
    rng.choice(["readmitted", "no_answer", "fine"], m, p=[0.8, 0.15, 0.05]),
    rng.choice(["readmitted", "no_answer", "fine"], m, p=[0.02, 0.28, 0.70]))

cols = ["admission_id", "patient_id", "admit_date", "age", "chronic_conditions", "length_of_stay",
        "n_medications", "prior_admissions_12m", "discharge_to", "creatinine",
        "followup_call_outcome", "readmitted_30d"]
visits[cols].sort_values("admit_date").to_csv("sample-readmissions.csv", index=False)
print(f"wrote sample-readmissions.csv: {m} admissions, {n_patients} patients, "
      f"{visits['readmitted_30d'].mean():.1%} positive")
