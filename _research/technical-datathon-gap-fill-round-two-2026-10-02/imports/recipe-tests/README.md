# Recipe test scripts (2026-10-02)

The exact scripts used to run the recipes in
[`08-datathon-handbook/03-modeling/baseline-recipes.md`](../../../../08-datathon-handbook/03-modeling/baseline-recipes.md)
on synthetic data. Environment: Python 3.12.3, the pinned
`08-datathon-handbook/PROJECT_TEMPLATE/requirements.txt`, plus statsforecast 2.0.1,
mlforecast 1.0.2, utilsforecast 0.2.15, pyod 3.6.6, causalml 0.17.0 and verde 1.9.0.

All five exited 0. Scores are on synthetic data and only show that the code runs.

| Script | Recipe |
| --- | --- |
| `ts.py` | Time series: statsforecast vs mlforecast on rolling origins |
| `text.py` | Text: TF-IDF + logistic regression |
| `anom.py` | Anomaly: PyOD ECOD + IForest, PR-AUC |
| `uplift.py` | Uplift: T-learner + causalml Qini/AUUC |
| `spatial.py` | Spatial block CV (GroupKFold, Verde) |
