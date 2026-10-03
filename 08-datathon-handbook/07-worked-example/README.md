# 🏆 07 · Worked Example

<!-- markdownlint-disable MD013 -->

> See the [runbook](../00-start-here/runbook.md) gates executed: once as a story, once as code you can run.

| File | What it is |
| --- | --- |
| [`worked-example.md`](worked-example.md) | A **fictional** hour-by-hour 48-hour clinical datathon, told as a story |
| [`run_end_to_end.py`](run_end_to_end.py) | **Runnable** gates 2-6 on the sample data: grouped folds, data audit, leak detection, baseline ladder, feature families with a log, $-based threshold, evidence table. ~20 s on a laptop |
| [`model_zoo.py`](model_zoo.py) | Trains **every classic model** (dummy, linear/logistic, Ridge/Lasso, Poisson, Naive Bayes, kNN, SVM, trees, forests, boosting, MLP, clustering, PCA, Isolation forest) on the same folds and prints a fair comparison. ~25 s |
| [`evaluate_and_explain.py`](evaluate_and_explain.py) | Confusion matrix, PR/ROC with baselines, calibration, learning curve, error slices, experiment log |
| [`sample-readmissions.csv`](sample-readmissions.csv) | 2,622 synthetic admissions for 900 synthetic patients (11% readmitted), with a planted post-outcome leak. No real people |
| [`make_sample_data.py`](make_sample_data.py) | Regenerates the CSV deterministically |

```bash
pip install -r ../PROJECT_TEMPLATE/requirements.txt   # Python 3.12
python run_end_to_end.py                              # writes ./outputs/
```

What you should see, and why it matters:

- **The planted leak is caught:** `followup_call_outcome` lifts PR-AUC by about +0.6, which is impossible in real life. It is recorded after the outcome, so it is dropped (runbook leakage test, prompt `DT16`).
- **Random folds flatter you:** random KFold scores clearly higher than folds grouped by patient, because the same patient's other admissions leak into training. Report the grouped score.
- **The simpler model can win:** on this data logistic regression beats LightGBM, so the script keeps it. Follow the folds, not the fashion.
- **Feature families are kept only if they help**, and each decision is logged in `outputs/feature_log.csv`.
- **The threshold comes from dollars, not 0.5**, using stated assumptions you should replace with your brief's numbers.

The numbers are synthetic; the procedure is the point. It runs in CI via `tools/smoke_test.py`.
