# 🖨️ Datathon Cheat Sheet

<!-- markdownlint-disable MD013 -->

> One page to print or keep open. Full detail: [`runbook.md`](runbook.md). Runnable example: [`../07-worked-example/run_end_to_end.py`](../07-worked-example/run_end_to_end.py).

## The 8 gates (do not pass a gate until its check is true)

| # | Gate | 48 h | Exit check |
| - | --- | --- | --- |
| 1 | Frame | 0-2 h | "Who does what differently because of the model?" in one sentence; $ per FP and FN written down |
| 2 | Lock evaluation | 2-4 h | Folds saved once; no entity in two folds; metric chosen |
| 3 | Ingest and audit | 4-8 h | Row counts reconcile after every join; target rate and missingness logged |
| 4 | Baseline | 8-12 h | Dummy < linear < GBDT on the **same** folds; OOF saved |
| 5 | Features | 12-24 h | One family at a time; keep only real OOF gains; logged |
| 6 | Improve | 24-34 h | Tuning/blend beats best single model on most folds; threshold from $ on OOF |
| 7 | Ship the UI | 34-42 h | App runs from a clean clone; what-if slider, SHAP, $ ticker |
| 8 | Pitch | 42-48 h | Timed run under limit; submission matches the sample file; every number has a file |

## Pick the split

| Data has… | Use |
| --- | --- |
| Time that predicts the future | `TimeSeriesSplit` (never shuffle) |
| Repeated entities (patient, store, user) | `StratifiedGroupKFold` on the entity ID |
| Neither, classification | `StratifiedKFold(5)` |
| Neither, regression | `KFold(5, shuffle=True)` |

## Pick the metric

Balanced classes → ROC-AUC · Positives < 5% → **PR-AUC** · Regression → RMSE (MAE if outliers are real) · Always also report **$ value at your chosen threshold**.

## Leakage alarms (stop and check)

- [ ] One feature jumps OOF score by more than ~0.15, or owns > 60% of importance
- [ ] A column only exists after the outcome (follow-up notes, final status, "days until…")
- [ ] Scaler, imputer or target encoding fitted on the full data instead of inside each fold
- [ ] Random-split score is much higher than grouped/time-split score
- [ ] Duplicate rows or IDs across train and test

## Imbalance and threshold

`scale_pos_weight` or class weights → tune the threshold on OOF by dollars → calibrate before quoting probabilities. SMOTE only if those fail and you can defend synthetic rows.

## Model defaults

Tabular → LightGBM or CatBoost baseline, then blend · Small data + GPU → try TabPFN/TabICL (check licence) · Text → TF-IDF + logistic regression first · Time series → seasonal naive, then LightGBM with lags · Anomaly → ECOD + Isolation Forest. Details: [`../03-modeling/method-selection-guide.md`](../03-modeling/method-selection-guide.md).

## Rules (Kaggle-style)

One account per person · no private sharing outside your team · external data only if free and available to all · pick your 2 final submissions yourself · never hand-label test rows. Details: [`../01-playbook/datathon-rules-and-licences.md`](../01-playbook/datathon-rules-and-licences.md).

**Prompts:** [`../08-ai-agent-kit/PROMPTS.md`](../08-ai-agent-kit/PROMPTS.md) (`DT00` first, `DT16` leakage audit, `DT19` pre-submission audit) · **Stuck:** [`troubleshooting.md`](troubleshooting.md)
