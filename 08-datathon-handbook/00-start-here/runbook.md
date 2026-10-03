# 🧭 Datathon Runbook: The Formulaic Path

<!-- markdownlint-disable MD013 -->

> Every datathon, same 8 gates. Each gate has: **Input → Action → Output artifact → Exit check → Trusted reference**. Do not start gate N+1 until gate N's exit check passes. Fill in the `PROJECT_TEMPLATE/PROJECT_CONTEXT.md` as you go.

**Trusted source for fundamentals:** [`rohitg00/ai-engineering-from-scratch`](https://github.com/rohitg00/ai-engineering-from-scratch) (MIT, 20 phases, 523 lessons; verified 2026-10-02). Each lesson lives at `phases/<NN>-<phase>/<NN>-<lesson>/` with `docs/en.md` (concepts), `code/` (runnable), `outputs/` (reusable artifacts) and follows Motto → Problem → Concept → Build It → Use It → Ship It. Use it to **learn the why**; use this handbook for the **how, fast**. Paths below are under `phases/02-ml-fundamentals/` unless stated.

---

## The 8 Gates

| # | Gate | Hours (48h) | Action | Output artifact | Exit check | Trusted lesson (AIE-from-scratch) | Handbook file |
| - | --- | --- | --- | --- | --- | --- | --- |
| 1 | **Frame** | 0-2 | Write: decision, decision-maker, metric, $ per FP/FN, hypothesis tree | `PROJECT_CONTEXT.md` §1-3 | Can state the problem as "Who does what differently because of the model?" in 1 sentence | `01-what-is-machine-learning` | `01-playbook/hypothesis-and-problem-framing.md` |
| 2 | **Lock evaluation** | 2-4 | Choose split (Stratified / Group / TimeSeries), metric, seed; save fold IDs | `folds.parquet`, `config.yaml` | Split matches data generating process; no entity in both train & valid | `09-model-evaluation` | `03-modeling/cross-validation-guide.md` |
| 3 | **Ingest & audit** | 4-8 | Load with Polars/DuckDB; profile; missingness + dtype + duplicate audit | `data_audit.md` | Row counts reconcile after every join; target rate logged | `13-ml-pipelines` | `02-data-wrangling/recipes.md` (no data? `02-data-wrangling/getting-data.md`) |
| 4 | **Baseline** | 8-12 | Dummy → linear → LightGBM with default params on locked folds | `oof_baseline.parquet` + score | Baseline beats dummy; OOF saved | `02-linear-regression`, `03-logistic-regression`, `04-decision-trees` | `03-modeling/baseline-pipeline.py` · metrics explained: `03-modeling/ml-fundamentals-and-metrics.md` · models explained: `03-modeling/ml-models-explained.md` |
| 5 | **Features** | 12-24 | Add one feature family at a time; keep only if OOF improves | `feature_log.csv` (feature, family, Δscore) | Leakage test: no jump >0.15 / no feature >60% importance | `08-feature-engineering`, `18-feature-selection`, `15-time-series` | `02-data-wrangling/feature-engineering-cookbook.md` |
| 6 | **Improve** | 24-34 | Imbalance handling, Optuna (10-min budget), rank-average ensemble, threshold tuning on OOF | `final_oof.parquet`, `threshold.json` | Ensemble beats best single model on OOF, not just on one fold | `17-imbalanced-data`, `11-ensemble-methods`, `12-hyperparameter-tuning`, `10-bias-variance` | `03-modeling/hyperparameter-tuning-and-ensembling.md` · `03-modeling/error-analysis-and-tracking.md` |
| 7 | **Ship the UI** | 34-42 | Streamlit: what-if sliders, $ ticker, SHAP waterfall, cohort filter, action button | `app.py` running from a clean clone | Cold start < 10 s; slider recompute < 300 ms; runs offline | phase `00` (Setup & Tooling) | `04-solutions-and-ui/streamlit-app-template.py` |
| 8 | **Pitch** | 42-48 | 10 slides, 3-min script, 5 judge Qs rehearsed, submission file validated | `slides`, `report.pdf`, `submission.csv` | Timed run ≤ 3:00; submission schema matches sample | n/a | `01-playbook/pitch-and-presentation-guide.md` |

Tool and algorithm verdicts for gates 4-6: [`03-modeling/ml-toolbox.md`](../03-modeling/ml-toolbox.md).

Situational lessons (use only when the dataset demands it): `16-anomaly-detection` (fraud/failure with few labels), `07-unsupervised-learning` (segmentation tasks), `14-naive-bayes` (small text baselines), `06-knn-and-distances` (similarity/lookup tasks), `05-support-vector-machines` (small, high-dimensional data).

---

**Hour 0 intake:** before Gate 1, run the checklist in [`01-playbook/judging-and-winning-evidence.md`](../01-playbook/judging-and-winning-evidence.md) (scoring type, official criteria, AI-use and data-access rules).

---

## Decision Tables (no judgment calls)

### Choosing the split
| If the data has… | Use |
| --- | --- |
| a timestamp that predicts the future | `TimeSeriesSplit` (never shuffle) |
| repeated entities (patient/store/user) | `StratifiedGroupKFold` on the entity ID |
| neither, binary target | `StratifiedKFold(5)` |
| neither, regression target | `KFold(5, shuffle=True)` |

### Choosing the metric
| Task | Optimize | Report to judges |
| --- | --- | --- |
| Balanced classification | ROC-AUC | + $ net value at chosen threshold |
| Positives < 5% | PR-AUC | + recall at fixed precision, $ net value |
| Regression | RMSE (or MAE if outliers are real) | + error in business units |
| Ranking / top-k | NDCG / precision@k | + $ captured in top-k |

### Choosing the model (first match wins)
_Full scenario matrix with confidence labels: [`03-modeling/method-selection-guide.md`](../03-modeling/method-selection-guide.md)._

1. Tabular → LightGBM or CatBoost baseline, then blend diverse models; consider TabPFN/TabICL only for small data with a GPU (check licence).
2. Tabular, need a leaderboard push → AutoGluon `best_quality` on locked folds.
3. Text/image columns → embed with a pretrained model, then feed to LightGBM.
4. Never start with deep nets on tabular data.

### Handling imbalance
`scale_pos_weight` → threshold tuning on OOF → class-weighted loss. SMOTE only if the first three failed **and** you can justify synthetic rows to judges.

---

## Per-Gate Rules of Evidence

Every claim in the pitch needs one row in `PROJECT_CONTEXT.md`:

| Claim | Evidence file | Reproduce command |
| --- | --- | --- |
| "OOF PR-AUC = X" | `final_oof.parquet` | `python eval.py --oof final_oof.parquet` |
| "Feature F adds Δ" | `feature_log.csv` | `python ablate.py --drop F` |
| "Saves $Y" | `threshold.json` + unit-economics cell | `python value.py` |

No evidence row → claim is cut from the pitch.

---

## Source Trust Ladder

When two sources disagree, trust in this order:

1. **Dataset/organizer rules** (the brief wins over everything).
2. **Library official docs** (scikit-learn, LightGBM, Polars, Streamlit).
3. **[`rohitg00/ai-engineering-from-scratch`](https://github.com/rohitg00/ai-engineering-from-scratch)**: MIT-licensed, structured, runnable code per lesson.
4. **This handbook's recipes** (tested in `07-worked-example`).
5. **Prior winning-solution repos** in `05-repo-catalog/README.md` (pattern inspiration only; check the license and re-derive).
6. **Blog posts / LLM output**: never copy without running on your folds.

Before using any GitHub repo: check license, last commit date, stars, and that its example runs. Record the check in `PROJECT_CONTEXT.md` §Sources.

---

## Pre-Submission Checklist (copy into your issue tracker)

- [ ] Folds frozen since Gate 2; no feature computed using validation labels
- [ ] Preprocessing fitted inside each fold (no global scaler/encoder on full data)
- [ ] Target-derived columns and post-event columns removed
- [ ] Seeds fixed; `python train.py` reproduces the OOF score
- [ ] Submission file matches the sample's columns, dtypes and row count
- [ ] App runs from a clean clone with only `requirements.txt`
- [ ] Every pitch number has an evidence row
