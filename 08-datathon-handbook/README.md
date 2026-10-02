# 📊 Datathon & ML Hackathon Playbook

<!-- markdownlint-disable MD013 -->

> **A beginner-friendly guide to winning data science hackathons and datathons** by turning messy real-world datasets into high-performing machine learning models and interactive, decision-grade user solutions.

---

## 👋 New to Datathons? Start Here in 3 Steps!

If this is your first time competing in a data hackathon, don't feel overwhelmed. Follow these 3 steps:

| Step | Open This | Time |
| --- | --- | --- |
| **1. Understand the Event** | [`first-datathon.md`](00-start-here/first-datathon.md): What a datathon actually is, what judges care about, and how to survive your first 24 hours | 10 min |
| **2. Learn the Lingo** | [`GLOSSARY.md`](00-start-here/GLOSSARY.md): Jargon-free explanations of scary words like *Overfitting*, *Data Leakage*, *SHAP*, and *Cross-Validation* | 10 min |
| **3. Run the 1-Click Starter** | Run `python 00-start-here/quickstart-1-click.py`: Trains a real model and launches an interactive web dashboard in 30 seconds! | 1 min |

> 🚨 **Something broke at 2 AM?** Open the [`troubleshooting.md`](00-start-here/troubleshooting.md) Panic Button for instant 60-second fixes!

---

## 🏆 The Core Datathon Philosophy

Most teams lose datathons because they make one of two critical mistakes:
1. **The Kaggle Notebook Trap:** Spending 36 hours obsessing over the 4th decimal place of ROC-AUC in a `.ipynb` notebook, showing static matplotlib bar charts, and offering zero operational tools for non-technical judges.
2. **The Dummy UI Trap:** Building an aesthetically gorgeous dashboard that has no statistical rigor, runs on fake hardcoded numbers, and falls apart under basic questioning from technical judges.

**The Winning Formula:**
```text
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ Real-Life Data  │ ────► │ High-Speed ML   │ ────► │ "The Last-Mile" │ ────► │  Winning Pitch  │
│  Ingestion &    │       │  & Leak-Free    │       │ Interactive App │       │  & Actionable   │
│ Wrangling (Duck)│       │  Models (LGBM)  │       │ (Streamlit/SHAP)│       │   Business ROI  │
└─────────────────┘       └─────────────────┘       └─────────────────┘       └─────────────────┘
```
Winning teams treat the datathon as a rapid **decision-support product challenge**. They give the judges an interactive interface to touch, simulate "what-if" scenarios, inspect model explainability (SHAP), and calculate quantified financial/operational impact.

---

## 🗺️ What's in This Playbook

Folders are numbered in the order you will need them.

| Folder | Open it when… | Start with |
| --- | --- | --- |
| 🟢 [`00-start-here/`](00-start-here/README.md) | You are new, or it is Hour 0 | [`first-datathon.md`](00-start-here/first-datathon.md) → [`runbook.md`](00-start-here/runbook.md) |
| 📘 [`01-playbook/`](01-playbook/README.md) | Planning, framing, judging, pitching | [`datathon-battle-plan.md`](01-playbook/datathon-battle-plan.md) |
| 🧹 [`02-data-wrangling/`](02-data-wrangling/README.md) | Loading, cleaning, feature engineering | [`recipes.md`](02-data-wrangling/recipes.md) |
| 🤖 [`03-modeling/`](03-modeling/README.md) | Choosing methods, CV, tuning, ensembling | [`method-selection-guide.md`](03-modeling/method-selection-guide.md) |
| 🖥️ [`04-solutions-and-ui/`](04-solutions-and-ui/README.md) | Building the interactive app | [`streamlit-app-template.py`](04-solutions-and-ui/streamlit-app-template.py) |
| 📦 [`05-repo-catalog/`](05-repo-catalog/README.md) | Looking for tools, datasets, winning repos | [`README.md`](05-repo-catalog/README.md) |
| 🏆 [`07-worked-example/`](07-worked-example/README.md) | You want to see a full run | [`worked-example.md`](07-worked-example/worked-example.md) |
| 🧠 [`08-ai-agent-kit/`](08-ai-agent-kit/README.md) | Using an AI assistant (prompts, skill spec, SOPs) | [`PROMPTS.md`](08-ai-agent-kit/PROMPTS.md) |
| 📦 [`PROJECT_TEMPLATE/`](PROJECT_TEMPLATE/README.md) | Starting your team's repo | [`PROJECT_CONTEXT.md`](PROJECT_TEMPLATE/PROJECT_CONTEXT.md) |

(`06-` is intentionally unused: its quick prompts moved to [`08-ai-agent-kit/quick-prompts.md`](08-ai-agent-kit/quick-prompts.md).)

### 🟢 00. Start Here
* [`00-start-here/first-datathon.md`](00-start-here/first-datathon.md) — What a datathon is, the 4 vital roles, the 10 Golden Rules.
* [`00-start-here/runbook.md`](00-start-here/runbook.md) — The 8 gates with exit checks, decision tables, evidence rules and source trust ladder.
* [`00-start-here/GLOSSARY.md`](00-start-here/GLOSSARY.md) — Plain-English cheat sheet for ML jargon.
* [`00-start-here/quickstart-1-click.py`](00-start-here/quickstart-1-click.py) — Trains a LightGBM model and launches a Streamlit UI in 30 seconds.
* [`00-start-here/troubleshooting.md`](00-start-here/troubleshooting.md) — 60-second fixes for out-of-memory, NaN, merge blow-ups, Streamlit crashes.

### 📘 01. Playbook & Strategy
* [`datathon-battle-plan.md`](01-playbook/datathon-battle-plan.md) — Hour-by-hour 24h and 48h timelines, roles.
* [`hypothesis-and-problem-framing.md`](01-playbook/hypothesis-and-problem-framing.md) — Hypothesis tree and economic bottleneck mapping.
* [`judging-and-winning-evidence.md`](01-playbook/judging-and-winning-evidence.md) — Sourced findings on rubrics, winners, failure modes, logistics; event intake checklist.
* [`pitch-and-presentation-guide.md`](01-playbook/pitch-and-presentation-guide.md) — 10-slide blueprint, 3-minute script, judge Q&A.
* [`executive-report-template.md`](01-playbook/executive-report-template.md) — Fill-in 2-page executive summary.
* [`team-git-and-notebook-workflow.md`](01-playbook/team-git-and-notebook-workflow.md) — Anti-merge-conflict Git practice and role contracts.

### 🧹 02. Data Wrangling & Feature Engineering
* [`recipes.md`](02-data-wrangling/recipes.md) — Polars and DuckDB ingestion, 10-minute EDA.
* [`handling-dirty-data.md`](02-data-wrangling/handling-dirty-data.md) — Missingness, cardinality, outliers, imbalance.
* [`feature-engineering-cookbook.md`](02-data-wrangling/feature-engineering-cookbook.md) — 10 high-yield feature families.

### 🤖 03. Modeling & Validation
* [`method-selection-guide.md`](03-modeling/method-selection-guide.md) — Scenario → method matrix with confidence labels.
* [`ml-toolbox.md`](03-modeling/ml-toolbox.md) — Tool verdicts by job, beat-the-baseline ladder, learning resources.
* [`tooling-2026-update.md`](03-modeling/tooling-2026-update.md) — Version pins, install smoke test, breaking changes, hosting terms.
* [`cross-validation-guide.md`](03-modeling/cross-validation-guide.md) — Stratified vs Group vs TimeSeries CV, leakage test.
* [`hyperparameter-tuning-and-ensembling.md`](03-modeling/hyperparameter-tuning-and-ensembling.md) — Optuna budgets, rank-average blends.
* [`baseline-pipeline.py`](03-modeling/baseline-pipeline.py) and [`automl-autogluon.py`](03-modeling/automl-autogluon.py) — Runnable pipelines.

### 🖥️ 04. Solutions & Interactive UI
* [`streamlit-app-template.py`](04-solutions-and-ui/streamlit-app-template.py) — Runnable app with scenario sliders, SHAP, ROI ticker.
* [`dashboard-design-patterns.md`](04-solutions-and-ui/dashboard-design-patterns.md) — The 5 UI patterns judges like.

### 📦 05. Repo Catalog · 🏆 07. Worked Example
* [`05-repo-catalog/README.md`](05-repo-catalog/README.md) — Trusted learning source, winning solutions, starter kits, data portals.
* [`07-worked-example/worked-example.md`](07-worked-example/worked-example.md) — A 4-person team's 48-hour run.

### 🧠 08. AI Agent Kit
* [`PROMPTS.md`](08-ai-agent-kit/PROMPTS.md) — 20 macros (`DT01`-`DT20`) plus `DT00` context and `DT-R` review.
* [`quick-prompts.md`](08-ai-agent-kit/quick-prompts.md) — Short co-pilot prompts.
* [`SKILLS.md`](08-ai-agent-kit/SKILLS.md) — Agent skill spec to drop into a project.
* [`STANDARDS.md`](08-ai-agent-kit/STANDARDS.md) — SOPs: repo taxonomy, data hygiene, CV integrity, seeds, pre-submission audit.

---

## ⚡ 2026 Recommended Datathon Tech Stack

| Need | Recommended Tool | Why It Wins at Hackathons | Level |
| --- | --- | --- | --- |
| **Instant 1-Click EDA** | [ydata-profiling](https://github.com/ydataai/ydata-profiling) | Complete interactive HTML report in 1 line of Python | 🟢 |
| **Speed ML Workhorse**| [LightGBM](https://github.com/microsoft/LightGBM) | Fastest CPU gradient boosted tree training; trains in seconds | 🟢 |
| **Messy Categoricals**| [CatBoost](https://github.com/catboost/catboost) | Native handling of categorical & text features with zero leakage | 🟢 |
| **Interactive App** | [Streamlit](https://github.com/streamlit/streamlit) | Instant Python web apps with sliders, scorecards, and free cloud URLs | 🟢 |
| **Explainability (XAI)**| [SHAP](https://github.com/slundberg/shap) | Proves to judges *why* the model made a prediction | 🟡 |
| **Data Engine** | [Polars](https://github.com/pola-rs/polars) | 10x-50x faster than Pandas; zero memory crashes on multi-GB CSVs | 🟡 |
| **Analytical SQL** | [DuckDB](https://github.com/duckdb/duckdb) | In-process analytical SQL; queries Parquet files with zero copy | 🟡 |
| **AutoML Stacking** | [AutoGluon](https://github.com/autogluon/autogluon) | Multi-layer stacking ensemble; tops leaderboards in 15–30 min | 🟡 |

---

## 🚀 Quickstart: 3 Steps for Hour 0

1. **Install Core Datathon Environment:**
   ```bash
   pip install polars duckdb lightgbm catboost shap streamlit ydata-profiling
   ```
2. **Run the 1-Click Starter:**
   ```bash
   python 00-start-here/quickstart-1-click.py
   ```
3. **Open the Battle Plan:**
   Read [`01-playbook/datathon-battle-plan.md`](01-playbook/datathon-battle-plan.md) and assign your 4 team roles before writing a single line of code!
