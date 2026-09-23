# 📊 Datathon & ML Hackathon Playbook

<!-- markdownlint-disable MD013 -->

> **A beginner-friendly guide to winning data science hackathons and datathons** by turning messy real-world datasets into high-performing machine learning models and interactive, decision-grade user solutions.

---

## 👋 New to Datathons? Start Here in 3 Steps!

If this is your first time competing in a data hackathon, don't feel overwhelmed. Follow these 3 steps:

| Step | Open This | Time |
| --- | --- | --- |
| **1. Understand the Event** | [`first-datathon.md`](first-datathon.md): What a datathon actually is, what judges care about, and how to survive your first 24 hours | 10 min |
| **2. Learn the Lingo** | [`GLOSSARY.md`](GLOSSARY.md): Jargon-free explanations of scary words like *Overfitting*, *Data Leakage*, *SHAP*, and *Cross-Validation* | 10 min |
| **3. Run the 1-Click Starter** | Run `python quickstart-1-click.py`: Trains a real model and launches an interactive web dashboard in 30 seconds! | 1 min |

> 🚨 **Something broke at 2 AM?** Open the [`troubleshooting.md`](troubleshooting.md) Panic Button for instant 60-second fixes!

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

| Folder / File | What It Covers | Level | Who Needs It |
| --- | --- | --- | --- |
| 🟢 [`first-datathon.md`](first-datathon.md) | **The Survival Guide:** What actually happens, team roles, and 10 golden rules | 🟢 Beginner | Everyone |
| 🟢 [`GLOSSARY.md`](GLOSSARY.md) | **Plain English Dictionary:** Scary data science terms translated into real-world analogies | 🟢 Beginner | Everyone |
| 🟢 [`quickstart-1-click.py`](quickstart-1-click.py) | **1-Click Starter:** Clean data, train a model, and launch a web demo in 30 seconds | 🟢 Beginner | Everyone |
| 📘 [`01-playbook/`](01-playbook/datathon-battle-plan.md) | **The Game Plan:** Hour-by-hour 24h & 48h timeline, judging criteria rubric, and team matrix | 🟢/🟡 | Team Leads |
| 🧹 [`02-data-wrangling/`](02-data-wrangling/recipes.md) | **Data Triage:** Polars + DuckDB high-speed recipes, automated 10-minute EDA, and dirty data cleaning | 🟡 Intermediate | Data Engineers |
| 🤖 [`03-modeling/`](03-modeling/baseline-pipeline.py) | **Rapid ML:** LightGBM, CatBoost, AutoGluon, leak-free CV pipelines, and threshold tuning | 🟡 Intermediate | ML Engineers |
| 🖥️ [`04-solutions-and-ui/`](04-solutions-and-ui/streamlit-app-template.py) | **The Last Mile:** Ready-to-run interactive Streamlit dashboard with "What-If" simulator & SHAP | 🟢/🟡 | UI & Demo Leads |
| 📦 [`05-repo-catalog/`](05-repo-catalog/README.md) | **The Shopping List:** Curated high-star libraries, open data portals, and starters | 🟢 Beginner | Everyone |
| 💬 [`06-prompts/`](06-prompts/datathon-prompts.md) | **LLM Co-Pilots:** Copy-paste prompts for fast EDA, feature brainstorming, and pitch narrative | 🟢 Beginner | Everyone |
| 🚨 [`troubleshooting.md`](troubleshooting.md) | **The Panic Button:** 60-second solutions when code crashes or laptops freeze | 🟢 Beginner | Everyone |

**Level key:** 🟢 Beginner-friendly (Start here!) · 🟡 Read when you need it · 🔴 Advanced

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
   python quickstart-1-click.py
   ```
3. **Open the Battle Plan:**
   Read [`01-playbook/datathon-battle-plan.md`](01-playbook/datathon-battle-plan.md) and assign your 4 team roles before writing a single line of code!
