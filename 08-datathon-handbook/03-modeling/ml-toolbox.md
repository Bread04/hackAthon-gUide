# 🧰 ML Toolbox: What to Use, When, and What to Skip

<!-- markdownlint-disable MD013 -->

> A hackathon-sized map of the machine learning ecosystem. The structure is inspired by the topic-by-topic taxonomy of [`mikeroyal/Machine-Learning-Guide`](https://github.com/mikeroyal/Machine-Learning-Guide) (frameworks, algorithms, MLOps, explainability, learning resources), but cut down to **one verdict per tool for a 24-48 hour event**. That repo is a very broad awesome-list; use it to discover options, then come back here to decide.

**Verdict key:** ✅ default pick · 🟡 use when the situation matches · ⛔ skip in a datathon

---

## 1. Pick the Algorithm Family (first match wins)

| Your data / task | Start with | Then | Skip |
| --- | --- | --- | --- |
| Tabular, classification or regression | LightGBM | CatBoost, then blend | Deep nets, SVM on large data |
| Tabular, many high-cardinality categoricals | CatBoost | LightGBM with target encoding **inside folds** | One-hot on 1000+ levels |
| Tiny data (< 5k rows) | Regularized linear / logistic | Random Forest, ExtraTrees | Boosting with defaults, big ensembles |
| Time series / forecasting | Lag + rolling features into LightGBM | Prophet / statsforecast baselines | Shuffled CV, LSTMs as a first move |
| Anomaly / fraud, few labels | Isolation Forest + supervised GBM if labels exist | Autoencoder if > 100k rows | Accuracy as a metric |
| Text column | TF-IDF + logistic regression | Sentence-embedding + LightGBM | Fine-tuning a transformer in < 12 h |
| Image data | Pretrained backbone, frozen embeddings + linear head | Light fine-tune | Training from scratch |
| Segmentation / grouping, no label | KMeans / HDBSCAN on scaled features | UMAP for visualization | Reporting clusters without a business label |
| Recommendation / similarity | Item-item cosine on interactions | Implicit ALS, embeddings | Neural CF as first model |
| Uplift / "who to target" | Two-model (T-learner) with GBM | CausalML / EconML | Plain propensity as if it were causal |

> Rule: the model that wins on your locked OOF folds wins. Never defend a model by its fame.

---

## 2. Tool Map by Job

### Data & Features
| Job | ✅ Default | 🟡 Situational | ⛔ Skip |
| --- | --- | --- | --- |
| Load/transform big CSV/Parquet | Polars | DuckDB for SQL joins | Pandas > 1 GB |
| Auto-profile | ydata-profiling | Sweetviz | Hand-written plot loops |
| Validate schema | pandera / `assert` | Great Expectations | Nothing at all |
| Version data | Git + dated Parquet snapshots | DVC if the data is large and shared | Emailing CSVs |
| Notebook | Jupyter / marimo | VS Code notebooks | Only notebooks, no `.py` pipeline |

### Modeling
| Job | ✅ Default | 🟡 Situational | ⛔ Skip |
| --- | --- | --- | --- |
| Pipeline & CV | scikit-learn | `sklearn.pipeline.Pipeline` + `ColumnTransformer` | Ad-hoc scripts that fit on full data |
| Gradient boosting | LightGBM | CatBoost, XGBoost | Untuned deep nets |
| AutoML | AutoGluon | FLAML (very fast budgets) | Running AutoML before locking folds |
| Hyperparameters | Optuna with `timeout=` | Hyperopt | GridSearchCV |
| Class imbalance | `scale_pos_weight`, threshold tuning | imbalanced-learn | SMOTE as the first move |
| Calibration | `CalibratedClassifierCV` (isotonic/Platt) | `venn-abers` | Using raw scores as probabilities for $ math |

### Explainability & Trust
| Job | ✅ Default | 🟡 Situational | ⛔ Skip |
| --- | --- | --- | --- |
| Tree model explanations | SHAP `TreeExplainer` | LightGBM `pred_contrib=True` | Plain MDI importance only |
| Partial dependence / what-if | `sklearn.inspection` PDP/ICE | Google What-If Tool | Static screenshots |
| Fairness slices | OOF metrics per cohort | Fairlearn | Claiming "unbiased" without a check |
| Deep-model attribution | Captum (PyTorch) | | Anything for a tree model |

### Experiment Tracking & MLOps (Keep It Tiny)
| Job | ✅ Default | 🟡 Situational | ⛔ Skip |
| --- | --- | --- | --- |
| Track runs | `feature_log.csv` + `config.yaml` in Git | MLflow local / Weights & Biases free tier | Kubeflow, Airflow, Spark clusters |
| Serve the model | Load `.pkl`/`.joblib` in Streamlit | FastAPI endpoint | Model-serving platforms (BentoML, Ray Serve, vLLM) for tabular |
| Distributed | Single machine | Ray only if a single machine truly cannot finish | Horovod, DeepSpeed |

> The big MLOps and distributed-training sections in large ML guides are **not** where datathons are won. Judges reward leak-free validation and decisions, not infrastructure.

### Deep Learning (Only When the Data Demands It)
| Data | ✅ Default | Notes |
| --- | --- | --- |
| Images | PyTorch + `timm` pretrained | Freeze backbone first |
| Text | Hugging Face Transformers / `sentence-transformers` | Embed once, cache to Parquet |
| Audio | `torchaudio` + pretrained encoder | Resample consistently |
| Time series | PyTorch Forecasting / Darts | Beat a seasonal-naive baseline first |

---

## 3. The "Beat the Baseline" Ladder

Run in order and stop climbing when the gain per hour drops below your team's next-best use of time.

1. **Dummy** (constant / prior) → know the floor.
2. **Seasonal-naive / last-value** for time series.
3. **Linear / logistic** with sensible scaling.
4. **Default LightGBM** on locked folds.
5. **Feature families** one at a time (log every Δ).
6. **Tuned LightGBM + CatBoost**.
7. **Blend** by OOF rank-average.
8. **Calibrate + choose threshold** on OOF.

Every rung must beat the rung below on **OOF**, not on the public leaderboard.

---

## 4. Learning Resources (Curated Short List)

Use these to fill a specific gap, not to binge. Starred entries are named in [`mikeroyal/Machine-Learning-Guide`](https://github.com/mikeroyal/Machine-Learning-Guide).

| Need | Resource |
| --- | --- |
| Hands-on, lesson-by-lesson with runnable code (MIT) | [`rohitg00/ai-engineering-from-scratch`](https://github.com/rohitg00/ai-engineering-from-scratch), `phases/02-ml-fundamentals/` |
| ML basics, free ⭐ | [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course/) |
| Structured course ⭐ | [Stanford Machine Learning (Coursera)](https://www.coursera.org/learn/machine-learning) |
| Cloud-oriented path ⭐ | [AWS Machine Learning learning path](https://aws.amazon.com/training/learning-paths/machine-learning/) |
| Deep learning framework tutorials ⭐ | [PyTorch tutorials](https://pytorch.org/tutorials/) · [TensorFlow tutorials](https://www.tensorflow.org/tutorials) |
| Wide discovery list | [`mikeroyal/Machine-Learning-Guide`](https://github.com/mikeroyal/Machine-Learning-Guide) (19 sections: frameworks, algorithms, CV, NLP, RL, per-language) |

Vet any new resource with the **Source Trust Ladder** in [`../00-datathon-runbook.md`](../00-datathon-runbook.md): license, last commit, runs end-to-end.

---

## 5. Fast Decision Checklist

- [ ] Is a tree model on a locked CV already the baseline? If not, do that before anything fancy.
- [ ] Does the new tool or model beat the baseline on **OOF** by more than fold-to-fold noise?
- [ ] Can a teammate reproduce it with one command and `requirements.txt`?
- [ ] Can you explain it to a judge in 20 seconds?

If any answer is no, drop it.
