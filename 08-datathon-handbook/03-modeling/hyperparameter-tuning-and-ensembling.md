# ⚡ Hyperparameter Tuning & Ensembling Under the Clock

<!-- markdownlint-disable MD013 -->

> 🟢 **In plain English:** In a 24-hour datathon, spending 6 hours running brute-force grid searches is a recipe for defeat. Winning teams use **time-capped Bayesian optimization** and **rank-based ensembling** to boost their score in the final hours.

## 🧁 What tuning actually is (read this first)

A model has **parameters it learns** from the data (for a tree: where to split) and **settings you choose before training** (how deep a tree may grow, how big each learning step is). Those settings are **hyperparameters**. Tuning means trying different settings and keeping the ones that score best on validation data.

Think of baking: the recipe is the model; oven temperature and baking time are hyperparameters. You cannot learn them from the flour, so you try a few combinations and taste the result (the validation score).

**When to tune:** after features are frozen (runbook Gate 6). Features usually move the score more than tuning, and every feature change makes old tuning results stale. Budget **10-20 minutes** of compute, not hours.

**How to explain it to a judge:** *"We capped tuning at 10 minutes and used Bayesian search over the settings that control tree complexity and randomness, on the same leak-free folds as everything else. Most of our gain came from features, not tuning."*

---

## ⏱️ Tuning Rule #1: Never Use GridSearchCV!

Standard `GridSearchCV` tries every combination. Five values for each of four settings is 5⁴ = 625 models; with 5-fold CV that is 3,125 trainings, most of them on settings that were obviously bad after the first few results.

### What to Use Instead:
- **FLAML (Fast & Lightweight AutoML):** Automatically finds near-optimal hyperparameters within a strict budget (e.g. `time_budget=300` seconds).
- **Optuna with a time cap:** Bayesian search: it looks at the scores so far and proposes the next settings near the good ones. `timeout=` stops it after a fixed time, and pruning can abandon clearly bad trials early. (Optuna 5's default sampler is multivariate TPE, so results differ from older tutorials.)

---

## 🎯 The Four Settings Worth Tuning First

Don't tune 20 parameters. As a practical rule of thumb (our experience, not a measured figure), these four, plus a minimum leaf size, capture most of what tuning can give a gradient-boosted model:

| Parameter | LightGBM Name | CatBoost Name | Recommended Search Range | What It Controls |
| --- | --- | --- | --- | --- |
| **Tree Depth / Size** | `num_leaves` | `depth` | `num_leaves`: [15, 63]<br>`depth`: [4, 8] | Model complexity (lower = less overfitting) |
| **Step Size** | `learning_rate` | `learning_rate` | [0.03, 0.10] | Shrinkage per tree (0.05 is the sweet spot) |
| **Feature Sampling** | `colsample_bytree` | `rsm` | [0.6, 0.9] | Fraction of columns sampled per tree |
| **Row Sampling** | `subsample` (+ `subsample_freq=1`) | `subsample` | [0.6, 0.95] | Fraction of rows sampled per tree |
| **Minimum leaf size** | `min_child_samples` | `min_data_in_leaf` | [20, 100] | Higher = smoother, safer trees |

**What each one does, and what goes wrong at either end:**

| Setting | Too low | Too high |
| --- | --- | --- |
| Tree size (`num_leaves` / `depth`) | Misses patterns (underfits) | Memorises noise (overfits) |
| `learning_rate` | Needs many trees; slow | Overshoots; unstable |
| Row and column sampling | Each tree too noisy | All trees see the same data and lean on the same top feature, so they overfit together |
| Minimum leaf size | Leaves built on a handful of rows (noise) | Model too coarse to see real patterns |

The sampling settings add deliberate randomness so trees make *different* mistakes that average out. The number of trees is **not** tuned: set a high cap and let early stopping pick it, because it depends on the learning rate.

---

## 🛠️ Rapid Tuning Recipe with Optuna (10-Minute Cap, Locked Folds)

Three rules make tuning honest:

1. **Reuse the locked folds from Gate 2** (`folds.csv` / `folds.parquet`). Fresh random folds leak when the same patient, user or store appears in several rows, and the search then chases a fake score.
2. **Early stopping picks the number of trees**, using a high cap.
3. **Don't report the best tuning score.** Trying many settings and keeping the best is itself a form of fitting, so that number is optimistic. Report the score of one clean re-run with the chosen settings, ideally checked on a holdout you never tuned on.

```python
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score

# X, y: features and target (pandas). folds: a Series of fold numbers aligned with X,
# loaded from the file you saved at Gate 2, e.g.
#   folds = pd.read_csv("outputs/folds.csv")["fold"]
METRIC = average_precision_score          # PR-AUC; use roc_auc_score etc. to match your brief
# text columns (pandas 3 gives them dtype "str", not "object", so test "not numeric")
CAT_COLS = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]


def cv_score(params, return_oof=False):
    oof = np.zeros(len(y))
    for k in sorted(folds.unique()):
        tr, va = (folds != k).values, (folds == k).values
        Xtr, Xva = X[tr].copy(), X[va].copy()
        for c in CAT_COLS:  # categories learned from the training fold only
            cats = Xtr[c].astype("category").cat.categories
            Xtr[c] = Xtr[c].astype(pd.CategoricalDtype(cats))
            Xva[c] = Xva[c].astype(pd.CategoricalDtype(cats))
        model = lgb.LGBMClassifier(n_estimators=2000, random_state=42, verbose=-1, **params)
        model.fit(Xtr, y[tr], eval_X=(Xva,), eval_y=(y[va],),   # LightGBM 4.7+ API
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va] = model.predict_proba(Xva)[:, 1]
    return (METRIC(y, oof), oof) if return_oof else METRIC(y, oof)


def objective(trial):
    return cv_score({
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 0.95),
        "subsample_freq": 1,                       # without this, subsample does nothing
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
    })


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.enqueue_trial({"learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20,
                     "subsample": 0.8, "colsample_bytree": 0.8})   # start from sane defaults
study.optimize(objective, timeout=600)        # stops after 10 minutes

default_score = cv_score({"subsample_freq": 1, "subsample": 0.8, "colsample_bytree": 0.8})
tuned_score, tuned_oof = cv_score({**study.best_params, "subsample_freq": 1}, return_oof=True)
print(f"trials run: {len(study.trials)}")
print(f"default settings OOF: {default_score:.4f} | tuned OOF: {tuned_score:.4f}")
print("best params:", study.best_params)
# Keep the tuned settings only if the gain is bigger than fold-to-fold noise.
# Log both numbers in feature_log.csv / PROJECT_CONTEXT.md; save tuned_oof for blending.
```

**Reading the output:** if the tuned score beats the defaults by less than the spread between folds, tuning didn't really help: keep the defaults and spend the time on features or the demo. `tuned_score` is still slightly optimistic because it was selected on these folds; for a strict number, score once on a holdout you never touched.

**Try it on the sample data:** with the runnable example's data ([`07-worked-example/`](../07-worked-example/README.md)) and a 60-second cap, our run (2026-10-03, Python 3.12, pinned requirements) completed 212 trials and lifted LightGBM's grouped OOF PR-AUC from **0.200 to 0.238**. Logistic regression still scores **0.254** on that data, so the baseline ladder keeps the simpler model: tuning cannot rescue the wrong model family. This code block is executed by `tools/smoke_test.py` so it stays runnable.

**Rare positive class?** Imbalance-specific additions (minimum hessian in leaf, `first_metric_only`, class-balanced bagging, whether to search `scale_pos_weight`) are in [`lightgbm-imbalance-and-tuning.md`](lightgbm-imbalance-and-tuning.md).

**Prompt shortcut:** [`DT18`](../08-ai-agent-kit/PROMPTS.md) asks an AI assistant to generate this with your folds, metric and time budget.

---

## 🤝 Winning Ensembling Techniques

Combining predictions from multiple distinct model architectures (e.g. LightGBM + CatBoost + XGBoost) almost always outperforms any single model.

### Technique 1: Rank-Averaging (The Secret Weapon)
Raw predicted probabilities from different models often have different calibrations (e.g., CatBoost might output probabilities between 0.10 and 0.40, while LightGBM outputs 0.02 to 0.85). If you take a simple arithmetic average, the model with higher variance will dominate.

**The Solution:** Convert probabilities to percentiles (ranks) before averaging!

```python
import numpy as np
from scipy.stats import rankdata

def rank_average_predictions(preds_list, weights=None):
    """
    Blends multiple prediction arrays by ranking them first.
    Prevents miscalibrated probabilities from breaking the ensemble.
    """
    if weights is None:
        weights = [1.0 / len(preds_list)] * len(preds_list)
        
    ranked_sum = np.zeros(len(preds_list[0]))
    for pred, w in zip(preds_list, weights):
        # Normalize ranks between 0 and 1
        normalized_ranks = rankdata(pred) / len(pred)
        ranked_sum += normalized_ranks * w
        
    return ranked_sum

# Usage
ensemble_preds = rank_average_predictions(
    [lgb_test_preds, catboost_test_preds, xgb_test_preds],
    weights=[0.45, 0.35, 0.20]
)
```

### Technique 2: Out-of-Fold Stacking
Train a meta-classifier (like Logistic Regression or Ridge) using the Out-Of-Fold predictions of your base models as inputs:

```python
from sklearn.linear_model import LogisticRegression

# 1. Stack OOF predictions from Phase 2
X_meta_train = np.column_stack([oof_lgb, oof_catboost, oof_xgb])
X_meta_test = np.column_stack([test_lgb, test_catboost, test_xgb])

# 2. Train simple meta-model
# (to *score* the stack honestly, fit the meta-model inside the same locked folds;
#  fitting it on all OOF rows and scoring on those rows is optimistic)
meta_model = LogisticRegression(C=1.0)
meta_model.fit(X_meta_train, y_train)

# 3. Final ensemble prediction
final_stacked_preds = meta_model.predict_proba(X_meta_test)[:, 1]
```
This multi-layer architecture is why AutoGluon consistently tops tabular benchmarks!
