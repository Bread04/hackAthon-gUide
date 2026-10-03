# 📚 ML Models Explained: From Linear Regression to Gradient Boosting

<!-- markdownlint-disable MD013 -->

> Every classic machine-learning model a hackathon or datathon team is likely to use, explained in plain English: what it does, how it learns, when to use it, the settings that matter, and what goes wrong. All of them are trained side by side on the guide's sample data by [`07-worked-example/model_zoo.py`](../07-worked-example/model_zoo.py), so the results quoted below are real (tested 2026-10-03, Python 3.12, pinned [`requirements.txt`](../PROJECT_TEMPLATE/requirements.txt)).
>
> Read [`ml-fundamentals-and-metrics.md`](ml-fundamentals-and-metrics.md) first if words like *overfitting*, *validation* or *PR-AUC* are new. To choose a model for a task, use [`method-selection-guide.md`](method-selection-guide.md); this page explains the models themselves. Background references: the [scikit-learn user guide](https://scikit-learn.org/stable/user_guide.html) and the [INRIA scikit-learn MOOC](https://inria.github.io/scikit-learn-mooc/).

---

## 🗺️ The map: every model on one table

| Model | Task | Learns… | Needs scaling? | Handles missing values? | Explainable? | Speed | Use it when… |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Dummy** | Both | Nothing (mean / base rate) | No | Yes | Trivially | Instant | Always, as the floor every model must beat |
| **Linear regression** | Regression | A weighted sum of features | Yes, for comparable coefficients | No (impute) | ✅ Coefficients | Instant | First regression baseline; relationships are roughly additive |
| **Ridge / Lasso / Elastic Net** | Regression | A weighted sum, with weights pulled toward zero | **Yes** | No | ✅ | Instant | Many or correlated features; Lasso also drops useless ones |
| **GLMs (Poisson, Gamma, Tweedie)** | Regression | A weighted sum through a link (e.g. log) | Yes | No | ✅ | Fast | Counts, positive amounts, insurance-style targets |
| **Logistic regression** | Classification | A weighted sum turned into a probability | **Yes** | No | ✅ Odds ratios | Instant | First classification baseline; you need probabilities and explanations |
| **Naive Bayes** | Classification | Per-class feature distributions | No | No | Partly | Instant | Text (word counts), tiny data, very fast baselines |
| **k-nearest neighbours** | Both | Nothing: it stores the data and compares | **Yes** | No | By example ("these 5 similar cases") | Slow to predict on big data | Small data, similarity search, recommendations |
| **Support vector machine** | Both | A maximum-margin boundary, optionally curved (kernel) | **Yes** | No | ❌ (kernel) | Slow above ~50k rows | Small-to-medium, high-dimensional data (text, genomics) |
| **Decision tree** | Both | Yes/no questions on features | No | Some implementations | ✅ Draw it | Fast | Explaining rules to judges; never as your best model |
| **Random forest** | Both | Many deep trees on random samples, averaged | No | Recent scikit-learn, partly | Partly (importances) | Medium | Robust no-tuning baseline |
| **Extra trees** | Both | Like random forest, with random split points | No | Partly | Partly | Fast | Noisy data; quick strong baseline |
| **AdaBoost** | Both | Trees trained in sequence, re-weighting mistakes | No | No | Partly | Medium | Mostly historical; gradient boosting usually wins |
| **Gradient boosting** (HistGradientBoosting, XGBoost, LightGBM, CatBoost) | Both | Trees in sequence, each fixing the last one's errors | No | ✅ Native | Partly (SHAP) | Fast | **Default for tabular data** once a linear baseline exists |
| **Neural network (MLP)** | Both | Layers of weighted sums with non-linearities | **Yes** | No | ❌ | Medium | Mostly for images, text, audio; rarely best on small tables |
| **k-means, GMM, DBSCAN/HDBSCAN** | Clustering | Groups without labels | **Yes** | No | By cluster profile | Fast | Segmentation, exploration |
| **PCA, t-SNE, UMAP** | Dimension reduction | A few new axes that summarise many features | **Yes** | No | PCA partly | Fast | Visualising, compressing, de-noising |
| **Isolation forest, LOF, one-class SVM** | Anomaly detection | What "normal" looks like | IF: no; others: yes | No | Partly | Fast | Fraud, faults, data-quality checks with few or no labels |

**Rule of thumb for a hackathon:** dummy → linear or logistic regression → one tree ensemble (LightGBM or CatBoost) → keep whichever wins on *your* folds. The rest of this page explains why each one behaves the way it does.

---

## 🧪 What happened when we trained all of them

[`model_zoo.py`](../07-worked-example/model_zoo.py) trains each model on the same grouped folds (no patient in two folds) with preprocessing fitted inside each fold. About 25 seconds on a laptop.

### Classification: will this admission be readmitted within 30 days? (11.2% positive)

| Model | PR-AUC | ROC-AUC | Time (s) |
| --- | --- | --- | --- |
| Logistic regression (L1, C=0.1) | 0.259 | 0.701 | 0.09 |
| Logistic regression | 0.255 | 0.700 | 0.09 |
| Naive Bayes (Gaussian) | 0.250 | 0.689 | 0.08 |
| k-nearest neighbours (k=50) | 0.211 | 0.668 | 0.16 |
| Extra trees | 0.210 | 0.677 | 2.90 |
| AdaBoost | 0.210 | 0.660 | 1.86 |
| CatBoost | 0.207 | 0.658 | 1.00 |
| LightGBM | 0.197 | 0.631 | 0.54 |
| XGBoost | 0.196 | 0.646 | 0.41 |
| Random forest | 0.196 | 0.654 | 3.69 |
| SVM (RBF kernel) | 0.192 | 0.551 | 1.64 |
| Decision tree (depth 4) | 0.178 | 0.645 | 0.08 |
| HistGradientBoosting | 0.176 | 0.608 | 1.05 |
| Decision tree (no depth limit) | 0.119 | 0.525 | 0.09 |
| Dummy (base rate) | 0.112 | 0.499 | 0.09 |
| Neural net (MLP 32-16) | 0.107 | 0.469 | 0.33 |

### Regression: how many medications at discharge?

| Model | MAE | RMSE | R² |
| --- | --- | --- | --- |
| Lasso (alpha=0.05) | 2.064 | 2.610 | 0.203 |
| Ridge (alpha=10) | 2.074 | 2.612 | 0.202 |
| Linear regression | 2.074 | 2.612 | 0.202 |
| Poisson regression (GLM) | 2.084 | 2.624 | 0.195 |
| Decision tree (depth 4) | 2.103 | 2.647 | 0.180 |
| Random forest | 2.145 | 2.702 | 0.146 |
| LightGBM | 2.145 | 2.703 | 0.145 |
| k-nearest neighbours (k=30) | 2.140 | 2.704 | 0.144 |
| HistGradientBoosting | 2.182 | 2.760 | 0.109 |
| Dummy (mean) | 2.313 | 2.925 | -0.001 |

### How to read these results honestly

- ⚠️ **Linear models win here because of how the data was made.** The sample data was *generated* from a linear (logistic) formula, so a linear model is the right shape for it. On real tabular data with interactions and thresholds, gradient-boosted trees usually lead (see the evidence in [`method-selection-guide.md`](method-selection-guide.md)). This is exactly why you run the comparison on your own folds instead of trusting any leaderboard, including this one.
- **The unlimited decision tree is barely better than guessing (0.119 vs 0.112)**: it memorised the training rows. Limiting depth to 4 nearly doubles its useful skill. That gap is overfitting in one line.
- **The small neural net scored below the dummy.** With ~2,600 rows, an imbalanced target and default settings, it never learned. Neural nets need more data and tuning; they are not a free upgrade.
- **SVM ranks well on PR-AUC but poorly on ROC-AUC (0.551).** Different metrics can disagree; report the one that matches your task, next to its baseline.
- **Every regression model beats the dummy**, and linear regression's coefficient for chronic conditions (+1.38 medications per standard deviation) matches the rule the data was generated with: one extra medication per condition, and the standard deviation of conditions is 1.32. That is a good sanity check: a simple model recovered the true relationship.

---

## 0. The dummy model (always first)

**What it does:** predicts the average (regression) or the base rate (classification), ignoring every feature.

**Why it matters:** it is the score you get for doing nothing. If your model doesn't clearly beat it, the model has learned nothing useful, whatever the headline number says. With 11% positives, a dummy gets 89% accuracy; it gets PR-AUC 0.112 and ROC-AUC 0.5.

**scikit-learn:** `DummyClassifier(strategy="prior")`, `DummyRegressor()`.

---

## 1. Linear regression

**Intuition:** draw the best straight line (or flat plane, with many features) through the data. Prediction = intercept + w₁·x₁ + w₂·x₂ + …

**How it learns:** chooses weights that minimise the **sum of squared errors** between predictions and true values (ordinary least squares). There is an exact formula, so training is instant.

**Reading it:** each weight says how much the prediction changes when that feature goes up by one unit, *holding the others fixed*. Standardise features first and the weights become comparable ("per +1 standard deviation"); `model_zoo.py` section C prints them.

**Use it when:** you need a fast regression baseline, or an explanation judges can follow.

**Watch out for:**
- **Outliers** pull the line strongly, because errors are squared. Consider `HuberRegressor` or a log-transformed target.
- **Correlated features** make individual weights unstable, even when predictions are fine. Use Ridge.
- **Curves and interactions** aren't captured unless you add them as features (e.g. `age²`, `age × conditions`).
- **Coefficients are associations, not causes.**
- A skewed positive target (prices, durations) often works better on `log1p(y)`; transform predictions back with `expm1`.

**scikit-learn:** `LinearRegression()`.

---

## 2. Ridge, Lasso and Elastic Net (regularised linear models)

**Intuition:** linear regression with a "keep it simple" penalty that pulls weights toward zero, trading a little fit on training data for better behaviour on new data.

| Variant | Penalty | Effect | Key setting |
| --- | --- | --- | --- |
| **Ridge** (L2) | Sum of squared weights | Shrinks all weights; stable with correlated features | `alpha` (higher = simpler) |
| **Lasso** (L1) | Sum of absolute weights | Sets some weights exactly to zero, so it selects features | `alpha` |
| **Elastic Net** | Mix of both | Selection plus stability | `alpha`, `l1_ratio` |

**Use it when:** you have many features, correlated features, or more features than you'd like to explain. Always scale features first: the penalty treats every weight equally, so units matter.

**Choosing `alpha`:** use `RidgeCV`, `LassoCV` or `ElasticNetCV`, which pick it by cross-validation. Grouped or time data needs your own folds passed via `cv=`.

**In our test:** Lasso edged out plain linear regression (MAE 2.064 vs 2.074), a small gain that's typical when a few features are noise.

**scikit-learn:** `Ridge`, `Lasso`, `ElasticNet` (+ `…CV` versions).

---

## 3. Generalised linear models (Poisson, Gamma, Tweedie) and quantile regression

**Intuition:** a linear model whose output goes through a *link function*, so predictions respect the target's shape. Poisson uses a log link, so predictions are always positive and effects multiply instead of add.

| Target looks like | Model | scikit-learn |
| --- | --- | --- |
| Counts (visits, defects, medications) | Poisson regression | `PoissonRegressor` |
| Positive, right-skewed amounts (claim size, time to event) | Gamma regression | `GammaRegressor` |
| Many exact zeros plus positive amounts (insurance, sales) | Tweedie (power between 1 and 2) | `TweedieRegressor(power=1.5)`; LightGBM `objective="tweedie"` |
| You need a range, not one number ("90th percentile wait time") | Quantile regression | `QuantileRegressor`, or LightGBM `objective="quantile"` |

**In our test:** Poisson was close to linear regression on the medication count (MAE 2.084 vs 2.074). It guarantees non-negative predictions, which a plain linear model doesn't.

---

## 4. Logistic regression (the classification baseline)

**Intuition:** the same weighted sum as linear regression, squeezed through an S-shaped curve (the sigmoid) into a probability between 0 and 1. Despite the name, it is a classifier.

**How it learns:** chooses weights that make the observed labels as likely as possible (minimises **log loss**). There's no exact formula, so it uses a fast iterative solver.

**Reading it:** `exp(weight)` is an **odds ratio**: the factor by which the odds of the positive class multiply when the feature rises by one unit (or one standard deviation, if scaled). In our sample, +1 SD of age multiplies the odds of readmission by 1.45 (`model_zoo.py` section C). An odds ratio of 1.0 means no effect.

**Use it when:** always, as the first classification baseline. It's fast, explainable and gives probabilities that are usually well calibrated, as long as you don't use `class_weight="balanced"`; see [calibration](ml-fundamentals-and-metrics.md).

**Key settings:** `C` = inverse penalty strength (smaller = simpler). Since scikit-learn 1.8, choose the penalty type with `l1_ratio` (0 = L2, 1 = L1) rather than `penalty=`. Raise `max_iter` if it warns about convergence. Scale features.

**Multi-class:** works out of the box for more than two classes (multinomial).

**In our test:** the winner on this (linearly generated) data, PR-AUC 0.255; the L1 version scored 0.259.

**scikit-learn:** `LogisticRegression(max_iter=2000)`.

---

## 5. Naive Bayes

**Intuition:** for each class, learn what each feature typically looks like, then ask which class makes this row most likely. "Naive" because it assumes features are independent given the class, which is rarely true but often good enough.

**Variants:** `GaussianNB` (numeric features), `MultinomialNB` / `ComplementNB` (word counts, the classic spam filter), `BernoulliNB` (yes/no features).

**Use it when:** text classification baselines, tiny datasets, or when you need something that trains in milliseconds.

**Watch out for:** its probabilities are usually over-confident (pushed toward 0 or 1). Rank with it, but calibrate before quoting probabilities.

**In our test:** PR-AUC 0.250, almost matching logistic regression in 0.08 s.

---

## 6. k-nearest neighbours (kNN)

**Intuition:** to predict a new row, find the *k* most similar rows in the training data and take their majority class or average value. No training at all: the "model" is the data.

**Key settings:** `n_neighbors` (small k = jumpy, overfits; large k = smooth), the distance metric, and `weights="distance"` to trust closer neighbours more.

**Use it when:** small data, "find similar cases" features for a demo ("patients like this one…"), and recommendations.

**Watch out for:**
- **Scaling is essential.** Otherwise a feature measured in thousands dominates the distance.
- Prediction gets slow on large data, because every prediction compares against all rows.
- It degrades with many features (the "curse of dimensionality").

**In our test:** PR-AUC 0.211 with k=50.

---

## 7. Support vector machines (SVM)

**Intuition:** find the boundary that separates classes with the **widest margin**. With a *kernel* (RBF is the usual default), the boundary can curve.

**Key settings:** `C` (higher = fits training data harder), `gamma` for RBF (higher = wigglier boundary). Both need tuning, and features must be scaled.

**Use it when:** small-to-medium datasets with many features (text, genomics). `LinearSVC` scales well for text.

**Watch out for:** training time grows quickly beyond tens of thousands of rows. Probabilities need `probability=True`, which adds an internal calibration step and makes it slower.

**In our test:** decent PR-AUC (0.192) but weak ROC-AUC (0.551), a reminder that metrics can disagree.

**scikit-learn:** `SVC`, `LinearSVC`, `SVR`.

---

## 8. Decision tree

**Intuition:** a flowchart of yes/no questions ("age > 65?", "prior admissions ≥ 2?") learned from the data. Each leaf gives a prediction.

**How it learns:** at each step, picks the question that best separates the targets (Gini impurity or entropy for classification, squared error for regression), then repeats inside each branch.

**Key settings:** `max_depth`, `min_samples_leaf`. Without limits, a tree keeps splitting until it memorises the data.

**Use it when:** you want rules a judge can read (`sklearn.tree.plot_tree` or `export_text`), or as the building block for the ensembles below. A single tree is rarely your best model.

**In our test:** unlimited depth 0.119 (barely above the 0.112 dummy); depth 4 scored 0.178. That is overfitting made visible.

**scikit-learn:** `DecisionTreeClassifier`, `DecisionTreeRegressor`.

---

## 9. Random forest and Extra trees (bagging)

**Intuition:** train hundreds of deep trees, each on a random sample of rows and considering a random subset of features at each split, then average them. Individual trees overfit in *different* ways, and averaging cancels much of it out ("wisdom of crowds").

**Extra trees** ("extremely randomised trees") also pick split points at random, which makes them faster and sometimes better on noisy data.

**Key settings:** `n_estimators` (more is better until it stops helping; 300-500 is usually plenty), `min_samples_leaf` (raise it for smoother predictions), `max_features`.

**Use it when:** you want a strong model with almost no tuning. It's hard to break.

**Watch out for:** large files and slower prediction with many deep trees. The built-in `feature_importances_` favour features with many distinct values; prefer permutation importance or SHAP for explanations.

**In our test:** random forest 0.196, Extra trees 0.210, with no tuning.

---

## 10. Boosting: AdaBoost and gradient boosting

**Intuition:** build small trees **one after another**, each trying to fix the mistakes of the trees so far, then add them up. Where bagging averages independent trees, boosting chains dependent ones.

**AdaBoost** re-weights the rows the previous trees got wrong. It's historically important, but gradient boosting has largely replaced it.

**Gradient boosting** fits each new tree to the *remaining error* of the current model (technically the gradient of the loss). It's the strongest general-purpose family for tabular data. Four implementations:

| Library | Strength | Note |
| --- | --- | --- |
| `HistGradientBoosting` (scikit-learn) | No extra install; native missing values and categoricals | Good default inside scikit-learn pipelines |
| **LightGBM** | Very fast on CPU; leaf-wise trees | Watch `num_leaves` to avoid overfitting small data |
| **XGBoost** | Mature, widely used; strong regularisation options | Level-wise trees by default |
| **CatBoost** | Best out of the box with categorical features (ordered target statistics) | Often needs the least tuning |

**Key settings:** `learning_rate` together with the number of trees (use early stopping), tree size (`num_leaves` / `max_depth`), row and column sampling, and minimum leaf size. See [`hyperparameter-tuning-and-ensembling.md`](hyperparameter-tuning-and-ensembling.md).

**Use it when:** tabular data after your linear baseline. This family wins most real tabular competitions.

**Watch out for:** overfitting on small data with default settings, as in our test, where CatBoost 0.207, LightGBM 0.197 and XGBoost 0.196 all trailed logistic regression on linearly generated data. Explain them with SHAP, and keep the leak-free folds.

---

## 11. Neural networks (multi-layer perceptron)

**Intuition:** layers of weighted sums, each followed by a simple non-linear function (such as ReLU), stacked so the network can learn complicated shapes. Trained by gradient descent on the loss (backpropagation).

**Key settings:** layer sizes, `alpha` (penalty), `learning_rate_init`, `early_stopping=True`. Features must be scaled.

**Use it when:** images, text, audio or very large datasets, usually by *fine-tuning a pretrained network* ([`deep-learning-quickstart.md`](deep-learning-quickstart.md)) rather than training an MLP from scratch on a small table.

**In our test:** below the dummy (0.107). On small tables, a plain MLP is rarely worth the tuning time; modern tabular nets (TabM, RealMLP) and tabular foundation models are covered in [`method-selection-guide.md`](method-selection-guide.md).

---

## 12. Interpretable-but-strong: GAMs and EBMs

**Intuition:** a sum of *one curve per feature* (plus a few pairwise interactions). Each feature's effect can be plotted exactly, yet the model captures non-linear shapes.

**Use it when:** judges or users must trust the model (health, finance) and logistic regression is too rigid. Explainable Boosting Machines (EBMs) from [`interpretml`](https://github.com/interpretml/interpret) often get close to gradient boosting on tabular data. They aren't in the pinned requirements; install separately.

---

## 13. Clustering (no labels)

| Model | Finds | Must choose | Watch out for |
| --- | --- | --- | --- |
| **k-means** | Round-ish groups around centres | `k`, the number of groups | Scale features; try several `k` and seeds |
| **Gaussian mixture (GMM)** | Overlapping elliptical groups, with soft membership probabilities | Number of components (compare BIC: lower is better) | Slower; can over-fit components |
| **DBSCAN / HDBSCAN** | Groups of any shape, plus "noise" points | Density settings (`eps`, `min_samples`; HDBSCAN needs fewer choices) | Noise points can flatter metrics |
| **Hierarchical (agglomerative)** | A tree of merges you can cut at any level | Linkage and cut level | Slow on large data |

**Judging a clustering:** there's no right answer without labels. Use the silhouette score (−1 to 1, higher = better-separated) to compare settings, check that clusters are stable across random seeds, and, most important, **describe each cluster in business terms** (its median feature values). In our sample data, silhouette scores were low (≈0.17 for k = 2-6), which is honest evidence there are no strong natural groups. Saying that is better than forcing segments.

---

## 14. Dimension reduction

- **PCA:** finds a few new axes (combinations of features) that keep most of the variance. Use it for plotting, de-noising or compressing correlated features. In our sample, 5 components keep 78% of the variance. Scale first.
- **t-SNE / UMAP:** non-linear maps for **visualising** high-dimensional data (e.g. embeddings) in 2-D. Distances between far-apart clusters in these plots are not meaningful; don't feed t-SNE output into a model. UMAP needs the separate `umap-learn` package.

---

## 15. Anomaly detection

| Model | Idea | Notes |
| --- | --- | --- |
| **Isolation forest** | Anomalies are easy to isolate with few random splits | Fast, no scaling needed; set `contamination` to the expected anomaly share |
| **Local outlier factor (LOF)** | Points in much sparser neighbourhoods than their neighbours | Scale first; `novelty=True` to score new data |
| **One-class SVM** | A boundary around "normal" data | Sensitive to settings; slower |
| **ECOD and others (PyOD)** | Statistical tail scores | See [`baseline-recipes.md`](baseline-recipes.md) |

In our sample, Isolation forest flagged 53 admissions (2%). They had far more conditions (median 4 vs 2), medications (11 vs 7) and prior admissions (3 vs 1) than average, which is a sensible, explainable result. With a few labels, check PR-AUC against them; with none, read the flagged rows by hand.

---

## 16. Ensembles of different models

- **Voting / averaging:** average the predictions of different model types (e.g. logistic regression + CatBoost). Use rank-averaging when probability scales differ.
- **Stacking:** a simple model learns how to combine the others' out-of-fold predictions. AutoGluon automates this.

Both help most when the models make *different* mistakes. Details and code: [`hyperparameter-tuning-and-ensembling.md`](hyperparameter-tuning-and-ensembling.md).

---

## 17. Beyond this page

| Problem | Go to |
| --- | --- |
| Time series and forecasting (seasonal naive, ETS, ARIMA, LightGBM with lags) | [`method-selection-guide.md`](method-selection-guide.md) · [`baseline-recipes.md`](baseline-recipes.md) |
| Images, text, audio (CNNs, transformers, transfer learning) | [`deep-learning-quickstart.md`](deep-learning-quickstart.md) |
| Uplift / "who to target" (T-learner, causal forests) | [`baseline-recipes.md`](baseline-recipes.md) |
| LLMs, RAG and agents | [`../../04-ai-and-rag/docs/how-llms-work.md`](../../04-ai-and-rag/docs/how-llms-work.md) |
| Tabular foundation models (TabPFN, TabICL) | [`tooling-2026-update.md`](tooling-2026-update.md) |

---

## ✅ Quick checklist before you pick a model

- [ ] A dummy score is printed next to every result
- [ ] Linear or logistic regression was tried first, with scaled features
- [ ] Every model used the **same locked folds** and preprocessing fitted inside each fold
- [ ] You can say in one sentence why the winner wins (shape of the data, categoricals, interactions)
- [ ] The winner beats the runner-up by more than the fold-to-fold spread, or you keep the simpler one
- [ ] You can explain it to a judge: coefficients, a small tree, or SHAP for ensembles
