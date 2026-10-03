# 📏 ML Fundamentals & Metrics, Explained

<!-- markdownlint-disable MD013 -->

> How a model learns, why you need three data splits, and which metric answers which question, in plain English with formulas, scikit-learn functions and pitfalls. Companions: [`cross-validation-guide.md`](cross-validation-guide.md), [`method-selection-guide.md`](method-selection-guide.md).
>
> 📚 Source research: [`technical-ml-gaps-2026-10-03`](../../_research/technical-ml-gaps-2026-10-03/research.md). **Label key used throughout.** **SNIPPET-ONLY** means the claim comes from a search-result snippet because the primary page could not be fetched; check the source before relying on it. **UNVERIFIED** means it comes from general knowledge or an unconfirmed attribution. **RE-CHECK** marks prices, quotas and limits that change often; confirm them on the day. **Vendor-reported** or **author-reported** marks numbers from the people selling or publishing the tool. Every code block is labelled **not run by us**: each is either quoted from the cited source or assembled from cited calls, and none was executed. All version and licence tables are dated **2026-10-03** and were taken from PyPI or npm JSON on that date. Nothing here is legal advice.

## 1. ML fundamentals and metrics: score what you are judged on, against a dummy

### Plain-English explanation

A supervised model learns by adjusting its internal numbers to shrink a **loss**, which measures how wrong it is on the training examples. What matters is whether it **generalises**, meaning whether it predicts well on data it has never seen. scikit-learn puts the core trap bluntly: testing a model on the data it learned from is "a methodological mistake," because a model that "would just repeat the labels" would score perfectly yet be useless. That failure is called **overfitting** ([scikit-learn cross-validation](https://github.com/scikit-learn/scikit-learn/blob/main/doc/modules/cross_validation.rst)).

Data is therefore split three ways:

- The **training** set is used to fit the model.
- The **validation** set is used to compare settings and choose between them.
- The **test** set is used once, at the end. Once you have tuned on a validation score, "the validation score is biased," so only an untouched test set estimates real performance ([scikit-learn learning curves](https://github.com/scikit-learn/scikit-learn/blob/main/doc/modules/learning_curve.rst)).

The one rule to memorise is "**never call `fit` on the test data**." That rule covers preprocessing too, such as scalers, imputers and PCA ([scikit-learn common pitfalls](https://github.com/scikit-learn/scikit-learn/blob/main/doc/common_pitfalls.rst)). The guide's cross-validation page covers how to split. This section covers how to read the numbers.

Error has two sources a beginner can diagnose. **Bias** is a model that is too simple and misses the pattern, which is **underfitting**. **Variance** is a model so sensitive to its particular training sample that it memorises noise, which is overfitting. A **learning curve** plots training and validation scores as the training set grows, and it tells you which problem you have ([scikit-learn learning curves](https://github.com/scikit-learn/scikit-learn/blob/main/doc/modules/learning_curve.rst)):

- If both curves converge at a low score, more data "will probably not" help. The model is too simple.
- If training is far above validation, "adding more training samples will most likely increase generalization."

Google's Machine Learning Crash Course (MLCC) teaches the same diagnosis through loss curves: training loss keeps falling while validation loss turns upward (**SNIPPET-ONLY**, [MLCC interpreting loss curves](https://developers.google.com/machine-learning/crash-course/overfitting/interpreting-loss-curves)).

Separate two things: **predicting** a probability and **deciding** at a threshold. `predict_proba` gives the probability, and `predict` turns it into a decision. Most classification metrics score the decision, and they all derive from the confusion matrix ([scikit-learn model evaluation](https://github.com/scikit-learn/scikit-learn/blob/main/doc/modules/model_evaluation.rst)).

scikit-learn's advice on which metric to use is the right first sentence for any hackathon: "if the scoring function is given, e.g. in a kaggle competition or in a business context, use that one." If you are free to choose, pick a "strictly consistent" score for what you are predicting (mean, median, quantile or class), and ideally use it as both the training loss and the evaluation metric (same source).

### Decision table: which metric answers which question

| Your situation | Metric (scikit-learn function) | Formula / meaning | Chance or dummy level |
|---|---|---|---|
| Organisers named a metric | That metric, exactly | Optimise what is scored | Compute it for `DummyClassifier` / `DummyRegressor` |
| Balanced classes, every error costs the same | Accuracy (`accuracy_score`) | Share of correct predictions | Majority-class rate |
| Imbalanced classes, need one headline number | Balanced accuracy (`balanced_accuracy_score`) | Mean of per-class recall; binary = ½(TP/(TP+FN) + TN/(TN+FP)) | 1 / n_classes |
| False alarms are costly (spam filter, fraud queue review cost) | Precision (`precision_score`) | TP / (TP+FP) | Prevalence |
| Missing a positive is costly (disease screening) | Recall (`recall_score`) | TP / (TP+FN) | — |
| Need one number that balances precision and recall | F1 / F-beta (`f1_score`, `fbeta_score`) | (1+β²)·P·R / (β²·P + R); β<1 favours precision, β>1 favours recall | — |
| Multiclass: care about rare classes equally | `average="macro"` | Unweighted mean over classes | — |
| Multiclass: overall per-sample performance | `average="micro"` or `"weighted"` | Pooled, or weighted by class size | — |
| Ranking quality across all thresholds | ROC-AUC (`roc_auc_score`) | Area under TPR vs FPR | 0.5 |
| Ranking quality where only flagged positives matter | Average precision / PR-AUC (`average_precision_score`) | Σ (Rₙ − Rₙ₋₁)·Pₙ | **Equals prevalence** |
| Probabilities will be used as numbers | Log loss (`log_loss`), Brier (`brier_score_loss`) plus a reliability diagram (`CalibrationDisplay`) | Negative log-likelihood; mean squared probability error | Score of predicting the base rate |
| Regression, error in target units, robust to outliers | MAE (`mean_absolute_error`) | mean \|y − ŷ\|; targets the median | Predict the median |
| Regression, large errors matter more | RMSE (`root_mean_squared_error`) | √mean (y − ŷ)²; same units as target | Predict the mean |
| Regression, exponential-growth targets, under-prediction worse | RMSLE (`root_mean_squared_log_error`) | On ln(1+y) | — |
| Regression, relative error | MAPE (`mean_absolute_percentage_error`) | Returns a **fraction**: 0.27 means 27% | — |
| "How much variance explained" | R² (`r2_score`) | 1 − SS_res/SS_tot; can be negative | 0.0 for predicting the mean |
| Search or recommendation with graded relevance | NDCG@k (`ndcg_score`) | DCG / ideal DCG, 0–1 | — |
| "How many of my top k are relevant" | Precision@k (no sklearn function; see code) | relevant in top k / k | Prevalence |

Formulas and functions are from [scikit-learn model evaluation](https://github.com/scikit-learn/scikit-learn/blob/main/doc/modules/model_evaluation.rst). Its worked F-beta example: precision 1.0 and recall 0.5 give F1 = 0.66, F0.5 = 0.83 and F2 = 0.55. Precision@k comes from [Manning et al., IR book ch. 8](https://nlp.stanford.edu/IR-book/pdf/08eval.pdf) (**SNIPPET-ONLY**). scikit-learn has **no `precision_at_k`**. Its `top_k_accuracy_score` is a different metric: whether the true label is among the top k predicted classes.

**Contested: PR-AUC vs ROC-AUC under imbalance.**

- **For PR plots.** Saito and Rehmsmeier (PLoS ONE 2015) argue that PR plots are more informative than ROC plots on imbalanced data, because ROC "can be deceptive" (**SNIPPET-ONLY**, [PLoS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432)).
- **Against "PR-AUC is always better".** McDermott et al. (NeurIPS 2024) "refute" the idea that AUPRC is generally superior under imbalance. They show AUPRC can unduly favour improvements in subpopulations where positives are more frequent. They also traced the claim through more than 1.5 million papers and found it often "made without citation, misattributed ... and aggressively overgeneralized" (**SNIPPET-ONLY**, [arXiv 2401.06091](https://arxiv.org/abs/2401.06091)).
- **For ROC.** A 2024 *Patterns* paper is titled "The receiver operating characteristic curve accurately assesses imbalanced datasets" (**SNIPPET-ONLY**, title only, [Patterns](https://www.cell.com/patterns/fulltext/S2666-3899(24)00109-0)).

The guide should teach that the two metrics **answer different questions**, not that one is right:

- Use PR-AUC when only the positives you flag matter, such as alert or review queues.
- Use ROC-AUC when ranking quality across both classes matters, or when prevalence may shift.
- Under heavy imbalance, report both.

One thing is certain: the chance levels differ. Random average precision equals the positive fraction, while a dummy's ROC-AUC is 0.5. A PR-AUC of 0.3 can therefore be strong at 1% prevalence ([scikit-learn](https://github.com/scikit-learn/scikit-learn/blob/main/doc/modules/model_evaluation.rst); [INRIA MOOC metrics notebook](https://github.com/INRIA/scikit-learn-mooc/blob/main/python_scripts/metrics_classification.py)).

### Minimal code (tested by us)

**Tested 2026-10-03** (Python 3.12.3, scikit-learn 1.9.1) on the guide's sample data; runs unchanged. Assembled from function names in [scikit-learn model evaluation](https://github.com/scikit-learn/scikit-learn/blob/main/doc/modules/model_evaluation.rst), [learning curves](https://github.com/scikit-learn/scikit-learn/blob/main/doc/modules/learning_curve.rst) and [calibration](https://github.com/scikit-learn/scikit-learn/blob/main/doc/modules/calibration.rst). 
```python
# tested 2026-10-03 — assembled from scikit-learn docs
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (classification_report, balanced_accuracy_score,
    average_precision_score, roc_auc_score, log_loss, brier_score_loss,
    ConfusionMatrixDisplay)
from sklearn.calibration import CalibrationDisplay
from sklearn.model_selection import LearningCurveDisplay

dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
for name, m in [("dummy", dummy), ("model", model)]:
    p = m.predict_proba(X_val)[:, 1]
    yhat = m.predict(X_val)
    print(name,
          "bal_acc", balanced_accuracy_score(y_val, yhat),
          "AP", average_precision_score(y_val, p),     # chance = prevalence
          "ROC-AUC", roc_auc_score(y_val, p),          # chance = 0.5
          "logloss", log_loss(y_val, p),
          "brier", brier_score_loss(y_val, p))
print("prevalence", y_val.mean())
print(classification_report(y_val, model.predict(X_val)))
ConfusionMatrixDisplay.from_estimator(model, X_val, y_val)  # rows = true class
CalibrationDisplay.from_estimator(model, X_val, y_val, n_bins=10)
LearningCurveDisplay.from_estimator(model, X_train, y_train)  # bias or variance?
```

Precision@k is illustrative code, not from a source (tested: runs).

```python
# tested 2026-10-03 — illustrative
import numpy as np
def precision_at_k(y_true, scores, k):
    top = np.argsort(-np.asarray(scores))[:k]
    return float(np.mean(np.asarray(y_true)[top]))
```

### Pitfalls

**Accuracy misleads on imbalanced data.** In the INRIA MOOC, a most-frequent `DummyClassifier` reaches 76% accuracy and predicts "as accurately as our logistic regression model". The MOOC's rule is that "when the classes are imbalanced, accuracy should not be used" ([MOOC metrics_classification](https://github.com/INRIA/scikit-learn-mooc/blob/main/python_scripts/metrics_classification.py)). Google MLCC gives a 99.93%-accurate model with no predictive power (**SNIPPET-ONLY**, [MLCC](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall)).

**Low Brier or log loss does not mean well calibrated.** A lower Brier loss "does not necessarily mean a better calibrated model". It can come from a worse-calibrated model with more discriminative power, so check a reliability diagram ([scikit-learn calibration](https://github.com/scikit-learn/scikit-learn/blob/main/doc/modules/calibration.rst)).

**Trapezoidal AUPRC is optimistic.** Computing `auc(recall, precision)` interpolates linearly, which is "overly-optimistic". Different tools also give conflicting AUPRC values. Use `average_precision_score` instead ([scikit-learn](https://github.com/scikit-learn/scikit-learn/blob/main/doc/modules/model_evaluation.rst)).

**MAPE problems.**

- scikit-learn returns MAPE as a fraction, so multiply by 100 before putting it on a slide.
- Near-zero targets explode the score, because scikit-learn divides by max(ε, |y|) (same source).
- Hyndman and Koehler (2006) propose **MASE**, which "never gives infinite or undefined values" (**SNIPPET-ONLY**, [Hyndman](https://robjhyndman.com/papers/foresight.pdf)).

**R² limits.**

- R² is not comparable across datasets: "As such variance is dataset dependent, R² may not be meaningfully comparable" ([scikit-learn](https://github.com/scikit-learn/scikit-learn/blob/main/doc/modules/model_evaluation.rst)).
- It is also the default `.score()` for scikit-learn regressors, so a teammate may report it without realising ([MOOC metrics_regression](https://github.com/INRIA/scikit-learn-mooc/blob/main/python_scripts/metrics_regression.py)).
- Trending time series regressed on each other give high R² with no real relationship ("spurious regression") (**SNIPPET-ONLY**, [Stata blog](https://blog.stata.com/2016/09/06/cointegration-or-spurious-regression/)). The classic citation, Granger and Newbold (1974), is **UNVERIFIED**.

**RMSLE is asymmetric.** It "penalizes an under-predicted estimate greater than an over-predicted estimate" ([scikit-learn](https://github.com/scikit-learn/scikit-learn/blob/main/doc/modules/model_evaluation.rst)).

**Confusion-matrix orientation varies.** In scikit-learn, rows are the true class, and the docs warn that other references flip the axes (same source).

**Explaining metrics to non-technical judges (our inference; no sourced guide was found).**

- Translate the metric into confusion-matrix counts: "of 100 fraud cases we catch 82; of every 10 we flag, 7 are real."
- Show the dummy baseline next to your score.
- Quote MAE or RMSE in the target's units. RMSE is "in the same units as the target variable" ([scikit-learn](https://github.com/scikit-learn/scikit-learn/blob/main/doc/modules/model_evaluation.rst)).
- Say which error is costlier.
- Explain calibration as "when we say 80%, it happens about 80% of the time."

### Versions and licences (as of 2026-10-03)

| Package | Version | Released | Licence | Note |
|---|---|---|---|---|
| scikit-learn | 1.9.1 | 2026-09-10 | BSD-3-Clause (UNVERIFIED this session; long-standing) | Needs Python ≥3.11 per the guide's tooling page; `root_mean_squared_error` and `root_mean_squared_log_error` are the current RMSE/RMSLE functions |

Version from [PyPI scikit-learn](https://pypi.org/pypi/scikit-learn/json).

### Learning resources

**Best first stop.** The **INRIA scikit-learn MOOC** is free and CC-BY, so excerpts can be reused with attribution. It has Binder support and dedicated metrics notebooks that use dummy baselines ([repo](https://github.com/INRIA/scikit-learn-mooc); [Jupyter Book](https://inria.github.io/scikit-learn-mooc)).

**Short visual primers.** **Google MLCC**: the classification modules ([accuracy/precision/recall](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall), [ROC and AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)), the [overfitting](https://developers.google.com/machine-learning/crash-course/overfitting) module and the [metrics glossary](https://developers.google.com/machine-learning/glossary/metrics). These URLs are **SNIPPET-ONLY**: the pages were blocked from fetch.

**Reference.** The **scikit-learn User Guide** pages on [model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html), [learning curves](https://scikit-learn.org/stable/modules/learning_curve.html), [calibration](https://scikit-learn.org/stable/modules/calibration.html) and [common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html).

**Depth.** **rasbt STAT 451** lectures L08–L12 on evaluation ([repo](https://github.com/rasbt/stat451-machine-learning-fs20)).

**Ranking metrics.** Manning, Raghavan and Schütze, *Introduction to Information Retrieval*, ch. 8 ([PDF](https://nlp.stanford.edu/IR-book/pdf/08eval.pdf), **SNIPPET-ONLY**).

### See it on real numbers

[`07-worked-example/evaluate_and_explain.py`](../07-worked-example/evaluate_and_explain.py) runs all of this on the sample data with out-of-fold predictions: a labelled confusion matrix with precision and recall worked out, PR-AUC and ROC-AUC next to their no-skill baselines, calibration before and after removing `class_weight="balanced"` (mean predicted risk 0.443 vs 0.112 against an actual rate of 0.112), and a learning curve with an automatic verdict. Plain accuracy there is 0.673 while predicting "nobody" scores 0.888, which is the accuracy trap in one line.
