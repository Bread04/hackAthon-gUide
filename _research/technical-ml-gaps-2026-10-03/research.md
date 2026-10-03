# Filling six ML gaps hackathon guides skip

A hackathon guide that already covers tabular cross-validation, leakage, tuning and method choice still has six gaps. Beginners need to know which metric answers which question. They need a routine for looking at what their model gets wrong. They need a cheap path into pretrained deep learning for images, text and audio, and a way to put a model inside an app without it falling over on demo day. They need to know how to get labelled data when nobody hands them a CSV, and how to keep track of what they tried. The evidence favours one approach across all six: **use the score you will be judged on, compare it with a dummy baseline, and hand-inspect the errors before adding complexity** ([scikit-learn model evaluation](https://github.com/scikit-learn/scikit-learn/blob/main/doc/modules/model_evaluation.rst); [cleanlab README](https://raw.githubusercontent.com/cleanlab/cleanlab/master/README.md)). For deep learning that means a ladder that starts with frozen embeddings and a linear head ([CS231n transfer learning](https://github.com/cs231n/cs231n.github.io/blob/master/transfer-learning.md)). For apps it means a hosted API or a small pretrained model, served from ONNX or run in the browser, rather than training your own. For data it means a held-out human-labelled sample that validates every shortcut, LLM labelling included ([Pangakis et al.](https://arxiv.org/pdf/2306.00176)). Three facts changed in 2026 and break older advice. Hugging Face Gradio and Docker Spaces now need a paid plan to create ([HF Spaces overview](https://raw.githubusercontent.com/huggingface/hub-docs/main/docs/hub/spaces-overview.md)). transformers is on 5.x, so old notebooks need API fixes ([PyPI](https://pypi.org/pypi/transformers/json)). MLflow now defaults to SQLite rather than a folder of files ([MLflow CHANGELOG](https://raw.githubusercontent.com/mlflow/mlflow/master/CHANGELOG.md)). Several popular tools carry licences a team must flag before turning a demo into a product: Ultralytics is AGPL-3.0, SDV is BUSL-1.1, DINOv3 is gated under a custom licence, and deepchecks is AGPL.

**Label key used throughout.** **SNIPPET-ONLY** means the claim comes from a search-result snippet because the primary page could not be fetched; check the source before relying on it. **UNVERIFIED** means it comes from general knowledge or an unconfirmed attribution. **RE-CHECK** marks prices, quotas and limits that change often; confirm them on the day. **Vendor-reported** or **author-reported** marks numbers from the people selling or publishing the tool. Every code block is labelled **not run by us**: each is either quoted from the cited source or assembled from cited calls, and none was executed. All version and licence tables are dated **2026-10-03** and were taken from PyPI or npm JSON on that date. Nothing here is legal advice.

**What this report does not repeat.** The guide already covers cross-validation schemes, out-of-fold (OOF) predictions and leakage tests (`08-datathon-handbook/03-modeling/cross-validation-guide.md`), tuning and ensembling (`hyperparameter-tuning-and-ensembling.md`), method choice, imbalance and calibration method choice (`method-selection-guide.md`), the frozen-embedding baseline recipes for text and images (`baseline-recipes.md`), free compute and free hosting terms (`tooling-2026-update.md`), and RAG architecture and evaluation (`04-ai-and-rag/docs/rag-architecture.md`). The sections below link to those pages instead of restating them.

---

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

### Minimal code (not run by us)

Assembled from function names in [scikit-learn model evaluation](https://github.com/scikit-learn/scikit-learn/blob/main/doc/modules/model_evaluation.rst), [learning curves](https://github.com/scikit-learn/scikit-learn/blob/main/doc/modules/learning_curve.rst) and [calibration](https://github.com/scikit-learn/scikit-learn/blob/main/doc/modules/calibration.rst). Not run by us.

```python
# not run by us — assembled from scikit-learn docs
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

Precision@k is illustrative code, not from a source. Not run by us.

```python
# not run by us — illustrative
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

---

## 2. Error analysis: tag 100 mistakes before you add a feature

### Plain-English explanation

Error analysis means looking at the specific rows your model gets wrong and asking why. The canonical method comes from Andrew Ng's *Machine Learning Yearning*, chapters 13–19 (all **SNIPPET-ONLY**; the PDF was blocked, [MLY PDF](https://home-wordpress.deeplearning.ai/wp-content/uploads/2022/03/andrew-ng-machine-learning-yearning.pdf), [KDnuggets summary](https://www.kdnuggets.com/2018/05/7-useful-suggestions-machine-learning-yearning.html)):

1. Gather about **100 misclassified validation examples**.
2. Tag each one in a spreadsheet. Rows are examples, columns are error categories that emerge as you look, and a comments column holds anything else.
3. Count the categories. Each category's share is the **ceiling** on what fixing it could gain: if only 5% of errors are blurry photos, a perfect blur fix gains at most 5% of the errors.

Ng also suggests splitting the dev set into an "eyeball" set you inspect by hand and a "blackbox" set you only score. At a 5% error rate, you need about 2,000 eyeball examples to get 100 errors.

Google's *Rules of ML* adds the next step. Rule #26 says "Look for patterns in the measured errors, and create new features" (**SNIPPET-ONLY**, [Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml)).

Tools automate the "find the bad group" step, but they do not replace looking:

- **Slicing** computes your metric per segment (region, device, age band).
- **Slice finding** searches for segment combinations you did not think of.
- **Label-noise detection** flags rows whose labels are probably wrong. Real benchmark datasets contain such errors ([labelerrors.com](https://labelerrors.com), linked from the [cleanlab README](https://raw.githubusercontent.com/cleanlab/cleanlab/master/README.md)).

Error analysis also gives you an honest **limitations slide**, which judges tend to reward. That last point is our inference.

### Decision table: which error-analysis tool for which question

| Question | Tool | Effort | Output |
|---|---|---|---|
| "Is the model worse for some known group?" | pandas `groupby` or fairlearn `MetricFrame` (any column works as `sensitive_features`) | Minutes | Per-group metric table with counts |
| "Which group did I not think to check?" | `sliceline` (Slicefinder) on binned features plus a 0/1 or absolute-residual error vector | ~15 min | Top-k worst slices |
| "Are some labels wrong?" | `cleanlab` `find_label_issues` or `Datalab` on out-of-fold `pred_probs` | ~15 min | Ranked suspect rows |
| "Does test look different from train?" | Evidently `DataDriftPreset`, or adversarial validation (see the guide's CV page) | ~15 min | HTML drift report |
| "One-click evaluation report for the deck" | Deepchecks `model_evaluation()` suite (**AGPL**) | 15–30 min if it installs | HTML report |
| "Interactive error tree and heatmap" | Microsoft `raiwidgets` `ErrorAnalysisDashboard` | 30+ min; stale releases | Notebook widget |
| "Why are these 100 rows wrong?" | Spreadsheet plus your eyes (Ng) | 1 hour | Error-category counts |

Sources: [fairlearn assessment docs](https://raw.githubusercontent.com/fairlearn/fairlearn/main/docs/user_guide/assessment/perform_fairness_assessment.rst), [sliceline README](https://raw.githubusercontent.com/DataDome/sliceline/main/README.rst), [cleanlab README](https://raw.githubusercontent.com/cleanlab/cleanlab/master/README.md), [Evidently README](https://raw.githubusercontent.com/evidentlyai/evidently/main/README.md), [deepchecks README](https://raw.githubusercontent.com/deepchecks/deepchecks/main/README.md) and the [RAI Error Analysis README](https://raw.githubusercontent.com/microsoft/responsible-ai-toolbox/main/docs/erroranalysis-dashboard-README.md). The effort column is our estimate.

### A 10-step 48-hour error-analysis routine (our synthesis; not a published checklist)

1. **Score on out-of-fold predictions.** Use OOF predictions, so every training row has an honest prediction (see the guide's CV page).
2. **Compare segments.** Build a per-segment metric table and show counts next to every metric, so small slices are not over-read.
3. **Find slices you missed.** Run Sliceline on binned features.
4. **Confusion by slice.** Compute a confusion matrix for each top slice.
5. **Read the worst errors.** Sort by loss (confident-wrong classifications, largest residuals), then hand-tag 50–100 of them (Ng).
6. **Check regression residuals.** Plot residuals against the prediction, key features and time:
   - A fan shape suggests a log target.
   - Bias within one slice suggests a missing feature (Rule #26).
7. **Check labels.** Run cleanlab on OOF `pred_probs`. Clean only training rows, never test or leaderboard data.
8. **Check drift.** Look for drift between train and test.
9. **Act on each top category.** Turn it into a feature, a segment-specific threshold, a data fix, or a decision to "accept and disclose".
10. **Put it on a slide.** For example: "X% worse on segment S (n=…), cause Y, fix Z gave +Δ."

The source for the data-centric loop in steps 7–9 is cleanlab's own suggested workflow: train an initial model, use it to diagnose data issues, retrain the same model on the improved data, and only then try other modelling techniques ([cleanlab README](https://raw.githubusercontent.com/cleanlab/cleanlab/master/README.md)).

### Minimal code (not run by us)

Per-segment table, quoted from the [fairlearn docs](https://raw.githubusercontent.com/fairlearn/fairlearn/main/docs/user_guide/assessment/perform_fairness_assessment.rst) with the doctest prompts removed. Not run by us.

```python
# not run by us — quoted from fairlearn docs
from fairlearn.metrics import MetricFrame, count, false_positive_rate, selection_rate
from sklearn.metrics import recall_score
my_metrics = {'tpr': recall_score, 'fpr': false_positive_rate, 'sel': selection_rate, 'count': count}
mf = MetricFrame(metrics=my_metrics, y_true=y_true, y_pred=y_pred, sensitive_features=sf_data)
mf.by_group; mf.difference(method='to_overall')
```

Automatic slices, quoted from the [sliceline README](https://raw.githubusercontent.com/DataDome/sliceline/main/README.rst). That X must be categorical or binned is **UNVERIFIED**. Not run by us.

```python
# not run by us — README calls; binning step is our assumption (UNVERIFIED requirement)
import pandas as pd
from sliceline.slicefinder import Slicefinder
X_binned = X_val.copy()
X_binned["income"] = pd.qcut(X_binned["income"], 5, duplicates="drop").astype(str)
errors = (y_val != y_pred).astype(int)          # or abs(y_val - y_pred) for regression
sf = Slicefinder()
sf.fit(X_binned, errors)
print(sf.top_slices_)
```

Label issues on out-of-fold probabilities. This is assembled from the cleanlab README and `filter.py`. The `return_indices_ranked_by` value is **UNVERIFIED**, as is the requirement that `pred_probs` be out-of-sample; the latter matches the confident-learning setup but its docs page was blocked. Not run by us.

```python
# not run by us — assembled
from sklearn.model_selection import cross_val_predict
from cleanlab.filter import find_label_issues
pred_probs = cross_val_predict(model, X, y, cv=5, method="predict_proba")
idx = find_label_issues(labels=y, pred_probs=pred_probs,
                        return_indices_ranked_by="self_confidence")  # value UNVERIFIED
print(idx[:50])  # hand-review these rows first
# one-call audit alternative (README): lab = cleanlab.Datalab(data=df, label="label");
# lab.find_issues(features=emb, pred_probs=pred_probs); lab.report()
```

Drift report, quoted from the [Evidently README](https://raw.githubusercontent.com/evidentlyai/evidently/main/README.md). Not run by us.

```python
# not run by us — quoted from Evidently README
from evidently import Report
from evidently.presets import DataDriftPreset
report = Report([DataDriftPreset(method="psi")], include_tests="True")
my_eval = report.run(train_df, test_df)
my_eval.save_html("drift.html")
```

### Pitfalls

**Error analysis on training predictions.** It finds nothing, because the model has memorised those rows. Use OOF or validation predictions.

**Over-reading tiny slices.** A segment with n=12 and a 40% error rate is noise until shown otherwise. Always print counts.

**"Cleaning" the test set.** This is leakage dressed up as data quality. The cleanlab loop applies to training data only.

**Deepchecks licence and staleness.** Deepchecks is **AGPL-3.0**, and its premium `ee` features need a commercial licence. Its last PyPI release was December 2024 ([deepchecks README](https://raw.githubusercontent.com/deepchecks/deepchecks/main/README.md); [PyPI](https://pypi.org/pypi/deepchecks/json)). The AGPL is unlikely to matter in a hackathon notebook, but it matters if the team later ships a hosted product (our inference).

**Microsoft's Error Analysis widgets are stale.** The last releases were raiwidgets 0.36.0 (July 2024) and erroranalysis 0.5.5 (January 2025) ([PyPI](https://pypi.org/pypi/raiwidgets/json)). That their older dependency pins may clash with current stacks is **UNVERIFIED**.

**Unverified source details.** Google's rule numbers other than #26, and Ng's advice on mislabelled dev examples, were not checked against the sources (**UNVERIFIED**).

### Versions and licences (as of 2026-10-03)

| Package | Version | Released | Licence | Python |
|---|---|---|---|---|
| cleanlab | 2.9.0 | 2026-01-13 | Apache-2.0 | ≥3.10 |
| sliceline | 0.3.0 | 2026-02-03 | BSD-3-Clause | ≥3.10,<4 |
| fairlearn | 0.14.0 | 2026-06-07 | MIT | ≥3.10 |
| evidently | 0.7.23 | 2026-09-11 | Apache-2.0 | ≥3.10 |
| deepchecks | 0.19.1 | 2024-12-15 | **AGPLv3+** | n/a |
| raiwidgets / responsibleai | 0.36.0 | 2024-07-08 | MIT | ≥3.7 |
| erroranalysis | 0.5.5 | 2025-01-27 | MIT | ≥3.7 |

All versions are from PyPI JSON: [cleanlab](https://pypi.org/pypi/cleanlab/json), [sliceline](https://pypi.org/pypi/sliceline/json), [fairlearn](https://pypi.org/pypi/fairlearn/json), [evidently](https://pypi.org/pypi/evidently/json), [deepchecks](https://pypi.org/pypi/deepchecks/json), [raiwidgets](https://pypi.org/pypi/raiwidgets/json) and [erroranalysis](https://pypi.org/pypi/erroranalysis/json).

### Learning resources

**Method.**

- Andrew Ng, *Machine Learning Yearning*, chapters 13–19 on basic error analysis ([PDF](https://home-wordpress.deeplearning.ai/wp-content/uploads/2022/03/andrew-ng-machine-learning-yearning.pdf), **SNIPPET-ONLY**).
- Zinkevich, *Rules of ML*, the "Human Analysis of the System" section ([PDF](https://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf), **SNIPPET-ONLY**).

**Papers behind the tools.**

- SliceLine (SIGMOD 2021, [PDF](https://mboehm7.github.io/resources/sigmod2021b_sliceline.pdf)).
- Confident Learning (JAIR 2021, [arXiv 1911.00068](https://arxiv.org/abs/1911.00068)).

**Hands-on.** The Microsoft Error Analysis dashboard notebooks ([GitHub](https://github.com/microsoft/responsible-ai-toolbox/blob/main/notebooks/individual-dashboards/erroranalysis-dashboard/)).

---

## 3. Deep learning for images, text and audio: climb a four-rung ladder from frozen embeddings

### Plain-English explanation

At a 24–48 hour event you almost never train a neural network from scratch. You **borrow one**: a "pretrained backbone" that has already learned general features from millions of images, documents or hours of audio. You then adapt it to your task. This is **transfer learning**, and there are four levels of adaptation, from cheapest to most expensive:

1. **Frozen embeddings plus a linear head.** Run your data through the backbone once, save the output vectors, then train a logistic regression on them. It takes minutes, often runs on CPU, and is hard to overfit. The guide's `baseline-recipes.md` already has code for this.
2. **Few-shot or head-and-top-layer training.** Examples include SetFit for text, and fastai's freeze-then-unfreeze. Use these when you have very few labels.
3. **Full fine-tuning of a small or base backbone.** Use this when you have thousands of labels or your domain looks unlike the pretraining data.
4. **LoRA/PEFT.** Train small "adapter" matrices instead of the whole model. Use this mainly when the model is too large to fully fine-tune on a free 16 GB T4: LLMs, Whisper-large, diffusion models.

The CS231n notes give the classic rule of thumb for choosing a level ([CS231n transfer learning](https://github.com/cs231n/cs231n.github.io/blob/master/transfer-learning.md)):

- With a **small dataset similar to the pretraining data**, it is "not a good idea to fine-tune ... due to overfitting concerns". Train a linear classifier on the features instead.
- With a **large dataset**, fine-tune the whole network.
- Even when your data is **very different**, initialising from pretrained weights is "very often still beneficial".

The PyTorch tutorial names the same two basic modes, "finetuning the ConvNet" and "ConvNet as fixed feature extractor" ([PyTorch transfer learning tutorial](https://github.com/pytorch/tutorials/blob/main/beginner_source/transfer_learning_tutorial.py)).

A useful nuance comes from Kumar et al. (ICLR 2022). Full fine-tuning can beat a linear probe in-distribution yet underperform it out-of-distribution. Their fix, **LP-FT**, trains the linear head first and then fine-tunes everything (**SNIPPET-ONLY**, [paper](https://par.nsf.gov/servlets/purl/10337813)). fastai's `fine_tune` already does exactly this by default: it trains frozen for `freeze_epochs`, then unfreezes with discriminative learning rates ([fastai schedule.py](https://github.com/fastai/fastai/blob/main/fastai/callback/schedule.py)).

### Decision table: which rung of the ladder

| Labels you have | Domain vs pretraining | First move | Upgrade if time allows | Compute (free T4) |
|---|---|---|---|---|
| None | Any | Zero-shot: SigLIP 2 / OpenCLIP (images), CLAP (audio), an LLM or zero-shot pipeline (text) | Label 100–300 items (section 5) | Inference only |
| 8–100 per class (text) | Similar | **SetFit** | Frozen embeddings + LogisticRegression as a comparison | Seconds to minutes; V100 timing below |
| Hundreds | Similar | Frozen embeddings (DINOv2, sentence-transformers, CLAP/BEATs) + LogisticRegression | LP-FT (`fastai fine_tune`) | CPU after one embedding pass |
| Hundreds | Very different (X-ray, satellite, machine sounds) | Linear probe, possibly on earlier-layer features (CS231n) | LP-FT with strong augmentation | T4, ≤1 h |
| Thousands | Any | Full fine-tune of a base model (ViT-B / ConvNeXt-T, DeBERTa-v3-base / ModernBERT-base, wav2vec2-base / AST) | Ensemble 2–3 seeds | T4, hours |
| Object detection / segmentation | Any | Ultralytics YOLO26n/s from COCO weights (**AGPL**) | Larger YOLO26 size | T4, hours |
| Speech-to-text | Any | Whisper zero-shot (pick size by VRAM) | Fine-tune whisper-tiny/base/small, or LoRA on large | Whisper-small fine-tune quoted at 5–10 h |
| Any, model >1B params | Any | LoRA / QLoRA via PEFT | — | 16 GB GPU possible for 7B with QLoRA |

The rungs follow [CS231n](https://github.com/cs231n/cs231n.github.io/blob/master/transfer-learning.md), the [SetFit README](https://github.com/huggingface/setfit), the [PEFT README](https://github.com/huggingface/peft) and the [HF Whisper fine-tuning blog](https://github.com/huggingface/blog/blob/main/fine-tune-whisper.md). The sources give **qualitative** rules based on data size and similarity. No authoritative row-count cutoffs were found, so the label counts in the first column are our inference.

**Author- and vendor-reported numbers to label as such.**

- **SetFit (author-reported).** With 8 labelled examples per class, SetFit is "competitive with fine-tuning RoBERTa Large on the full training set of 3k examples". Training takes "30 seconds" on a **V100** (not a T4) and costs about $0.025 ([SetFit README](https://github.com/huggingface/setfit); [HF blog](https://github.com/huggingface/blog/blob/main/setfit.md)).
- **PEFT (author-reported, measured on an A100 80GB, not a T4).**
  - Training T0_3B fully needs 47.14 GB, against 14.4 GB with LoRA.
  - The LoRA checkpoint is 19 MB, against 11 GB for the full model.
  - LoRA on Qwen2.5-3B trains 0.1193% of the parameters ([PEFT README](https://github.com/huggingface/peft)).
- **Whisper-small fine-tuning (author-reported).** About 8 h of Hindi audio, 5,000 steps, "approximately 5-10 hours" on a Colab-class GPU. That is a large share of a hackathon, so cut `max_steps` or use tiny/base ([HF blog](https://github.com/huggingface/blog/blob/main/fine-tune-whisper.md)).

### Decision table: recommended backbones in 2026, with licences

| Modality / job | Backbone | Licence (weights) | Load with | Notes |
|---|---|---|---|---|
| Image features (frozen) | **DINOv2** | Apache-2.0 (XRay-DINO weights excepted: FAIR non-commercial) | `torch.hub.load('facebookresearch/dinov2','dinov2_vits14')` | Frictionless default ([dinov2](https://github.com/facebookresearch/dinov2)) |
| Image features (frozen) | **DINOv3** | **Custom DINOv3 License, gated** (request access; URLs by e-mail) | `AutoModel.from_pretrained("facebook/dinov3-convnext-tiny-pretrain-lvd1689m")` (transformers ≥4.56) | Authors claim it beats specialised SOTA "without fine-tuning" (author-reported); approval delay at a hackathon ([dinov3](https://github.com/facebookresearch/dinov3); [LICENSE](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)) |
| Image classification fine-tune | timm (ConvNeXt, EVA, ViT, etc.) | Code Apache-2.0; **weights may inherit dataset licence; some CC-BY-NC** | `timm.create_model(name, pretrained=True, num_classes=N)` | "assume that the original dataset license applies to the weights" ([timm licences](https://github.com/huggingface/pytorch-image-models#licenses)) |
| Zero-shot image / image–text | SigLIP 2, OpenCLIP | OpenCLIP repo MIT-style; SigLIP 2 card licence **UNVERIFIED** | `pipeline("zero-shot-image-classification", model="google/siglip2-base-patch16-224")` | Text must be padded to `max_length=64` ([siglip2 doc](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/siglip2.md); [open_clip](https://github.com/mlfoundations/open_clip)) |
| Detection / segmentation / pose | **Ultralytics YOLO26** | **AGPL-3.0** or paid Enterprise licence | `YOLO("yolo26n.pt")` | Vendor-reported COCO: YOLO26n 40.9 mAP, 2.4M params, 1.7 ms T4 TensorRT; YOLO26s 48.6; YOLO26m 53.1; YOLO26l 55.0; YOLO26x 57.5 ([ultralytics](https://github.com/ultralytics/ultralytics)) |
| Text classification | DeBERTa-v3 (xsmall 22M → large 304M; mDeBERTa for 102 languages) | Repo MIT; HF card licence **UNVERIFIED** | `AutoModelForSequenceClassification` | Author-reported MNLI-m: base 90.6, large 91.8 ([DeBERTa](https://github.com/microsoft/DeBERTa)) |
| Text classification, long inputs | ModernBERT-base/large | **UNVERIFIED** (card not fetched) | `answerdotai/ModernBERT-base` | 8,192-token context; HF blog claims ~2× faster than DeBERTa and first base model to beat DeBERTaV3 on GLUE, but "slightly lags" it on NLU (author-reported) ([modernbert doc](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/modernbert.md); [blog](https://github.com/huggingface/blog/blob/main/modernbert.md)) |
| Text embeddings / few-shot | sentence-transformers (`all-MiniLM-L6-v2`, 384-d), SetFit | Apache-2.0 (libraries) | `SentenceTransformer(...)`, `SetFitModel.from_pretrained(...)` | ([sentence-transformers](https://github.com/UKPLab/sentence-transformers)) |
| Speech-to-text | **Whisper** | MIT (code and weights) | `openai/whisper` | VRAM: tiny/base ~1 GB, small ~2 GB, medium ~5 GB, large ~10 GB, turbo ~6 GB; turbo "not trained for translation" ([whisper](https://github.com/openai/whisper)) |
| Audio classification | AST; wav2vec2-base; BEATs | AST BSD-3-Clause; BEATs repo MIT (weights via OneDrive) | transformers audio-classification guide | AST ESC-50 95.75% in attached log (author-reported) ([AST](https://github.com/YuanGongND/ast); [BEATs](https://github.com/microsoft/unilm/tree/master/beats)) |
| Zero-shot audio / audio embeddings | CLAP (`laion-clap`) | Repo CC0-1.0 | `laion_clap.CLAP_Module(enable_fusion=False)` | Separate checkpoints for general audio <10 s, music, and speech ([CLAP](https://github.com/LAION-AI/CLAP)) |

**Licence warnings to put in a callout.**

- **Ultralytics: AGPL-3.0.** The README offers AGPL-3.0 for "students, researchers, and enthusiasts" and a paid Enterprise licence for business use, "including internal tools, automated workflows, and production deployments" ([Ultralytics licence](https://github.com/ultralytics/ultralytics#license)). In practice, a team that serves a modified YOLO over a network and later commercialises must open-source it or buy a licence (our reading; not legal advice).
- **DINOv3: custom licence, gated download.** Use DINOv2 if you cannot wait for approval.
- **timm: some weights are non-commercial.**
- **Hugging Face model cards: not checked.** Licences for SigLIP 2, ModernBERT, wav2vec2, the AST HF port and the DeBERTa-v3 weights could not be fetched. Many cards are Apache-2.0 or MIT, but that is **UNVERIFIED**.

### Free-GPU training settings (T4, 16 GB, Turing)

| Setting | Use | Why / source |
|---|---|---|
| Precision | `fp16=True`, **not** bf16 | "bf16 requires Ampere, Ada, or Hopper GPUs" ([flash-attention](https://github.com/Dao-AILab/flash-attention)) |
| Batch | `per_device_train_batch_size=16` + `gradient_accumulation_steps`; or `auto_find_batch_size=True` (needs accelerate) | Effective batch = per-device × devices × accumulation ([training_args.py](https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py)) |
| Memory | `gradient_checkpointing=True` for big models; on OOM halve batch, double accumulation | ([Whisper blog](https://github.com/huggingface/blog/blob/main/fine-tune-whisper.md)) |
| Early stopping | `EarlyStoppingCallback(early_stopping_patience=...)` with `metric_for_best_model` and `load_best_model_at_end=True` | Stops only at save points if `save_steps` ≠ `eval_steps` ([trainer_callback.py](https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_callback.py)) |
| Pick checkpoint by task metric | `metric_for_best_model="wer", greater_is_better=False` | In the Whisper log, validation loss rose 0.3075 → 0.4519 while WER improved 34.63 → 32.01 ([Whisper blog](https://github.com/huggingface/blog/blob/main/fine-tune-whisper.md)) |
| FlashAttention / ModernBERT on T4 | FA2 not supported on Turing (needs separate flash-attention-turing repo) | Measured speed gains may shrink on a T4 (our inference) |
| YOLO | Set `patience=10–20`; `batch=-1` (AutoBatch, ~60% memory); `cache=True`; `cls_pw` for class imbalance | Defaults are `epochs: 100, patience: 100`, which effectively disables early stopping (our inference from [default.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml); [train docs](https://github.com/ultralytics/ultralytics/blob/main/docs/en/modes/train.md)) |
| Augmentation | `RandomResizedCrop` + flip (images); YOLO already applies mosaic, flip and HSV | ([HF image guide](https://github.com/huggingface/transformers/blob/main/docs/source/en/tasks/image_classification.md)) |
| Cache embeddings | Embed once under `torch.no_grad()`/autocast, save `.npy`, train head on CPU | Saves GPU quota (our inference); Kaggle quota is in the guide's tooling page |

### Minimal code (not run by us)

Frozen image embeddings with timm. Quoted from the [timm feature extraction docs](https://github.com/huggingface/pytorch-image-models/blob/main/hfdocs/source/feature_extraction.mdx), with the head training assembled by us. Not run by us.

```python
# not run by us — timm call quoted; rest assembled
import timm, torch, numpy as np
from sklearn.linear_model import LogisticRegression
m = timm.create_model('resnet50', pretrained=True, num_classes=0).eval()  # -> (N, 2048)
with torch.no_grad():
    feats = torch.cat([m(xb) for xb in loader]).numpy()
np.save("feats.npy", feats)
clf = LogisticRegression(max_iter=2000).fit(feats[train_idx], y[train_idx])
```

Few-shot text with SetFit, condensed from the [SetFit README](https://github.com/huggingface/setfit), which prints about 0.869 accuracy on SST-2 with 8 examples per class (author-reported). Not run by us.

```python
# not run by us — condensed from SetFit README
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
train_ds = sample_dataset(dataset["train"], label_column="label", num_samples=8)
model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2",
                                    labels=["negative", "positive"])
args = TrainingArguments(batch_size=16, num_epochs=4, eval_strategy="epoch",
                         save_strategy="epoch", load_best_model_at_end=True)
trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                  metric="accuracy", column_mapping={"sentence": "text", "label": "label"})
trainer.train()
```

Full fine-tune with T4 settings. Assembled from the [transformers text-classification guide](https://github.com/huggingface/transformers/blob/main/docs/source/en/tasks/sequence_classification.md) plus `fp16` and `EarlyStoppingCallback` from the transformers source; `push_to_hub` was dropped. Not run by us.

```python
# not run by us — assembled; transformers 5.x argument names
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
args = TrainingArguments(output_dir="out", learning_rate=2e-5,
    per_device_train_batch_size=16, gradient_accumulation_steps=2,
    num_train_epochs=3, weight_decay=0.01, fp16=True,          # T4: fp16, not bf16
    eval_strategy="epoch", save_strategy="epoch",
    load_best_model_at_end=True, metric_for_best_model="f1")
trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
    processing_class=tokenizer, data_collator=collator, compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)])
trainer.train()
```

LoRA, quoted from the [PEFT README](https://github.com/huggingface/peft). Not run by us.

```python
# not run by us — quoted from PEFT README
from peft import LoraConfig, TaskType, get_peft_model
peft_config = LoraConfig(r=16, lora_alpha=32, task_type=TaskType.CAUSAL_LM)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```

Detection, quoted from the [Ultralytics README](https://github.com/ultralytics/ultralytics), with `patience` added by us. The licence is **AGPL-3.0**. Not run by us.

```python
# not run by us — README calls; patience added
from ultralytics import YOLO
model = YOLO("yolo26n.pt")
model.train(data="data.yaml", epochs=100, imgsz=640, patience=15)
metrics = model.val()
model.export(format="onnx")
```

Audio preprocessing, quoted from the [transformers audio-classification guide](https://github.com/huggingface/transformers/blob/main/docs/source/en/tasks/audio_classification.md). Not run by us.

```python
# not run by us — quoted
from datasets import Audio
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))  # wav2vec2/BEATs expect 16 kHz
```

### Pitfalls

**Near-duplicates across train and validation.** This is the silent killer. Barz and Denzler report that 3.3% of CIFAR-10 and 10% of CIFAR-100 test images have near-duplicates in training, and accuracy fell "between 9% and 14% relative" on duplicate-free test sets (**SNIPPET-ONLY**, [arXiv 1902.00423](https://arxiv.org/pdf/1902.00423)). Our inference for a fix:

- Split by source (patient, video, user).
- Embed every item with DINOv2 or CLIP.
- Group cosine near-neighbours above a threshold.
- Use `GroupKFold` so a group never straddles train and validation.

**Fine-tuning a small dataset.** It overfits. Start frozen (CS231n), and use LP-FT rather than unfreezing everything at once.

**Models left in train mode.** OpenCLIP models start in train mode, so call `model.eval()` before extracting embeddings ([open_clip](https://github.com/mlfoundations/open_clip)).

**Wrong SigLIP 2 text padding.** If you call the model manually and do not pad to `max_length=64`, accuracy drops with no error message ([siglip2 doc](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/siglip2.md)).

**Old notebooks on transformers 5.x.** Use `processing_class=` instead of `tokenizer=`, and `eval_strategy` instead of `evaluation_strategy`. The Whisper blog still uses the old name ([Whisper blog](https://github.com/huggingface/blog/blob/main/fine-tune-whisper.md); [sequence_classification guide](https://github.com/huggingface/transformers/blob/main/docs/source/en/tasks/sequence_classification.md)). Pin versions at the start of the event.

**Choosing checkpoints on loss.** Pick on the task metric instead (see the Whisper example in the table above).

**Class imbalance.**

- In YOLO, use `cls_pw` ([train docs](https://github.com/ultralytics/ultralytics/blob/main/docs/en/modes/train.md)).
- Elsewhere, class-weighted cross-entropy plus macro-F1 is general practice, **UNVERIFIED** here. No built-in class-weight argument in the transformers Trainer was confirmed.

**Ultralytics `optimizer: auto` ignores your learning rate.** It picks MuSGD for runs longer than about 10k iterations and AdamW otherwise, and ignores the user's `lr0` ([default.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml)).

### Versions and licences (as of 2026-10-03)

| Package | Version | Released | Licence (PyPI) | Python |
|---|---|---|---|---|
| torch | 2.14.1 | 2026-09-30 | Apache-2.0 and BSD/MIT (composite) | ≥3.10 |
| transformers | 5.18.0 | 2026-09-30 | Apache-2.0 | ≥3.10 |
| timm | 1.0.30 | 2026-09-22 | Apache-2.0 (code; weights vary) | ≥3.8 |
| peft | 0.21.2 | 2026-10-01 | Apache | ≥3.10 |
| setfit | 1.2.0 | 2026-09-04 | Apache-2.0 | ≥3.9 |
| sentence-transformers | 6.1.0 | 2026-09-18 | Apache-2.0 | ≥3.10 |
| accelerate | 1.15.0 | 2026-09-09 | Apache | ≥3.10 |
| fastai | 2.8.12 | 2026-09-09 | Apache-2.0 | ≥3.10 |
| ultralytics | 8.4.171 | 2026-10-01 | **AGPL-3.0** | ≥3.8 |

All versions are from PyPI JSON, e.g. [transformers](https://pypi.org/pypi/transformers/json) and [ultralytics](https://pypi.org/pypi/ultralytics/json). Ultralytics also lists YOLO27 as "Coming Soon" with no launch date ([README](https://github.com/ultralytics/ultralytics)). Colab and Kaggle preinstalled versions were not checked.

### Learning resources

**Concepts.**

- [CS231n transfer learning notes](https://github.com/cs231n/cs231n.github.io/blob/master/transfer-learning.md).
- [PyTorch transfer learning tutorial](https://github.com/pytorch/tutorials/blob/main/beginner_source/transfer_learning_tutorial.py). It quotes 15–25 minutes on CPU for the feature-extractor case.

**Fastest hands-on.**

- The [fastai Quick Start](https://github.com/fastai/fastai/blob/main/nbs/quick_start.ipynb) (5-line models).
- [fastbook](https://github.com/fastai/fastbook). The code is GPL v3, but the prose "is not licensed for any redistribution or change of format", so link to it rather than copying it.

**Per-modality notebooks.**

- Hugging Face task guides for [image](https://github.com/huggingface/transformers/blob/main/docs/source/en/tasks/image_classification.md), [text](https://github.com/huggingface/transformers/blob/main/docs/source/en/tasks/sequence_classification.md) and [audio](https://github.com/huggingface/transformers/blob/main/docs/source/en/tasks/audio_classification.md).
- [ViT on beans](https://github.com/huggingface/blog/blob/main/fine-tune-vit.md).
- The [Whisper fine-tuning Colab](https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/fine_tune_whisper.ipynb).
- [SetFit notebooks](https://github.com/huggingface/setfit/tree/main/notebooks).
- The [DINOv3 linear-probe segmentation Colab](https://colab.research.google.com/github/facebookresearch/dinov3/blob/main/notebooks/foreground_segmentation.ipynb) (gated weights).
- [AST](https://github.com/YuanGongND/ast) (Colab inference).
- The [Ultralytics Colab tutorial](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb).
- The [PEFT README](https://github.com/huggingface/peft), which links a Whisper-large LoRA Colab.

---

## 4. ML inside a hackathon app: call an API first, ship ONNX second, train last

### Plain-English explanation

A software hackathon has a different goal from a datathon. You need a feature that works live in front of judges, not a leaderboard score. There are three ways to get ML into an app:

1. **Call a hosted API.** This means an LLM provider, or Hugging Face Inference Providers for open-weights models. It is fastest to build, but it costs per call, adds seconds of latency, and fails if the Wi-Fi does.
2. **Run a pretrained open model yourself.** You can run it on a server, or directly in the user's browser. It is free per call and private, but you handle the hosting.
3. **Train your own.** This is realistic mainly for small tabular or classical-ML problems where you already have data. scikit-learn trains in minutes, and the model exports to a tiny ONNX file.

For LLM features specifically, the order of attempts is:

1. Prompting.
2. Putting the documents in the prompt.
3. Retrieval-augmented generation (RAG).
4. Fine-tuning, last.

Anthropic's guidance is that if a knowledge base is "smaller than 200,000 tokens (about 500 pages of material), you can just include the entire knowledge base in the prompt ... with no need for RAG." Prompt caching cuts latency by more than 2× and cost by up to 90% (**vendor-reported**, [Anthropic contextual retrieval](https://www.anthropic.com/news/contextual-retrieval)). Fine-tuning helps format, tone, consistency, cost and latency. Secondary sources summarise OpenAI's position as "Fine-tuning doesn't make the model smarter" (**SNIPPET-ONLY**, not checked against OpenAI docs, [secondary source](https://theneuralbase.com/openai-fine-tuning/qna/when-to-fine-tune-vs-prompt-engineer/)).

### Decision table: API vs pretrained vs train your own

| Criterion | Hosted API (LLM API / HF Inference Providers) | Pretrained open model (server or browser) | Train your own (sklearn / small net) |
|---|---|---|---|
| Labelled data needed | None | None (zero-shot) or a few hundred for embeddings + kNN/logistic | Yes, clean and relevant |
| Build time | Hours | Hours to a day | A day+ including evaluation |
| Latency | Seconds per call | Fast once warm; browser has a first-load download | Near-instant on CPU (ONNX) |
| Cost at demo | Per call; HF free credits are **$0.10/month (RE-CHECK)** | Free per call; hosting may cost | Free |
| Privacy | Data leaves the device | Browser or on-device keeps data local | Local |
| Demo risk | Rate limits, Wi-Fi, keys | Cold starts, model download size | Lowest |
| Best for | Generation, open-ended text and vision tasks | Classification, embeddings, speech, detection; offline or privacy pitches | Tabular prediction on your own data |

The table synthesises [HF Inference Providers pricing](https://raw.githubusercontent.com/huggingface/hub-docs/main/docs/inference-providers/pricing.md), the [ONNX Runtime Web README](https://raw.githubusercontent.com/microsoft/onnxruntime/main/js/web/README.md) and [scikit-learn model persistence](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/model_persistence.rst). No primary source gives a hackathon-specific table, so the arrangement is ours.

### Decision table: fine-tuning vs prompting vs RAG

| Need | Approach | 24–48 h realism | Source |
|---|---|---|---|
| Behaviour, format, few examples | Prompt + few-shot + structured output | Hours | [Anthropic prompt engineering overview](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview) |
| Answers grounded in <200k tokens of docs | Whole corpus in context + prompt caching | Hours | [Anthropic](https://www.anthropic.com/news/contextual-retrieval) (vendor-reported) |
| Larger corpus | RAG (see the guide's `04-ai-and-rag/docs/rag-architecture.md`) | Half a day for basic | Contextual Retrieval cuts failed retrievals 49%, 67% with reranking (vendor-reported, same source) |
| Consistent style or format, lower cost or latency at scale | LoRA/QLoRA of a 1–8B model (Unsloth free Colab, Axolotl) | Feasible: a few hours of training **after** building hundreds of clean examples | [PEFT](https://raw.githubusercontent.com/huggingface/peft/main/README.md); [Unsloth](https://raw.githubusercontent.com/unslothai/unsloth/main/README.md); [Axolotl](https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/README.md) |
| New knowledge | Not fine-tuning; use context or RAG | — | (secondary sources, SNIPPET-ONLY) |
| Hosted fine-tuning on a vendor API | Check tier gating and price | **RE-CHECK** | OpenAI Cookbook notes GPT-4o mini fine-tuning was gated to tiers 4–5 at time of writing, possibly outdated ([cookbook](https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/How_to_finetune_chat_models.ipynb)) |

Before choosing any of these, follow Anthropic's instruction: have "a clear definition of the success criteria" and "ways to empirically test" ([prompt engineering overview](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview)). In practice that means a 20–50-case evaluation set. Unsloth's "2× faster with 70% less VRAM with no accuracy loss" is a **vendor claim**. Its free notebooks cover Gemma 4 E2B, Qwen3.5 4B, gpt-oss 20B, Llama 3.1 8B and Llama 3.2 1B/3B ([Unsloth README](https://raw.githubusercontent.com/unslothai/unsloth/main/README.md)). Axolotl requires Python ≥3.12 and PyTorch ≥2.13. Its quickstart is `axolotl fetch examples` followed by `axolotl train examples/llama-3/lora-1b.yml` ([Axolotl README](https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/README.md)). The bottleneck is building the dataset, not GPU time (our inference). Serving a fine-tuned open model also brings back GPU hosting and cold-start risk.

### Decision table: serving pattern

| Model | Serve with | Why |
|---|---|---|
| scikit-learn / GBDT, your own | **skl2onnx → onnxruntime inside FastAPI** | No pickle code-execution risk; no sklearn version lock; "much less RAM than Python" ([sklearn persistence](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/model_persistence.rst)) |
| scikit-learn, same environment guaranteed | joblib in FastAPI or Streamlit | Simple, but needs identical versions; "Loading can execute arbitrary code" (same source) |
| PyTorch model | `torch.onnx.export(..., dynamo=True)` → onnxruntime | Dynamo is "the recommended exporter"; TorchScript path "no longer recommended" ([PyTorch ONNX tutorial](https://raw.githubusercontent.com/pytorch/tutorials/main/beginner_source/onnx/export_simple_model_to_onnx_tutorial.py)) |
| HF transformers model, need batching | BentoML (`@bentoml.api(batchable=True)`) or LitServe | Packaging plus batching ([BentoML](https://raw.githubusercontent.com/bentoml/BentoML/main/README.md); [LitServe](https://raw.githubusercontent.com/Lightning-AI/LitServe/main/README.md)) |
| Need a UI and an API in 10 lines | Gradio (`gr.Interface(..., api_name="predict")`) | ([Gradio README](https://raw.githubusercontent.com/gradio-app/gradio/main/README.md)) |
| Need it to run with no server | transformers.js or onnxruntime-web in the browser | Privacy, no hosting; WASM runs all `ai.onnx` and `ai.onnx.ml` ops, so sklearn ONNX runs in-browser ([ORT Web](https://raw.githubusercontent.com/microsoft/onnxruntime/main/js/web/README.md)) |

### Hugging Face in 2026: what changed

There are three ways to use a Hugging Face model:

- **Locally:** `pipeline(task, model)` is the fastest path ([pipeline tutorial](https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/en/pipeline_tutorial.md)).
- **Hosted:** **Inference Providers** exposes "200+ models" through an OpenAI-compatible router at `https://router.huggingface.co/v1`, with "no markup from Hugging Face" ([Inference Providers](https://raw.githubusercontent.com/huggingface/hub-docs/main/docs/inference-providers/index.md)). You need a fine-grained token with the "Make calls to Inference Providers" permission. Monthly credits are **$0.10 free and $2.00 PRO, "subject to change" (RE-CHECK)**, so the free credits will not cover a demo day. A "Custom Provider Key" is billed by the provider ([pricing](https://raw.githubusercontent.com/huggingface/hub-docs/main/docs/inference-providers/pricing.md)).
- **On Spaces:** these terms changed in 2026 and are covered next.

**The Spaces paid-plan change (RE-CHECK).** The current docs say:

> "Static Spaces are free for everyone. Gradio and Docker Spaces run on compute and require a paid plan to create ... Free personal accounts in good standing can still host up to 2 Gradio Spaces running on ZeroGPU."

Free hardware still "go[es] to sleep" when unused, and the 50 GB disk is not persistent ([Spaces overview](https://raw.githubusercontent.com/huggingface/hub-docs/main/docs/hub/spaces-overview.md)). ZeroGPU details ([ZeroGPU docs](https://raw.githubusercontent.com/huggingface/hub-docs/main/docs/hub/spaces-zerogpu.md)):

- It supports the Gradio SDK only.
- Functions get 60 s of GPU by default, adjustable with `@spaces.GPU(duration=120)`.
- Free accounts need a verified email and must be older than 30 days.
- PRO gets 8× the daily quota.

The date the change took effect was not confirmed here. The guide's `tooling-2026-update.md` hosting table cites forum posts placing the Docker restriction around July 2026. **Older tutorials that say "deploy free to a Gradio Space" are now wrong for new free accounts.**

**Licences and safety on the Hub.**

- Hub licence tags include `apache-2.0` and `mit`, the OpenRAIL family, the Llama community licences, `gemma` and `other` ([Hub licences](https://raw.githubusercontent.com/huggingface/hub-docs/main/docs/hub/repositories-licenses.md)). Prefer `apache-2.0` or `mit`.
- Gated models require you to "share contact information", so request access before the event ([gated models](https://raw.githubusercontent.com/huggingface/hub-docs/main/docs/hub/models-gated.md)).
- Prefer `.safetensors` weights. Pickle files allow "arbitrary code execution attacks" ([HF pickle security](https://raw.githubusercontent.com/huggingface/hub-docs/main/docs/hub/security-pickle.md)).

### In-browser models

transformers.js v4 mirrors the Python `pipeline` API ([transformers.js README](https://raw.githubusercontent.com/huggingface/transformers.js/main/README.md)):

- It runs on WASM (CPU) by default, or on WebGPU with `device: 'webgpu'`.
- The default dtype is `q8` on WASM and `fp32` on WebGPU, and `q4` is available.

The transformers.js WebGPU guide states: "As of March 2026, global WebGPU support is around 85% (according to caniuse.com)". Firefox needs a flag, and Safari support is version-dependent ([WebGPU guide](https://raw.githubusercontent.com/huggingface/transformers.js/main/packages/transformers/docs/source/guides/webgpu.md)). ONNX Runtime Web's compatibility table lists WebGPU on Chrome and Edge but not on Safari, iOS or Firefox. It also calls WebGL "in maintenance mode" ([ORT Web](https://raw.githubusercontent.com/microsoft/onnxruntime/main/js/web/README.md)). The two sources may differ in how recent they are. Either way, **make WASM the default and WebGPU an enhancement.**

TensorFlow.js has had no npm release since October 2024 ([npm](https://registry.npmjs.org/@tensorflow/tfjs)). For new projects, prefer transformers.js, onnxruntime-web or MediaPipe (`@mediapipe/tasks-vision` 1.0.1). The MediaPipe web code pattern could not be fetched and is **UNVERIFIED**.

### Minimal code (not run by us)

Train, export to ONNX and serve with FastAPI. This is assembled from the [skl2onnx README](https://raw.githubusercontent.com/onnx/sklearn-onnx/main/README.md) and the [FastAPI README](https://raw.githubusercontent.com/fastapi/fastapi/master/README.md). The pydantic body model and the output format are **UNVERIFIED**; skl2onnx classifiers also emit probabilities in a second output. Not run by us.

```python
# not run by us — assembled. export.py
import numpy as np
from skl2onnx import to_onnx
onx = to_onnx(model, X_train[:1].astype(np.float32))   # inputs must be float32
with open("model.onnx", "wb") as f:
    f.write(onx.SerializeToString())
```

```python
# not run by us — assembled. app.py ; run with: uvicorn app:app (UNVERIFIED command)
import numpy as np, onnxruntime as rt
from fastapi import FastAPI
from pydantic import BaseModel
sess = rt.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])  # load once
inp = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
app = FastAPI()
class Rows(BaseModel):
    rows: list[list[float]]
@app.post("/predict")
def predict(body: Rows):
    X = np.asarray(body.rows, dtype=np.float32)
    return {"label": sess.run([label_name], {inp: X})[0].tolist()}
```

PyTorch to ONNX, quoted from the [PyTorch ONNX tutorial](https://raw.githubusercontent.com/pytorch/tutorials/main/beginner_source/onnx/export_simple_model_to_onnx_tutorial.py). It needs `pip install --upgrade onnx onnxscript`. Not run by us.

```python
# not run by us — quoted
example_inputs = (torch.randn(1, 1, 32, 32),)
onnx_program = torch.onnx.export(torch_model, example_inputs, dynamo=True)
```

Hosted open-weights model through HF Inference Providers, quoted from the [HF docs](https://raw.githubusercontent.com/huggingface/hub-docs/main/docs/inference-providers/index.md). Not run by us.

```python
# not run by us — quoted
import os
from openai import OpenAI
client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=os.environ["HF_TOKEN"])
completion = client.chat.completions.create(model="openai/gpt-oss-120b:fastest",
    messages=[{"role": "user", "content": "How many 'G's in 'huggingface'?"}])
```

In-browser model, quoted from the [transformers.js README](https://raw.githubusercontent.com/huggingface/transformers.js/main/README.md) and [dtypes guide](https://raw.githubusercontent.com/huggingface/transformers.js/main/packages/transformers/docs/source/guides/dtypes.md). Not run by us.

```js
// not run by us — quoted
import { pipeline } from '@huggingface/transformers';
const pipe = await pipeline('sentiment-analysis');            // WASM, q8 by default
const out = await pipe('I love transformers!');
const gen = await pipeline("text-generation", "onnx-community/Qwen2.5-0.5B-Instruct",
                           { dtype: "q4", device: "webgpu" });
```

Gradio UI plus API, quoted from the [Gradio README](https://raw.githubusercontent.com/gradio-app/gradio/main/README.md). Not run by us.

```python
# not run by us — quoted
import gradio as gr
def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)
demo = gr.Interface(fn=greet, inputs=["text", "slider"], outputs=["text"], api_name="predict")
demo.launch()
```

### Pitfalls

**Pickle is not safe to load from untrusted sources.** Python's docs say "The `pickle` module **is not secure** ... malicious pickle data ... execute arbitrary code during unpickling" ([Python pickle docs](https://raw.githubusercontent.com/python/cpython/main/Doc/library/pickle.rst)). Never accept user-uploaded model files. `skops.io` is a safer format for scikit-learn models ([sklearn persistence](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/model_persistence.rst)).

**Version mismatches.** A model saved with one scikit-learn version and loaded with another raises `InconsistentVersionWarning`. None of the pickle-based methods support loading across versions (same source). Pin the training version in the serving requirements, or ship ONNX.

**Not every model converts to ONNX.** Some models and pipelines are unsupported (same source), and skl2onnx needs float32 inputs. The last supported opset is 21 ([skl2onnx](https://raw.githubusercontent.com/onnx/sklearn-onnx/main/README.md)).

**Cold starts.**

- Free hosts sleep. Render's free tier reportedly spins down after 15 minutes idle (**SNIPPET-ONLY, RE-CHECK**).
- Reported wake times of 30–90 s are anecdotal and **UNVERIFIED** ([GitHub issue](https://github.com/Sponti-App/Sponti/issues/125)).
- Mitigations (our inference):
  - Load the model once at startup, never per request.
  - Bake weights into the image at build time.
  - Hit the endpoint about 5 minutes before judging.
  - Keep a recorded fallback.
- The guide's `tooling-2026-update.md` has the free-host table.

**Browser model size.** The model download is the main cost. Our inference:

- Choose models in the tens to low hundreds of MB.
- Use `q4` or `q8`.
- Load in a Web Worker so the UI does not freeze.
- Show a progress bar.
- Preload before the demo.

No primary source on per-tab memory limits was found.

**Batching does not always help.** In `pipeline`, batching "may improve speed ... but it isn't guaranteed" ([pipeline tutorial](https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/en/pipeline_tutorial.md)).

### Versions and licences (as of 2026-10-03; RE-CHECK on the day)

| Package | Version | Released | Licence |
|---|---|---|---|
| fastapi | 0.142.2 | 2026-09-30 | not in PyPI `license` field (not checked) |
| onnxruntime | 1.30.0 | 2026-09-10 | MIT |
| onnx | 1.23.1 | — | not checked |
| skl2onnx | 1.20.0 | 2026-01-30 | Apache-2.0 |
| bentoml | 1.4.39 | 2026-05-07 | Apache-2.0 |
| litserve | 0.2.19 | 2026-09-09 | Apache-2.0 |
| gradio | 6.29.1 | 2026-10-02 | not in PyPI `license` field (not checked) |
| streamlit | 1.65.0 | 2026-10-02 | not in PyPI `license` field (not checked) |
| huggingface_hub | 2.1.1 | — | not checked |
| unsloth | 2026.9.14 | 2026-10-01 | not in PyPI `license` field (not checked) |
| axolotl | 0.20.0 | 2026-09-30 | not in PyPI `license` field (not checked) |
| peft | 0.21.2 | 2026-10-01 | Apache |
| npm @huggingface/transformers | 4.3.0 | 2026-09-16 | not checked |
| npm onnxruntime-web | 1.30.0 | 2026-09-14 | not checked (ORT is MIT) |
| npm @mediapipe/tasks-vision | 1.0.1 | 2026-07-31 | not checked |
| npm @tensorflow/tfjs | 4.22.0 | **2024-10-21 (stale)** | not checked |

Versions are from the PyPI JSON API (e.g. [fastapi](https://pypi.org/pypi/fastapi/json), [onnxruntime](https://pypi.org/pypi/onnxruntime/json)) and the npm registry (e.g. [@huggingface/transformers](https://registry.npmjs.org/@huggingface/transformers)).

### Learning resources

**Python serving.**

- transformers [pipeline tutorial](https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/en/pipeline_tutorial.md).
- scikit-learn [model persistence](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/model_persistence.rst).
- PyTorch [ONNX export tutorial](https://raw.githubusercontent.com/pytorch/tutorials/main/beginner_source/onnx/export_simple_model_to_onnx_tutorial.py).

**Hugging Face hosting.** HF [Inference Providers](https://raw.githubusercontent.com/huggingface/hub-docs/main/docs/inference-providers/index.md) and [Spaces overview](https://raw.githubusercontent.com/huggingface/hub-docs/main/docs/hub/spaces-overview.md).

**In-browser.** [transformers.js README](https://raw.githubusercontent.com/huggingface/transformers.js/main/README.md) and [WebGPU guide](https://raw.githubusercontent.com/huggingface/transformers.js/main/packages/transformers/docs/source/guides/webgpu.md).

**LLM strategy.**

- Anthropic [contextual retrieval](https://www.anthropic.com/news/contextual-retrieval) and [prompt engineering overview](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview).
- OpenAI Cookbook [fine-tuning chat models](https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/How_to_finetune_chat_models.ipynb).
- [Unsloth notebooks](https://raw.githubusercontent.com/unslothai/unsloth/main/README.md).

---

## 5. Getting data with no dataset provided: hold out a human-labelled test set first

### Plain-English explanation

Many software hackathons hand you a problem but no data. You have six ways to get some:

1. **Find a public dataset.** Sources include Kaggle, the Hugging Face Hub, UCI, OpenML, government portals and Google Dataset Search.
2. **Label your own** in a labelling tool.
3. **Write rules that label for you** (weak supervision), then combine their noisy votes.
4. **Ask an LLM to label.**
5. **Generate synthetic data.**
6. **Scrape the web.**

One rule ties all six together. **Before you use any shortcut, hand-label a small random sample yourself and set it aside as the test set.** That sample is how you prove the shortcut works. It answers the judge's inevitable question, "how do you know the labels are right?" Pangakis et al.'s paper title states the principle: "Automated Annotation with Generative AI **Requires Validation**" (**SNIPPET-ONLY**; author names **UNVERIFIED**, [arXiv 2306.00176](https://arxiv.org/pdf/2306.00176)).

### Decision table: where data can come from

| Route | When | Tooling | Main caveat |
|---|---|---|---|
| Public dataset | A close-enough dataset exists | Kaggle CLI, HF `datasets`, `ucimlrepo`, `openml`, Data.gov, Google Dataset Search | Licence per dataset; re-uploads may lack rights to relicense |
| Hand labelling | Fewer than a few hundred items needed; any modality | Label Studio (mixed), doccano (text), CVAT (boxes, video), Argilla (needs a server) | Time: plan 100–300 items first |
| Weak supervision | Domain rules are expressible as code (keywords, regexes, lookups) | Snorkel (stale since Feb 2024) | Label quality unknown until checked against gold |
| Active learning | Big unlabelled pool, small labelling budget | modAL (stale), small-text | Setup time; needs a decent initial model |
| LLM labelling | Text tasks with clear categories | Any LLM API; distilabel; review in Argilla | Non-deterministic; must validate with kappa |
| Synthetic | Demo or UI data; filling rare classes; schema testing | Faker (format-valid), SDV (fitted tabular, **BUSL**), LLM text, augmentation (nlpaug, albumentations, AugLy) | Distribution mismatch; leakage; judge scepticism |
| Scraping | No API or dataset exists, and ToS allow it | requests/BeautifulSoup (not researched) | ToS enforceable as contract; robots.txt is etiquette, not permission |

The sources are cited in the subsections below.

**Public sources and their licences.**

- **Kaggle** shows a licence on every dataset page (**SNIPPET-ONLY**, [secondary](https://labelyourdata.com/articles/machine-learning/kaggle-datasets)). Competition data is stricter: "Participants must not use data other than the Data" unless the rules say otherwise (**SNIPPET-ONLY**, [example rules](https://www.kaggle.com/competitions/data-science-and-ai/rules)).
- **Papers with Code** was sunsetted on 24 July 2025 and now redirects to Hugging Face Trending Papers. Its historical data survives in the paperswithcode-data GitHub repo (**SNIPPET-ONLY**, [Coursera](https://www.coursera.org/articles/papers-with-code)).
- **Google Dataset Search** indexes pages marked up with schema.org `Dataset` or DCAT metadata, so the licence is whatever the hosting site states (**SNIPPET-ONLY**, [Google](https://developers.google.com/search/docs/appearance/structured-data/dataset)).
- **Data.gov** harvests "over 1000 different sources". A dataset count seen in a snippet was not confirmed (**SNIPPET-ONLY**, [catalog](https://catalog.data.gov/)).
- **Hugging Face** dataset-card licences are declared by uploaders, and some datasets are gated. Both points are **UNVERIFIED** because huggingface.co was blocked.
- **UCI**'s default licence is **UNVERIFIED**.

A practical licence check (our inference):

1. Read the dataset page's licence.
2. Trace it back to the upstream source, because re-uploads may not have the right to relicense.
3. Record the URL and licence in the README for the judges.

**How much to label.** No authoritative universal number exists. The available evidence points to diminishing returns after a few hundred examples for simple text classification:

- One study says 300 examples can give high accuracy, and that more "rarely substantially increases" performance. The quote's attribution to this exact paper is unconfirmed (**SNIPPET-ONLY**, [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0747563213002823)).
- A single-task blog experiment found going from 2 to 5 examples per class gained 15 points, and from 5 to 10 gained only 5 more (**SNIPPET-ONLY**, [TDS](https://towardsdatascience.com/how-many-labeled-examples-does-a-text-classifier-actually-need-i-measured-it/)).
- In few-shot settings, "simple baseline classifiers can get surprisingly close to state-of-the-art" (**SNIPPET-ONLY**, [arXiv 2101.12073](https://arxiv.org/pdf/2101.12073)).

Teach the method rather than a magic number:

1. Label 100–300 items.
2. Plot a learning curve at 50, 100, 200 and 400 labels, using `LearningCurveDisplay` from section 1.
3. Stop when the curve flattens.

Combine this with SetFit or frozen embeddings from section 3. No guidance for images was found.

**LLM labelling: promising, but it needs validating.**

- **Promising (SNIPPET-ONLY).** Gilardi, Alizadeh and Kubli (PNAS 2023) tested 6,183 tweets and news articles. ChatGPT zero-shot beat crowd workers by about 25 points on average. Intercoder agreement was about 56% for MTurk, 79% for trained annotators, 91% for ChatGPT at temperature 1 and 97% at temperature 0.2. Cost was under $0.003 per annotation ([PNAS](https://www.pnas.org/doi/pdf/10.1073/pnas.2305016120)).
- **But inconsistent.** Reiss finds ChatGPT non-deterministic enough that consistency can fall below scientific reliability thresholds, and that pooling repeated runs by majority vote helps (**SNIPPET-ONLY**, [arXiv 2304.11085](https://arxiv.org/pdf/2304.11085)).
- **And biased.** Törnberg warns of bias and unreliable results (**SNIPPET-ONLY**, via [Frontiers 2025](https://www.frontiersin.org/journals/social-psychology/articles/10.3389/frsps.2025.1460277/full)).
- **The literature's consistent recommendation** is to validate against human labels with agreement statistics. No specific sample sizes or kappa thresholds were confirmed, and none are invented here.

A validation recipe synthesised from these sources:

1. Have humans label a random gold sample independently, with two annotators overlapping on part of it.
2. Compute LLM-vs-human accuracy and Cohen's kappa (`cohen_kappa_score`), plus human-vs-human kappa as the ceiling.
3. Run the LLM at low temperature, or several times with a majority vote, and report run-to-run agreement.
4. Review the disagreements by hand.
5. **Never let the LLM alone label the test set.**

**Synthetic data.**

- **Faker** produces format-realistic values such as names and addresses. They are not drawn from any real distribution, so Faker is for demos and pipeline tests, not for training a model meant to generalise ([Faker README](https://raw.githubusercontent.com/joke2k/faker/master/README.rst)).
- **SDV** fits tabular data and reports "Column Shapes" and "Column Pair Trends" quality scores. These measure fidelity, not downstream usefulness ([SDV README](https://raw.githubusercontent.com/sdv-dev/SDV/main/README.md)).
- **Model collapse (contested scope).** Shumailov et al. (Nature 2024) warn that "indiscriminate use of model-generated content in training causes irreversible defects", in which the tails of the distribution disappear (**SNIPPET-ONLY**, [Nature](https://www.nature.com/articles/s41586-024-07566-y)). A follow-up questions how universal that is (**SNIPPET-ONLY**, [arXiv 2410.12954](https://arxiv.org/html/2410.12954v2)).
- **Three rules (our inference):**
  - Split before you synthesise: fit generators on the train split only, and deduplicate synthetic items against test.
  - Always evaluate on real data.
  - State the synthetic fraction in the pitch.
- No source on what ratio of synthetic to real data is safe was found.

**Scraping basics (not legal advice; US-centric).**

- RFC 9309 says robots.txt rules "are not a form of access authorization". They are an honour-based convention (**SNIPPET-ONLY**, [RFC 9309](https://datatracker.ietf.org/doc/html/rfc9309)).
- Terms of service bite. In *hiQ v. LinkedIn*, hiQ won its Computer Fraud and Abuse Act (CFAA) arguments at the Ninth Circuit, but the district court held LinkedIn's anti-scraping user agreement enforceable as a contract. The December 2022 settlement included a permanent injunction, $500,000 and deletion of the scraped data (**SNIPPET-ONLY**, [ZwillGen](https://www.zwillgen.com/alternative-data/hiq-v-linkedin-wrapped-up-web-scraping-lessons-learned/); [Ninth Circuit opinion](https://law.justia.com/cases/federal/appellate-courts/ca9/17-16783/17-16783-2022-04-18.html)).
- Checklist (our inference):
  - Prefer official APIs and published datasets.
  - Respect robots.txt and rate limits.
  - Read the ToS, especially behind a login.
  - Avoid personal data. GDPR exposure is **UNVERIFIED** here.
  - Do not commit scraped content to a public repo.
  - Cite the source in the README.
- EU/UK database rights and text-and-data-mining exceptions were not researched.

### Minimal code (not run by us)

Public data loaders, quoted from the [ucimlrepo README](https://raw.githubusercontent.com/uci-ml-repo/ucimlrepo/main/README.md) and the [Argilla README](https://raw.githubusercontent.com/argilla-io/argilla/develop/README.md). The OpenML and Kaggle lines are **UNVERIFIED**. Not run by us.

```python
# not run by us
from ucimlrepo import fetch_ucirepo
heart = fetch_ucirepo(id=45); X, y = heart.data.features, heart.data.targets
from datasets import load_dataset
ds = load_dataset("imdb", split="train[:100]")
# UNVERIFIED: openml.datasets.get_dataset(<id>).get_data(target=...)
# UNVERIFIED: kaggle datasets download -d <owner>/<slug>   (needs API token)
```

Labelling tool start-up commands, quoted from the [Label Studio README](https://raw.githubusercontent.com/HumanSignal/label-studio/develop/README.md). The tool uses SQLite by default. Not run by us.

```bash
# not run by us — quoted
pip install label-studio && label-studio            # http://localhost:8080
docker run -it -p 8080:8080 -v $(pwd)/mydata:/label-studio/data heartexlabs/label-studio:latest
```

Active learning, quoted from the [modAL README](https://raw.githubusercontent.com/modAL-python/modAL/master/README.md). Not run by us.

```python
# not run by us — quoted
from modAL.models import ActiveLearner
from sklearn.ensemble import RandomForestClassifier
learner = ActiveLearner(estimator=RandomForestClassifier(), X_training=X_training, y_training=y_training)
query_idx, query_inst = learner.query(X_pool)
learner.teach(X_pool[query_idx], y_new)
```

Validating LLM labels against a human gold set. This is assembled from scikit-learn functions and is illustrative. Not run by us.

```python
# not run by us — assembled
from sklearn.metrics import cohen_kappa_score, accuracy_score
from collections import Counter
# gold: human labels on a random sample; llm_runs: list of 3 label lists from repeated LLM runs
llm_vote = [Counter(v).most_common(1)[0][0] for v in zip(*llm_runs)]   # majority vote (Reiss)
print("LLM vs human  acc", accuracy_score(gold, llm_vote), "kappa", cohen_kappa_score(gold, llm_vote))
print("human vs human kappa (ceiling)", cohen_kappa_score(gold_annot_a, gold_annot_b))
print("run-to-run kappa", cohen_kappa_score(llm_runs[0], llm_runs[1]))
```

Synthetic data, quoted from the [SDV README](https://raw.githubusercontent.com/sdv-dev/SDV/main/README.md) and the [Faker README](https://raw.githubusercontent.com/joke2k/faker/master/README.rst). The licence is **BUSL-1.1** for SDV. Not run by us.

```python
# not run by us — quoted (fit on the TRAIN split only)
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.evaluation.single_table import evaluate_quality
synth = GaussianCopulaSynthesizer(metadata)
synth.fit(data=train_df)
fake = synth.sample(num_rows=500)
evaluate_quality(train_df, fake, metadata)
from faker import Faker
fake_person = Faker(); fake_person.name(); fake_person.address()
```

The SDV import paths are from background knowledge, because the README snippet in the notes showed calls but not imports (**UNVERIFIED**).

### Pitfalls

**Labelling the test set with the same shortcut you are testing.** This makes the evaluation circular. Hold out the human gold set first.

**Leakage from synthesis.** Synthesising or paraphrasing from data that includes test rows leaks the test set into training.

**Training and testing on synthetic data.** That mostly measures how well the model learned the generator.

**Stale tools.**

- Snorkel's last release was February 2024, and the team "is now focusing their efforts on Snorkel Flow", its commercial product ([Snorkel README](https://raw.githubusercontent.com/snorkel-team/snorkel/main/README.md)).
- modAL's last release was June 2023, and doccano's July 2023.
- Use an isolated virtual environment and pin versions.
- The Snorkel `@labeling_function` / `LabelModel` pattern is **UNVERIFIED** here.

**Argilla needs a server.** The README calls the HF Spaces deployment "easiest" ([Argilla README](https://raw.githubusercontent.com/argilla-io/argilla/develop/README.md)), but that is now affected by the Spaces paid-plan change in section 4 (our inference; **RE-CHECK**). huggingface.co may also be blocked on some venue networks.

**CVAT self-hosting is heavy.** It needs Docker Compose with several services. CVAT Online's free plan has limits that "vary by plan" (**RE-CHECK**, [CVAT README](https://raw.githubusercontent.com/cvat-ai/cvat/develop/README.md)).

**Setup times are estimates, not measurements** (our judgement from the install paths):

- Under 5 minutes: Faker, ucimlrepo, `datasets`, modAL.
- About 10–20 minutes: Label Studio and doccano.

### Versions and licences (as of 2026-10-03)

| Package | Version | Released | Licence | Python |
|---|---|---|---|---|
| label-studio | 1.23.2 | 2026-09-29 | Apache-2.0 | ≥3.10,<4 |
| doccano | 1.8.4 | 2023-07-20 | MIT | ≥3.8,<4 |
| argilla (client) | 2.8.0 | 2025-03-10 | Apache-2.0 | ≥3.9 |
| cvat-cli | 2.77.0 | — | MIT (core MIT; some assets separately licensed) | ≥3.10 |
| snorkel | 0.10.0 | 2024-02-27 | Apache-2.0 | ≥3.11 |
| modAL-python | 0.4.2.1 | 2023-06-01 | MIT | none declared |
| small-text | 1.4.1 (2.0.0.dev4 pre-release) | 2024-08-18 | MIT | ≥3.7 (README says 3.10+) |
| Faker | 40.40.0 | 2026-09-29 | MIT | ≥3.10 |
| **sdv** | 1.38.5 | 2026-09-28 | **BUSL-1.1** | ≥3.9,<3.15 |
| distilabel | 1.5.3 | 2025-01-28 | Apache-2.0 | ≥3.9 |
| cleanlab | 2.9.0 | — | Apache-2.0 | ≥3.10 |
| nlpaug | 1.1.11 | — | MIT | ≥3.7 |
| albumentations | 2.0.8 | — | MIT | ≥3.9 |
| augly | 1.0.0 | — | MIT | ≥3.6 |
| datasets | 5.0.1 | — | Apache-2.0 | ≥3.10 |
| openml | 0.15.1 | — | BSD-3-Clause | ≥3.8 |
| ucimlrepo | 0.0.7 | — | MIT | ≥3.7 |
| kaggle | 2.2.4 | — | Apache-2.0 | ≥3.11 |

All versions are from PyPI JSON, e.g. [sdv](https://pypi.org/pypi/sdv/json) and [label-studio](https://pypi.org/pypi/label-studio/json). Release dates are given only where they were fetched.

**SDV licence warning.** SDV's Business Source License 1.1 "is not an Open Source license". Its Additional Use Grant permits use provided you do not offer "a Synthetic Data Service", meaning a commercial offering that gives third parties synthetic-data-creation functionality. Each release converts to MIT "four years from release date" ([SDV LICENSE](https://raw.githubusercontent.com/sdv-dev/SDV/main/LICENSE)). A hackathon prototype is fine under the grant. A team pitching a commercial synthetic-data product must flag it (our reading; not legal advice).

### Learning resources

**Loaders and labelling tools.** The [ucimlrepo README](https://raw.githubusercontent.com/uci-ml-repo/ucimlrepo/main/README.md), [Label Studio README](https://raw.githubusercontent.com/HumanSignal/label-studio/develop/README.md) and [small-text README](https://raw.githubusercontent.com/webis-de/small-text/main/README.md).

**Weak supervision.** The Snorkel tutorials linked from the [Snorkel README](https://raw.githubusercontent.com/snorkel-team/snorkel/main/README.md).

**LLM-labelling evidence.** [Gilardi et al., PNAS 2023](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10372638/), [Pangakis et al.](https://arxiv.org/pdf/2306.00176) and [Reiss](https://arxiv.org/pdf/2304.11085) (all **SNIPPET-ONLY**).

**Scraping law.** [RFC 9309](https://datatracker.ietf.org/doc/html/rfc9309) and the [ZwillGen hiQ wrap-up](https://www.zwillgen.com/alternative-data/hiq-v-linkedin-wrapped-up-web-scraping-lessons-learned/).

---

## 6. Lightweight experiment tracking: one JSONL line per run beats an unshared dashboard

### Plain-English explanation

Experiment tracking means writing down, for every model you train, exactly what you did and what score you got. The goal is that at 3 a.m. you can still answer three questions: which run was best, what code produced it, and was the gain real or noise?

A 2–4 person team with no infrastructure does not need a platform. The guide's toolbox already recommends a `feature_log.csv` plus a config file in Git. This section adds what to log, how to share it, and when a real tracker is worth it. Our recommended default, an inference from the sources, is:

- An **append-only JSONL or CSV file committed to Git**. It has zero dependencies, merges trivially, and loads straight into pandas for the leaderboard slide.
- Add **Trackio** when you want live curves or a shared dashboard. It is "local-first ... you shouldn't need to make an account", stores runs in SQLite, and is API-compatible with W&B, so `import trackio as wandb` works ([Trackio README](https://raw.githubusercontent.com/gradio-app/trackio/main/README.md)).

### Decision table: which tracker

| Option | Account needed | Shared across laptops | Setup | Best when | Caveat |
|---|---|---|---|---|---|
| JSONL/CSV in Git (our default) | No | Yes, via Git | 2 min | Any team | No curves; discipline needed |
| **Trackio** | No (local); HF token with write permission for a shared Space | Yes, via `space_id="user/space"` (auto-deploys an HF Space) or a self-hosted `server_url` | 5 min | Want W&B-style curves for free | README says hosting on HF is free; Space limits **RE-CHECK** in light of the Spaces changes in section 4 |
| **MLflow** (local) | No | Only if someone runs a reachable server (`mlflow server --host 0.0.0.0`, venue LAN **UNVERIFIED**) | 5–10 min | One person running many tabular runs; want model registry | New projects default to **SQLite since 3.7.0**; the file store is deprecated |
| **W&B** | Yes (every member) | Yes | 5 min | Team already uses W&B | Free-tier limits disputed, **RE-CHECK** |
| **DVC experiments** (dvclive) | No | Via Git | 20+ min | Team already uses DVC | Needs a Git repo; `dvc exp run` needs a DVC pipeline |
| **Aim** | No | Self-hosted (`aim up`) | 10 min | Want local queries over runs | Last release May 2025 |

Sources: [Trackio README](https://raw.githubusercontent.com/gradio-app/trackio/main/README.md), [MLflow README](https://raw.githubusercontent.com/mlflow/mlflow/master/README.md), [MLflow CHANGELOG](https://raw.githubusercontent.com/mlflow/mlflow/master/CHANGELOG.md), [W&B README](https://raw.githubusercontent.com/wandb/wandb/main/README.md), [DVC experiment tracking](https://raw.githubusercontent.com/iterative/dvc.org/main/content/docs/start/experiments/experiment-tracking.md), [DVC running experiments](https://raw.githubusercontent.com/iterative/dvc.org/main/content/docs/user-guide/experiment-management/running-experiments.md) and the [Aim README](https://raw.githubusercontent.com/aimhubio/aim/main/README.md).

**MLflow documentation conflict.** The tracking docs still say MLflow "logs data to the local `mlruns` directory" by default ([tracking docs](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/classic-ml/tracking/index.mdx)). The 3.7.0 changelog (2025-12-05) says new projects default to "SQLite as Default Backend ... unless existing mlruns data is detected". It also deprecates the file store in favour of a `mlflow migrate-filestore` command ([CHANGELOG](https://raw.githubusercontent.com/mlflow/mlflow/master/CHANGELOG.md)). Trust the changelog for new installs. MLflow records the Git commit in the tag `mlflow.source.git.commit` ([mlflow_tags.py](https://raw.githubusercontent.com/mlflow/mlflow/master/mlflow/utils/mlflow_tags.py)).

**W&B free tier: sources conflict (all RE-CHECK).**

- One source says up to 5 model seats, 5 GB of storage a month and 1 GB of Weave ingestion, with Pro at $60 a month (**SNIPPET-ONLY**, [ZenML](https://www.zenml.io/blog/wandb-pricing)).
- Another says a 100 GB limit with single-user-only access (**SNIPPET-ONLY**, [Spheron](https://www.spheron.network/blog/weights-biases-pricing-vs-self-hosted-mlflow-2026/)).
- An academic plan is described with 100 GB of free storage (**SNIPPET-ONLY**, [wandb pricing](https://wandb.ai/site/pricing/)).
- The pricing page title appeared as "CoreWeave Forge Pricing" in search results, which suggests a rebrand (**UNVERIFIED**).

### What to log for every run (our synthesis)

| Field | Why |
|---|---|
| run_id, timestamp, author | Who did what, when |
| `git rev-parse HEAD` + dirty flag | Reproduce the exact code |
| Data version: file names + hash or row counts of train/test; feature-set name | Catch "same code, different data" |
| Full config / hyperparameters | Reproduce the exact model |
| Random seed(s), including the CV split seed | Separate signal from seed noise |
| CV scheme (n_folds, group or time key) | Make scores comparable |
| **Metric per fold**, mean ± std | A gain smaller than the fold std is noise |
| Public leaderboard score, if submitted | Track CV vs leaderboard agreement |
| Runtime | Budget the remaining hours |
| Path to OOF predictions and submission file | Enables error analysis (section 2) and ensembling without retraining |
| Free-text note: what changed and why | The pitch narrative writes itself |

### Minimal code (not run by us)

A dependency-free logger. This is illustrative code, not from a source. Not run by us.

```python
# not run by us — illustrative
import json, subprocess, time, statistics
def log_run(cfg, fold_scores, note="", path="runs.jsonl"):
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"]).strip())
    rec = {"ts": time.strftime("%F %T"), "commit": commit, "dirty": dirty, **cfg,
           "folds": fold_scores, "cv_mean": statistics.mean(fold_scores),
           "cv_std": statistics.pstdev(fold_scores), "note": note}
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
# leaderboard slide: pd.read_json("runs.jsonl", lines=True).sort_values("cv_mean")
```

Trackio as a drop-in for W&B. Calls are from the [Trackio README](https://raw.githubusercontent.com/gradio-app/trackio/main/README.md) and the [W&B README](https://raw.githubusercontent.com/wandb/wandb/main/README.md). Not run by us.

```python
# not run by us — assembled from README calls
import trackio as wandb
wandb.init(project="hackathon", config={"lr": 3e-4, "model": "lgbm"},
           space_id="username/hackathon-dash")   # optional: shared dashboard on an HF Space
for epoch, (tr, va) in enumerate(history):
    wandb.log({"train_loss": tr, "val_loss": va})
wandb.finish()
# locally: `trackio show` ; query: trackio query project --project hackathon --sql "SELECT ..."
```

MLflow, local. The tracking URI comes from the [MLflow README](https://raw.githubusercontent.com/mlflow/mlflow/master/README.md). The `log_params` and `log_metric` calls are standard API from background knowledge, not re-checked this session (**UNVERIFIED**). Not run by us.

```python
# not run by us
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")   # after: uvx mlflow server
mlflow.set_experiment("hackathon")
with mlflow.start_run():
    mlflow.log_params(cfg)
    mlflow.log_metric("cv_mean", cv_mean)
```

DVCLive, quoted from the [DVC get-started docs](https://raw.githubusercontent.com/iterative/dvc.org/main/content/docs/start/experiments/experiment-tracking.md). `live.next_step()` is **UNVERIFIED** because the quoted snippet was truncated. Not run by us.

```python
# not run by us
from dvclive import Live
with Live() as live:
    live.log_param("epochs", NUM_EPOCHS)
    live.log_metric("val_f1", score); live.next_step()
# then: dvc exp show --md --sort-by val_f1
```

### Pitfalls

**Reporting seed noise as progress.** If a "gain" is smaller than the standard deviation across folds or seeds, do not put it in the pitch.

**Trackers nobody else can see.** A local MLflow on one laptop is invisible to teammates. Decide on the shared location in hour one.

**Committing large files.** Never commit large OOF or prediction files to Git. Log their paths, and keep the files in a shared drive or Kaggle dataset (our inference).

**Account friction mid-event.** W&B needs every member to have an account. A shared HF Space for Trackio needs an HF token with write permission, plus whatever the current Spaces plan allows (**RE-CHECK**).

**Stale trackers.** Aim's release pace has slowed: the last release was 3.29.1 in May 2025 ([PyPI](https://pypi.org/pypi/aim/json)).

**Not evaluated here.** Comet, Neptune and ClearML were out of scope.

### Versions and licences (as of 2026-10-03)

| Package | Version | Released | Licence | Python |
|---|---|---|---|---|
| trackio | 0.40.0 | 2026-09-30 | MIT | ≥3.10 |
| mlflow | 3.16.1 | 2026-09-16 | Apache-2.0 | ≥3.10 |
| wandb | 0.30.0 | 2026-09-09 | MIT (client library) | ≥3.10 |
| dvc | 3.67.1 | 2026-03-31 | Apache-2.0 | ≥3.9 |
| aim | 3.29.1 | 2025-05-08 | Apache-2.0 | ≥3.7 |

All versions are from PyPI JSON: [trackio](https://pypi.org/pypi/trackio/json), [mlflow](https://pypi.org/pypi/mlflow/json), [wandb](https://pypi.org/pypi/wandb/json), [dvc](https://pypi.org/pypi/dvc/json) and [aim](https://pypi.org/pypi/aim/json). The W&B hosted service's terms are separate from the MIT-licensed client.

### Learning resources

**Trackers.**

- [Trackio README](https://raw.githubusercontent.com/gradio-app/trackio/main/README.md).
- [MLflow README](https://raw.githubusercontent.com/mlflow/mlflow/master/README.md) and [tracking docs](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/classic-ml/tracking/index.mdx). Read these alongside the 3.7.0 changelog entry.
- [W&B README](https://raw.githubusercontent.com/wandb/wandb/main/README.md).
- [Aim README](https://raw.githubusercontent.com/aimhubio/aim/main/README.md).

**DVC.** [DVC experiment tracking](https://raw.githubusercontent.com/iterative/dvc.org/main/content/docs/start/experiments/experiment-tracking.md) and the [`dvc exp show` reference](https://raw.githubusercontent.com/iterative/dvc.org/main/content/docs/command-reference/exp/show.md).

---

## Conclusion

The six gaps share one failure mode: confidence without a comparison. Each has a matching check:

- A metric with no dummy baseline beside it.
- A fine-tuned model never compared against a frozen linear probe.
- LLM labels never scored against a human sample.
- A synthetic dataset evaluated on more synthetic data.
- A "gain" smaller than the fold-to-fold spread.

The guide's most valuable addition is therefore not any single library. It is a habit that recurs in every section: **hold out something real, and measure against the simplest alternative before believing a number.** That habit is also what judges can verify in a five-minute pitch.

The second lesson concerns how stale this kind of guidance gets. In roughly fifteen months, Hugging Face moved Gradio Spaces behind a paid plan, transformers and sentence-transformers moved to new major versions (transformers 5.x renamed arguments that old notebooks use), MLflow changed its default storage, and Papers with Code disappeared. Licence terms (AGPL, BUSL, gated custom licences) now matter as much as accuracy when a demo might become a startup. Every version table, price and quota in these pages therefore needs a "checked on" date and a named owner who re-checks it before each event. Without those, the guide's advice will start to break copy-pasted code within a few months.
