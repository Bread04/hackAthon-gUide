# ⚖️ LightGBM on Imbalanced Data: Probabilities, Thresholds & Tuning

<!-- markdownlint-disable MD013 -->

> How class imbalance, the quality of probability predictions, precision/recall trade-offs and false-positive vs false-negative costs fit together in a LightGBM model, plus the tuning settings that matter for rare classes. Every "our test" number comes from [`../07-worked-example/lightgbm_imbalance_lab.py`](../07-worked-example/lightgbm_imbalance_lab.py) (runs in ~7 s, in the smoke test). 📚 Source research: [`technical-lightgbm-imbalance-2026-10-03`](../../_research/technical-lightgbm-imbalance-2026-10-03/research.md).

## In one line

**Train plain, calibrate honestly, threshold on money.**


For an imbalanced binary problem in LightGBM, the safest default is to **train on the natural class mix with the plain `binary` objective, early-stop on a ranking metric, calibrate the out-of-fold probabilities, then choose the decision threshold from the costs of false positives and false negatives**. LightGBM's imbalance switches (`is_unbalance`, `scale_pos_weight`, sklearn `class_weight`) mostly do not improve ranking. They do inflate the predicted probabilities, and LightGBM's own docs warn that they cause "poor estimates of the individual class probabilities" ([Parameters.rst](https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/Parameters.rst)). Our lab on the guide's synthetic sample data (11.2% positives) shows the pattern clearly. Weighting moved PR-AUC by at most 0.006 (0.187 to 0.193). It roughly **2.6-fold inflated the mean predicted probability (0.101 to 0.266)** and worsened the Brier score from 0.103 to 0.157. Undersampling to 1:1 hurt both ranking and probabilities. This matters because a cost-based threshold only works on honest probabilities. The same "flag if p > 0.20" rule earned **+$5,700 on calibrated probabilities and lost $58,350 on uncalibrated weighted ones**. Tuning comes last and adds less than these choices. The guide's existing tuning recipe applies, plus a few imbalance-specific settings (`min_sum_hessian_in_leaf`, the early-stopping metric, `n_jobs`).

**Label key.** **Our test on synthetic data** marks results from [`../07-worked-example/lightgbm_imbalance_lab.py`](../07-worked-example/lightgbm_imbalance_lab.py). It was run on 2026-10-03 with Python 3.12, LightGBM 4.7.0 and scikit-learn 1.9.1 on the guide's synthetic sample data: about 2.6k rows, 11.2% positives, grouped 5-fold out-of-fold (OOF) predictions with no patient in two folds. It is one dataset and one seed, so read it as a demonstration of the pattern, not as a benchmark. **SNIPPET-ONLY** means the claim comes from a search snippet because the paper was blocked, so check it before relying on it. **UNVERIFIED** means it comes from a standard derivation or memory and has not been confirmed against a primary source. Code labelled **not run by us** was assembled from cited APIs and never executed. Versions are as of **2026-10-03**.

**What this page does not repeat.** Metric definitions, the confusion matrix, the PR-AUC versus ROC-AUC debate and the basic `CalibrationDisplay` call are in [`ml-fundamentals-and-metrics.md`](ml-fundamentals-and-metrics.md). The four core LightGBM settings, the 10-minute Optuna recipe on locked folds and rank-averaging are in [`hyperparameter-tuning-and-ensembling.md`](hyperparameter-tuning-and-ensembling.md). Fold design is in [`cross-validation-guide.md`](cross-validation-guide.md).

---

## 1. Imbalance switches move probabilities far more than ranking

### Plain-English explanation

A model trained on data that is 11% positive learns that positives are rare, and that is correct. Every imbalance "fix" tells the model to act as if positives were more common. It does this by counting each positive more heavily (weights) or by throwing negatives away (undersampling). The model then ranks people in much the same order, but it **reports higher probabilities for everyone**. Elkan's classic result is that rebalancing the training data is equivalent to moving the decision threshold (**SNIPPET-ONLY**, [Elkan 2001](https://www.researchgate.net/publication/2365611_The_Foundations_of_Cost-Sensitive_Learning)). A strong learner such as LightGBM therefore gains little from rebalancing that you could not get more cleanly by picking a lower threshold.

Mechanically, `is_unbalance` and `scale_pos_weight` multiply both the gradient and the hessian of each positive row's log-loss by a class weight. With `is_unbalance=True` the minority class weight is majority count divided by minority count. The two options are mutually exclusive: setting both raises `Cannot set is_unbalance and scale_pos_weight at the same time`. Neither changes the starting score, which `boost_from_average` still sets from the unweighted prevalence, so the upward shift is learned through the trees ([binary_objective.hpp](https://raw.githubusercontent.com/microsoft/LightGBM/master/src/objective/binary_objective.hpp)). The scikit-learn wrapper's `class_weight` is different. It is converted into per-row `sample_weight` before training, so it also moves the starting score to logit(0.5) = 0. Its docstring says to use it "only for multi-class classification", and for binary tasks to use `is_unbalance` or `scale_pos_weight` instead ([sklearn.py](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/sklearn.py)).

### Decision table: the options and what they do

| Option (exact name) | What it does | Effect on probabilities | Use it when |
|---|---|---|---|
| Nothing (default `objective="binary"`) | Plain log-loss, start score = logit(prevalence) | Closest to honest. Check anyway | **Default.** Handle imbalance at the threshold |
| `scale_pos_weight=w` (default 1.0, must be > 0) | Multiplies positive rows' gradient and hessian by `w` | Inflated. Odds rise by roughly a factor of `w` | You only need ranking and want to try a mild weight such as √(n_neg/n_pos) |
| `is_unbalance=True` | Sets minority weight = majority/minority automatically | Same as `scale_pos_weight=n_neg/n_pos` | Same as above. Cannot be combined with `scale_pos_weight` |
| `class_weight="balanced"` (sklearn wrapper) | Becomes `sample_weight` = n/(2·class count); start score 0 | Inflated | Multiclass softmax. For binary, prefer the two above |
| `pos_bagging_fraction`, `neg_bagging_fraction` + `bagging_freq>0` | Samples each class at its own rate each round, with no reweighting. Overrides `bagging_fraction` | Mildly inflated | Very large, very imbalanced data where you want speed |
| Under/oversampling, SMOTE (imbalanced-learn) | Changes the training rows | Strongly inflated | Rarely. Only inside each training fold. SMOTE needs `SMOTENC` for categoricals |
| Focal loss (custom objective) | Down-weights easy examples | Not calibrated. Returns raw margins | Experiments only. No evidence it beats `binary` on tabular data |
| Threshold moving | Leaves the model alone and changes the cut-off | Unchanged | **Always.** See sections 3–4 |

Sources for the table: [Parameters.rst](https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/Parameters.rst), [sklearn.py](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/sklearn.py), [bagging.hpp](https://raw.githubusercontent.com/microsoft/LightGBM/master/src/boosting/bagging.hpp), [imbalanced-learn over-sampling docs](https://raw.githubusercontent.com/scikit-learn-contrib/imbalanced-learn/master/doc/over_sampling.rst).

### Our test on synthetic data

| Setting | PR-AUC | ROC-AUC | Brier | Log loss | Mean predicted p (actual 0.112) |
|---|---|---|---|---|---|
| No weighting (default) | 0.187 | **0.625** | **0.103** | **0.371** | **0.101** |
| `scale_pos_weight=7.9` (= n_neg/n_pos) | 0.191 | 0.619 | 0.157 | 0.482 | 0.266 |
| `is_unbalance=True` | 0.193 | 0.617 | 0.157 | 0.482 | 0.266 |
| `class_weight="balanced"` | 0.191 | 0.617 | 0.159 | 0.487 | 0.275 |
| Undersample negatives 1:1 (train folds only) | 0.174 | 0.626 | 0.249 | 0.711 | 0.444 |

The ranking columns barely move. PR-AUC differences of 0.002–0.006 on about 290 positives are within fold noise, and ROC-AUC actually fell slightly with weighting. The probability columns move a lot. Weighting **inflated the mean predicted risk from 0.101 to 0.27**, and undersampling inflated it to 0.44, four times the real rate, while also losing PR-AUC because the model saw fewer negatives. `is_unbalance` and `scale_pos_weight=7.9` give the same calibration numbers, which matches the source code treating them alike. The guide's [`evaluate_and_explain.py`](../07-worked-example/evaluate_and_explain.py) shows the same effect for `class_weight="balanced"` (mean risk 0.443 against 0.112).

The outside evidence points the same way, but most of it is snippet-level. In logistic-regression simulations, undersampling, oversampling and SMOTE "yielded poorly calibrated models" that strongly overestimated minority probability, "did not result in higher areas under the ROC curve", and gave a sensitivity/specificity balance that "shifting the probability threshold" matched (**SNIPPET-ONLY**, [van den Goorbergh et al., JAMIA 2022](https://academic.oup.com/jamia/article/29/9/1525/6605096)). That study used logistic regression, not boosted trees. Across 73 datasets including LightGBM, XGBoost and CatBoost, Elor and Averbuch-Elor found balancing "does not improve prediction performance for the strong ones", and that oversampling helps with a fixed threshold but not once the threshold is optimised (**SNIPPET-ONLY**, [arXiv 2201.08528](https://arxiv.org/pdf/2201.08528)). **The evidence on oversampling conflicts.** Another snippet from the same paper says oversampling "significantly improved prediction for the weak decision tree and LightGBM". The likely reconciliation is fixed versus tuned threshold, but that is **UNVERIFIED**. A separate check by our researcher on a 2.5%-positive synthetic set also found ranking differences under 0.01 AUC/AP across all settings, focal loss included.

### Minimal code (not run by us)

The researcher confirmed that `eval_X`/`eval_y` works in LightGBM 4.7.0, where `eval_set` is now deprecated ([sklearn.py](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/sklearn.py)).

```python
# not run by us — LightGBM >= 4.7.0
import lightgbm as lgb
clf = lgb.LGBMClassifier(objective="binary", n_estimators=5000, learning_rate=0.03,
                         n_jobs=1, random_state=42, verbose=-1)   # no is_unbalance / scale_pos_weight
clf.fit(X_tr, y_tr, eval_X=X_es, eval_y=y_es,
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(200, first_metric_only=True)])
```

### Pitfalls

**Early stopping silently uses two metrics.** The sklearn wrapper always adds `binary_logloss`, and early stopping checks *all* metrics unless `first_metric_only=True` ([callback.py](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/callback.py)). Under weighting, the unweighted validation log loss penalises the inflated probabilities. Training then stops on a calibration signal while you think you are optimising ranking.

**Resampling before splitting inflates CV.** In imbalanced-learn's own example, resampling before cross-validation showed a CV balanced accuracy of 0.724 that fell to 0.698 on left-out data ([common_pitfalls.rst](https://raw.githubusercontent.com/scikit-learn-contrib/imbalanced-learn/master/doc/common_pitfalls.rst)). Resample only inside an `imblearn.pipeline.Pipeline`, and keep the early-stopping set at the natural class mix.

**Weights change regularisation.** Weights multiply hessians, so `min_sum_hessian_in_leaf` (alias `min_child_weight`) and `lambda_l2` bind differently after you add or change a weight. Retune them together.

**Focal loss has three traps.** If you try focal loss, pass it as a callable in `params["objective"]`, because `fobj=` no longer exists in `lgb.train`. `predict_proba` then returns raw scores with a warning, so apply the sigmoid yourself ([engine.py](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/engine.py)). The focal hessian can go negative and must be clipped, and there is no `boost_from_average`, so training starts from p = 0.5. Popular community code ([jrzaurin](https://raw.githubusercontent.com/jrzaurin/LightGBM-with-Focal-Loss/master/README.md)) uses the removed `fobj=` API.

---

## 2. Calibration: the cheapest fix is a cross-fitted Platt map on OOF scores

### Plain-English explanation

A model is **calibrated** when its numbers mean what they say. Among people scored about 0.2, about 20% should turn out positive ([scikit-learn calibration docs](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/modules/calibration.rst)). Ranking metrics cannot see calibration. You can multiply every score by two and ROC-AUC stays the same. That is why a team can win on AUC and still hand the judges a "35% risk" that really means 12%. Calibration matters whenever a probability is multiplied by money, shown to a user, or compared with a cost threshold. It matters less when you only take the top *k* cases. Van Calster and colleagues put it bluntly: "Miscalibration always reduces Net Benefit" (**SNIPPET-ONLY**, [BMC Medicine 2019](https://link.springer.com/article/10.1186/s12916-019-1466-7)).

Old evidence that boosted trees are badly miscalibrated (Niculescu-Mizil and Caruana 2005, **SNIPPET-ONLY**, [ICML05](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf)) concerns AdaBoost-style margins. LightGBM's `binary` objective minimises log loss directly, so unweighted LightGBM is usually close to calibrated. That is **UNVERIFIED** as a general claim, and our lab bears it out only roughly: the unweighted model's mean p was 0.101 against an actual 0.112. A 2026 large-scale study warns that Platt and isotonic "can systematically degrade proper scoring performance for strong modern tabular models" (**SNIPPET-ONLY**, [arXiv 2601.19944](https://arxiv.org/pdf/2601.19944)). So keep a calibrator only if it improves log loss on data it was not fitted on.

### How to measure it

Use four checks together, all on OOF or held-out data. Compare the **mean predicted p with the actual positive rate** (calibration-in-the-large). Draw a **reliability diagram with quantile bins** (`calibration_curve(..., strategy="quantile")`), because uniform bins leave the upper bins empty under imbalance. Report **log loss and Brier**, which are proper scores, but remember that a lower Brier "does not necessarily mean a better calibrated model" ([calibration.rst](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/modules/calibration.rst)). Treat **ECE** as secondary, since binning makes it biased and bin-count dependent (**SNIPPET-ONLY**, [Nixon et al. 2019](https://arxiv.org/abs/1904.01685)). scikit-learn ships no ECE function. `n_bins="cube_root"` arrives only in scikit-learn 1.10 and is not in 1.9.1 ([calibration.py](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/calibration.py)).

### Our test on synthetic data: four ways to fix probabilities

| Probabilities | PR-AUC | ROC-AUC | Brier | Log loss | Mean p (actual 0.112) |
|---|---|---|---|---|---|
| Weighted (`scale_pos_weight=7.9`), uncorrected | 0.191 | 0.619 | 0.157 | 0.482 | 0.266 |
| Weighted + odds correction p/(p+(1−p)w) | 0.191 | 0.619 | 0.102 | 0.403 | **0.066 (over-corrected)** |
| Weighted + cross-fitted Platt on OOF scores | 0.180 | 0.616 | **0.098** | **0.341** | **0.112** |
| Unweighted + cross-fitted Platt on OOF scores | 0.178 | 0.621 | **0.097** | **0.341** | **0.112** |
| Unweighted + 25% calibration slice held out per fold, sigmoid | 0.161 | 0.610 | 0.099 | 0.345 | — |
| Unweighted + 25% calibration slice held out per fold, isotonic | 0.158 | 0.606 | 0.100 | 0.352 | — |

The reliability diagram for weighted + cross-fitted Platt (quantile bins, predicted → observed) was **0.05→0.05, 0.08→0.10, 0.11→0.11, 0.13→0.12, 0.18→0.18**, which is essentially on the diagonal.

Three lessons follow. First, **the textbook odds correction overshot**. It halved the true mean (0.066 against 0.112), because the formula is exact only for an otherwise ideal model with uniform within-class weights. In LightGBM the weights also change which trees get built, through hessian-based leaf constraints ([Dal Pozzolo et al. 2015](https://www.semanticscholar.org/paper/Calibrating-Probability-with-Undersampling-for-Pozzolo-Caelen/e36bb7fbe1b4f7c521608e93a2215e2062dae5b1), **SNIPPET-ONLY**, for the undersampling version of the formula). Second, **cross-fitting Platt on OOF scores gave the best probabilities at no training-data cost**. For each fold, fit a two-parameter logistic map on the *other* folds' OOF scores and apply it to this fold. Third, **carving out a calibration slice inside each fold cost ranking**: PR-AUC fell from 0.187 to 0.161 because each model trained on 25% less data. On a dataset this small, losing training data hurts more than imperfect calibration. Isotonic did slightly worse than sigmoid with only a few hundred calibration rows per fold, as scikit-learn's "≪1000 samples" warning predicts ([calibration.py](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/calibration.py)).

Within a single fold, Platt is monotone and cannot change the ranking. The small PR-AUC drop after cross-fitted Platt (0.191 to 0.180) is, on our reading, because each fold gets a slightly different map, which reorders scores *across* folds when they are pooled. This is our interpretation, not a tested claim.

### Decision table: which calibrator

| Situation | Method (exact name) | Why |
|---|---|---|
| Few calibration rows, or few positives (the usual datathon case) | `method="sigmoid"` (Platt) | 2 parameters, keeps ranking. Best for "small sample sizes" ([calibration.rst](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/modules/calibration.rst)) |
| More than ~1,000 calibration rows *with plenty of positives*, non-S-shaped error | `method="isotonic"` | Flexible, but introduces ties that can change AUC |
| Platt's symmetric-error assumption fails (common under imbalance) | `betacal.BetaCalibration(parameters="abm")` (betacal 1.1.0) | 3 parameters ([betacal](https://raw.githubusercontent.com/betacal/python/master/README.md)) |
| Want the largest log-loss gains, accept a dependency | `venn_abers.VennAbersCalibrator` (1.5.4) | Ranked first in a 2026 benchmark, which is **SNIPPET-ONLY** and whose first author maintains a Venn-Abers ecosystem ([arXiv 2601.19944](https://arxiv.org/pdf/2601.19944); [README](https://raw.githubusercontent.com/ip200/venn-abers/main/README.md)) |
| Multiclass | `method="temperature"` (scikit-learn ≥ 1.8) | One parameter, keeps the argmax |
| You weighted with a known `w` and need a quick fix | p/(p+(1−p)·w) | Exact only in theory. **It over-corrected in our test**, so check the reliability diagram |

With heavy imbalance, the binding constraint is the **number of positives** in the calibration data, not total rows. "1,000 rows" with 10 positives is tiny. This is our inference; no source gives a minimum positive count.

### Minimal code (run by us, excerpts from the lab script)

```python
# from lightgbm_imbalance_lab.py — run 2026-10-03 (scikit-learn 1.9.1)
import numpy as np
from sklearn.linear_model import LogisticRegression

def crossfit_platt(p_oof, y, folds):
    """Fit Platt on the OTHER folds' OOF scores, apply to this fold: no leakage, no data lost."""
    z = np.log(np.clip(p_oof, 1e-6, 1 - 1e-6) / (1 - np.clip(p_oof, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    out = np.zeros(len(y))
    for tr, va in folds:
        out[va] = LogisticRegression(C=1e6).fit(z[tr], y[tr]).predict_proba(z[va])[:, 1]
    return out
```

To calibrate a single already-fitted model for deployment, the current API is `CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid").fit(X_cal, y_cal)`, which the lab also uses. `cv="prefit"` was deprecated in 1.6 and **removed in 1.8** ([v1.6 whats_new](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/whats_new/v1.6.rst); [calibration.py @1.8.0](https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.8.0/sklearn/calibration.py)). For deployment, fit the final Platt map on *all* OOF scores of the model trained the same way.

### Pitfalls

**Never calibrate on rows the model saw.** Fitting the calibrator on training-set outputs gives "a biased calibrator that maps to probabilities closer to 0 and 1 than it should" ([calibration.rst](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/modules/calibration.rst)). Also do not calibrate on the same rows used for early stopping.

**Never resample the calibration set.** It must carry the deployment base rate, or the calibrator learns the wrong prior.

**Groups and time need special handling.** `cv=GroupKFold()` inside `CalibratedClassifierCV` gets no `groups` unless metadata routing is on, so pass precomputed splits with `cv=list(StratifiedGroupKFold(5).split(X, y, groups))`. Under `ensemble=False`, groups may not reach the splitter even with routing on (our code reading, **UNVERIFIED**). `TimeSeriesSplit` with `ensemble=False` raises "cross_val_predict only works for partitions" ([_validation.py](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/model_selection/_validation.py)).

**Do not pass a fixed `eval_set` into an ensembled calibrator.** With `ensemble=True`, each clone is refit in each fold. A fixed early-stopping set passed through `fit_params` can then overlap test folds. Fix `n_estimators` beforehand instead.

---

## 3. Precision and recall trade off along one fixed curve

### Plain-English explanation

Training produces a score. The threshold turns the score into a yes/no decision. Raising the threshold flags fewer cases: precision (how many flags are real) usually rises and recall (how many real cases you catch) falls. **The threshold does not change the model.** scikit-learn notes that tuned and untuned classifiers "provide the same predict_proba outputs and thus the same ROC and Precision-Recall curves" ([classification_threshold.rst](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/modules/classification_threshold.rst)). It also says the default 0.5 cut-off is "most certainly not ideal for most use cases" (same source). On an 11%-positive problem, 0.5 flags almost nobody. Reweighting is a roundabout way to move along this same curve, so section 1's switches and this section's threshold are two handles on one decision.

### Our test on synthetic data

The lab's calibrated probabilities (weighted + cross-fitted Platt) show how sharply the trade-off bites with a weak model (ROC-AUC 0.62):

| Threshold | Flagged | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|
| 0.20 | 112 | 30 | 82 | 264 | 0.268 | 0.102 |
| 0.10 | 1,463 | 202 | 1,261 | 92 | 0.138 | 0.687 |

Both rows are printed by the lab. Halving the threshold multiplies flags by 13 and recall by about 7, and precision halves. PR-AUC (0.187) should always be read against its no-skill baseline, the prevalence (0.112) ([model_evaluation.rst](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/modules/model_evaluation.rst)).

### Decision table: how to choose an operating point

| What the problem gives you | How to pick the threshold | Tool |
|---|---|---|
| Costs of a false positive and the value of a true positive | Formula on calibrated p, or scan a business metric (section 4) | `TunedThresholdClassifierCV(scoring=make_scorer(...))` |
| A capacity: "we can act on k cases" | Rank and take the top k. Report precision@k | `np.argsort(-p)[:k]` |
| A floor on recall ("catch ≥ 80%") | Highest threshold with recall ≥ target, which maximises precision | `precision_recall_curve` on OOF |
| A floor on precision ("≥ 1 in 3 alerts real") | Lowest threshold with precision ≥ target | `precision_recall_curve` on OOF |
| Nothing at all | F-beta (β > 1 favours recall) as a stated fallback | `fbeta_score` |
| A full sweep table for the slides | Per-threshold confusion counts | `confusion_matrix_at_thresholds` (≥ 1.8), `metric_at_thresholds` (≥ 1.9) |

Sources: [classification_threshold.rst](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/modules/classification_threshold.rst), [v1.8](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/whats_new/v1.8.rst) and [v1.9 whats_new](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/whats_new/v1.9.rst). `TunedThresholdClassifierCV` has no built-in "precision at fixed recall" mode, so encode a constraint as a custom scorer or apply it by hand, then freeze it with `FixedThresholdClassifier`.

### Pitfalls

**Defaulting to F1 or balanced accuracy.** `TunedThresholdClassifierCV` defaults to `scoring="balanced_accuracy"`, and the docs say to "choose a meaningful metric for their use case" (same source). F1 treats FP and FN as interchangeable and ignores true negatives.

**Using threshold metrics for early stopping.** F1 and recall@precision are step functions of the score. Optimise them by threshold choice after training, not as an early-stopping metric.

**Expecting precision to transfer.** Precision and PR curves depend on prevalence, while TPR and FPR do not. A threshold set for "30% precision" will drift if the base rate changes between training and judging data.

---

## 4. A false positive's cost and a true positive's *net* benefit set the threshold

### Plain-English explanation

Flag a case when its expected gain beats its expected loss. If p is the calibrated probability, a flag earns the true-positive net benefit B with probability p and costs C_FP with probability 1 − p. So flag when p·B > (1 − p)·C_FP:

**threshold t\* = C_FP / (C_FP + B)**

Here B must be the **net benefit of a true positive compared with not flagging**. It is not the raw "cost of a miss". In the lab's story, a flag (a nurse visit) costs $150 whether or not the patient was going to be readmitted. A correct flag saves 25% × $3,000 = $750, so it **nets $600**. The threshold is 150/(150 + 600) = **0.20**. Plugging in the $750 "cost of a miss" as if it were B gives 150/900 ≈ 0.167. That is wrong here, because it forgets that the $150 is also paid on true positives. The general cost-matrix form is t\* = (c10 − c00)/[(c10 − c00) + (c01 − c11)], which gives 0.20 either way if every cell is filled in honestly (Elkan 2001; formula **UNVERIFIED** against the paper, which was blocked; [ResearchGate](https://www.researchgate.net/publication/2365611_The_Foundations_of_Cost-Sensitive_Learning)).

**A wording discrepancy to know about.** scikit-learn's credit example uses costs FP = 1 and FN = 5 and says the threshold can be set to "1/5 of the cost ratio, as stated by Eq. (2) in Elkan's paper" ([plot_cost_sensitive_learning.py](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/examples/model_selection/plot_cost_sensitive_learning.py)). The formula gives 1/(1 + 5) = **1/6 ≈ 0.17**, not 0.20. That is our arithmetic, and we read the docs' phrase as loose wording. The same example says the closed form is safe only "given that our model is calibrated, our dataset is representative and large enough".

### Our test on synthetic data

| Policy (on the same OOF predictions) | Flagged | Net value |
|---|---|---|
| Theory threshold 0.20 on **calibrated** probabilities | 112 (TP 30, FP 82) | **+$5,700** |
| Best threshold found by scanning (0.19) | — | **+$7,050** |
| Threshold 0.10 on calibrated probabilities | 1,463 | **−$67,950** |
| Same 0.20 rule on **uncalibrated weighted** probabilities | 1,299 | **−$58,350** |
| Fixed budget: top 100 by risk | 100 | precision 0.270 (2.4× base rate), recall 0.092 |

Three points stand out. **Theory and scan agree (0.20 versus 0.19) only because the probabilities were calibrated.** The identical rule on inflated weighted probabilities flagged 11 times as many patients and turned a profit into a large loss. Second, **the money-optimal policy flags few people and catches only about 10% of readmissions**. With a weak model, most flags are not worth $150, and saying so honestly is better than presenting a high-recall threshold that loses money. Third, one in 3.7 flags is real (1/0.268), which is the sentence judges understand.

### Minimal code (not run by us)

```python
# not run by us — scikit-learn >= 1.5 (TunedThresholdClassifierCV); 1.9.1 current
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import TunedThresholdClassifierCV, StratifiedGroupKFold

def net_value(y_true, y_pred, cost_fp=150, benefit_tp=600):
    tp = np.sum((y_pred == 1) & (y_true == 1)); fp = np.sum((y_pred == 1) & (y_true == 0))
    return tp * benefit_tp - fp * cost_fp

tuned = TunedThresholdClassifierCV(
    model, scoring=make_scorer(net_value), thresholds=100,
    cv=list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, groups)))
tuned.fit(X, y); print(tuned.best_threshold_, tuned.best_score_)
```

An empirical scan like this absorbs miscalibration, because it searches on whatever scores the model produces. The closed form does not. When costs differ per row, as in scikit-learn's fraud example where the cost depends on the transaction amount, route the amount into the scorer through metadata routing and tune one global threshold ([plot_cost_sensitive_learning.py](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/examples/model_selection/plot_cost_sensitive_learning.py)).

**When stakeholders cannot agree on one cost ratio**, plot net benefit across a range of thresholds (decision curve analysis). The formula is NB = TP/n − (FP/n)·t/(1 − t), compared against "flag everyone" and "flag no one" ([dcurves dca.py](https://raw.githubusercontent.com/MSKCC-Epi-Bio/dcurves/main/dcurves/dca.py), dcurves 1.1.7). The threshold encodes a trade: t = 0.20 means "one true positive is worth four false alarms". Ask "how many false alarms would you accept to catch one real case?" An answer of W implies t ≈ 1/(1 + W). That mapping is our inference, consistent with the dcurves vignette's "30 tests per cancer → harm 1/30" ([dca.Rmd](https://raw.githubusercontent.com/ddsjoberg/dcurves/main/vignettes/dca.Rmd)).

### Pitfalls

**Tuning the threshold on training or test rows.** "You should never use the same data for training the classifier and tuning the decision threshold." A flat plateau of near-optimal thresholds "is symptomatic of overfitting" ([classification_threshold.rst](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/modules/classification_threshold.rst); [cost-sensitive example](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/examples/model_selection/plot_cost_sensitive_learning.py)). Use OOF predictions, as the lab does.

**Applying the formula to weighted or resampled scores.** The lab's −$58,350 row shows the result. Calibrate first, or scan empirically.

**Reporting the noisy dollar figure as exact.** With about 30 true positives at the chosen threshold, the dollar figure has wide error bars. scikit-learn warns that business-metric estimates are unreliable with few minority samples and should be confirmed by A/B testing (same source). Present a range or the fold-to-fold spread.

**Assuming the base rate holds.** If the judging data's base rate differs, recalibrate or apply a prior-shift correction (formula **UNVERIFIED**) before reusing t\*.

---

## 5. Tuning: early-stop on the right metric and guard the rare-class leaves

### Plain-English explanation

The guide's [tuning page](hyperparameter-tuning-and-ensembling.md) already covers the core: `num_leaves`, `learning_rate`, `colsample_bytree`, `subsample` and `min_child_samples`, a time-capped Optuna search on locked folds, and early stopping to choose the number of trees. Imbalance adds three concerns. First, **which metric stops training and selects settings**. Second, **leaves that form around a handful of positives**. Third, **whether class weight is itself a hyperparameter**. For binary log loss, each row's hessian is p(1 − p). When positives are rare, most rows have p near 0 and contribute almost no hessian. The default `min_sum_hessian_in_leaf=1e-3` then almost never binds, and a leaf can form on a few positives with an extreme value. This is derived from the loss math, not a documented claim. Raising `min_sum_hessian_in_leaf`, `min_data_in_leaf` or `lambda_l2`, or setting `path_smooth`, damps it ([Parameters.rst](https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/Parameters.rst)).

### Decision table: imbalance-specific additions to the guide's search space

| Parameter (alias) | Default | Suggested range | Why it matters with rare positives |
|---|---|---|---|
| Early-stopping metric | `binary_logloss` | `average_precision` first, with `first_metric_only=True`. Or log loss if probabilities matter | Selection and stopping should use the same metric |
| `min_sum_hessian_in_leaf` (`min_child_weight`) | 1e-3 | log-uniform 1e-3 to 10 | Stops leaves built on a few positives |
| `min_data_in_leaf` (`min_child_samples`) | 20 | log 5 to 500, capped well below the positive count | Docs say "hundreds or thousands" for large data, but leaves must still be able to isolate positives |
| `lambda_l2` (`reg_lambda`) | 0.0 | log 1e-8 to 10 | Adds to the hessian denominator and shrinks extreme leaves |
| `path_smooth` | 0 | 0 to ~10–100 (no official range, **UNVERIFIED**) | Shrinks small leaves toward their parent. Needs `min_data_in_leaf ≥ 2` |
| `scale_pos_weight` | 1.0 | Fixed at 1, or log 1 to n_neg/n_pos if ranking is all you need | If searched, retune hessian limits, and calibrate afterwards |
| `pos_bagging_fraction` / `neg_bagging_fraction` | 1.0 / 1.0 | 1.0 / 0.05–1.0, with `bagging_freq ≥ 1` | Speed on large, very imbalanced data. Ignores `bagging_fraction` |
| `feature_pre_filter` | true | **false** when tuning `min_data_in_leaf` on a reused `Dataset` | Otherwise features filtered by the first trial's value stay lost |
| `n_jobs` (`num_threads`) | 0 (all cores) | **1 on small data** or under parallel trials | See the speed result below |

Ranges are synthesised from Optuna's LightGBMTuner, FLAML's search space and the LightGBM docs. They are not an official recommendation ([optimize.py](https://raw.githubusercontent.com/optuna/optuna-integration/main/optuna_integration/lightgbm/_lightgbm_tuner/optimize.py); [FLAML model.py](https://raw.githubusercontent.com/microsoft/FLAML/main/flaml/automl/model.py); [Parameters-Tuning.rst](https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/Parameters-Tuning.rst)).

### Our test on synthetic data: threads

On about 2.6k rows with a busy CPU, **LightGBM with default `n_jobs` took 162 s per fit, against 0.2 s with `n_jobs=1`**. That is roughly 800 times slower, enough to wreck a 10-minute tuning budget. LightGBM's docs warn not to use many threads on small data, giving the example "do not use 64 threads for a dataset with 10,000 rows". They also say not to change `num_threads` "when running multiple jobs simultaneously by external packages" ([Parameters.rst](https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/Parameters.rst)). When Optuna runs trials in parallel, keep trials × `n_jobs` at or below the number of physical cores.

### How much to tune, and with which tool

Two snippet-level findings pull in different directions. Koster and Sigrist (2026, 59 datasets) report that ">100 trials is typically required", that defaults or small grids "often yield very inaccurate models", and that choosing the number of trees by early stopping beats searching over it (**SNIPPET-ONLY**, [arXiv 2602.05786](https://arxiv.org/abs/2602.05786v1)). TabZilla found that *light* tuning of a GBDT matters more than choosing between GBDT and neural nets on about a third of datasets (**SNIPPET-ONLY**, [arXiv 2305.02997](https://arxiv.org/abs/2305.02997)). Under a datathon clock, the guide's 10-minute cap still stands: after `n_jobs=1`, a fit on small data takes a fraction of a second, so 10 minutes buys hundreds of trials. On larger data, prefer random search or TPE with a trial budget over a grid.

**Optuna's LightGBMTuner has moved.** It now lives in `optuna-integration` (`import optuna_integration.lightgbm as lgb`). The old `optuna.integration.lightgbm` path is a shim, deprecated since Optuna 4.9.0, that emits a FutureWarning and is scheduled for removal in 6.0.0 ([optuna/integration/lightgbm.py](https://raw.githubusercontent.com/optuna/optuna/master/optuna/integration/lightgbm.py)). It tunes in about 68 stepwise trials: feature_fraction → num_leaves → bagging → feature_fraction again → lambda_l1/l2 → min_child_samples. It sets `feature_pre_filter=False` itself. It does **not** tune `learning_rate`, `min_sum_hessian_in_leaf`, `max_bin`, `path_smooth` or class weights ([optimize.py](https://raw.githubusercontent.com/optuna/optuna-integration/main/optuna_integration/lightgbm/_lightgbm_tuner/optimize.py)).

### Minimal code (not run by us)

Add these lines to the guide's Optuna objective, which already handles locked folds, early stopping and the time cap:

```python
# not run by us — extra search dimensions for imbalanced binary LightGBM (Optuna 5.0.0, LightGBM 4.7.0)
params.update(
    min_child_weight=trial.suggest_float("min_child_weight", 1e-3, 10, log=True),
    reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
    path_smooth=trial.suggest_float("path_smooth", 0.0, 50.0),
    n_jobs=1, deterministic=True, force_row_wise=True, random_state=42,
)
# score each trial on OOF average_precision (ranking) — threshold and calibration come AFTER tuning
```

### Pitfalls

**Picking the best of many trials is optimistic.** Prefer the simplest configuration within about one standard error of the best, or average the top few. Confirm once on an untouched holdout.

**Early stopping on the scored fold leaks a little.** Early-stopping on the same fold that scores the trial biases it slightly upward. Accept the bias and confirm on the holdout, or early-stop on an inner split.

**Reproducibility needs several settings.** Set `seed`, `deterministic=True` together with `force_row_wise=True` or `force_col_wise=True`, a fixed `num_threads`, and a seeded `TPESampler` ([Parameters.rst](https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/Parameters.rst)).

**Retune hessian limits after any weight change.** Changing `scale_pos_weight` changes leaf hessian sums, so earlier `min_child_weight` results are stale.

### Versions (as of 2026-10-03, from PyPI)

| Package | Version | Note |
|---|---|---|
| lightgbm | 4.7.0 (2026-07-18) | `eval_set` deprecated in favour of `eval_X`/`eval_y`. Project now at `github.com/lightgbm-org/LightGBM` |
| scikit-learn | 1.9.1 | `cv="prefit"` removed (1.8). `method="temperature"` (1.8). `metric_at_thresholds` (1.9) |
| optuna / optuna-integration | 5.0.0 / 5.0.0 | LightGBMTuner in `optuna_integration.lightgbm` |
| FLAML | 2.7.0 | Linked from LightGBM's tuning page |
| venn-abers / betacal / dcurves | 1.5.4 / 1.1.0 / 1.1.7 | Optional calibrators and decision curves |

Sources: [PyPI lightgbm](https://pypi.org/pypi/lightgbm/json), [PyPI scikit-learn](https://pypi.org/pypi/scikit-learn/json), [PyPI optuna](https://pypi.org/pypi/optuna/json), [PyPI optuna-integration](https://pypi.org/pypi/optuna-integration/json), [PyPI flaml](https://pypi.org/pypi/flaml/json), [PyPI dcurves](https://pypi.org/pypi/dcurves/json).

---

## Recommended recipe

1. **Lock grouped, stratified folds** (`StratifiedGroupKFold`) so every fold has positives and no entity appears in two folds.
2. **Train plain LightGBM**: `objective="binary"`, no `is_unbalance`, `scale_pos_weight` or `class_weight`, `n_jobs=1` on small data, and early stopping on `average_precision` with `first_metric_only=True`.
3. **Tune briefly** with the guide's Optuna recipe, adding `min_child_weight`, `reg_lambda` and `path_smooth`. Score trials on OOF PR-AUC.
4. **Collect OOF probabilities** from the final settings.
5. **Calibrate with cross-fitted Platt on OOF scores.** Keep it only if OOF log loss and Brier improve. Check that mean p ≈ the base rate and the quantile reliability diagram sits on the diagonal.
6. **Choose the threshold.** With costs, use t\* = C_FP/(C_FP + net TP benefit) and confirm it with a scan of the business metric on calibrated OOF predictions. With a capacity, take the top k. With neither, state your F-beta choice.
7. **Report** PR-AUC next to the prevalence, the confusion matrix at the chosen threshold, "1 in N flags is real", and net value against the flag-nobody and flag-everyone baselines.
8. **Refit on all training data** with the mean best iteration from CV. Apply the Platt map fitted on all OOF scores. Never touch the threshold again using test data.

Try weighting or `pos/neg_bagging_fraction` only as an experiment that must beat step 2 on OOF PR-AUC by more than fold noise, and always follow it with step 5.

---

## Conclusion

The common datathon instinct is to "fix" imbalance inside the model. It solves a problem LightGBM mostly does not have, which is ranking, and it creates one that matters at decision time: dishonest probabilities. Our lab shows the cost concretely. The threshold rule that makes money on calibrated scores loses tens of thousands of dollars on weighted ones, while the ranking metrics that teams usually watch barely move. So the competitive edge on an imbalanced problem lies less in the training switches than in the steps after training: cross-fitted calibration that spends no training data, and a threshold derived from a correctly specified net benefit.

The open questions are about scale and generality. Our evidence is one synthetic dataset with a weak signal (ROC-AUC 0.62), and most outside evidence is snippet-level, logistic-regression-based or in conflict on oversampling. On much larger or rarer-positive data, weighting or balanced bagging may earn a ranking gain, and isotonic or Venn-Abers may beat Platt. The lab's `--rare` flag (about 3% positives) is the cheap way to test that before a team commits.
