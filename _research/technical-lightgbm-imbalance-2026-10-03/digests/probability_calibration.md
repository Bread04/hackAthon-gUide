# Probability calibration for gradient-boosted trees (LightGBM): measuring and fixing it

Research date: 2026-10-03. Access note: the egress proxy blocked arxiv.org, cs.cornell.edu, ncbi/PMC, semanticscholar, scikit-learn.org, readthedocs, and journal sites. Primary sources fetched in full are limited to **raw.githubusercontent.com** (the scikit-learn source and docs, LightGBM docs, the venn-abers and betacal READMEs) and the **PyPI JSON API**. Claims about papers come from search-engine snippets and are marked **SNIPPET-ONLY**. Claims from memory or my own reasoning that I could not check are marked **UNVERIFIED**.

Package versions from the PyPI JSON API on 2026-10-03: scikit-learn **1.9.1** (stable); `main` is `1.10.dev0`. lightgbm **4.7.0**, now hosted at github.com/lightgbm-org/LightGBM. venn-abers **1.5.4**, betacal **1.1.0**, netcal **1.4.0** — [PyPI scikit-learn](https://pypi.org/pypi/scikit-learn/json), [PyPI lightgbm](https://pypi.org/pypi/lightgbm/json), [PyPI venn-abers](https://pypi.org/pypi/venn-abers/json), [PyPI betacal](https://pypi.org/pypi/betacal/json), [PyPI netcal](https://pypi.org/pypi/netcal/json), [sklearn `__init__` on main](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/__init__.py)

## What calibration is, how to measure it, and whether boosted trees (LightGBM) are miscalibrated by default

### Takeaway
A calibrated classifier's outputs can be read as frequencies: among cases scored about 0.8, about 80% are positive. Measure calibration with a reliability diagram plus proper scores (log loss, and Brier with its reliability/resolution/uncertainty decomposition). Treat a single binned ECE number with caution. Classic AdaBoost-style boosted trees are strongly miscalibrated, with a sigmoid-shaped distortion. Modern LightGBM optimises log loss directly, so it is usually closer to calibrated, but class weighting or resampling breaks calibration badly, and the LightGBM docs say so explicitly.

### Cited Findings
**Definition**
- scikit-learn defines it this way: "Well calibrated classifiers are probabilistic classifiers for which the output of the predict_proba method can be directly interpreted as a confidence level… among the samples to which it gave a predict_proba value close to, say, 0.8, approximately 80% actually belong to the positive class." — [sklearn calibration.rst (main)](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/modules/calibration.rst)

**Proper scores and the Brier decomposition**
- The scikit-learn docs say: "Strictly proper scoring rules for probabilistic predictions like brier_score_loss and log_loss assess calibration (reliability) and discriminative power (resolution) of a model, as well as the randomness of the data (uncertainty) at the same time. This follows from the well-known Brier score decomposition of Murphy… A lower Brier loss, for instance, does not necessarily mean a better calibrated model, it could also mean a worse calibrated model with much more discriminatory power." The reference is Murphy (1973), "A New Vector Partition of the Probability Score", J. Appl. Meteor. 12(4):595-600 — [sklearn calibration.rst](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/modules/calibration.rst)
  - The decomposition is BS = Reliability − Resolution + Uncertainty, where Uncertainty = p̄(1−p̄) and p̄ is the base rate. This is the standard Murphy form; the doc names the three terms but does not print the equation. **UNVERIFIED** as a verbatim quote.
- `brier_score_loss` handles binary and multiclass predictions. The docs say it "is equivalent to the mean squared error" and is "a strictly proper scoring rule" — [sklearn model_evaluation.rst](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/modules/model_evaluation.rst)
- The 1.7 changelog adds a metric for multiclass problems with a `scale_by_half` argument. It is described as "notably useful to assess both sharpness and calibration of probabilistic classifiers" — [sklearn v1.7 whats_new](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/whats_new/v1.7.rst)
  - The snippet does not name the metric. It is `brier_score_loss` gaining multiclass support (**UNVERIFIED** from the snippet alone).
- 1.9 fixed how `pos_label` is inferred in `brier_score_loss` and `d2_brier_score` — [sklearn v1.9 whats_new](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/whats_new/v1.9.rst)

**Reliability diagrams and binning**
- A reliability diagram plots "the frequency of the positive label (… an estimation of the conditional event probability P(Y=1|predict_proba)) on the y-axis against the predicted probability". scikit-learn bins predictions to build it.
  - `n_bins` "is subject to the usual bias-variance trade-off".
  - `n_bins="cube_root"` sets ⌈n^{1/3}⌉ bins. The docs call the exponent 1/3 "asymptotically optimal" and warn that for about 100 samples it "might return slightly too low number of bins".
  - Source: [sklearn calibration.rst](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/modules/calibration.rst)
- The `"cube_root"` option for `calibration_curve` and `CalibrationDisplay.from_estimator` / `from_predictions` is marked `.. versionadded:: 1.10`, so it is on `main` and **not in 1.9.1**. In released versions, `n_bins` defaults to 5 and `strategy` is `'uniform'` or `'quantile'` — [sklearn calibration.py (main)](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/calibration.py)
- The docs cite Dimitriadis, Gneiting & Jordan (2021), "Stable reliability diagrams for probabilistic classifiers", PNAS 118(8) (the CORP approach) — [sklearn calibration.rst](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/modules/calibration.rst)

**ECE and its pitfalls**
- Expected calibration error (ECE) is "the most popular metric" but "has numerous flaws".
  - Binning introduces bias.
  - The number of bins trades bias against variance.
  - Equal-mass (adaptive) binning (ECE_em; Kumar et al. 2019; Nixon et al. 2019) and thresholded adaptive calibration error (TACE) were proposed as fixes.
  - Source: Nixon et al. 2019, "Measuring Calibration in Deep Learning", CVPRW — **SNIPPET-ONLY** [CVF PDF](https://openaccess.thecvf.com/content_CVPRW_2019/papers/Uncertainty%20and%20Robustness%20in%20Deep%20Visual%20Learning/Nixon_Measuring_Calibration_in_Deep_Learning_CVPRW_2019_paper.pdf), [arXiv 1904.01685](https://arxiv.org/abs/1904.01685)
- scikit-learn does not ship an ECE function. Its public calibration tools are `calibration_curve`, `CalibrationDisplay` and the proper scores. Basis: a search of `sklearn/calibration.py` and the docs. **UNVERIFIED** as an exhaustive check of `sklearn.metrics`.

**Boosted trees by default: Niculescu-Mizil & Caruana (2005) and later evidence**
- Niculescu-Mizil & Caruana, quoted in the scikit-learn docs, explain why bagging and random forests rarely predict near 0 or 1 (variance makes the errors one-sided). The docs also say max-margin methods produce "an even more sigmoid curve… (compare Niculescu-Mizil and Caruana)" — [sklearn calibration.rst](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/modules/calibration.rst)
- "Maximum margin methods such as boosted trees and boosted stumps push probability mass away from 0 and 1 yielding a characteristic sigmoid shaped distortion." Boosting has "good accuracy, precision, and ROC area" but "poor squared error and cross-entropy". After Platt scaling or isotonic calibration, boosted trees "yield the best average squared error and cross-entropy performance" of the algorithms compared.
  - Logistic Correction and boosting with log-loss "work well when boosting weak models such as decision stumps, but yield poor performance when boosting more complex models such as full decision trees".
  - Sources: Niculescu-Mizil & Caruana, ICML 2005 / UAI 2005 — **SNIPPET-ONLY** [ICML05 PDF](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf), [Obtaining Calibrated Probabilities from Boosting, arXiv 1207.1403](https://arxiv.org/abs/1207.1403)
- Caveat on that evidence: those results are for AdaBoost-style boosting, where scores come from an exponential-loss margin. LightGBM's `binary` objective minimises log loss, a proper scoring rule. With sensible early stopping its raw probabilities are often reasonably calibrated. I found no primary benchmark to quantify this, so it is **UNVERIFIED** and should be checked on your own data.
- A 2026 large-scale study (Manokhin & Grønhaug, "Classifier Calibration at Scale", arXiv 2601.19944) found that "commonly used calibration procedures, most notably Platt scaling and isotonic regression, can systematically degrade proper scoring performance for strong modern tabular models" — **SNIPPET-ONLY** [arXiv 2601.19944](https://arxiv.org/pdf/2601.19944). This supports the point that a strong GBDT may not need recalibration, and that recalibrating it can make things worse.
- A diabetes study using LightGBM as its GBDT reported reliability measured by ECE, log loss and reliability diagrams; in a search snippet, LightGBM had ECE 0.0018 ± 0.00033 and log loss 0.167 on big data. Title: "Gradient boosting decision tree becomes more reliable than logistic regression in predicting probability for diabetes with big data" — **SNIPPET-ONLY** [PMC9553945](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9553945/). The numbers come from the search summariser, so verify them before quoting.

**LightGBM documentation on imbalance options**
- `is_unbalance` (default `false`) and `scale_pos_weight` (default `1.0`) carry the same warning: "**Note**: while enabling this should increase the overall performance metric of your model, it will also result in poor estimates of the individual class probabilities". The two options cannot be used together — [LightGBM docs/Parameters.rst](https://raw.githubusercontent.com/lightgbm-org/LightGBM/master/docs/Parameters.rst)
- LightGBM's `sigmoid` parameter (default 1.0) is the "parameter for the sigmoid function" in binary/multiclassova — [LightGBM Parameters.rst](https://raw.githubusercontent.com/lightgbm-org/LightGBM/master/docs/Parameters.rst)

### Inferences
- A practical metric stack for LightGBM:
  - log loss and Brier on held-out data (proper scores, so they are the primary objective);
  - a reliability diagram, using quantile bins when probabilities are skewed (common under imbalance);
  - calibration-in-the-large (mean predicted vs observed rate) and calibration slope/intercept, from logistic recalibration of the logit;
  - ECE only as a secondary summary, with its bin count and binning strategy stated.
- Under heavy imbalance, uniform bins leave the upper bins nearly empty. Quantile bins, or a smooth or isotonic (CORP-style) curve, are more informative.
- Early-stopping a LightGBM model on validation log loss (not AUC) protects calibration as well as ranking. Many extra trees with a high learning rate tend to make the model overconfident. This is general knowledge; I did not find a primary source for it.

### Gaps
- I could not fetch the Niculescu-Mizil & Caruana PDFs, so the exact figures (dataset counts, calibration-set-size curves) are not quoted.
- I found no peer-reviewed benchmark specifically on the out-of-the-box calibration of LightGBM, XGBoost or CatBoost under log loss. The 2601.19944 and PMC9553945 findings are snippet-level only.
- I could not confirm the formal statement of Kumar et al. 2019 ("Verified Uncertainty Calibration": the debiased estimator, and the result that binned ECE underestimates).

## Class weighting and resampling distort probabilities; the prior-shift correction

### Takeaway
Undersampling, oversampling, SMOTE, `is_unbalance` and `scale_pos_weight` all change the effective class prior the model learns. The result is that minority-class probabilities are systematically **overestimated**. If the only change was to the prior or the weights, a closed-form odds correction undoes it. Otherwise, recalibrate on untouched, naturally distributed data. Simulation studies find these corrections usually do not improve AUC.

### Cited Findings
- Dal Pozzolo, Caelen, Johnson & Bontempi (IEEE SSCI 2015), "Calibrating Probability with Undersampling for Unbalanced Classification", make these points — **SNIPPET-ONLY**:
  - Undersampling "modifies the priors of the training set and consequently biases the posterior probabilities".
  - They derive a correction formula from Bayes' rule and set thresholds using Bayes Minimum Risk.
  - Search snippets give the correction as κ(γ) = γπ₀/(1−γ+γπ₀).
  - Sources: [Semantic Scholar](https://www.semanticscholar.org/paper/Calibrating-Probability-with-Undersampling-for-Pozzolo-Caelen/e36bb7fbe1b4f7c521608e93a2215e2062dae5b1), [PDF](https://pdfs.semanticscholar.org/e36b/b7fbe1b4f7c521608e93a2215e2062dae5b1.pdf)
- The formula as usually written (Dal Pozzolo 2015) — **UNVERIFIED** verbatim, but it follows from Bayes' rule:
  - Keep each negative with probability β (0<β≤1) and all positives.
  - The undersampled-model posterior p_s relates to the true posterior p by p_s = p / (p + β(1−p)).
  - Inverse: **p = β·p_s / (β·p_s − p_s + 1)**.
  - The Bayes threshold moves the same way.
- General prior-shift odds correction (Elkan 2001, "Foundations of cost-sensitive learning"; Saerens et al. 2002) — **UNVERIFIED** citation, but an algebraic identity:
  - odds_true(x) = odds_model(x) × [π_true/(1−π_true)] / [π_train/(1−π_train)], where π_train is the effective positive rate after resampling or weighting.
  - For `scale_pos_weight = w` with log loss, the weighted optimum has odds_w = w·odds_true, so **p = p_w / (p_w + w·(1−p_w))**. This is the β-formula with β = 1/w.
  - The identity holds only if the model is otherwise calibrated and the weighting is uniform within each class.
- LightGBM's own docs warn that `is_unbalance` / `scale_pos_weight` give "poor estimates of the individual class probabilities" — [LightGBM Parameters.rst](https://raw.githubusercontent.com/lightgbm-org/LightGBM/master/docs/Parameters.rst)
  - `is_unbalance` is commonly said to weight positives by the negative-to-positive ratio, but the docs do not state the formula, so this is **UNVERIFIED**.
- van den Goorbergh, van Smeden, Timmerman & Van Calster (JAMIA 2022; 29(9):1525-1534) report that "random undersampling, random oversampling, or SMOTE yielded poorly calibrated models: the probability to belong to the minority class was strongly overestimated". The corrections "did not result in higher areas under the ROC curve" — **SNIPPET-ONLY** [JAMIA](https://academic.oup.com/jamia/article/29/9/1525/6605096)
- A follow-up for ML models: Carriero et al. 2025, "The Harms of Class Imbalance Corrections for Machine Learning Based Prediction Models: A Simulation Study", Statistics in Medicine. Title only; **SNIPPET-ONLY** [Wiley](https://onlinelibrary.wiley.com/doi/full/10.1002/sim.10320)
- Zewen Liu (arXiv 2606.29720, June 2026) studied random forest and gradient boosting on 5 datasets, with imbalance ratios 1.9-70 — **SNIPPET-ONLY** preprint, not peer reviewed [arXiv 2606.29720](https://arxiv.org/abs/2606.29720)
  - SMOTE's calibration cost was "real but small (ECE +0.009)".
  - Random undersampling inflated ECE "from 0.008 to 0.395 on a dataset with ratio 70".
  - "A single post-hoc recalibration step (Platt or isotonic)… reduc[es] ECE by up to 66% at a negligible ranking-power cost (AUC -0.002)."
- scikit-learn notes that sigmoid calibration fits an intercept, which "helps shift decision boundaries appropriately when the classifier being calibrated is biased towards the majority class". It also notes Platt "works best if the calibration error is symmetrical… This can be a problem for highly imbalanced classification problems" — [sklearn calibration.py](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/calibration.py), [calibration.rst](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/modules/calibration.rst)

### Inferences
- For LightGBM, the cleanest route is to train on the natural class distribution with the plain `binary` objective. Handle imbalance at the **decision threshold**, not in the probabilities.
- If `scale_pos_weight = w` is kept for ranking or recall reasons, apply p = p_w / (p_w + w(1−p_w)) first. Then check the reliability diagram, and add Platt or isotonic calibration on a natural-prior hold-out if residual miscalibration remains. The analytic fix is exact only for the prior/weight effect, not for any interaction with tree growth (`min_sum_hessian_in_leaf` and similar are weight-dependent).
- The calibration set must **never** be resampled. It must reflect the deployment prior; otherwise the calibrator learns the wrong base rate.

### Gaps
- I could not verify the exact notation in Dal Pozzolo 2015 or Elkan 2001 against the primary PDFs.
- No primary source confirms LightGBM's internal `is_unbalance` weight formula.

## Calibration methods (Platt/sigmoid, isotonic, Venn-Abers, beta, temperature) and which to use at what sample size

### Takeaway
- Sigmoid/Platt (2 parameters) is the safe choice for small calibration sets and for under-confident, symmetric distortions. It preserves ranking.
- Isotonic is more flexible but overfits below roughly 1,000 calibration samples, and its ties can change AUC.
- Beta calibration (3 parameters) fixes Platt's symmetry assumption cheaply and suits probability-valued GBDT outputs.
- Venn-Abers is an isotonic-based method with validity guarantees under exchangeability. It does well in a 2026 large tabular benchmark.
- Temperature scaling (in scikit-learn since 1.8) is a single-parameter multiclass method that preserves the argmax.

### Cited Findings
**Sigmoid / Platt**
- The model is p(y=1|f) = 1/(1+exp(A f + B)). When `predict_proba` exists, f = logit(p̂); otherwise f is the `decision_function` output.
- The docs say it is "most effective for small sample sizes or when the un-calibrated model is under-confident and has similar calibration errors for both high and low outputs". It assumes symmetric error, citing Kull et al. 2017.
- Source: [sklearn calibration.rst](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/modules/calibration.rst)

**Isotonic**
- The docs say isotonic "is more prone to overfitting, especially on small datasets" and "will perform as well as or better than 'sigmoid' when there is enough data (greater than ~ 1000 samples) to avoid overfitting", citing Niculescu-Mizil & Caruana 2005.
- The docstring says: "Isotonic calibration is not recommended when the number of calibration samples is too low (≪1000) since it then tends to overfit."
- Isotonic "introduces ties", so ROC-AUC can change. Use sigmoid "in case, you strictly want to keep the ranking".
- Sources: [sklearn calibration.rst](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/modules/calibration.rst), [calibration.py](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/calibration.py)

**Temperature scaling**
- `method="temperature"` computes softmax(z/T), with T fitted by minimising log loss on held-out data.
- It "does not alter the accuracy", and it is "a natural way to obtain (better) calibrated multi-class probabilities with just one free parameter". Sigmoid and isotonic instead use one-vs-rest plus renormalisation.
- It was added in **1.8** (PR #31068), with array-API support in 1.8 (PR #32246). The reference is Guo et al., ICML 2017.
- Sources: [calibration.rst](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/modules/calibration.rst), [v1.8 whats_new](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/whats_new/v1.8.rst)

**Beta calibration**
- Kull, Silva Filho & Flach, "Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers", AISTATS 2017 (also EJS 2017, "Beyond sigmoids").
- The `betacal` Python package provides `BetaCalibration(parameters="abm")` (the default), with `"am"` and `"ab"` variants. It is sklearn-compatible (`BaseEstimator, RegressorMixin`).
- Sources: [betacal source](https://raw.githubusercontent.com/betacal/python/master/betacal/__init__.py), [betacal README](https://raw.githubusercontent.com/betacal/python/master/README.md), [betacal.github.io](https://betacal.github.io/)

**Venn-Abers**
- Vovk & Petej, UAI 2014. Isotonic regression is fitted twice per test point, with the point added as a dummy label 0 and then as 1. This gives an interval [p0, p1] and calibration guarantees under i.i.d./exchangeability. It is described as overcoming isotonic's overfitting — **SNIPPET-ONLY** [UAI 2014 PDF](https://www.auai.org/uai2014/proceedings/individuals/166.pdf)
- The `venn-abers` package (v1.5.4) provides `VennAbersCalibrator(estimator=..., inductive=True, cal_size=0.2, random_state=...)` for binary and multiclass. It is sklearn-compatible (inherits `BaseEstimator`, works in `Pipeline`, `GridSearchCV`), and fit kwargs are routed to the underlying estimator (e.g., `eval_set`) — [venn-abers README](https://raw.githubusercontent.com/ip200/venn-abers/main/README.md)
- Manokhin & Grønhaug (2026, arXiv 2601.19944) compared isotonic, Platt, Beta, Venn-Abers and Pearsonify on tabular binary tasks — **SNIPPET-ONLY** [arXiv](https://arxiv.org/pdf/2601.19944). Findings:
  - "Venn-Abers predictors achieve the largest average reductions in log-loss, followed closely by Beta calibration, while Platt scaling exhibits weaker and less consistent effects."
  - "Beta calibration improves log-loss most frequently across tasks, whereas Venn-Abers displays fewer instances of extreme degradation."
  - Caution on authorship: the first author also maintains a related Venn-Abers ecosystem. The venn-abers README lists the paper under "Research Using Venn-Abers".

**netcal**
- netcal (v1.4.0, EFS-OpenSource/calibration-framework) offers binning methods, scaling methods (including beta and temperature) and ECE/MCE metrics plus reliability diagrams. The README fetch failed, so this is **UNVERIFIED** beyond the PyPI metadata — [PyPI netcal](https://pypi.org/pypi/netcal/json)

**Other surveys and benchmarks**
- Silva Filho, Song, Perello-Nieto, Santos-Rodriguez, Kull & Flach, "Classifier calibration: a survey on how to assess and improve predicted class probabilities", Machine Learning 112(9):3211-3260, 2023 — **SNIPPET-ONLY** [Bristol](https://research-information.bris.ac.uk/en/publications/classifier-calibration-a-survey-on-how-to-assess-and-improve-pred/)
- A 2026 "CalArena" large-scale post-hoc calibration benchmark ([arXiv 2605.30188](https://arxiv.org/html/2605.30188v2)) and a "Small-Data Classifier Calibration Benchmark: When Post-Hoc Methods Help and Hurt" ([Zenodo 20140793](https://zenodo.org/records/20140793)) exist. I saw titles only — **SNIPPET-ONLY**, results not read.

### Inferences
Decision guide, synthesised from the sources above:
- **Fewer than about 1,000 calibration samples, or very few positives:** use sigmoid or beta. Isotonic will overfit.
- **More than about 1,000 samples with ample positives, and a non-sigmoid distortion:** isotonic or Venn-Abers.
- **Need to keep the exact ranking/AUC:** sigmoid, beta or temperature.
- **Multiclass:** temperature scaling. Its one parameter is far fewer than one-vs-rest needs.
- **Very imbalanced data:** the binding constraint is the number of minority-class examples in the calibration set, not total n. "1000 samples" with 10 positives is effectively tiny. This is my inference; I found no source giving a minority-count threshold.
- **Always:** compare calibrated against uncalibrated log loss on an independent test fold. The 2026 benchmark warns that recalibrating a strong GBDT can hurt.

### Gaps
- I found no quantitative sample-size threshold from primary sources other than scikit-learn's "~1000", which cites Niculescu-Mizil & Caruana 2005.
- I could not read Kull 2017, so its empirical comparisons are not quoted.
- Venn-Abers interval width as a function of calibration-set size was not quantified.

## Correct scikit-learn usage in 2026 (CalibratedClassifierCV, prefit → FrozenEstimator, ensemble, CalibrationDisplay) and leakage-free calibration with grouped/time folds

### Takeaway
- `cv="prefit"` was **deprecated in 1.6 and removed in 1.8**. For a model that is already fitted, use `CalibratedClassifierCV(FrozenEstimator(model))`. With a FrozenEstimator, `ensemble="auto"` (the default since 1.6) resolves to `False`.
- Current stable is 1.9.1; methods are `sigmoid` (default), `isotonic`, and `temperature` (since 1.8).
- For grouped or time-ordered data, the safest pattern is: fit LightGBM on the train window, then calibrate a FrozenEstimator on a later or disjoint-group calibration window, then evaluate on a third.
  - Alternatively, pass `cv` as a precomputed list of (train, test) index splits.
  - Note that `cross_val_predict` (used when `ensemble=False`) rejects non-partition splitters such as TimeSeriesSplit.

### Cited Findings
**Version history**
- **1.6**:
  - "`cv="prefit"` is deprecated for CalibratedClassifierCV. Use FrozenEstimator instead, as `CalibratedClassifierCV(FrozenEstimator(estimator))`" (PR #30171).
  - `sklearn.frozen.FrozenEstimator` was introduced as a MajorFeature: "calling .fit on it has no effect, and doing a clone(frozenestimator) returns the same estimator instead of an unfitted clone" (PR #29705).
  - Source: [v1.6 whats_new](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/whats_new/v1.6.rst)
- **1.6.1 and 1.7.2** code: `# TODO(1.8): Remove this code branch and cv='prefit'`, plus the warning "The `cv='prefit'` option is deprecated in 1.6 and will be removed in 1.8. You can use CalibratedClassifierCV(FrozenEstimator(estimator)) instead." — [calibration.py @1.6.1](https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.6.1/sklearn/calibration.py), [@1.7.2](https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.7.2/sklearn/calibration.py)
- **1.7**: the prefit warning changed from `UserWarning` to `FutureWarning` — [v1.7 whats_new](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/whats_new/v1.7.rst)
- **1.8.0**: `calibration.py` no longer contains "prefit" (I grepped the 1.8.0 tag), which confirms the removal — [calibration.py @1.8.0](https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.8.0/sklearn/calibration.py)
- **1.9**: `SVC(probability=True)` is deprecated, to be removed in 1.11. The changelog says: "Use CalibratedClassifierCV with the respective estimator and ensemble=False instead" (PR #32050) — [v1.9 whats_new](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/whats_new/v1.9.rst)

**Current API (main and 1.9.1)**
- Signature: `CalibratedClassifierCV(estimator=None, *, method='sigmoid', cv=None, n_jobs=None, ensemble='auto')`.
- `ensemble` docstring: "'auto' will use False if the estimator is a FrozenEstimator, and True otherwise" (versionchanged 1.6).
- `cv=None` means 5-fold, and StratifiedKFold is used for binary or multiclass y.
- Docstring for frozen models: "Already fitted classifiers can be calibrated by wrapping the model in a FrozenEstimator. In this case all provided data is used for calibration. The user has to take care manually that data for model fitting and calibration are disjoint."
- Example in the docstring: `CalibratedClassifierCV(FrozenEstimator(fitted_clf)).fit(X_calib, y_calib)`, after which `len(calibrated_classifiers_) == 1`.
- Source: [calibration.py (main)](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/calibration.py)
- In 1.9.1, `method` accepts 'sigmoid', 'isotonic' and 'temperature' — [calibration.py @1.9.1](https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.9.1/sklearn/calibration.py)

**ensemble=True vs False**
- With `ensemble=True`, each fold trains a clone on the train split and fits a calibrator on the test split. `predict_proba` averages the k pairs, giving "the traditional ensembling effect… slightly more accurate".
- With `ensemble=False`, `cross_val_predict` produces out-of-fold predictions, a single calibrator is fitted on them, and the base estimator is refit on all data. This is cheaper, smaller and faster to predict.
- Warning from the docs: "All classes should be present in both train and test subsets for every split"; otherwise probabilities are skewed or calibration becomes ineffective.
- The docs say the calibrator should be fit on data "independent of the training data… Using the classifier output of training data to fit the calibrator would thus result in a biased calibrator that maps to probabilities closer to 0 and 1 than it should."
- Source: [calibration.rst](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/modules/calibration.rst)

**Code-level details relevant to grouped and time folds** (from reading [calibration.py main](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/calibration.py))
- **Without metadata routing:**
  - The splitter receives no extra parameters (`_manual_routing({"splitter": {}, ...})`), so `cv=GroupKFold()` would get no `groups`.
  - `fit(X, y, sample_weight=None, **fit_params)` forwards `**fit_params` only to the estimator's `fit`.
- **With metadata routing** (`sklearn.set_config(enable_metadata_routing=True)`):
  - `get_metadata_routing` adds the `cv` splitter with `fit → split`. In the `ensemble=True` path, routed split params are passed to `cv.split(X, y, **routed_params.splitter.split)`.
  - In the `ensemble=False` path, `cross_val_predict(... cv=cv, params=routed_params.estimator.fit)` is called **without** the splitter's routed params. Groups may therefore not reach the splitter in that path. This is my reading of the code only, not tested — **UNVERIFIED** behaviour.
- **TimeSeriesSplit:** `cross_val_predict` raises "cross_val_predict only works for partitions" — [_validation.py](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/model_selection/_validation.py). So TimeSeriesSplit cannot be combined with `ensemble=False`.
- **Leave-one-out:** `LeaveOneOut` is explicitly rejected. A check also raises an error if any class has fewer examples than `n_folds` (integer cv or a splitter with `n_splits`).
- **Sample weights:** if the estimator's `fit` does not accept `sample_weight`, the weights are used "only for the calibration itself" and a warning is issued (sklearn issue #21134). LightGBM's `LGBMClassifier.fit` does accept `sample_weight`.

**CalibrationDisplay**
- `CalibrationDisplay.from_estimator(estimator, X, y, *, n_bins=5, strategy='uniform', pos_label=None, name=None, ax=None, ...)` and `from_predictions(y_true, y_prob, ...)`, added in 1.0. `strategy` is 'uniform' or 'quantile'.
- `n_bins="cube_root"` arrives in 1.10. 1.6 fixed handling of Matplotlib style aliases.
- Sources: [calibration.py](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/calibration.py), [v1.6 whats_new](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/whats_new/v1.6.rst)

### Inferences
Recommended leakage-safe recipes for LightGBM. These follow from the API facts above; the code was not executed here.

1. **Three-way split (time or group aware), the simplest approach:**
   ```python
   from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
   from sklearn.frozen import FrozenEstimator  # sklearn >= 1.6
   clf = lgb.LGBMClassifier(objective="binary", ...).fit(X_tr, y_tr, eval_set=[(X_es, y_es)], callbacks=[lgb.early_stopping(100)])
   cal = CalibratedClassifierCV(FrozenEstimator(clf), method="sigmoid")  # or "isotonic" if n_cal >> 1000 positives permitting
   cal.fit(X_cal, y_cal)          # X_cal: later time window / unseen groups, natural class prior
   CalibrationDisplay.from_estimator(cal, X_test, y_test, n_bins=10, strategy="quantile")
   ```
   - The calibration window must not overlap the early-stopping set, or the early-stopping choice leaks into it.
   - For time data the order is train < early-stop < calibrate < test.
2. **Grouped CV calibration with an ensemble:** pass precomputed splits, `cv=list(GroupKFold(n_splits=5).split(X, y, groups))`, with the default `ensemble=True`. This avoids relying on metadata routing for `groups`, since `cv` accepts "an iterable yielding (train, test) splits as arrays of indices". For stratification plus groups, `StratifiedGroupKFold` gives every fold both classes, which the docs require.
3. **Time series:** do not use `ensemble=False` with TimeSeriesSplit; `cross_val_predict` requires a partition. Either use recipe 1, or pass expanding-window splits with `ensemble=True` and accept that early folds have small calibration sets.
4. **LightGBM-specific leakage:**
   - Do not early-stop and calibrate on the same rows.
   - Under `ensemble=True`, each clone is refit inside the fold. Fixed `eval_set` fit params passed via `**fit_params` would then be the same rows for every fold, which can leak test-fold data. Prefer a fixed `n_estimators` (chosen beforehand) inside CalibratedClassifierCV.
5. `FrozenEstimator` makes `clone()` return the same fitted object. That is safe in `cross_validate` or `GridSearchCV` over calibrator settings, but the base model then never sees the CV folds. Make sure the base model's training data is excluded from every evaluation fold.

### Gaps
- I did not run code to confirm how `groups` routing behaves under `ensemble=False`; this is inferred from source.
- I did not find official scikit-learn guidance on calibration for time-series data specifically.

## Why calibrated probabilities matter for decisions (expected cost, risk communication)

### Takeaway
Thresholds derived from costs (Bayes minimum risk) and expected-value calculations assume the probabilities are calibrated. Miscalibration always lowers net benefit at a given threshold, and it misleads patients or users when risks are communicated as numbers. Ranking metrics (AUC) cannot detect it.

### Cited Findings
- Van Calster, McLernon, van Smeden, Wynants & Steyerberg, "Calibration: the Achilles heel of predictive analytics", BMC Medicine 2019 — **SNIPPET-ONLY** [BMC Med](https://link.springer.com/article/10.1186/s12916-019-1466-7), [PubMed 31842878](https://pubmed.ncbi.nlm.nih.gov/31842878/)
  - Calibration "is assessed far less often than discrimination".
  - "Miscalibration always reduces Net Benefit."
  - Calibration "is especially important when the aim is to support decision-making".
  - "Poorly calibrated risk estimates lead to false expectations with patients and healthcare professionals."
- Dal Pozzolo et al. 2015 use "Bayes Minimum Risk theory to find the correct classification threshold". The threshold depends on the probabilities being correct, so it must be adjusted after undersampling — **SNIPPET-ONLY** [Semantic Scholar](https://www.semanticscholar.org/paper/Calibrating-Probability-with-Undersampling-for-Pozzolo-Caelen/e36bb7fbe1b4f7c521608e93a2215e2062dae5b1)
- van den Goorbergh et al. 2022 say "inaccurate probability estimates reduce the clinical utility of the model, because decisions about treatment are ill-informed" — **SNIPPET-ONLY** [JAMIA](https://academic.oup.com/jamia/article/29/9/1525/6605096)
- scikit-learn says sigmoid preserves ranking, and that calibration is "generally expected" not to change ROC-AUC (isotonic ties aside). AUC is therefore blind to calibration — [calibration.rst](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/modules/calibration.rst)

### Inferences
- Expected-cost decision rule: act if p·C_FN > (1−p)·C_FP, i.e. threshold t* = C_FP/(C_FP + C_FN). This is the standard Elkan 2001 result; **UNVERIFIED** citation, but it is elementary algebra.
- The rule is only optimal if p is calibrated. With `scale_pos_weight = w` uncorrected, the effective threshold silently becomes the cost-optimal one for a different cost ratio.
- For a hackathon or product setting: calibrate first, then choose the threshold from costs, or tune it on a validation set using the calibrated scores. Report calibrated risks (e.g., "12% chance") only after checking a reliability diagram on held-out data from the deployment distribution.

### Gaps
- I could not fetch the full texts to quote Van Calster's sample-size guidance for calibration curves (e.g., a minimum number of events) or the definitions of the calibration levels (mean/weak/moderate/strong).
- No primary source was retrieved for Elkan 2001's cost-sensitive threshold theorem.
