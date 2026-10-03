# Precision/recall trade-offs, FP vs FN costs, and decision thresholds (as of Oct 2026)

Source-access note: GitHub raw (scikit-learn, dcurves) and PyPI were fetched in full. The original papers (Elkan 2001 at cseweb.ucsd.edu, Vickers & Elkin 2006 on PMC, Saito & Rehmsmeier 2015 on PLOS ONE, arXiv, Springer) were all **blocked by the egress proxy**. Claims from those papers come from search-result snippets or from secondary code/docs and are marked SNIPPET-ONLY or UNVERIFIED.

## Q1. Confusion matrix, threshold movement, and reading PR/ROC curves

### Takeaway
A binary classifier gives a score; `predict` then applies a fixed cut-off (0.5 for `predict_proba`, 0 for `decision_function`). Moving that cut-off trades FP for FN. ROC and PR curves show every possible cut-off, and both stay the same when only the threshold changes. On imbalanced data the PR curve (whose baseline is the prevalence) usually tells you more than ROC about how useful the alerts are.

### Cited Findings
- scikit-learn splits classification into "the statistical problem of learning a model to predict, ideally, class probabilities" and "the decision problem to take concrete action based on those probability predictions" (the "chance of rain vs. take an umbrella" analogy). — [classification_threshold.rst](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/modules/classification_threshold.rst)
- Default rule: "a positive class is predicted when the conditional probability P(y|X) is greater than 0.5 (obtained with predict_proba) or if the decision score is greater than 0 (obtained with decision_function)". The docs say these hard-coded rules "are most certainly not ideal for most use cases." — [same](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/modules/classification_threshold.rst)
- Medical example from the docs: physicians prioritise recall because "the cost of a missed cancer is much higher than the cost of further diagnostic tests," so it "may be more beneficial to classify them as positive for cancer when the conditional probability estimate is much lower than 0.5." — [same](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/modules/classification_threshold.rst)
- "The vanilla and tuned classifiers provide the same predict_proba outputs and thus the same ROC and Precision-Recall curves, [but] the class label predictions differ because of the tuned decision threshold." In the insurance/credit figure the tuned threshold is "a very low probability (around 0.02)". — [same](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/modules/classification_threshold.rst)
- Formulas: precision = tp/(tp+fp); recall (sensitivity) = tp/(tp+fn). — [model_evaluation.rst](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/modules/model_evaluation.rst)
- Average precision: AP = Σ_n (R_n − R_{n−1}) P_n. "With random predictions, the AP is the fraction of positive samples." `average_precision_score` "does not implement any interpolated variant". Linear interpolation of PR points (the trapezoidal `auc`) "provides an overly-optimistic measure" (citing Davis 2006 and Flach 2015). — [model_evaluation.rst](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/modules/model_evaluation.rst)
- `precision_recall_curve` is "restricted to the binary case". `PrecisionRecallDisplay.from_estimator/from_predictions(..., plot_chance_level=True)` draws the prevalence baseline. — [model_evaluation.rst](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/modules/model_evaluation.rst); [plot_precision_recall.py](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/examples/model_selection/plot_precision_recall.py)
- The scikit-learn cost-sensitive example plots a custom FPR scorer next to precision/recall/TPR, because "scikit-learn does not provide a scorer for the FPR". It marks the 0.5 operating point on both curves to show "the statistical performance of the model when using `model.predict`". — [plot_cost_sensitive_learning.py](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/examples/model_selection/plot_cost_sensitive_learning.py)
- New per-threshold tools: `metrics.confusion_matrix_at_thresholds` ("returns the number of true negatives, false positives, false negatives and true positives per threshold") was added in **1.8**. `metrics.metric_at_thresholds` ("compute a metric's values across all possible thresholds", a MajorFeature) was added in **1.9**. Also in 1.9: `PrecisionRecallDisplay.from_cv_results`, and passing args other than the first positionally to `confusion_matrix_at_thresholds` is deprecated, with removal in 1.11. — [whats_new v1.8](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/whats_new/v1.8.rst); [whats_new v1.9](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/whats_new/v1.9.rst)
- The latest scikit-learn on PyPI is **1.9.1**. — [PyPI JSON](https://pypi.org/pypi/scikit-learn/json)
- Saito & Rehmsmeier (PLoS ONE 10(3):e0118432, 2015) argue that ROC plots on imbalanced data "can be deceptive ... owing to an intuitive but wrong interpretation of specificity," and that PRC plots "can provide the viewer with an accurate prediction of future classification performance" because they evaluate the fraction of true positives among positive predictions. SNIPPET-ONLY. — [Semantic Scholar](https://www.semanticscholar.org/paper/The-Precision-Recall-Plot-Is-More-Informative-than-Saito-Rehmsmeier/904627c2d5a91ab8cb1b682e42f06f1ca192aea6); [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0118432)
- Counterpoint (titles only, not read; SNIPPET-ONLY): "The receiver operating characteristic curve accurately assesses imbalanced datasets" (Patterns, 2024) and "A Closer Look at AUROC and AUPRC under Class Imbalance" (arXiv 2401.06091) dispute that AUPRC is always better. — [Cell Patterns](https://www.cell.com/patterns/fulltext/S2666-3899(24)00109-0); [arXiv 2401.06091](https://arxiv.org/pdf/2401.06091)

### Inferences
- How to read the curves:
  - ROC (TPR vs FPR) does not depend on prevalence. A small FPR can still mean many false alarms when negatives vastly outnumber positives.
  - PR (precision vs recall) does depend on prevalence. Its chance line sits at the positive rate, which is why AP should always be reported next to the prevalence.
  - Neither curve picks the operating point. The threshold is a separate decision.
- For hackathon-style imbalanced problems, a practical approach is to report both ROC-AUC and AP, plus the confusion matrix at the chosen threshold. `confusion_matrix_at_thresholds` (≥1.8) and `metric_at_thresholds` (≥1.9) make the threshold sweep table easy to build.

### Gaps
- Could not read the Saito & Rehmsmeier full text (blocked), so no numeric illustrations from it.
- Could not read the 2024 counterpoint papers' arguments in detail.

## Q2. Choosing thresholds: (a) cost-sensitive decision theory, (b) constraints / budgets, (c) F-beta

### Takeaway
If probabilities are calibrated and costs are known, the Bayes-optimal rule is "predict positive when p ≥ p*". Here p* = C_FP/(C_FP + C_FN), or more generally (c10 − c00)/[(c10 − c00) + (c01 − c11)] once benefits of correct decisions are included. If probabilities are not trustworthy, or costs vary per example, tune the threshold empirically on held-out data against a business metric (`TunedThresholdClassifierCV`). With operational constraints (minimum recall, alert budget), pick the threshold from the PR curve or by score rank on validation data. F-beta is a fallback, not a default.

### Cited Findings
**(a) Cost-sensitive / Elkan 2001**
- Elkan (2001), "The Foundations of Cost-Sensitive Learning" (IJCAI). The optimal prediction is class 1 "if the expected cost of this prediction is less than or equal to the expected cost of predicting class 0". The threshold is p* = c10/(c10 + c01) when correct decisions cost 0. Rebalancing training examples by sampling can make the 0.5 threshold equivalent to p*. SNIPPET-ONLY (paper blocked). — [Elkan PDF (blocked)](https://cseweb.ucsd.edu/~elkan/rescale.pdf); [ResearchGate](https://www.researchgate.net/publication/2365611_The_Foundations_of_Cost-Sensitive_Learning)
- General form with benefits: with cost c_ij for predicting i when the truth is j, predict positive when p ≥ p* = (c10 − c00)/(c10 − c00 + c01 − c11). This is Elkan's Eq. (2). UNVERIFIED: written from the standard derivation and not confirmed against the PDF, but it reduces to the snippet's form when c00 = c11 = 0. Elkan also states "reasonableness" conditions (c10 > c00 and c01 > c11) and a Theorem 1 for resampling negatives to a proportion p*(1−p0)/(p0(1−p*))-style factor. UNVERIFIED (exact statement not seen).
- scikit-learn credit example (German credit, OpenML id 31, "bad" = positive):
  - Gain matrix: TN 0, FP −1, FN −5, TP 0, because "classifying a 'bad' credit as 'good' is 5 times more costly on average than the opposite".
  - The docs note that "given that our model is calibrated, our dataset is representative and large enough, we do not need to tune the threshold, but can safely set it to 1/5 of the cost ratio, as stated by Eq. (2) in Elkan's paper."
  - — [plot_cost_sensitive_learning.py](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/examples/model_selection/plot_cost_sensitive_learning.py)
- In the same example, tuning the threshold with `TunedThresholdClassifierCV(scoring=credit_gain_scorer)` "improves our business gains by almost a factor of 2". The optimum lies "much lower than 0.5: the tuned model enjoys a much higher recall at the cost of significantly lower precision". — [same](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/examples/model_selection/plot_cost_sensitive_learning.py)
- **Fraud example with example-dependent costs.** OpenML id 1597 credit card fraud; fraud is "only 0.17% of the data"; "around 500" fraud samples; 50/50 stratified split.
  - Business metric: accept legit = +2% of amount; accept fraud = −amount; refuse legit = −5€; refuse fraud = +50€.
  - Baselines on the test half: "always accept" ≈ +220,000€ profit; "always reject" ≈ −700,000€ ("almost 700,000€" loss).
  - Model: LogisticRegression with C tuned by `neg_log_loss` "to ensure that the model's probabilistic predictions ... are as accurate as possible". With the default 0.5 threshold it already beats always-accept.
  - Then `TunedThresholdClassifierCV(estimator=model.best_estimator_, scoring=business_scorer, thresholds=100)` is fit with `amount=` routed as metadata (`sklearn.set_config(enable_metadata_routing=True)`, `make_scorer(business_metric).set_score_request(amount=True)`). The tuned threshold "is far away from the default 0.5" and "increases the expected profit".
  - The exact tuned euro figures are printed at runtime and not stated in the source text.
  - — [same](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/examples/model_selection/plot_cost_sensitive_learning.py)
- The example warns that "the estimate of the (average) business metric itself can be unreliable, in particular when the number of data points in the minority class is very small", and that offline estimates "should ideally be confirmed by A/B testing on live data". — [same](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/examples/model_selection/plot_cost_sensitive_learning.py)

**scikit-learn API (current main / 1.9.x)**
- `FixedThresholdClassifier`, `TunedThresholdClassifierCV` and their base class are all `.. versionadded:: 1.5`. — [_classification_threshold.py](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/model_selection/_classification_threshold.py)
- `TunedThresholdClassifierCV` parameters and attributes:
  - `scoring` default "balanced_accuracy"
  - `response_method` {"auto","decision_function","predict_proba"}
  - `thresholds` int or array-like, default 100
  - `cv` None (5-fold stratified), int, float (single shuffle split, validation fraction), splitter, iterable, or "prefit"
  - `refit=True`; `refit=False` with a multi-split cv raises an error, and `refit=True` with cv="prefit" raises an error
  - `n_jobs`, `random_state`, `store_cv_results=False`
  - Attributes: `estimator_`, `best_threshold_`, `best_score_`, `cv_results_`
  - Across folds, scores are averaged by `np.interp` onto common thresholds (`_mean_interpolated_score`).
  - — [_classification_threshold.py](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/model_selection/_classification_threshold.py)
- `FixedThresholdClassifier(estimator, threshold="auto", pos_label=None, response_method="auto")`. "auto" means 0.5 for predict_proba and 0 for decision_function. The docs recommend `FixedThresholdClassifier(FrozenEstimator(estimator), ...)` so fit does not refit. — [classification_threshold.rst](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/modules/classification_threshold.rst); [_classification_threshold.py](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/model_selection/_classification_threshold.py)
- Default-metric caveat from the docs: "By default, the balanced accuracy is the metric used but be aware that one should choose a meaningful metric for their use case". Scorers carry default `pos_label`, so for a different positive label you need `make_scorer(f1_score, pos_label=0)`. — [classification_threshold.rst](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/modules/classification_threshold.rst)
- Diabetes example: "a decision threshold around 0.32 maximizes the balanced accuracy", and "the metric used to tune the decision threshold should be chosen carefully". — [plot_tuned_decision_threshold.py](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/examples/model_selection/plot_tuned_decision_threshold.py)

**(c) F-beta**
- F_β = (1+β²)·P·R/(β²P + R). scikit-learn computes it as (1+β²)tp / ((1+β²)tp + fp + β²fn), and it is undefined with no true positives. β=1 means "recall and the precision are equally important". — [model_evaluation.rst](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/modules/model_evaluation.rst)

### Inferences
- Worked check of the credit example: costs FP=1 and FN=5 give p* = 1/(1+5) = 1/6 ≈ 0.17. The docs' phrase "1/5 of the cost ratio" is loosely worded; the Elkan formula gives 1/6. This is arithmetic from the cited costs, not a source quote.
- Fraud costs vary per transaction, so p* differs per transaction. Using the example's cost values, refusing beats accepting when p·50 − (1−p)·5 > −p·A + (1−p)·0.02A. This yields p* = (5 + 0.02A)/(55 + 1.02A), which falls as the amount A grows. That is why the example tunes a single global threshold against the summed business metric rather than using a closed form. Derived arithmetic, not a source quote.
- (b) Constraints:
  - **Minimum recall:** on a validation set, take `precision_recall_curve` and choose the highest threshold whose recall ≥ target. That maximises precision subject to the constraint.
  - **Minimum precision:** symmetric — choose the lowest threshold with precision ≥ target.
  - **Fixed alert budget (k alerts/day):** rank by score and set the threshold at the k-th highest score of the expected daily volume (a quantile threshold). Precision@k then becomes the key metric.
  - None of these is a built-in `TunedThresholdClassifierCV` option. You can encode one as a custom scorer (e.g. precision with a penalty if recall < target) or apply it by hand, then freeze the result with `FixedThresholdClassifier`.
- F-beta encodes a cost ratio only implicitly; β>1 weights recall more. Prefer an explicit gain matrix when costs are known.

### Gaps
- Could not verify Elkan's exact Eq. (2) with benefits, the reasonableness conditions or the Theorem 1 formula against the PDF (blocked).
- No authoritative source found stating which F_β corresponds to which cost ratio. Avoid claiming an exact equivalence.
- Could not confirm whether earlier development versions of `TunedThresholdClassifierCV` had "precision at recall constraint" options that were later removed (UNVERIFIED recollection). The current API has no such options.

## Q3. Decision curve analysis / net benefit (Vickers)

### Takeaway
Net benefit is TP/n − (FP/n)·pt/(1−pt). The threshold probability pt encodes how much worse a miss is than an unnecessary intervention (odds pt/(1−pt) = harm/benefit). Plot it across a clinically plausible pt range against "treat all" and "treat none" (NB = 0). The best strategy is the highest curve.

### Cited Findings
- Vickers & Elkin (2006, *Medical Decision Making*) define NB(t) = μ·sens(t) − (1−μ)(1−spec(t))·t/(1−t), equivalently TP/n − FP/n × pt/(1−pt). SNIPPET-ONLY for the paper itself. — [PubMed 21696604 (Baker et al., revisited)](https://pubmed.ncbi.nlm.nih.gov/21696604); [ResearchGate](https://www.researchgate.net/publication/6698706_Decision_Curve_Analysis_A_Novel_Method_for_Evaluating_Prediction_Models)
- The MSKCC `dcurves` Python package (v1.1.7 on PyPI) computes the same thing in code: `net_benefit = tp_rate − (threshold/(1−threshold))·fp_rate − harm`. Its `tp_rate` is TP/n (sensitivity × prevalence), and `net_intervention_avoided = (NB − NB_treat_all)/(threshold/(1−threshold)) × nper`. — [dcurves/dca.py](https://raw.githubusercontent.com/MSKCC-Epi-Bio/dcurves/main/dcurves/dca.py); [PyPI dcurves](https://pypi.org/pypi/dcurves/json)
- Interpretation from the official DCA tutorial (dcurves vignette, biopsy example):
  - "Net benefit ... of 0.03 at a threshold probability of 20% can be interpreted as: 'Comparing to conducting no biopsies, biopsying on the basis of the marker is the equivalent of a strategy that found 3 cancers per hundred patients without conducting any unnecessary biopsies.'"
  - "At a probability threshold of 15%, the net reduction in interventions is about 0.33 ... equivalent of a strategy that reduced the biopsy rate by 33%, without missing any cancers."
  - The treat-all curve "crosses the y axis at the prevalence".
  - Show only clinically reasonable thresholds (0–35% in the example), since "it is unlikely that a patient would demand that they had at least a 50% risk of cancer before they would accept a biopsy".
  - Testing harm: if "few clinicians would conduct more than 30 tests to predict one cancer diagnosis", then harm = 1/30 = 0.0333, subtracted from NB.
  - A published "Brown" model was "harmful in patients with more moderate threshold probabilities", i.e. below treat-all/none.
  - — [dcurves dca.Rmd vignette](https://raw.githubusercontent.com/ddsjoberg/dcurves/main/vignettes/dca.Rmd)

### Inferences
- NB is the cost-sensitive rule in clinical units. pt plays the role of Elkan's p*, since pt/(1−pt) = C_FP/C_FN. So a DCA curve is an expected-utility curve over a range of plausible cost ratios. It is useful when stakeholders cannot agree on a single cost ratio.
- DCA assumes calibrated risks. A miscalibrated model can have NB below treat-all/none, as the Brown example shows.

### Gaps
- Original Vickers & Elkin 2006 text and the 2019 "step-by-step guide" (Diagn Progn Res) were blocked; their exact worked numbers were not retrieved.

## Q4. Pitfalls

### Takeaway
The main failure modes:
- tuning the threshold on the same data used for training or final test;
- applying cost thresholds to probabilities that are uncalibrated or distorted by reweighting/resampling;
- ignoring a prevalence shift between training and deployment;
- defaulting to F1 or balanced accuracy instead of a cost-based metric.

### Cited Findings
- On threshold overfitting, the docs say: "You should never use the same data for training the classifier and tuning the decision threshold due to the risk of overfitting." `cv="prefit"` "should only be used when the provided classifier was already trained, and you just want to find the best decision threshold using a new validation set." — [classification_threshold.rst](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/modules/classification_threshold.rst)
- In the example, tuning with prefit on training data shows "a large plateau of near-optimal 0 gain for a large span of decision thresholds. This behavior is symptomatic of overfitting." A single split (`cv=0.75`) "does not account for the variability ... we are unable to know if there is any variance in the cut-off point." — [plot_cost_sensitive_learning.py](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/examples/model_selection/plot_cost_sensitive_learning.py)
- On calibration: the closed-form Elkan threshold is valid only "given that our model is calibrated, our dataset is representative and large enough". The fraud example tunes C with log loss to get accurate probabilities. — [plot_cost_sensitive_learning.py](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/examples/model_selection/plot_cost_sensitive_learning.py)
- Elkan: rebalancing the training set by sampling makes 0.5 on the rebalanced model equivalent to p* on the original. The corollary is that resampled/reweighted models already embed a shifted threshold. SNIPPET-ONLY. — [ResearchGate](https://www.researchgate.net/publication/2365611_The_Foundations_of_Cost-Sensitive_Learning)
- On the default metric: "By default, the balanced accuracy is the metric used but be aware that one should choose a meaningful metric for their use case." — [classification_threshold.rst](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/modules/classification_threshold.rst)
- Offline business-metric estimates are noisy with few minority samples and should be confirmed by A/B testing. — [plot_cost_sensitive_learning.py](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/examples/model_selection/plot_cost_sensitive_learning.py)

### Inferences
- Do not pick the threshold on the test set. Tune via internal CV, or on a separate validation split, then report the test metric once.
- If LightGBM uses `is_unbalance`/`scale_pos_weight` or resampling, its scores are shifted toward the positive class. Either recalibrate on unweighted held-out data before applying p*, or tune the threshold empirically on the business metric, which absorbs the shift.
- Prevalence shift: precision and the PR curve change with prevalence, while TPR/FPR do not. A threshold set for "≥X% precision" or "k alerts/day" will drift if the base rate changes. The standard prior-shift correction for calibrated probabilities is p' = (π'/π)p / [(π'/π)p + ((1−π')/(1−π))(1−p)] (Saerens et al. 2002 style). UNVERIFIED (no source fetched). In practice, monitor the alert rate and realised precision after deployment.
- F1 assumes FP and FN matter equally (in a harmonic-mean sense). It ignores TN and fixed costs, and its optimum depends on prevalence. Use it only when no cost information exists.

### Gaps
- No fetched source for the prior-shift correction formula or for quantified F1-vs-cost regret.

## Q5. Presenting FP/FN trade-offs to non-technical stakeholders

### Takeaway
Translate thresholds into counts and money or units per period:
- an expected-value table (TP/FP/FN/TN counts × per-outcome gain), compared against the "do nothing" and "act on everything" baselines;
- alerts per day and how many are real (precision → "1 in N alerts is real" = 1/PPV, the number needed to evaluate);
- net benefit framed as "equivalent to finding X cases per 100 with no unnecessary interventions".

### Cited Findings
- The scikit-learn fraud example frames the decision in euros against the constant baselines first ("always accept" ≈ +220,000€, "always reject" ≈ −700,000€). A model should "make a profit larger than the 220,000€ of the best of our constant baseline policies." — [plot_cost_sensitive_learning.py](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/examples/model_selection/plot_cost_sensitive_learning.py)
- The credit example calls any metric quantifying "how the predictions (correct or wrong) might impact the business value" a "business metric", built by weighting the confusion matrix with a gain matrix (`np.sum(cm * gain_matrix)`). — [same](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/examples/model_selection/plot_cost_sensitive_learning.py)
- Net-benefit phrasing: "found 3 cancers per hundred patients without conducting any unnecessary biopsies". Interventions avoided: "reduced the biopsy rate by 33%, without missing any cancers". Harm phrasing: "few clinicians would conduct more than 30 tests to predict one cancer diagnosis" (an NNT-style tolerance turned into a harm weight of 1/30). — [dcurves vignette](https://raw.githubusercontent.com/ddsjoberg/dcurves/main/vignettes/dca.Rmd)
- "Number needed to evaluate" for alerts combines number needed to screen, 1/P(event | alert) (i.e. 1/PPV), and number needed to treat. A review reports clinical decision support alert PPVs ranging from 8% to 83%, mostly 20–40%. SNIPPET-ONLY. — [PMC8510333](https://pmc.ncbi.nlm.nih.gov/articles/PMC8510333/); [PMC5803531](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5803531/)

### Inferences
- A suggested one-slide format: a table with 3–5 candidate thresholds as rows and these columns:
  - alerts/day (volume × predicted-positive rate);
  - real cases caught/day (TP);
  - real cases missed/day (FN);
  - false alarms/day (FP);
  - "1 in N alerts is real" (1/precision);
  - expected net value/day (Σ count × gain);
  - always include the "act on none" and "act on all" baseline rows.
- An "NNT"-style tolerance question ("how many false alarms would you accept to catch one real case?") gives an answer W. That implies C_FN/C_FP ≈ W and p* ≈ 1/(1+W), the same mapping dcurves uses (harm 1/30 ↔ 30 tests per cancer). This mapping is an inference.

### Gaps
- Primary texts on "number needed to evaluate" and the Lancet Digital Health 2025 guidance on performance measures were not fetched (blocked or snippet-only). There are no sourced alerts-per-day case studies with numbers.
