# Cross-cutting modeling choices for datathons: imbalance, calibration, ensembling, validation, features, explainability

Note: all claims below come from search-result snippets (no full-page fetches were done), so each is SNIPPET-ONLY unless stated. No numbers are invented; where I lack a figure it is listed under Gaps.

## Class imbalance: weights vs threshold vs SMOTE/resampling vs focal loss

### Takeaway
With a strong gradient-boosted/consistent classifier, resampling (SMOTE) generally does not help ranking metrics and distorts calibration. Prefer: train unmodified (or with class weights if you only need ranking), then tune the decision threshold for your metric or cost. Resample only if you tune the oversampler hyperparameters carefully.

### Cited Findings
- SNIPPET-ONLY: Elor & Averbuch-Elor ran experiments on 73 datasets; generally, best prediction comes from a strong consistent classifier and balancing is not beneficial; balancing helps only when exceptionally good oversampler hyperparameters are available a priori. — [arXiv 2201.08528](https://arxiv.org/abs/2201.08528) (via [ADS](https://ui.adsabs.harvard.edu/abs/2022arXiv220108528E/abstract)); code [aws/to-smote-or-not](https://github.com/aws/to-smote-or-not)
- SNIPPET-ONLY: van den Goorbergh et al. (JAMIA 2022, logistic regression simulation incl. ridge): random under/oversampling and SMOTE gave no noticeable discrimination benefit, worsened calibration, strongly overestimated probabilities; miscalibration often not restored by recalibration; shifting the probability threshold had similar effect on sensitivity/specificity as imbalance corrections. — [PMC9382395](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9382395/)
- SNIPPET-ONLY: follow-up simulation study on harms of imbalance corrections for ML models (not just logistic regression). — [PMC11771573](https://pmc.ncbi.nlm.nih.gov/articles/PMC11771573/); related paper on random resampling and calibration/discrimination: [ScienceDirect S1532046424000844](https://www.sciencedirect.com/science/article/pii/S1532046424000844)
- SNIPPET-ONLY: a 2024 paper addresses "Platt's scaling for calibration after undersampling" (i.e. recalibration fix if you do undersample). — [arXiv 2410.18144](https://arxiv.org/pdf/2410.18144) (contents not verified)
- Practitioner pieces arguing SMOTE is not a silver bullet exist (e.g. [TrainInData 2026 blog](https://blog.trainindata.com/smote-in-python-a-guide-to-balanced-datasets/), [valeman/smote_is_what_you_dont_need](https://github.com/valeman/smote_is_what_you_dont_need/blob/main/README.md)) — UNVERIFIED content, blog-level quality.

### Inferences
- Class weights/scale_pos_weight act like resampling for calibration (shift predicted probabilities upward), so if probabilities feed dollar calculations, either skip weighting or recalibrate afterward. (Inference from the van den Goorbergh result; not directly tested in sources found.)
- Threshold tuning is the cheapest equivalent for hard-label metrics (F1, cost): train plain, pick threshold on out-of-fold predictions.

### Gaps
- No source found directly comparing focal loss with weights/threshold on tabular GBDT. Kaggle-experience evidence on scale_pos_weight not found. Effect sizes (AUC deltas) not retrieved.

## Metrics and calibration (ROC-AUC vs PR-AUC vs F-beta vs cost; Platt/isotonic/conformal; decision curves)

### Takeaway
ROC-AUC can look flattering under heavy imbalance; PR-AUC reflects precision at deployment prevalence, though it is not universally superior. If a dollar value is computed from probabilities, calibrate (Platt for small calibration sets, isotonic for >~1000) and evaluate with cost or decision-curve net benefit at the relevant threshold.

### Cited Findings
- SNIPPET-ONLY: Saito & Rehmsmeier (PLoS ONE 2015): ROC plots can be deceptive on imbalanced data because of intuitive misreading of specificity; PR plots evaluate the fraction of true positives among positive predictions and better reflect future performance. — [PLoS ONE](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0118432)
- Counterpoint: papers argue ROC accurately assesses imbalanced data ([Patterns 2024](https://www.cell.com/patterns/fulltext/S2666-3899(24)00109-0)) and analyze AUROC vs AUPRC under imbalance ([arXiv 2401.06091](https://arxiv.org/pdf/2401.06091)) — titles only, findings UNVERIFIED.
- SNIPPET-ONLY: Niculescu-Mizil & Caruana: with fewer than roughly 200-1000 calibration cases Platt scaling beats isotonic (isotonic overfits); at 1000+ isotonic is as good or better; isotonic gives piecewise-constant output. — [ICML 2005 PDF](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf), [boosting calibration PDF](https://www.cs.cornell.edu/~caruana/niculescu.scldbst.crc.rev4.pdf)
- SNIPPET-ONLY: Decision curve analysis (Vickers & Elkin 2006): net benefit = sensitivity x prevalence - (1 - specificity) x (1 - prevalence) x w, w = odds at threshold probability; puts benefits and harms on one scale across thresholds. — [Med Decis Making](https://journals.sagepub.com/doi/10.1177/0272989X06295361); common errors: [Harrell, Seven Common Errors in DCA](https://www.fharrell.com/post/edca/) (content not read)
- SNIPPET-ONLY: Conformal prediction gives distribution-free, finite-sample coverage prediction sets via nonconformity score, calibration on held-out data, set construction. — [Angelopoulos & Bates, arXiv 2107.07511](https://ui.adsabs.harvard.edu/abs/2021arXiv210707511A/abstract)
- Temperature scaling: only surfaced via a [KDnuggets overview](https://www.kdnuggets.com/a-deep-dive-into-calibration-of-language-models-platt-scaling-isotonic-regression-temperature-scaling); it is mainly for neural nets (multiclass), not verified here.

### Inferences
- Conformal sets give coverage guarantees, not calibrated point probabilities, so they do not replace Platt/isotonic for expected-value arithmetic.
- Any expected-dollar = p x amount calculation is only as good as calibration of p; checking reliability curves/Brier on OOF data is cheap insurance.

### Gaps
- F-beta guidance and the exact conditions where uncalibrated probabilities break cost calculations: no source retrieved. scikit-learn calibration docs not fetched.

## Ensembling and stacking

### Takeaway
Averaging diverse models is robust; stacking can add more but risks leakage/overfit unless built on out-of-fold predictions. AutoGluon's recipe (multi-layer stacking with skip connections plus repeated k-fold bagging) is the best-documented design.

### Cited Findings
- SNIPPET-ONLY: AutoGluon-Tabular uses multi-layer stack ensembling with skip connections, repeated k-fold bagging on different random partitions with OOF predictions averaged, to curb overfitting; reported up to 50% relative error reduction vs frameworks on OpenML AutoML benchmark and top-20% on Kaggle competitions. — [arXiv 2003.06505](https://arxiv.org/pdf/2003.06505)
- SNIPPET-ONLY: Kaggle grandmasters note stacking was historically a big gold-medal tactic; ensembling beats single models. — [NVIDIA blog](https://developer.nvidia.com/blog/kaggle-grandmasters-unveil-winning-strategies-for-data-science-superpowers/)
- A recent Kaggle Playground winner write-up stresses diversity, selection and trusting the CV-LB relation. — [PS S6E2 1st place](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/1st-place-solution-diversity-selection-and-t) (title only, not read)

### Inferences
- With few rows, simple (rank/weighted) average of a handful of diverse models is lower risk than a flexible meta-learner; fit blend weights on OOF predictions with constraints.

### Gaps
- No quantitative evidence found on blend-overfitting, rank averaging vs stacking, or how many seeds/folds are enough.

## Validation strategy

### Takeaway
Use CV you trust over the public leaderboard; the same CV loop used to tune and to report is optimistically biased, so use nested CV (or a held-out set) for honest estimates when many configs are tried.

### Cited Findings
- SNIPPET-ONLY: Cawley & Talbot (JMLR 2010): selecting hyperparameters and estimating performance with one CV loop yields optimistic bias; nested CV is the remedy; variance of the selection criterion causes overfitting in model selection. — [JMLR](https://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf)
- SNIPPET-ONLY: counterview that nested CV is overzealous for most practical classifier selection. — [arXiv 1809.09446](https://arxiv.org/pdf/1809.09446)
- "Overtuning in hyperparameter optimization" documents overfitting to validation. — [arXiv 2506.19540](https://arxiv.org/pdf/2506.19540) (title only)
- SNIPPET-ONLY: "Trust your CV" is the Kaggle norm; shake-up = rank difference between public and private LB; competitors simulate public/private splits across folds. — [Kaggle shake-up handbook (Medium)](https://medium.com/global-maksimum-data-information-technologies/kaggle-handbook-fundamentals-to-survive-a-kaggle-shake-up-3dec0c085bc8)

### Gaps
- Repeated vs single CV variance numbers and adversarial validation: no sources retrieved.

## Feature engineering vs model choice

### Takeaway
Evidence found only on the model-choice side: trees beat deep nets on medium tabular data. No citable source quantifying feature engineering's share was found.

### Cited Findings
- SNIPPET-ONLY: Grinsztajn et al. (NeurIPS 2022), 45 datasets: tree-based models remain state of the art on medium-sized (~10K) data; good models must be robust to uninformative features, preserve data orientation, learn irregular functions. — [researchgate](https://www.researchgate.net/publication/362123616_Why_do_tree-based_models_still_outperform_deep_learning_on_tabular_data)

### Gaps
- Out-of-fold target encoding, frequency encoding, aggregations, interactions: no sources retrieved. Cite general principle (leakage if target encoding is not OOF) only as UNVERIFIED practitioner knowledge.

## Explainability (SHAP, permutation importance, judges)

### Takeaway
Report SHAP with explicit method choice and treat results as model explanations, not causal effects; check permutation importance on grouped correlated features.

### Cited Findings
- SNIPPET-ONLY: Kumar et al. (ICML 2020): Shapley-based explanations fail as general feature-importance measures, mathematical problems arise, mitigations need causal reasoning, and they may not suit human-centric explainability goals. — [PMLR v119](https://proceedings.mlr.press/v119/kumar20e.html)
- SNIPPET-ONLY: TreeExplainer default is tree-path-dependent (fast, uses leaf training counts); interventional needs background data; with correlated features path-dependent can give later-split features larger contributions. — [SHAP docs](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html), [Alibi interventional example](https://docs.seldon.io/projects/alibi/en/latest/examples/interventional_tree_shap_adult_xgb.html)
- SNIPPET-ONLY: scikit-learn: with correlated features, permuting one leaves its twin available, so both can show low importance; remedy is clustering correlated features and keeping one per cluster. — [sklearn permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html), [multicollinear example](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html)

### Gaps
- What datathon judges accept: no source found; likely rubric-dependent. Interventional-vs-conditional (off-manifold) critique not directly sourced beyond Kumar.
