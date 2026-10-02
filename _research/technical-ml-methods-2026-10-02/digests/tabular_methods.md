# Supervised methods for tabular classification/regression: what wins, and when

Access note: arxiv.org and medium.com were blocked by the egress proxy (WebFetch). Most claims below come from WebSearch result summaries and are marked SNIPPET-ONLY (not confirmed against the full paper). Only github.com pages were fetched in full. Treat every number as snippet-level unless stated.

## 1. What does benchmark evidence say about when each family wins?

### Takeaway
Across benchmarks, tuned/ensembled GBDTs remain a strong default on medium-large, irregular data; but on the newest benchmarks (TabArena, TALENT) tabular foundation models win on small data, modern MLP-ensembles (TabM, RealMLP) match or beat GBDTs, and cross-model ensembles are best overall. Older results (Grinsztajn 2022, TabZilla 2023) predate these and favor trees more.

### Cited Findings
- Grinsztajn et al. (NeurIPS 2022): tree-based models remain state-of-the-art on medium-sized data (~10k samples), even ignoring their speed advantage. Challenges identified for NNs: be robust to uninformative features, preserve data orientation (non-rotation-invariant), learn irregular functions easily. SNIPPET-ONLY — [search summary](https://www.researchgate.net/publication/362123616_Why_do_tree-based_models_still_outperform_deep_learning_on_tabular_data)
- TabZilla / McElfresh et al. (NeurIPS 2023): 19 algorithms x 176 datasets. NN-vs-GBDT debate is often overemphasized: on many datasets the difference is negligible, or light GBDT hyperparameter tuning matters more than model-family choice. Dataset regularity (less skewed, less heavy-tailed features) predicts NN wins; GBDTs better on larger datasets and much better with skewed/heavy-tailed/irregular features. TabPFN (v1) had the best average performance and fastest training but average rank 4.88, not dominant. SNIPPET-ONLY — [arXiv 2305.02997](https://arxiv.org/pdf/2305.02997) (search summary)
- TabArena (NeurIPS 2025 D&B): 51 curated datasets, 16 models in the paper (repo now lists 27+ methods incl. 10+ tabular foundation models; 25M runs in paper). Headline findings: validation method and ensembling of hyperparameter configs matter a lot to benchmark models at full potential; foundation models excel on smaller datasets; ensembles across models advance state of the art. SNIPPET-ONLY for findings — [search summary of arXiv 2506.16791](https://arxiv.org/abs/2506.16791); repo description (fetched) — [GitHub autogluon/tabarena](https://github.com/autogluon/tabarena). Repo also lists BeyondArena: 142 datasets incl. temporal and grouped tasks, from tiny to 1M rows (same page).
- TALENT (300 datasets: 120 binary, 80 multiclass, 100 regression): recent deep models (ModernNCA, TabM, RealMLP) and small-data foundation models perform similar to or better than GBDTs; post-hoc ensembling per model and across models improves performance dramatically. SNIPPET-ONLY — [TALENT GitHub](https://github.com/LAMDA-Tabular/TALENT), [paper](https://arxiv.org/pdf/2407.00956), TabArena text quoted via search.
- TabM (ICLR 2025): MLP with BatchEnsemble-style parameter-efficient ensembling; reported average rank 1.7 vs 15+ models on 46 datasets, ahead of XGBoost/CatBoost/LightGBM; mean relative improvement over plain MLP 2.82 ± 4.0% (TabM†). Note: rank claim is from the authors' own benchmark and a social-media-sourced summary; SNIPPET-ONLY — [arXiv 2410.24210](https://arxiv.org/pdf/2410.24210), [GitHub](https://github.com/yandex-research/tabm)
- RealMLP ("Better by Default", NeurIPS 2024): improved MLP plus meta-tuned default hyperparameters for GBDTs and RealMLP, tuned on 71 classification + 47 regression meta-train datasets; competitive with GBDTs on medium-to-large datasets (1K-500K samples) with a good time-accuracy tradeoff; tuned defaults can rival expensive HPO. SNIPPET-ONLY — [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/2ee1c87245956e3eaa71aaba5f5753eb-Paper-Conference.pdf)
- A secondary benchmark aggregator (aimultiple, low-authority) claims on <1,000-row data foundation models take the top four slots on numeric and hybrid data; logistic regression 91.2% mean ROC-AUC on numeric but 77.5% on hybrid; LightGBM best classical model (mean rank 5.16). SNIPPET-ONLY, secondary source — [aimultiple](https://aimultiple.com/tabular-models)
- Categorical: CatBoost claimed strongest mean rank among GBDTs, especially for high-cardinality categoricals, mixed types and moderate-to-high noise; on Amazon dataset log-loss 0.1377 vs LightGBM 0.1636 (+18.8%). SNIPPET-ONLY, from a CatBoost-promoting source (Substack summarizing a book) — [valeman substack](https://valeman.substack.com/p/where-catboost-beats-xgboost-and); original method paper [CatBoost](https://arxiv.org/pdf/1706.09516). XGBoost needs numeric encoding for categoricals per same snippet (note: recent XGBoost has native categorical support; UNVERIFIED here).

### Inferences
- Pre-2024 "trees beat NNs" conclusions should be treated as superseded for default advice: MLP ensembles and foundation models now compete. Still, GBDTs are the safest single model for large, skewed, irregular data.
- Cross-model ensembling (GBDT + NN + FM) is the consistent top performer, at higher compute cost.
- Library defaults vs tuned matters: RealMLP's finding implies tuned defaults often suffice for a hackathon time budget.

### Gaps
- No exact Elo/rank tables from TabArena or TALENT retrieved (arxiv blocked; tabarena.ai leaderboard not fetched).
- No dedicated effect-size source for EBM/KNN/SVM/ExtraTrees families in these benchmarks; random forest/ExtraTrees generally rank below GBDT but I did not find a sourced number.
- FT-Transformer/ResNet specific ranks not retrieved.

## 2. Decision thresholds: size, and when do deep nets/ensembles help?

### Takeaway
Foundation models (TabPFN v2/2.5, TabICL) are the best bet below roughly 10k rows (TabPFN-2.5 designed up to 50k rows, 2k features) and remain competitive to ~100k with TabICL/TabPFN-2.5; above that, GBDTs/MLP-ensembles plus stacking. Exact "linear/ExtraTrees beat GBDT below N rows" thresholds were not found in sourced form.

### Cited Findings
- TabPFN-2.5 scales to 50,000 samples x 2,000 features (5x rows, 4x features over TabPFNv2); reports strong results up to 100k rows; 100% win rate vs default XGBoost on classification datasets <=10,000 rows and <=500 features, 87% win rate for up to 100K rows/2K features; claims forward-pass accuracy matching AutoGluon 1.4 tuned 4h. Vendor/author claim; SNIPPET-ONLY — [arXiv 2511.08667](https://arxiv.org/pdf/2511.08667)
- TabPFN-2.5 limitation: transformer attention is quadratic in input length so memory is high at 50k rows (same source).
- TabICL: 100K samples x 500 features with 5 GB GPU memory and 32 GB RAM; on 200 TALENT classification datasets on par with TabPFNv2 and up to 10x faster (10k x 100: ~20 s vs 1 min 40 s); on 53 datasets with >10K samples beats TabPFNv2 and CatBoost. Authors' claim; SNIPPET-ONLY — [TabICL arXiv 2502.05564](https://arxiv.org/abs/2502.05564). A v2 exists: [TabICLv2](https://arxiv.org/html/2602.11139v1), [GitHub soda-inria/tabicl](https://github.com/soda-inria/tabicl) (details not retrieved).
- TabArena: foundation models excel on smaller datasets (see Q1). Repo lists >10 foundation models.
- LightGBM improved more over logistic regression once sample size exceeded 10,000 in a diabetes-prediction study (medical, one domain). SNIPPET-ONLY — [Sci Rep](https://www.nature.com/articles/s41598-022-20149-z)
- RealMLP competitive with GBDT in 1K-500K sample range (see Q1).
- Ensembling: TALENT/TabArena both report ensembling (per model and across models) as a large gain (see Q1).

### Inferences
- Practical rule for datasets: n <= ~10k: try TabPFN-2.5/TabICL first, plus tuned CatBoost; 10k-100k: GBDT + TabM/RealMLP + TabICL if GPU; >100k: GBDT and MLP ensembles, FMs less practical. These are inferences from the cited thresholds, not published decision rules.
- Fixed GPU requirement for FMs matters under hackathon compute budgets.

### Gaps
- No sourced threshold for when ExtraTrees/linear models beat GBDTs; only the weak <10k logistic-regression hint above.
- Noise-level effects: only found CatBoost "moderate-to-high noise" claim (weak, SNIPPET-ONLY) and a GBDT label-noise paper ([arXiv 2409.08647](https://arxiv.org/html/2409.08647v1), not read).
- AutoML stacking (AutoGluon) numbers only via TabPFN-2.5 comparison; no direct benchmark pulled.

## 3. What did Kaggle Playground / tabular competition winners use (2024-2026)?

### Takeaway
Winners almost never use a single model: large blends/stacks of GBDTs (XGBoost, LightGBM, CatBoost), NNs, TabPFN and simple models, with heavy feature engineering; XGBoost is frequently the best single model.

### Cited Findings
- Chris Deotte, 1st place Playground April 2025 (Podcast): 3-level stack of 72 models using RAPIDS cuML on GPU: XGBoost, LightGBM, CatBoost, NNs, TabPFN, KNN, SVR, Ridge, Random Forest, combined by Ridge and GBDT. SNIPPET-ONLY, from a Medium blog (could not open) — [Medium](https://medium.com/@gauurab/kaggle-playground-how-top-competitors-actually-win-in-2025-c75d4b380bb5)
- Deotte also won S5E6 (Jun 2025) with 1 of 9 ensembled models being XGBoost trained on original data with depth 18; Mahog won S5E11 (Loan Payback, Nov 2025) with XGBoost as best single model. Same source, SNIPPET-ONLY.
- Same source: winners build thousands of features (groupby stats, interactions, binning, digit extraction), blending dozens (sometimes 70+) of models. SNIPPET-ONLY, blog opinion.
- S6E5 (Predicting F1 Pit Stops) winning solution was an ensemble including XGBoost, documented via PR in XGBoost repo (details not in the page) — [dmlc/xgboost PR 12267](https://github.com/dmlc/xgboost/pull/12267) (fetched; limited).
- Other write-ups exist but were not opened: [S5E8 top25 write-up](https://www.kaggle.com/competitions/playground-series-s5e8/writeups/top25), [S6E4 log](https://hittmg.hatenablog.jp/entry/2026/04/11/080000).

### Inferences
- Using the original (pre-synthetic) dataset as extra training data and GPU-accelerated stacking are recurring winning levers in Playground (synthetic-data competitions); this applies less to real-world datathons with small data.

### Gaps
- No non-Playground (main-track Kaggle) or datathon winner write-ups retrieved; primary Kaggle write-ups not opened (Kaggle likely blocked/not fetched).

## 4. When does a simple interpretable model (logistic regression, EBM) lose little accuracy and win on trust?

### Takeaway
EBMs (GAMs with boosted shape functions) are reported to be roughly on par with GBDTs on many tabular datasets while giving exact, editable explanations; logistic regression holds up on small, numeric-only data but degrades on mixed/categorical data.

### Cited Findings
- EBM vs XGBoost on Adult Income: AUROC 0.928±0.002 vs 0.927±0.001. SNIPPET-ONLY — [Accuracy, Interpretability, and DP via Explainable Boosting](https://arxiv.org/pdf/2106.09680) (search summary attribution uncertain)
- EBM described as comparable to RF/gradient-boosted trees with exact explanations, editable by domain experts; additive structure gives exact global/local interpretability. SNIPPET-ONLY — [interpretml GitHub](https://github.com/interpretml/interpret), [GAM trust paper](https://arxiv.org/pdf/2006.06466)
- Logistic regression 91.2% vs 77.5% mean ROC-AUC on numeric vs hybrid (see Q1; secondary source).
- GBDT beats logistic regression more clearly at >10k rows in one diabetes study (see Q2).

### Inferences
- EBM/logistic regression are justified when strong interactions are absent, features are mostly monotone/additive, data are small, or regulators/judges need explanations; use as a baseline and check gap to GBDT before committing. Inference from cited evidence, not a sourced rule.

### Gaps
- No multi-dataset benchmark quantifying EBM-vs-GBDT accuracy loss; TabArena/TALENT numbers for EBM, KNN, SVM not retrieved.
- Nothing sourced on SVM/KNN beyond TALENT/TabR context (ModernNCA, a neighbor-based deep model, reportedly outperforms tree models in TALENT per snippet) and the TabZilla note that logistic regression/KNN occasionally win on niche datasets (secondary, SNIPPET-ONLY, aimultiple).
