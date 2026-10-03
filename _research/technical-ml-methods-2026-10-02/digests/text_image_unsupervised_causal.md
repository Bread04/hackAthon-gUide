# Non-standard-tabular datathon tasks: text, images, anomaly, clustering, recsys, causal/uplift

Evidence caveat: all findings below come from WebSearch result summaries (no page was fetched in full), so every cited claim is effectively SNIPPET-ONLY unless noted. Recommendations marked "(practitioner judgement)" are inferences, not sourced. I found nothing on BEIR or on 2026 hosted-LLM pricing; see Gaps.

## (a) Text columns / NLP

### Takeaway
Start with TF-IDF + logistic regression/linear SVM (minutes, deterministic, strong baseline). Upgrade to frozen sentence embeddings + linear/GBDT head, then a fine-tuned small transformer if labels exist. Use zero-shot LLMs only for label-free or extraction tasks, and cache outputs, because LLM runs are not reliably reproducible.

### Cited Findings
- Fine-tuned small models (RoBERTa, DeBERTa-v3, Electra) were compared against zero-shot ChatGPT/Claude, and the paper's title states fine-tuned small LLMs "still significantly outperform" zero-shot generative models in text classification — [arXiv 2406.08660](https://arxiv.org/pdf/2406.08660) (SNIPPET-ONLY: title and abstract-level summary)
- One study reports TF-IDF at 64% F1 vs BERT-base-uncased at 82.18% on the same task; classical TF-IDF is described as effective for unambiguous text but weaker on ambiguous text — [search summary of various papers incl. IIETA study](https://www.iieta.org/journals/isi/paper/10.18280/isi.300606) (SNIPPET-ONLY; single task, don't generalise)
- Broader benchmark and comparison papers exist: [Text Classification in the LLM Era](https://arxiv.org/pdf/2502.11830), [A thorough benchmark of automatic text classification](https://arxiv.org/pdf/2504.01930) (SNIPPET-ONLY: titles only, results not read)
- MTEB classification protocol: embeddings feed a logistic regression, scored by F1 (accuracy in some versions); clustering uses mini-batch k-means with k = number of labels, scored by V-measure. So "frozen embeddings + logistic regression" is exactly how MTEB evaluates models, which supports it as a default recipe — [BGE MTEB docs / overview](https://bge-model.com/tutorial/4_Evaluation/4.2.2.html), [summary](https://zeroentropy.dev/concepts/mteb/) (SNIPPET-ONLY)
- MTEB is community-maintained; leaderboard at huggingface.co/spaces/mteb/leaderboard. Models named near the top in 2025-26 secondary sources: NV-Embed-v2, KaLM-Embedding-Gemma3-12B (72.32), Jina v5-text-small (71.7), "Harrier-OSS-v1-27B (74.3)" — [codesota summary](https://www.codesota.com/benchmarks/mteb), [futureagi](https://futureagi.com/blog/best-embedding-models-2025/) (SNIPPET-ONLY, aggregator blogs; scores unverified and leaderboard changes quickly, UNVERIFIED)
- LLM nondeterminism: temperature 0 plus fixed seed reduces but does not guarantee identical outputs, due to near-tie tokens, GPU kernel ordering and batching — [arXiv 2407.10457](https://arxiv.org/pdf/2407.10457), [arXiv 2601.06118](https://arxiv.org/pdf/2601.06118), [Sara Zan 2026](https://www.zansara.dev/posts/2026-03-24-temp-0-llm/) (SNIPPET-ONLY)
- LLM annotation benchmarking (toxicity/incivility) exists: [arXiv 2409.09741](https://arxiv.org/pdf/2409.09741) (SNIPPET-ONLY)

### Inferences
- Recommended order (practitioner judgement): (1) TF-IDF word+char n-grams + LogisticRegression, stratified/time-aware CV; (2) sentence-transformers embeddings (e.g. a small BGE/E5/MiniLM-class model chosen from MTEB classification column) + logistic regression, or concatenated with tabular features into LightGBM/XGBoost; (3) fine-tune DeBERTa-v3-small/base only if >~1-5k labels and a GPU; (4) zero-shot/few-shot LLM for extraction or no-label cases, run once, cache to disk, pin model version, and store raw outputs.
- Avoid: fine-tuning large models on tiny data; embedding dense 768-d vectors straight into GBDT without PCA/SVD on small N (overfit, slow); calling an LLM API row by row without caching or rate-limit handling; judging an LLM approach on a handful of examples.
- Reproducibility for judges: cached LLM outputs are the only way to make a final submission re-runnable.

### Gaps
- No BEIR evidence retrieved (relevant only for retrieval/search tasks; use MTEB retrieval column or BEIR as a model-selection guide).
- No sourced cost/latency numbers for LLM APIs; pricing changes, check provider pages.
- Did not read arXiv 2406.08660 or 2502.11830 in full, so no accuracy margins reported.

## (b) Images

### Takeaway
Extract frozen DINOv2 (or CLIP/SigLIP) embeddings once, fit a linear/logistic head (or GBDT with tabular features). Fine-tune only if the linear probe plateaus and you have GPU hours plus a few thousand labelled images.

### Cited Findings
- DINOv2 surpasses OpenCLIP and EVA-CLIP on linear evaluation with large ViTs; the DINOv2 authors note finetuning is optional, with only modest gains over linear probing — [DINOv2 paper](https://arxiv.org/html/2304.07193v2) (SNIPPET-ONLY)
- One study found linear probing improves with DINOv2 but full fine-tuning can worsen, suggesting overfitting risk — [search summary; likely from the PEFT/transfer studies such as arXiv 2409.16434](https://arxiv.org/pdf/2409.16434) (SNIPPET-ONLY; exact source of this claim not confirmed, UNVERIFIED attribution)
- DINOv2 on radiology benchmarks experimental study: [arXiv 2312.02366](https://arxiv.org/html/2312.02366v1) (SNIPPET-ONLY; title only)
- DINOv3 exists (Aug 2025) as the newer successor: [arXiv 2508.10104](https://arxiv.org/pdf/2508.10104) (SNIPPET-ONLY)

### Inferences
- First method: DINOv2 ViT-S/B (fast on a free GPU) CLS + mean patch features, L2-normalise, logistic regression; CLIP/SigLIP if text-image alignment or zero-shot labels are needed (domain-specific: medical/satellite may differ).
- Upgrade: light augmentation + test-time averaging of embeddings, ensemble of two backbones, then LoRA/last-blocks fine-tune with low LR.
- Avoid: training CNNs from scratch, full fine-tuning at high LR on small data, leaking near-duplicate images across train/val splits (group by source).

### Gaps
- No datathon-specific comparison with quantified margins found; domain-shift cases (medical, microscopy) not covered.

## (c) Anomaly / fraud with few labels

### Takeaway
Run ECOD and Isolation Forest (PyOD) as free unsupervised baselines. As soon as even ~1% anomaly labels exist, a semi-supervised or supervised method (e.g. XGBOD, GBDT) tends to win; split by time.

### Cited Findings
- ADBench (NeurIPS 2022): 30 algorithms, 57 datasets; no unsupervised algorithm statistically outperforms the rest — [ADBench arXiv 2206.09426](https://arxiv.org/pdf/2206.09426), [NeurIPS paper](https://papers.neurips.cc/paper_files/paper/2022/file/cf93972b116ca5268827d575f2cc226b-Paper-Datasets_and_Benchmarks.pdf) (SNIPPET-ONLY)
- With 1% labelled anomalies semi-supervised methods (DeepSAD, DevNet, XGBOD) frequently beat the best unsupervised ones (median AUROC reported 71% to 75%); fully supervised methods (tree ensembles, FTTransformer) need ~10% anomaly labels to match — [ADBench summary](https://www.alphaxiv.org/abs/2206.09426), [emergentmind](https://www.emergentmind.com/topics/adbench-anomaly-detection-benchmark) (SNIPPET-ONLY, secondary summary)
- Robustness: unsupervised methods degrade (-16% median AUROC at 6x anomaly duplication); (semi-)supervised more stable under duplicates, irrelevant features, label noise — same ADBench summary (SNIPPET-ONLY)
- ECOD: parameter-free, O(nd), reported +2% AUROC and +5% AP over second-best detector in its own paper (authors' own evaluation) — [ECOD arXiv 2201.00382](https://arxiv.org/pdf/2201.00382); PyOD README/docs recommend ECOD and IForest as starting points based on ADBench — [PyOD README](https://github.com/yzhao062/pyod/blob/master/README.rst), [PyOD docs 3.5.0](https://pyod.readthedocs.io/en/latest/examples/tabular.html) (SNIPPET-ONLY)
- Temporal leakage: one fraud paper reports random splits inflating results (0.81 PR-AUC random vs 0.553 a month later; "0.19 PR-AUC and 2x recall at 90% precision" over-reporting) — [arXiv 2603.06632](https://arxiv.org/html/2603.06632) (SNIPPET-ONLY; likely single dataset)
- Temporal drift in fraud: [PMC article](https://pmc.ncbi.nlm.nih.gov/articles/PMC13542085/) (SNIPPET-ONLY)

### Inferences
- First: ECOD + IForest, rank-average scores; check against any labels with PR-AUC (not accuracy). Upgrade: LightGBM/XGBoost on labels with class weights, time-based CV, plus unsupervised scores as features (XGBOD-style). Autoencoders: ADBench results above did not single them out; treat as optional.
- Avoid: random splits on temporal data, fitting scalers/encoders on full data, tuning thresholds on test, trusting "labels" that are only investigated cases (selection bias/label noise), oversampling before splitting.
- Label noise: ADBench says supervised methods are more stable than unsupervised under noise; do not over-read.

### Gaps
- No specific ADBench ranking list for IForest vs ECOD vs COPOD retrieved; no autoencoder-specific evidence.

## (d) Clustering / segmentation

### Takeaway
KMeans (scaled features, small k sweep) as the first pass, GMM for soft/elliptical groups, HDBSCAN when shapes are irregular or noise exists; validate with stability (resampling) and business interpretability, not silhouette alone.

### Cited Findings
- KMeans: fixed k, suits roughly spherical well-separated clusters, scales O(nKt), sensitive to outliers; HDBSCAN needs no k, labels outliers as noise, uses a stability score over density levels — [search summaries incl. Towards AI comparison](https://pub.towardsai.net/advanced-customer-segmentation-a-comprehensive-comparison-of-hdbscan-dbscan-and-k-means-ce3bcaa7f1a2?gi=54fd4a2a82ab), [letsdatascience](https://letsdatascience.com/blog/mastering-hdbscan-clustering-variable-density-data-made-easy) (SNIPPET-ONLY, blog-grade)
- Silhouette alone is unreliable; HDBSCAN noise points can inflate scores if excluded — marketing KMeans/HDBSCAN comparison, [ResearchGate](https://www.researchgate.net/publication/387934161_Comparison_of_K-Means_and_HDBSCAN_Clustering_Approaches_to_Enhance_Marketing_Strategies) (SNIPPET-ONLY)
- Stability-based relative validation package: [reval](https://www.sciencedirect.com/science/article/pii/S2666389921000428); scikit-learn clustering docs (v1.9.0 listed): [sklearn](https://sklearn.org/stable/modules/clustering.html) (SNIPPET-ONLY)
- Text-embedding effect on clustering: [arXiv 2305.03144](https://arxiv.org/pdf/2305.03144) (SNIPPET-ONLY)

### Inferences
- Validate: re-run on bootstrap subsamples and compare with adjusted Rand index; check profile differences on features not used for clustering; report cluster sizes; name segments by actionable traits. For high-dim/embedding data, reduce with PCA/UMAP before HDBSCAN (practitioner judgement).
- Avoid: clustering unscaled mixed features; choosing k by silhouette alone; presenting clusters as "real" categories; ignoring HDBSCAN's noise share.

### Gaps
- No peer-reviewed benchmark found comparing KMeans/GMM/HDBSCAN stability; evidence is blog/summary level.

## (e) Recommendation / similarity

### Takeaway
Popularity and item-item cosine/co-occurrence as baseline, implicit ALS as the main upgrade, content embeddings for cold-start; evaluate with time-based holdout.

### Cited Findings
- One 2024 e-commerce repo evaluation: ALS Recall@10 0.1182 vs item-item 0.0624, but item-item coverage 0.4836 vs ALS 0.1056 — [kirtis111 repo](https://github.com/kirtis111/e-commerce-recommendation-system) (SNIPPET-ONLY; single hobby project, weak evidence)
- iALS revisited: properly tuned iALS is competitive on standard benchmarks — [Revisiting the Performance of iALS](https://www.researchgate.net/publication/355698565_Revisiting_the_Performance_of_iALS_on_Item_Recommendation_Benchmarks) (SNIPPET-ONLY; 2021-22, pre-dates preferred range)
- Reproducibility concerns: some replication studies show large drops (-51% to -72%) from evaluation-protocol details — [arXiv 2501.10143](https://arxiv.org/pdf/2501.10143), [survey](https://arxiv.org/pdf/2607.26074) (SNIPPET-ONLY; exact attribution of the range unconfirmed, UNVERIFIED)
- Libraries: `implicit` (ALS, BPR, LMF; CPU and CUDA) — docs list 0.7.2; shown as updated ~2 months before search — [GitHub](https://github.com/benfred/implicit), [docs](https://benfred.github.io/implicit/) (SNIPPET-ONLY); RecTools baselines tutorial — [RecTools docs](https://rectools.readthedocs.io/en/latest/examples/tutorials/baselines_extended_tutorial.html)

### Inferences
- Tune regularisation/factors for ALS; apply BM25/TF-IDF weighting of the interaction matrix (common practice, unsourced here). Use leave-last-out or time split; compare against popularity always.
- Avoid: random splits on interaction logs, optimising only Recall (check coverage), deep sequence models in 24-48h.

### Gaps
- No strong 2023-26 benchmark comparing ALS vs item-item vs embeddings; LLM-based recsys evidence not gathered.

## (f) Causal / uplift ("who to target")

### Takeaway
A response model ranks people likely to convert, not people changed by treatment. With randomised treatment data, use a T-learner/X-learner with LightGBM as the first method, upgrade to causal forest/DR learners, evaluate with Qini/AUUC on held-out data. Without randomisation, state confounding risk and use propensity adjustment cautiously.

### Cited Findings
- Response models target "sure things"; only persuadables yield incremental response; uplift can't be observed per individual, so standard accuracy metrics don't apply; Qini/cumulative-gain curves are the evaluation — [ScienceDirect: Why you should stop predicting customer churn and start using uplift models](https://www.sciencedirect.com/science/article/pii/S0020025519312022), [Wikipedia: Uplift modelling](https://en.wikipedia.org/wiki/Uplift_modelling) (SNIPPET-ONLY)
- Criteo Uplift v2.1 comparisons: in one large-scale study, causal forest Qini 0.0877, X-learner 0.0778, T-learner 0.0712 (2.8M held-out rows), DR-learner indistinguishable from T-learner; another summary says response-LightGBM scored highest under conversion with causal forest close, while T/X trailed — [arXiv 2604.06123 listing](https://awesomepapers.io/federated-learning/papers/2604.06123), [GitHub omvyas77/uplift](https://github.com/omvyas77/uplift) (SNIPPET-ONLY; mixed results, secondary sources, UNVERIFIED; note one source shows a plain response model can match uplift models on this dataset)
- Libraries: CausalML 0.17.0 (July 2026, 4 maintainers) and EconML 0.17.0 (July 2026) per search summaries — [PyPI causalml](https://pypi.org/project/causalml/), [Snyk causalml](https://snyk.io/advisor/python/causalml) (SNIPPET-ONLY; version/dates UNVERIFIED); CausalML paper [arXiv 2002.11631](https://arxiv.org/pdf/2002.11631); UTBoost (GBDT for uplift) [arXiv 2312.02573](https://arxiv.org/pdf/2312.02573)

### Inferences
- Confounding: in observational data, features that drove who got treated also drive outcomes, so correlation-based models can recommend targeting people who were selected for treatment because they were already likely to respond. Check treatment balance/propensity overlap first; randomisation (RCT) is the clean case.
- Avoid: ranking by P(Y|X) alone; evaluating uplift with AUC; tuning on the same data as Qini; ignoring cost of treatment (use expected incremental profit); heavy HTE claims with small N (noisy; start with simple T-learner/ATE and a few segments).

### Gaps
- Benchmark results are inconsistent and come from secondary summaries; no rigorous independent confirmation that causal forest beats meta-learners generally.
- Observational-data (non-RCT) uplift pitfalls not backed by a retrieved source beyond the general causal-inference argument.
