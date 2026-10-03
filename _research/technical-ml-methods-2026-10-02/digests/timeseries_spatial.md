# Time-series, panel/temporal and spatial methods for short competitions

Research limits: arxiv.org and many sites were blocked for WebFetch; most claims below come from WebSearch result summaries (marked SNIPPET-ONLY) or GitHub README fetches. No numbers are invented; where a number was not retrieved it is listed under Gaps.

## 1. Which family wins by benchmark/horizon/series count/seasonality/intermittency (M4, M5, M6, GIFT-Eval, fev-bench, Kaggle)

### Takeaway
Evidence splits by regime. On large retail panels with exogenous drivers (M5, Favorita), LightGBM with lag/rolling features and ensembling won. On M4 (100k heterogeneous series) simple statistical combinations beat pure ML/NN and only a hybrid (ES-RNN) won. In 2025-26 zero-shot pretrained foundation models (Chronos-2, TiRex, TimesFM-2.5) lead fev-bench and GIFT-Eval over statistical and task-specific deep models. M6 (finance) shows most teams fail to beat benchmarks.

### Cited Findings
- M4: winner Slawek Smyl's hybrid ES-RNN (Holt-Winters-style ES plus dilated LSTM) won "with a solid margin"; pure ML/NN methods did worse than standard ARIMA/ETS, and worse still than combinations of statistical methods. SNIPPET-ONLY — [Uber blog](https://www.uber.com/at/en/blog/m4-forecasting-competition/), [search summary of M4 results](https://arxiv.org/html/2203.03279). ES-RNN claimed 9.4% sMAPE improvement (SNIPPET-ONLY, same sources). M4 metric OWA averages sMAPE and MASE relative to Naive2.
- M5 Accuracy: all top-50 methods were ML, the first M competition where this held, and significantly better than all statistical benchmarks; winner YeonJun In used equal-weighted combination of LightGBM models trained at different aggregation levels; improvement vs best benchmark exceeded one fifth (SNIPPET-ONLY) — [M5 accuracy paper, IJF](https://www.sciencedirect.com/science/article/pii/S0169207021001874), [PDF copy](https://statmodeling.stat.columbia.edu/wp-content/uploads/2021/10/M5_accuracy_competition.pdf). M5 data are hierarchical, intermittent retail sales with exogenous variables (prices, calendar/events).
- Favorita (Kaggle grocery, 16-day horizon): LightGBM was the standard method of the winners; a public write-up reports LightGBM wins short/mid horizons using origin lags (7/28-day means, own sales, promotion) (SNIPPET-ONLY; the latter is a single GitHub repo, low authority) — [search results incl. Kaggle learnings paper](https://arxiv.org/pdf/2009.07701), [repo](https://github.com/gersonlramos/Store-Sales---Time-Series-Forecasting).
- Kaggle Web Traffic (~145k Wikipedia series): a 2nd-place solution mixed Keras deep models, XGBoost and statistical baselines (SNIPPET-ONLY, see search above; [Kaggle learnings paper](https://arxiv.org/pdf/2009.07701)).
- M6 (finance, live 48 weeks): only 23.3% of teams beat benchmarks on forecast accuracy, 28.8% on portfolio, 6.7% on both; top forecasters had inefficient portfolios and vice versa; teams that updated frequently and combined forecasts did better (SNIPPET-ONLY) — [M6 paper](https://arxiv.org/html/2310.13357), [IJF record](https://ideas.repec.org/a/eee/intfor/v41y2025i4p1315-1354.html).
- fev-bench: 100 tasks, 96 datasets, frequencies 5 min to quarterly; Chronos-2, TiRex, TimesFM-2.5 generally dominate statistical and task-specific deep models; Chronos-2 statistically significantly ahead in probabilistic and point forecasting. Reported probabilistic win rate/skill score: TiRex 86.7%/42.6%, TimesFM-2.5 82.1%/42.3% (full benchmark); on a smaller covariate subset Chronos-2 90.3%/51.1%, TiRex 79.5%/43.4%. TabPFN-TS strong with dynamic covariates; Toto-1.0 in multivariate settings. SNIPPET-ONLY — [fev-bench paper](https://arxiv.org/pdf/2509.26468), [alphaxiv](https://www.alphaxiv.org/abs/2509.26468). (Note: the two reported full-benchmark rows come from a summary; verify against paper tables.)
- GIFT-Eval: 23 datasets, 97 tasks, >144k series, 177M points, 7 domains, 10 frequencies; metrics MASE and CRPS normalized by seasonal naive and aggregated by geometric mean; leaderboard lists statistical (Naive, Seasonal Naive, AutoARIMA, AutoETS, AutoTheta), deep-learning, and foundation models. SNIPPET-ONLY — [GIFT-Eval repo](https://github.com/SalesforceAIResearch/gift-eval), [overview](https://www.emergentmind.com/topics/gift-eval). TabPFN-TS reported ranked 1st on GIFT-Eval in January 2025 (SNIPPET-ONLY) — [TabPFN-TS repo](https://github.com/PriorLabs/tabpfn-time-series), [paper](https://arxiv.org/pdf/2501.02945).
- Chronos-2 (120M params) supports univariate, multivariate and covariate (past-only or known-future, real or categorical) forecasting zero-shot via group attention/in-context learning; "consistently outperforms baselines by a wide margin" on covariate tasks (SNIPPET-ONLY) — [paper](https://arxiv.org/abs/2510.15821), [repo](https://github.com/amazon-science/chronos-forecasting).
- Deep models: PatchTST (ICLR 2023) beats DLinear and other transformers in its own paper ([arXiv](https://arxiv.org/pdf/2211.14730)); N-BEATS from M4-era ([arXiv](https://arxiv.org/pdf/1905.10437)). No direct, sourced deep vs LightGBM comparison on a common panel was retrieved.

### Inferences
- Rule of thumb for a datathon: panel with many series plus known-future covariates (price, promo, weather, calendar) -> global LightGBM with lags/rolling, then compare to Chronos-2/TabPFN-TS with covariates; few or single series -> seasonal naive, ETS/Theta and a foundation model zero-shot, blend. Intermittent demand: M5 evidence favors LightGBM with tweedie-type objectives (objective not confirmed in sources here) and statsforecast has intermittent models (CrostonSBA etc. listed in README).
- Foundation-model leaderboard claims are largely self-reported and GIFT-Eval/fev-bench leakage risk exists for pretraining overlap; treat gaps of a few percent as noise.

### Gaps
- Exact M4/M5 WRMSSE/OWA numbers, GIFT-Eval current top-10 values and by-horizon breakdown not retrieved (HF leaderboard not reachable).
- No sourced evidence by horizon length for LightGBM vs deep (only Favorita anecdote).
- M5 "intermittency" breakdown not fetched.

## 2. When seasonal naive/simple statistics are hard to beat; minimum data length for deep/foundation models

### Takeaway
M-competition history says simple methods and their combinations are tough to beat on short, noisy or heterogeneous series (M4, M6). Foundation models are zero-shot and degrade when context is too short; supervised models need enough samples. Specific minimum thresholds were not found in a reliable source.

### Cited Findings
- "Recurring results of the M competitions show simple methods do as well or better than more advanced ones"; M4 combinations of statistical methods beat pure ML (SNIPPET-ONLY) — [Uber blog](https://www.uber.com/at/en/blog/m4-forecasting-competition/).
- Seasonal naive is the normalizer in GIFT-Eval, so MASE < 1 is the bar there — [GIFT-Eval repo](https://github.com/SalesforceAIResearch/gift-eval) (SNIPPET-ONLY).
- Zero-shot foundation models perform poorly with very small training segments and improve with more history; supervised models overtake zero-shot once enough samples exist; one study found accuracy degraded at context <= 48 hours (hourly) and improved up to ~480 steps (5 workdays) — SNIPPET-ONLY, task-specific (a building-energy study) — [search results](https://arxiv.org/pdf/2506.00630), [OpenReview](https://openreview.net/pdf?id=kQDzqIkXLO), [Context parroting](https://arxiv.org/pdf/2505.11349).
- Chronos repo gives no minimum context length; examples use tail(256) — [repo](https://github.com/amazon-science/chronos-forecasting).
- Context parroting paper argues a simple copy-the-context baseline is tough to beat for foundation models on dynamical systems (SNIPPET-ONLY) — [arXiv](https://arxiv.org/pdf/2505.11349).

### Inferences
- Always include seasonal naive (and a seasonal-window mean) as first submission; if a fancy model cannot beat it on rolling-origin folds, stop. Roughly: needs at least 2-3 full seasonal cycles for ETS/Theta (general practice, UNVERIFIED here), and a few hundred points per series (or many series) for global deep models (UNVERIFIED).

### Gaps
- No authoritative minimum-length figure for N-BEATS/NHITS/PatchTST/TFT or for Chronos/TimesFM found.

## 3. Correct validation: rolling origin, gap/embargo, leakage traps, spatial block CV

### Takeaway
Use rolling-origin (expanding or sliding window) folds that mimic the test horizon, with a gap if features use lags newer than the forecast availability; for spatial prediction to new areas, use spatial blocks at least as wide as the autocorrelation range. Random K-fold is over-optimistic with spatial or temporal dependence.

### Cited Findings
- Spatial CV partitions data into spatially independent folds (Roberts 2017, Valavi 2018, Ploton 2020); blocks should be separated by at least the autocorrelation range; random CV is overoptimistic for transfer to new areas but acceptable for within-area interpolation (SNIPPET-ONLY) — [SDM validation paper](https://www.sciencedirect.com/science/article/pii/S1574954125005308), [Spatial CV for GeoAI](https://www.acsu.buffalo.edu/~yhu42/papers/2023_GeoAIHandbook_SpatialCV.pdf), [Assessing performance of spatial CV](https://arxiv.org/pdf/2303.07334), [evaluation challenges](https://arxiv.org/pdf/2303.18087).
- Adversarial validation can quantify train/test dissimilarity for geospatial tasks — [arXiv 2404.12575](https://arxiv.org/pdf/2404.12575) (SNIPPET-ONLY).
- mlforecast offers cross_validation, lag/rolling and target transforms (Differences) — [repo](https://github.com/Nixtla/mlforecast).
- M6 showed live, out-of-sample evaluation exposes overfit; frequent updating helped — [M6](https://arxiv.org/html/2310.13357) (SNIPPET-ONLY).

### Inferences (general practice, no specific source retrieved)
- Leakage traps: (a) future covariates that are not truly known at forecast time (realized weather, post-hoc promo flags); (b) global scaling/target encoding/imputation fit on all data including test period; (c) rolling features computed without shift(1); (d) random split on panel; (e) duplicate entities across folds in spatial data; (f) lags shorter than the horizon in direct multi-step setups; (g) foundation-model pretraining overlap with public datasets.
- Mimic the test: same horizon, same gap between last-known and first-predicted date, same covariate availability. Fit scalers within each fold.
- Compare to a seasonal-naive fold score every time.

### Gaps
- No source fetched specifically on embargo/purging for time series CV (López de Prado style) or on target leakage via future covariates.

## 4. Spatial feature approaches (KNN neighbours, H3/geohash)

### Takeaway
Only validation evidence was retrieved; no sourced head-to-head of KNN-neighbour features vs H3/geohash was found.

### Cited Findings
- See section 3 for spatial CV sources. Nothing else sourced.

### Inferences (UNVERIFIED, standard practice)
- KNN neighbour features (mean target of k nearest labelled points, distance-weighted, excluding self and computed inside CV folds) often a strong baseline; H3 at multiple resolutions plus parent-cell target encoding captures hierarchical location effects; lat/lon directly in GBDT works for dense data; ensure neighbour features are built using only training-fold points or they leak.
- Spatial CV via GroupKFold on H3 cell at a coarse resolution (parent cell) approximates spatial blocking.

### Gaps
- No H3/geohash feature-engineering benchmark found; no Kaggle geospatial write-up retrieved.

## 5. Practical recommendations for 24-48 hour events and library status

### Takeaway
Fast plan: baselines in statsforecast (minutes), global LightGBM via mlforecast, zero-shot Chronos-2/TabPFN-TS, then a simple blend validated on rolling folds. Libraries are all Apache/BSD and actively listed on GitHub.

### Cited Findings
- statsforecast (Nixtla): Apache-2.0, ~4.9k stars, 1,477 commits; AutoARIMA/AutoETS/AutoCES/AutoTheta, MSTL, TBATS, baselines, intermittent models; claims 20x faster than pmdarima, 4x statsmodels, 500x Prophet; Spark/Dask/Ray — [repo](https://github.com/Nixtla/statsforecast). (Speed claims are the vendor's.)
- mlforecast: Apache-2.0, 507 commits; lag/rolling/expanding transforms, target transforms, cross_validation, conformal intervals, exogenous and static features, pandas/polars/Spark/Dask/Ray — [repo](https://github.com/Nixtla/mlforecast).
- darts: ~9.5k stars, Apache-2.0, Python 3.11+, 40+ models incl. statistical, regression (LightGBM/CatBoost/XGBoost), N-BEATS/N-HiTS/TFT/DLinear/TiDE/TSMixer, and foundation models (Chronos-2, TimesFM 2.5 & 3.0, TiRex, PatchTST-FM, T0 as listed), backtesting, past/future covariates — [repo](https://github.com/unit8co/darts).
- sktime: v1.2.0 shown, 10k+ stars, BSD-3, Python 3.10-3.14; fetched page did not mention foundation models — [repo](https://github.com/sktime/sktime).
- Chronos repo: Chronos-2 (120M), Chronos-Bolt (9M-205M, up to 250x faster than original), original T5 Chronos (8M-710M); `Chronos2Pipeline.predict_df` with `future_df` covariates; Apache-2.0 — [repo](https://github.com/amazon-science/chronos-forecasting).
- TabPFN-TS: zero-shot via TabPFN with lightweight feature engineering; good on small datasets — [repo](https://github.com/PriorLabs/tabpfn-time-series).
- Other recent foundation entrants referenced in search: Moirai 2.0 ([arXiv](https://arxiv.org/pdf/2511.11698)), Toto 2.0 ([arXiv](https://arxiv.org/pdf/2605.20119)), TempoPFN ([arXiv](https://arxiv.org/pdf/2510.25502)).

### Inferences
- Suggested 24-48h schedule: hour 0-2 data audit and validation scheme (rolling-origin matching test horizon); hour 2-4 seasonal naive/ETS/Theta via statsforecast; hour 4-14 mlforecast LightGBM with lags, rolling means (shifted), calendar, covariates; hour 14-20 zero-shot Chronos-2 and TabPFN-TS (with covariates), possibly NHITS/PatchTST only if series are long and GPU available; remaining time blend (average of GBDT and foundation model usually robust) and sanity check submission format. Avoid heavy deep-model tuning.
- Ensembling different families is supported by M5 (equal-weight LightGBM ensembles) and M6 (combination helped).

### Gaps
- Exact latest release dates/versions for statsforecast, mlforecast, neuralforecast, and maintenance activity (last commit dates) not retrieved; neuralforecast (N-BEATS/NHITS/PatchTST/TFT) not checked; TimesFM/Moirai/Lag-Llama repo status not checked (Lag-Llama maintenance UNVERIFIED).
- HF GIFT-Eval leaderboard and TabPFN-TS vs Chronos-2 ranking on current boards not verified.
