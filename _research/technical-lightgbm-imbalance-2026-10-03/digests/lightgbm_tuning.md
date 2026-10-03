# LightGBM hyperparameter tuning (focus: imbalanced binary classification)

Retrieval date: 2026-10-03. Primary sources were downloaded raw (curl) and read in full or grepped:
- Parameters.rst: https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/Parameters.rst
- Parameters-Tuning.rst: https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/Parameters-Tuning.rst
- LightGBM Python callback.py / sklearn.py (master): https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/callback.py , https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/sklearn.py
- PyPI JSON: https://pypi.org/pypi/lightgbm/json , https://pypi.org/pypi/optuna/json , https://pypi.org/pypi/optuna-integration/json , https://pypi.org/pypi/flaml/json
- Optuna / optuna-integration source (raw GitHub), FLAML model.py (raw GitHub)
- arxiv.org, jmlr.org, neurips proceedings and huggingface.co were BLOCKED by the egress proxy, so all paper claims below come from search-result snippets and are marked SNIPPET-ONLY.

## 1. Current versions and where things live

### Takeaway
As of 2026-10-03: LightGBM 4.7.0, Optuna 5.0.0, optuna-integration 5.0.0, FLAML 2.7.0. LightGBMTuner / LightGBMTunerCV now live in `optuna-integration` (`optuna_integration.lightgbm`); the old `optuna.integration.lightgbm` path is a deprecated shim that re-exports from optuna-integration and emits a FutureWarning. The LightGBM GitHub project now points to `lightgbm-org/LightGBM`.

### Cited Findings
- `lightgbm` latest = **4.7.0**, uploaded 2026-07-18 — [PyPI JSON](https://pypi.org/pypi/lightgbm/json)
- PyPI project_urls for lightgbm list homepage `https://github.com/lightgbm-org/LightGBM` (changelog `https://github.com/lightgbm-org/LightGBM/releases`, docs `https://lightgbm.readthedocs.io/en/latest/`); Parameters.rst issue links also point at `github.com/lightgbm-org/LightGBM` — [PyPI JSON](https://pypi.org/pypi/lightgbm/json), [Parameters.rst](https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/Parameters.rst). (The microsoft/LightGBM raw URL still served the file at time of retrieval.)
- `optuna` latest = **5.0.0**, uploaded 2026-09-07 — [PyPI JSON](https://pypi.org/pypi/optuna/json)
- `optuna-integration` latest = **5.0.0**, uploaded 2026-09-07 — [PyPI JSON](https://pypi.org/pypi/optuna-integration/json)
- `FLAML` latest = **2.7.0**, uploaded 2026-09-18 — [PyPI JSON](https://pypi.org/pypi/flaml/json)
- `optuna/integration/lightgbm.py` in Optuna master does `import optuna_integration.lightgbm as lgb` (raising ModuleNotFoundError with an install hint if missing) and emits a deprecation FutureWarning built from `_DEPRECATION_WARNING_TEMPLATE.format(name="`optuna.integration.lightgbm`", d_ver="4.9.0", r_ver="6.0.0")` plus "Use `optuna_integration.lightgbm` instead." It re-exports `LightGBMPruningCallback`, `LightGBMTuner`, `LightGBMTunerCV`, `train` — [optuna/integration/lightgbm.py](https://raw.githubusercontent.com/optuna/optuna/master/optuna/integration/lightgbm.py)
  - i.e. deprecated since Optuna 4.9.0, scheduled removal in Optuna 6.0.0 (per the template arguments in the source).
- `optuna_integration/lightgbm/__init__.py` exports `LightGBMTuner`, `LightGBMTunerCV`, `LightGBMPruningCallback`, `Dataset`, and wraps `train`, `LGBMModel`, `LGBMClassifier`, `LGBMRegressor` from the tuner (so `import optuna_integration.lightgbm as lgb; lgb.train(...)` is a drop-in for `lightgbm.train`) — [optuna_integration/lightgbm/__init__.py](https://raw.githubusercontent.com/optuna/optuna-integration/main/optuna_integration/lightgbm/__init__.py)
- Implementation file: `optuna_integration/lightgbm/_lightgbm_tuner/optimize.py` — [optimize.py](https://raw.githubusercontent.com/optuna/optuna-integration/main/optuna_integration/lightgbm/_lightgbm_tuner/optimize.py)
- LightGBMTuner docstring warning: "Arguments `feature_name` and `categorical_feature` were deprecated in v4.2.2 and will be removed in the future ... currently scheduled for v6.0.0" — [optimize.py](https://raw.githubusercontent.com/optuna/optuna-integration/main/optuna_integration/lightgbm/_lightgbm_tuner/optimize.py)
- LightGBM's own tuning page links FLAML and Optuna (the Optuna link is the LightGBM Tuner Medium post) "for automated hyperparameter tuning" — [Parameters-Tuning.rst](https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/Parameters-Tuning.rst)

### Inferences
- New code should `pip install optuna optuna-integration[lightgbm]` (extra name UNVERIFIED) or simply `optuna-integration lightgbm`, and `import optuna_integration.lightgbm as lgb`; `optuna.integration.lightgbm` will still work under Optuna 5.x but warns and is slated for removal in 6.0.

### Gaps
- Could not read the optuna-integration release notes / GitHub tree via API (gh access to optuna/optuna-integration not enabled for session), so the exact release in which LightGBMTuner first moved is not confirmed from release notes; only the current state of source is verified.
- LightGBM 4.7.0 changelog not reviewed.

## 2. What each parameter does (defaults quoted from Parameters.rst) and interactions

### Takeaway
`num_leaves` (default 31) is the main complexity knob for leaf-wise trees; keep it below 2^max_depth if you set max_depth. `min_data_in_leaf` (20) is the key anti-overfit knob; `min_sum_hessian_in_leaf` (1e-3) is a Hessian-weighted analogue that matters most for logloss with rare positives. Regularisers (bagging, feature_fraction, lambda_l1/l2, min_gain_to_split, path_smooth, extra_trees, max_bin) are all off or at "no-regularisation" values by default.

### Cited Findings (all from [Parameters.rst](https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/Parameters.rst) unless noted)
Core / boosting
- `num_iterations` default **100**; aliases include `n_estimators`, `num_boost_round`, `nrounds`, `max_iter`.
- `learning_rate` default **0.1**; aliases `shrinkage_rate`, `eta`; "in `dart`, it also affects on normalization weights of dropped trees".
- `num_leaves` default **31**; constraint `1 < num_leaves <= 131072`; aliases `max_leaves`, `max_leaf_nodes`.
- `max_depth` default **-1** ("<= 0 means no limit"); "This is used to deal with over-fitting when #data is small. Tree still grows leaf-wise".
- `min_data_in_leaf` default **20**; aliases `min_child_samples`, `min_samples_leaf`, `min_data`; "Note: this is an approximation based on the Hessian, so occasionally you may observe splits which produce leaf nodes that have less than this many observations".
- `min_sum_hessian_in_leaf` default **1e-3**; aliases `min_child_weight`, `min_hessian`; "Like `min_data_in_leaf`, it can be used to deal with over-fitting".
- `early_stopping_round` default **0** (`<= 0` disables); aliases `early_stopping_rounds`, `n_iter_no_change`; "will stop training if one metric of one validation data doesn't improve in last `early_stopping_round` rounds".
- `early_stopping_min_delta` default **0.0**, "New in version 4.4.0".
- `first_metric_only` default **false**: "Set this to true, if you want to use only the first metric for early stopping".
Sampling
- `bagging_fraction` default **1.0** (aliases `subsample`, `sub_row`); sampling "without resampling"; "to enable bagging, `bagging_freq` should be set to a non zero value as well".
- `bagging_freq` default **0** (alias `subsample_freq`); "`k` means perform bagging at every `k` iteration"; "bagging is only effective when 0.0 < bagging_fraction < 1.0".
- `pos_bagging_fraction` / `neg_bagging_fraction` default **1.0**: "used for imbalanced binary classification problem, will randomly sample `#pos_samples * pos_bagging_fraction` positive samples in bagging"; need `bagging_freq` too; "if balanced bagging is enabled, `bagging_fraction` will be ignored".
- `data_sample_strategy` default `bagging`, option `goss`; new in 4.0.0.
- `feature_fraction` default **1.0** (alias `colsample_bytree`): random subset of features per tree.
- `feature_fraction_bynode` default **1.0** (alias `colsample_bynode`): per-node; "unlike feature_fraction, this cannot speed up training"; combined fraction = product of the two.
- `extra_trees` default **false**: "when evaluating node splits LightGBM will check only one randomly-chosen threshold for each feature"; speeds training and reduces over-fitting.
Regularisation
- `lambda_l1` (alias `reg_alpha`) default **0.0**; `lambda_l2` (alias `reg_lambda`, `lambda`) default **0.0**.
- `min_gain_to_split` (alias `min_split_gain`) default **0.0**.
- `path_smooth` default **0**: "helps prevent overfitting on leaves with few samples"; "if path_smooth > 0 then min_data_in_leaf must be at least 2"; node weight = `w * (n / path_smooth) / (n / path_smooth + 1) + w_p / (n / path_smooth + 1)` (n = samples in node, w = optimal node weight ≈ -sum_grad/sum_hess, w_p = parent weight); smoothing "accumulates with the tree depth".
Binning / categorical
- `max_bin` default **255** (constraint > 1): "small number of bins may reduce training accuracy but may increase general power (deal with over-fitting)"; uses `uint8_t` storage at 255.
- `min_data_in_bin` default **3**: "avoid one-data-one-bin (potential over-fitting)".
- `feature_pre_filter` default **true**: ignores features unsplittable given `min_data_in_leaf`; "you may need to set this to false when searching parameters with min_data_in_leaf, otherwise features are filtered by min_data_in_leaf firstly if you don't reconstruct dataset object".
- `cat_smooth` default **10.0**: "reduce the effect of noises in categorical features, especially for categories with few data".
- `max_cat_threshold` default **32**: "limit number of split points considered for categorical features".
- `cat_l2` default **10.0** (L2 in categorical split); `min_data_per_group` default **100**; `max_cat_to_onehot` default **4** (one-vs-other split when #categories <= 4).
Imbalance / objective
- `is_unbalance` default **false**, `scale_pos_weight` default **1.0**; both: "while enabling this should increase the overall performance metric of your model, it will also result in poor estimates of the individual class probabilities"; mutually exclusive.
- sklearn wrapper `class_weight`: "Use this parameter only for multi-class classification task; for binary classification task you may use `is_unbalance` or `scale_pos_weight`... usage of all these parameters will result in poor estimates of the individual class probabilities. You may want to consider performing probability calibration" — [sklearn.py](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/sklearn.py)
- `boost_from_average` default **true** (binary: initial score set to label mean).
- Binary-relevant metrics available: `auc`, `average_precision`, `binary_logloss` (alias `binary`), `binary_error`.
- sklearn `eval_metric` default for LGBMClassifier: `'logloss'` — [sklearn.py](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/sklearn.py)
- sklearn wrapper `min_child_weight` default 1e-3, `min_split_gain` default 0 — [sklearn.py](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/sklearn.py)

Interactions per official tuning page — [Parameters-Tuning.rst](https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/Parameters-Tuning.rst)
- num_leaves vs max_depth: "Theoretically, we can set num_leaves = 2^(max_depth)... However, this simple conversion is not good in practice. A leaf-wise tree is typically much deeper than a depth-wise tree for a fixed number of leaves... when trying to tune the num_leaves, we should let it be smaller than 2^(max_depth). For example, when the max_depth=7 the depth-wise tree can get good accuracy, but setting num_leaves to 127 may cause over-fitting, and setting it to 70 or 80 may get better accuracy than depth-wise." "If you set max_depth, also explicitly set num_leaves to some value <= 2^max_depth."
- min_data_in_leaf: "a very important parameter to prevent over-fitting in a leaf-wise tree. Its optimal value depends on the number of training samples and num_leaves... In practice, setting it to hundreds or thousands is enough for a large dataset."
- min_sum_hessian_in_leaf: "For some regression objectives, this is just the minimum number of records that have to fall into each node. For classification objectives, it represents a sum over a distribution of probabilities" (links a stats.stackexchange answer on XGBoost min_child_weight).
- learning_rate x num_iterations: "if you reduce num_iterations, you should increase learning_rate"; values "often chosen ... through hyperparameter tuning".
- Bagging example: `{"bagging_freq": 5, "bagging_fraction": 0.75}` = "re-sample without replacement every 5 iterations, and draw samples of 75% of the training data".
- Early stopping is described in "For Faster Speed"; `early_stopping_round=1` means stop "the first time accuracy on the validation set does not improve".
- Feature pre-filtering example: a feature with 995/5 split and `min_data_in_leaf=10` is filtered at Dataset construction.

Early stopping callback (Python) — [callback.py](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/callback.py)
- `lightgbm.early_stopping(stopping_rounds, first_metric_only=False, verbose=True, min_delta=0.0)`; "Requires at least one validation data and one metric. If there's more than one, will check all of them... To check only the first metric set first_metric_only to True." Best iteration stored in `best_iteration`. "If using boosting_type="dart", this callback has no effect and early stopping will not be performed." `min_delta` may be a list per metric (added 4.0.0).

### Inferences
- Why min_sum_hessian_in_leaf matters with a small positive class (derived from standard logistic-loss math, not quoted from LightGBM docs): for binary logloss each row's Hessian is p(1-p). When the base rate is low, most rows have p near 0, so their Hessians are tiny; a leaf full of confidently predicted negatives (or a handful of positives) can contain many rows yet very little Hessian mass. The default 1e-3 essentially never binds, so leaves can form on a few rare positives with large leaf values (≈ -G/H with small H) → overfitting and extreme probabilities. Raising `min_sum_hessian_in_leaf` (e.g., log-uniform 1e-3..10), raising `min_data_in_leaf`, adding `lambda_l2` (which adds to H in the denominator), or `path_smooth` all damp this. Note that `scale_pos_weight`/`is_unbalance` multiply positives' gradients and Hessians, which changes how much Hessian a leaf of positives carries, so `min_sum_hessian_in_leaf` and weight parameters interact (tune them jointly or retune after changing weights).
- Because LightGBM's own note says `min_data_in_leaf` is "an approximation based on the Hessian", the two constraints are coupled even when only one is tuned.
- When tuning `min_data_in_leaf` with a reused `lgb.Dataset`, set `feature_pre_filter=False` (LightGBMTuner does this automatically; see section 3), otherwise trials with smaller values silently lose features filtered by the first value.
- Early stopping with multiple metrics checks *all* metrics unless `first_metric_only=True`; with `metric=["auc","binary_logloss"]` training stops when either stalls, so put the selection metric first and set `first_metric_only=True`.

### Gaps
- No official LightGBM guidance found on numeric ranges for `cat_smooth`, `max_cat_threshold`, `path_smooth`; ranges in section 3 come from tuner sources, not LightGBM docs.

## 3. Official tuning guidance (verbatim lists) and tuning strategies / search ranges

### Takeaway
LightGBM's docs give short qualitative lists (accuracy / speed / over-fitting) rather than ranges. Concrete ranges come from Optuna's LightGBMTuner (stepwise: feature_fraction → num_leaves → bagging → feature_fraction stage 2 → lambda_l1/l2 → min_child_samples) and FLAML's LGBMEstimator search space. Recent evidence (Koster & Sigrist 2026, SNIPPET-ONLY) says >100 trials are typically needed and early-stopping-chosen iterations beat searching n_estimators.

### Cited Findings
Official lists — [Parameters-Tuning.rst](https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/Parameters-Tuning.rst)
- **For Better Accuracy**: "Use large max_bin (may be slower)"; "Use small learning_rate with large num_iterations"; "Use large num_leaves (may cause over-fitting)"; "Use bigger training data"; "Try dart".
- **Deal with Over-fitting**: "Use small max_bin"; "Use small num_leaves"; "Use min_data_in_leaf and min_sum_hessian_in_leaf"; "Use bagging by set bagging_fraction and bagging_freq"; "Use feature sub-sampling by set feature_fraction"; "Use bigger training data"; "Try lambda_l1, lambda_l2 and min_gain_to_split for regularization"; "Try max_depth to avoid growing deep tree"; "Try extra_trees"; "Try increasing path_smooth".
- **For Faster Speed** (sections): add resources (`num_threads` = number of real CPU cores; GPU build; distributed); grow shallower trees (decrease max_depth, decrease num_leaves, increase min_gain_to_split, increase min_data_in_leaf and min_sum_hessian_in_leaf); grow fewer trees (decrease num_iterations, use early stopping); consider fewer splits (feature_pre_filter=True, decrease max_bin / max_bin_by_feature, increase min_data_in_bin, decrease feature_fraction, decrease max_cat_threshold); use less data (bagging); save_binary (CLI only). "The suggestions below will speed up training, but might hurt training accuracy."
- On min_gain_to_split: "in practice you might find that very small improvements in the training loss don't have a meaningful impact on the generalization error of the model."

Optuna LightGBMTuner / LightGBMTunerCV (optuna-integration main) — [optimize.py](https://raw.githubusercontent.com/optuna/optuna-integration/main/optuna_integration/lightgbm/_lightgbm_tuner/optimize.py)
- Docstring: "optimizes the following hyperparameters in a stepwise manner: lambda_l1, lambda_l2, num_leaves, feature_fraction, bagging_fraction, bagging_freq and min_child_samples"; algorithm/benchmarks in Kohei Ozaki's blog post ([Medium](https://medium.com/optuna/lightgbm-tuner-new-optuna-integration-for-hyperparameter-optimization-8b7095e99258), not fetched).
- `run()` order and trial budgets: `tune_feature_fraction` (GridSampler, 7 values = linspace(0.4,1.0,7)) → `tune_num_leaves` (TPE, 20 trials) → `tune_bagging` (TPE, 10 trials) → `tune_feature_fraction_stage2` (Grid, 6 values in best±0.08, clipped to [0.4,1.0]) → `tune_regularization_factors` (TPE, 20 trials) → `tune_min_data_in_leaf` (Grid over min_child_samples [5, 10, 25, 50, 100]). Total ≈ 68 trials.
- Ranges: `lambda_l1`, `lambda_l2` ∈ log-uniform [1e-8, 10]; `num_leaves` ∈ int [2, 2^max_depth] with default depth 8 → [2, 256]; `feature_fraction` ∈ [0.4, 1.0]; `bagging_fraction` ∈ [0.4, 1.0]; `bagging_freq` ∈ int [1, 7]; `min_child_samples` ∈ [5, 100].
- Forces `feature_pre_filter=False` with warning "This is required for the tuner to tune min_child_samples."
- Direction is maximize for metrics in `("auc", "auc_mu", "ndcg", "map", "average_precision")`, else minimize.
- `LightGBMTuner.__init__(params, train_set, num_boost_round=1000, valid_sets=None, valid_names=None, feval=None, feature_name=None, categorical_feature=None, keep_training_booster=False, callbacks=None, time_budget=None, sample_size=None, study=None, optuna_callbacks=None, model_dir=None, *, show_progress_bar=True, optuna_seed=None)`; `LightGBMTunerCV` is the cross-validation variant (wraps `lgb.cv`; supports `return_cvbooster`).
- `optuna_seed`: seeds TPESampler for num_leaves, bagging_fraction, bagging_freq, lambda_l1, lambda_l2; note: "The deterministic parameter of LightGBM makes training reproducible. Please enable it when you use this argument."
- `model_dir`: needed for `get_best_booster` in distributed settings.

FLAML LGBMEstimator default search space — [flaml/automl/model.py](https://raw.githubusercontent.com/microsoft/FLAML/main/flaml/automl/model.py)
- `n_estimators`: lograndint [4, min(32768, n_rows)], init 4; `num_leaves`: lograndint [4, same upper], init 4; `min_child_samples`: lograndint [2, 129], init 20; `learning_rate`: loguniform [1/1024, 1.0], init 0.1; `log_max_bin`: lograndint [3, 11] → `max_bin = 2^k - 1` (7..2047), init 8 (=255); `colsample_bytree`: uniform [0.01, 1.0], init 1.0; `reg_alpha`: loguniform [1/1024, 1024], init 1/1024; `reg_lambda`: loguniform [1/1024, 1024], init 1.0.
- FLAML's search starts from low-cost configs (4 trees, 4 leaves) and grows (its cost-frugal search design); n_estimators is part of the search space (ITER_HP), not purely early-stopped.

Benchmark / methodology evidence on tuning strategy
- Koster & Sigrist, "Selecting Hyperparameters for Tree-Boosting" (arXiv:2602.05786, Feb 2026; also listed on ScienceDirect): compared random grid search, TPE, GP-BO, Hyperband, SMAC and full grid on 59 regression + binary classification datasets; findings: "(i) a relatively large number of trials larger than 100 is typically required for accurate tuning, (ii) using default values for hyperparameters or a deterministic search over a small grid often yields very inaccurate models, (iii) all considered hyperparameters can have a material effect..., and (iv) choosing the number of boosting iterations using early stopping yields more accurate results compared to including it in the search space." — SNIPPET-ONLY [arXiv abs](https://arxiv.org/abs/2602.05786v1), [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2666827026001349). Which method won overall was not in the snippet (UNVERIFIED).
- TabArena tuning protocol: "200 random configurations using 8-fold inner cross-validation, followed by post-hoc ensembling using Greedy Ensemble Selection"; three regimes reported per model: default, tuned (random search), tuned+ensembled — SNIPPET-ONLY [TabArena NeurIPS 2025 D&B PDF](https://papers.neurips.cc/paper_files/paper/2025/file/1697e3fb412da11dc9488249f9e7bbc9-Paper-Datasets_and_Benchmarks_Track.pdf), [arXiv 2506.16791](https://arxiv.org/abs/2506.16791)

### Inferences — a sensible Optuna TPE search space for imbalanced binary LightGBM (synthesised from the above sources; not an official recommendation)
- Fix `learning_rate` (e.g. 0.02–0.05) and `num_boost_round` large (e.g. 5000) with early stopping (stopping_rounds ~ 100–200), rather than searching n_estimators — consistent with Koster & Sigrist (iv) and LightGBM's "small learning_rate with large num_iterations". Optionally tune learning_rate log-uniform [0.005, 0.2].
- `num_leaves` int log [8, 256] (Optuna tuner uses [2, 256]); optionally `max_depth` ∈ {-1, 4..12} with num_leaves constrained ≤ 2^max_depth (docs).
- `min_data_in_leaf` int log [5, 500] (tuner [5,100]; FLAML [2,129]; docs say "hundreds or thousands" for large data) — but cap relative to the positive count so leaves can still isolate positives.
- `min_sum_hessian_in_leaf` log [1e-3, 10].
- `feature_fraction` [0.4, 1.0] (tuner); `bagging_fraction` [0.4, 1.0] with `bagging_freq` [1, 7] (tuner) — or balanced bagging via `pos_bagging_fraction=1.0`, `neg_bagging_fraction` ∈ [0.05, 1.0] for heavy imbalance.
- `lambda_l1`, `lambda_l2` log [1e-8, 10] (tuner) or [1e-3, 1e3] (FLAML-like).
- `min_gain_to_split` [0, 1–5]; `path_smooth` [0, 10–100] (no official ranges; UNVERIFIED as best practice); `extra_trees` {False, True}; `max_bin` {63, 127, 255, 511} (FLAML spans 7..2047; must be set at Dataset construction).
- `scale_pos_weight` either fixed at 1 (preferred when probabilities matter, then threshold-tune) or searched log [1, n_neg/n_pos]; docs warn it degrades probability estimates.
- Categorical: `cat_smooth` [1, 100], `max_cat_threshold` [8, 64], `cat_l2` [1, 100] around defaults 10/32/10 (UNVERIFIED ranges).
- Use `optuna_integration.LightGBMPruningCallback` (exported) or MedianPruner to cut bad trials; seed `TPESampler(seed=...)`.
- Random search remains a strong baseline (it is what TabArena uses with 200 configs); budget >100 trials per Koster & Sigrist; Optuna LightGBMTuner (~68 trials, stepwise) is a cheap, decent starting point but does not tune learning_rate, min_sum_hessian, max_bin, path_smooth, or class weights.

### Gaps
- Could not fetch Ozaki's Medium benchmark post or Koster & Sigrist full text (blocked), so per-method winners and quantitative gains are not available.
- Did not verify FLAML's BlendSearch/CFO defaults beyond the search space file.

## 4. Metric choice for imbalanced data and early-stopping metric

### Takeaway
LightGBM supports `auc`, `average_precision` (PR-AUC) and `binary_logloss` as built-in early-stopping metrics; default for LGBMClassifier is logloss. For rare positives, optimise/select on average_precision (ranking) or logloss (if calibrated probabilities matter), not error/accuracy; avoid `is_unbalance`/`scale_pos_weight` if you need probabilities, or recalibrate afterwards.

### Cited Findings
- Built-in binary metrics: `auc`, `average_precision` ("average precision score", linked to sklearn), `binary_logloss`, `binary_error` — [Parameters.rst](https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/Parameters.rst)
- `metric=""` (default) uses the objective's metric; for binary that is logloss; LGBMClassifier eval default `'logloss'` — [Parameters.rst](https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/Parameters.rst), [sklearn.py](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/sklearn.py)
- Early stopping checks all metrics unless `first_metric_only`; `early_stopping_min_delta` (4.4.0) / callback `min_delta` (4.0.0) — [Parameters.rst](https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/Parameters.rst), [callback.py](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/callback.py)
- `is_unbalance` / `scale_pos_weight` / `class_weight` "will result in poor estimates of the individual class probabilities"; LightGBM suggests probability calibration — [sklearn.py](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/sklearn.py)
- Optuna LightGBMTuner treats `auc`, `average_precision` as maximize; everything else minimize — [optimize.py](https://raw.githubusercontent.com/optuna/optuna-integration/main/optuna_integration/lightgbm/_lightgbm_tuner/optimize.py)
- TabZilla: "GBDTs perform better when datasets are more class imbalanced" (relative to NNs) — SNIPPET-ONLY [TabZilla NeurIPS 2023](https://proceedings.nips.cc/paper_files/paper/2023/file/f06d5ebd4ff40b40dd97e30cee632123-Paper-Datasets_and_Benchmarks.pdf)

### Inferences
- Recommended pattern: early-stop on the same metric used for model selection (put it first, `first_metric_only=True`). With very few positives, AUC/AP on a validation fold is noisy; use `early_stopping_min_delta` / larger `stopping_rounds`, and consider logloss for early stopping (smoother) while selecting hyperparameters on AP averaged over CV folds.
- Threshold-dependent metrics (F1, recall@precision) should be optimised by threshold tuning after training, not used as early-stopping metrics (they are step functions of scores).
- Use stratified folds so every fold has positives (LightGBMTunerCV accepts `folds`/`stratified` via lgb.cv kwargs — `stratified` default in lgb.cv is True per LightGBM API, UNVERIFIED here).

### Gaps
- No primary-source benchmark found specifically comparing early-stopping metrics (AUC vs AP vs logloss) for imbalanced LightGBM.

## 5. How much does tuning help GBDTs vs defaults? Avoiding validation overfitting; reproducibility

### Takeaway
Evidence consistently says tuning matters for GBDTs (TabZilla: light tuning of a GBDT beats the NN-vs-GBDT choice on ~1/3 of datasets; Koster & Sigrist: defaults "often yield very inaccurate models"; TabArena: tuned+ensembled > tuned > default), though magnitudes are only available from snippets. Protect against validation overfitting with CV-based selection, an untouched test/holdout, and restraint in trial counts; for reproducibility set `seed`, `deterministic=True` + `force_row_wise`/`force_col_wise`, fixed `num_threads`, and seed the Optuna sampler.

### Cited Findings
Tuning value
- TabZilla (McElfresh et al., NeurIPS 2023 D&B; 19 algorithms, 176 datasets): "light hyperparameter tuning on a GBDT is more important than choosing between NNs and GBDTs" for "a surprisingly high number of datasets"; for about one-third of datasets light tuning yields a greater improvement than GBDT-vs-NN selection; recommendation to "conduct light hyperparameter tuning on CatBoost" — SNIPPET-ONLY [arXiv 2305.02997](https://arxiv.org/abs/2305.02997), [NeurIPS PDF](https://proceedings.nips.cc/paper_files/paper/2023/file/f06d5ebd4ff40b40dd97e30cee632123-Paper-Datasets_and_Benchmarks.pdf)
- Probst, Boulesteix, Bischl (JMLR 20, 2019) "Tunability": defines tunability as gain from best value vs default; provides data-based defaults; "For xgboost there are two parameters that are quite tunable: eta and booster" — SNIPPET-ONLY [JMLR PDF](https://www.jmlr.org/papers/volume20/18-444/18-444.pdf). (Study was on xgboost, not LightGBM.)
- TabArena (NeurIPS 2025 D&B; arXiv 2506.16791): "best performance for individual tabular machine learning models is generally achieved by post-hoc ensembling tuned hyperparameter configurations"; GBDTs "still strong contenders", but best DL methods with tuning+ensembling match or beat GBDTs — SNIPPET-ONLY [arXiv 2506.16791](https://arxiv.org/abs/2506.16791). TabArena repo now also has "BeyondArena" (142 datasets incl. temporal/grouped) intended to supersede TabArena-v0.1; TabArena-v0.1 has 51 curated datasets — [TabArena README](https://raw.githubusercontent.com/autogluon/tabarena/main/README.md)
- A third-party leaderboard summary claims tuning is worth "on the order of 200 Elo" for LightGBM and XGBoost on TabArena; the same search result gave inconsistent LightGBM T+E Elo values (1432 vs 1598) — SNIPPET-ONLY, conflicting, do not quote numbers [codesota summary](https://www.codesota.com/tasks/tabular-ml) / search snippet.
- Koster & Sigrist 2026: defaults or small-grid search "often yields very inaccurate models"; >100 trials typically needed — SNIPPET-ONLY [arXiv 2602.05786](https://arxiv.org/abs/2602.05786v1)

Validation overfitting
- TabArena protocol uses 8-fold inner CV for tuning and evaluates on separate outer splits (9–30 splits per dataset per README) — SNIPPET-ONLY for the 8-fold detail ([arXiv 2506.16791](https://arxiv.org/abs/2506.16791)); splits from [README](https://raw.githubusercontent.com/autogluon/tabarena/main/README.md)

Reproducibility — [Parameters.rst](https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/Parameters.rst)
- `seed` (aliases `random_seed`, `random_state`) default None: "used to generate other seeds, e.g. data_random_seed, feature_fraction_seed... has lower priority... overridden, if you set other seeds explicitly". Sub-seed defaults: `bagging_seed`=3, `feature_fraction_seed`=2, `extra_seed`=6, `data_random_seed`=1, `objective_seed`=5.
- `deterministic` default false, CPU only: "should ensure the stable results when using the same data and the same parameters (and different num_threads)"; different seeds, LightGBM versions, compilers or systems → different results expected; "may slow down the training"; "please set force_col_wise=true or force_row_wise=true when setting deterministic=true".
- `num_threads` default 0 (OpenMP default); set to number of real CPU cores; "do not set it too large if your dataset is small (for instance, do not use 64 threads for a dataset with 10,000 rows)"; "please don't change this during training, especially when running multiple jobs simultaneously by external packages".
- Optuna tuner: seed TPE with `optuna_seed` and enable LightGBM `deterministic` — [optimize.py](https://raw.githubusercontent.com/optuna/optuna-integration/main/optuna_integration/lightgbm/_lightgbm_tuner/optimize.py)

### Inferences
- Avoiding validation overfitting in practice: (1) select hyperparameters by stratified K-fold CV mean (and look at std), not a single small validation split, especially with few positives; (2) keep a final untouched holdout for one evaluation; (3) use nested CV only if you need an unbiased estimate of the whole tuning procedure with small data; (4) the more trials, the more the best CV score is optimistically biased — after ~100–200 trials gains are often within fold noise, so prefer the simplest config within ~1 SE of the best, or average several top configs (TabArena-style ensembling); (5) early stopping on the same fold used to score the trial leaks — either early-stop on an inner split, or accept small bias and confirm on the outer holdout; (6) after tuning, refit on all training data with num_boost_round = mean best_iteration from CV (scaled up slightly for more data is a common heuristic, UNVERIFIED).
- When running Optuna trials in parallel (n_jobs>1), set LightGBM `num_threads` per trial so total threads ≤ physical cores, consistent with the docs' warning.

### Gaps
- Exact quantitative default-vs-tuned gains for LightGBM (TabArena tables, Probst tunability numbers for GBDT parameters) could not be read because arxiv/jmlr/neurips were blocked; only snippet-level qualitative claims are reported.
- No primary source found specifically quantifying validation-overfitting as a function of Optuna trial count for LightGBM.
