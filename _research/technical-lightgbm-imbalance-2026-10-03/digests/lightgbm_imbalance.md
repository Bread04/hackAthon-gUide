# LightGBM class-imbalance options: exact semantics, effects on ranking and probabilities, evidence, and current API (as of Oct 2026)

Source abbreviations (all fetched from GitHub raw on 2026-10-03, `master` branch unless stated):
- [Parameters.rst](https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/Parameters.rst)
- [sklearn.py](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/sklearn.py)
- [engine.py](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/engine.py)
- [callback.py](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/callback.py)
- [binary_objective.hpp](https://raw.githubusercontent.com/microsoft/LightGBM/master/src/objective/binary_objective.hpp)
- [multiclass_objective.hpp](https://raw.githubusercontent.com/microsoft/LightGBM/master/src/objective/multiclass_objective.hpp)
- [xentropy_objective.hpp](https://raw.githubusercontent.com/microsoft/LightGBM/master/src/objective/xentropy_objective.hpp)
- [bagging.hpp](https://raw.githubusercontent.com/microsoft/LightGBM/master/src/boosting/bagging.hpp)
- [binary_metric.hpp](https://raw.githubusercontent.com/microsoft/LightGBM/master/src/metric/binary_metric.hpp)
- [Python-Intro.rst](https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/Python-Intro.rst)
- [PyPI JSON](https://pypi.org/pypi/lightgbm/json)
- "Own check" = I installed lightgbm 4.7.0 + scikit-learn 1.9.1 from PyPI in a scratch venv and ran small scripts (synthetic `make_classification`, 60k rows, ~2.5% positives, single seed). These are illustrations of mechanics, NOT benchmark evidence.

## Q1. Exact semantics of LightGBM's imbalance parameters (is_unbalance, scale_pos_weight, class_weight, sample_weight, pos/neg_bagging_fraction) and how they enter gradients/hessians

### Takeaway
`is_unbalance` and `scale_pos_weight` are objective-level per-class multipliers on the gradient AND hessian of the binary log-loss (also per-class inside `multiclassova`); they are mutually exclusive (fatal error), they do not change the initial score (boost_from_average uses unweighted prevalence) and they do not affect metrics. sklearn `class_weight` is converted to `sample_weight` (multiplied with any user `sample_weight`), which does change the initial score, the metrics on weighted data, and the hessian sums that `min_child_weight` constrains. `pos/neg_bagging_fraction` is per-iteration class-stratified row subsampling with no reweighting. The docs say all of these hurt individual class-probability estimates.

### Cited Findings
**is_unbalance**
- Default `false`; aliases `unbalance`, `unbalanced_sets`; "used only in `binary` and `multiclassova` applications"; "set this to `true` if training data are unbalanced"; "**Note**: while enabling this should increase the overall performance metric of your model, it will also result in poor estimates of the individual class probabilities"; "**Note**: this parameter cannot be used at the same time with `scale_pos_weight`, choose only **one** of them" — [Parameters.rst](https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/Parameters.rst)
- Implementation: in `BinaryLogloss::Init`, counts positives (label > 0) and negatives; if `is_unbalance`, the minority class gets label weight = majority_count / minority_count and the majority class gets 1.0 (if positives are the majority, the negatives get pos/neg instead). Then `label_weights_[1] *= scale_pos_weight_` — [binary_objective.hpp](https://raw.githubusercontent.com/microsoft/LightGBM/master/src/objective/binary_objective.hpp)
- Mutual exclusion is enforced in the constructor: `if (is_unbalance_ && std::fabs(scale_pos_weight_ - 1.0f) > 1e-6) Log::Fatal("Cannot set is_unbalance and scale_pos_weight at the same time")` — [binary_objective.hpp](https://raw.githubusercontent.com/microsoft/LightGBM/master/src/objective/binary_objective.hpp). Own check (4.7.0): `lgb.train` with both set raises `LightGBMError: Cannot set is_unbalance and scale_pos_weight at the same time`.

**scale_pos_weight**
- Default `1.0`, type double, constraint `> 0.0`; "used only in `binary` and `multiclassova` applications"; "weight of labels with positive class"; same two Notes as is_unbalance (poor probability estimates; cannot combine) — [Parameters.rst](https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/Parameters.rst)

**How the weights enter gradient/hessian (binary)**
- With labels mapped to ±1 and `sigmoid` σ (default 1.0): `response = -label*σ / (1 + exp(label*σ*score))`; `gradient = response * label_weight [* weights_[i]]`; `hessian = |response|*(σ - |response|) * label_weight [* weights_[i]]`. So class weight (`is_unbalance`/`scale_pos_weight`) and per-row `sample_weight` multiply both grad and hess multiplicatively — [binary_objective.hpp](https://raw.githubusercontent.com/microsoft/LightGBM/master/src/objective/binary_objective.hpp)
- `BoostFromScore` (the `boost_from_average` initial score) computes `pavg = sum(is_pos * weights_) / sum(weights_)` using only the per-row sample weights (unweighted prevalence if none), then `initscore = log(pavg/(1-pavg))/σ`. `label_weights_` (is_unbalance / scale_pos_weight) are NOT used here — [binary_objective.hpp](https://raw.githubusercontent.com/microsoft/LightGBM/master/src/objective/binary_objective.hpp). Own check: with `is_unbalance=True` or `scale_pos_weight=39` the log shows `pavg=0.025250 -> initscore=-3.653355` (same as unweighted). With sklearn `class_weight="balanced"` it shows `pavg=0.500000 -> initscore=0.000000`.
- `boost_from_average`: default `true`, "used only in `regression`, `binary`, `multiclassova` and `cross-entropy` applications", "adjusts initial score to the mean of labels for faster convergence" — [Parameters.rst](https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/Parameters.rst)
- Own check: is_unbalance and scale_pos_weight = n_neg/n_pos (=38.6, passed as 38.6) gave identical early-stopping iteration and identical test metrics (AUC 0.8919, AP 0.7581), which is consistent with the source code treating them identically when positives are the minority.

**multiclassova**
- `MulticlassOVA` builds `num_class` independent `BinaryLogloss` objectives, each with `is_pos = (label == i)` and the same `config` — [multiclass_objective.hpp](https://raw.githubusercontent.com/microsoft/LightGBM/master/src/objective/multiclass_objective.hpp). Therefore `is_unbalance` rebalances every one-vs-rest sub-problem using its own counts, and `scale_pos_weight` multiplies the "this class" side of every sub-problem.
- Softmax `multiclass` objective: gradients `(p-1)*w_i` or `p*w_i`, hessian `factor*p*(1-p)*w_i`. It reads only per-row `weights_`; there is no `is_unbalance`/`scale_pos_weight` handling in this objective — [multiclass_objective.hpp](https://raw.githubusercontent.com/microsoft/LightGBM/master/src/objective/multiclass_objective.hpp). The class-prior initialisation also uses weights (`class_init_probs_[label] += weights_[i]`) — same source.

**sklearn-API class_weight**
- Docstring (LGBMModel and LGBMClassifier): "`class_weight` : dict, 'balanced' or None ... Weights associated with classes in the form `{class_label: weight}`. Use this parameter only for multi-class classification task; for binary classification task you may use `is_unbalance` or `scale_pos_weight` parameters. Note, that the usage of all these parameters will result in poor estimates of the individual class probabilities. You may want to consider performing probability calibration (https://scikit-learn.org/stable/modules/calibration.html) of your model. The 'balanced' mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as `n_samples / (n_classes * np.bincount(y))`. If None, all classes are supposed to have weight one. Note, that these weights will be multiplied with `sample_weight` (passed through the `fit` method) if `sample_weight` is specified." — [sklearn.py](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/sklearn.py)
- Implementation: in `fit`, `class_sample_weight = _LGBMComputeSampleWeight(self._class_weight, y)`, where `_LGBMComputeSampleWeight = sklearn.utils.compute_sample_weight` — [sklearn.py](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/sklearn.py); [compat.py](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/compat.py). `class_weight` is popped from params before passing to the C++ core, i.e. it becomes Dataset weights, not an objective parameter — [sklearn.py](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/sklearn.py)
- `fit` also has `eval_class_weight` ("Class weights of eval data") and `eval_sample_weight` ("Weights of eval data. Weights should be non-negative.") — [sklearn.py](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/sklearn.py)
- Own check (binary, 4.7.0): `class_weight="balanced"` works on a binary problem despite the docstring advice; in-sample mean predicted probability after 300 rounds was 0.0887 vs 0.0825 for `is_unbalance=True` vs prevalence 0.0253 (unweighted model 0.0253).

**sample_weight / Dataset weights and metrics**
- Built-in binary point-wise metrics (e.g. binary_logloss) compute `sum(loss_i * w_i) / sum(w_i)` when the Dataset has weights — [binary_metric.hpp](https://raw.githubusercontent.com/microsoft/LightGBM/master/src/metric/binary_metric.hpp). The AUC metric class in the same file also reads `metadata.weights()` — same source.
- `min_sum_hessian_in_leaf` is "Minimum sum of the Hessian ... For classification objectives, it represents a sum over a distribution of probabilities" — [Parameters-Tuning.rst](https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/Parameters-Tuning.rst). sklearn docstring: `min_child_weight` "Minimum sum of instance weight (Hessian) needed in a child (leaf)" default 1e-3 — [sklearn.py](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/sklearn.py)

**pos_bagging_fraction / neg_bagging_fraction (balanced bagging)**
- `pos_bagging_fraction`: default 1.0, aliases `pos_sub_row`, `pos_subsample`, `pos_bagging`, constraint `0.0 < x <= 1.0`; "used only in `binary` application"; "used for imbalanced binary classification problem, will randomly sample `#pos_samples * pos_bagging_fraction` positive samples in bagging"; "should be used together with `neg_bagging_fraction`"; "to enable this, you need to set `bagging_freq` and `neg_bagging_fraction` as well"; "if both ... are set to `1.0`, balanced bagging is disabled"; "if balanced bagging is enabled, `bagging_fraction` will be ignored". `neg_bagging_fraction` mirrors this (aliases `neg_sub_row`, `neg_subsample`, `neg_bagging`) — [Parameters.rst](https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/Parameters.rst)
- `bagging_freq`: default 0 (disabled); "`k` means perform bagging at every `k` iteration" — [Parameters.rst](https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/Parameters.rst)
- Code: balanced bagging is on when `(pos_bagging_fraction < 1.0 || neg_bagging_fraction < 1.0) && num_pos_data > 0`; bag size = `num_pos*pos_frac + num_neg*neg_frac`; `BalancedBaggingHelper` samples each class at its own rate — [bagging.hpp](https://raw.githubusercontent.com/microsoft/LightGBM/master/src/boosting/bagging.hpp)

### Inferences
- Because `is_unbalance`/`scale_pos_weight` multiply both grad and hess by the same class factor, Newton leaf values become weighted log-odds fits, i.e. the model targets the odds of the reweighted problem (roughly prior-shifted by log(w_pos/w_neg)). Since the initial score stays at the true prevalence, the shift is learned through the trees; predicted probabilities end up inflated for the minority class (own check: in-sample mean P rose from 0.0253 to ~0.082 with weight 39).
- `class_weight="balanced"` on binary is not identical to `is_unbalance`: same weight ratio, but (a) init score becomes logit(0.5)=0, (b) absolute weights are rescaled (minority ~n/(2n_pos), majority ~n/(2n_neg) < 1), which changes how `min_child_weight`/`min_sum_hessian_in_leaf` and L1/L2 regularisation bind, and (c) if you pass `eval_class_weight`, validation metrics are computed on the weighted distribution. With `is_unbalance` metrics stay on the unweighted distribution (label weights live only in the objective).
- Balanced bagging subsamples negatives without reweighting them, so each tree is fit on a positive-enriched sample; this should also inflate probabilities, though less directly (init score still from the full data). Own check: test mean P 0.0181 vs 0.0155 baseline with neg_bagging_fraction=0.1 (single synthetic run).
- For softmax multiclass, the only native imbalance knob is per-row weights (sklearn `class_weight` dict/"balanced", or Dataset `weight`); `is_unbalance` applies only if you switch to `multiclassova`.

### Gaps
- No official doc gives a formula for "how much" probabilities are distorted; the docs only say "poor estimates".
- I did not find an official statement on whether `is_unbalance` interacts with `boost_from_average` by design; the behaviour above is read from source code.

## Q2. Objectives (binary, cross_entropy, custom focal loss), metrics (average_precision, auc, binary_logloss), and early stopping with them

### Takeaway
`binary` and `cross_entropy` both minimise log-loss (cross_entropy accepts labels in [0,1] and only per-row weights; no is_unbalance). Ranking metrics (`auc`, `average_precision`) are invariant to monotone prob distortion and are the natural early-stopping metrics when weighting is used; `binary_logloss` measures calibration and will look worse under weighting. Early stopping is now a callback (`lgb.early_stopping`) or the `early_stopping_round` param; it checks all metrics unless `first_metric_only=True`.

### Cited Findings
- Objectives: `binary` (log loss), `multiclass` (softmax, alias `softmax`), `multiclassova` (aliases `multiclass_ova`, `ova`, `ovr`; "`num_class` should be set as well"), `cross_entropy` ("objective function for cross-entropy (with optional linear weights)", alias `xentropy`), `cross_entropy_lambda` ("alternative parameterization of cross-entropy", alias `xentlambda`); for cross-entropy "label is anything in interval [0, 1]" — [Parameters.rst](https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/Parameters.rst)
- `cross_entropy` gradient/hessian multiply by per-row `weights_[i]`; no class-weight logic in the objective — [xentropy_objective.hpp](https://raw.githubusercontent.com/microsoft/LightGBM/master/src/objective/xentropy_objective.hpp)
- Metrics: `auc`; `average_precision` ("average precision score", linking sklearn's `average_precision_score`); `binary_logloss` (alias `binary`); `binary_error`; `auc_mu`; `multi_logloss`; `multi_error`; `cross_entropy`; `kullback_leibler`; "support multiple metrics, separated by `,`"; `r2` is "New in version 4.7.0" — [Parameters.rst](https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/Parameters.rst)
- `auc_mu_weights`: "used only with `auc_mu` metric"; "list representing flattened matrix (in row-major order) giving loss weights for classification errors"; `n*n` elements; "if not specified, will use equal weights for all classes" — [Parameters.rst](https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/Parameters.rst)
- Early stopping params: `early_stopping_round` default 0, aliases `early_stopping_rounds`, `early_stopping`, `n_iter_no_change`, "will stop training if one metric of one validation data doesn't improve in last `early_stopping_round` rounds", "`<= 0` means disable"; `early_stopping_min_delta` default 0.0, "*New in version 4.4.0*"; `first_metric_only` default false, "Set this to `true`, if you want to use only the first metric for early stopping" — [Parameters.rst](https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/Parameters.rst)
- Callback: `lgb.early_stopping(stopping_rounds, first_metric_only=False, verbose=True, min_delta=0.0)`; "Requires at least one validation data and one metric. If there's more than one, will check all of them. But the training data is ignored anyway."; "If using `boosting_type="dart"`, this callback has no effect" — [callback.py](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/callback.py)
- "Note that `train()` will return a model from the best iteration." and "This works with both metrics to minimize (L2, log loss, etc.) and to maximize (NDCG, AUC, etc.). Note that if you specify more than one evaluation metric, all of them will be used for early stopping." Example: `lgb.train(param, train_data, num_round, valid_sets=valid_sets, callbacks=[lgb.early_stopping(stopping_rounds=5)])` — [Python-Intro.rst](https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/Python-Intro.rst)
- sklearn `eval_metric`: "If list, it can be a list of built-in metrics, a list of custom evaluation metrics, or a mix of both. In either case, the `metric` from the model parameters will be evaluated and used as well. Default: ... 'logloss' for LGBMClassifier" — [sklearn.py](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/sklearn.py)
- `feval` signature for `lgb.train`: `(preds, eval_data) -> (metric_name, metric_value, maximize)`; "If custom objective function is used, predicted values are returned before any transformation, e.g. they are raw margin instead of probability"; "To ignore the default metric corresponding to the used objective, set the `metric` parameter to the string `"None"`" — [engine.py](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/engine.py)
- Own check, one synthetic dataset (2.5% positives, early stopping on validation `average_precision` with `first_metric_only=True`, test set n=30k), illustrative only:

| setting | best_iter | test AUC | test AP | test logloss | Brier | mean P (true 0.0253) |
|---|---|---|---|---|---|---|
| binary, no correction | 587 | 0.8915 | 0.7651 | 0.0765 | 0.00953 | 0.0155 |
| is_unbalance | 761 | 0.8919 | 0.7581 | 0.0728 | 0.00880 | 0.0173 |
| scale_pos_weight=38.6 | 761 | 0.8919 | 0.7581 | 0.0728 | 0.00880 | 0.0173 |
| scale_pos_weight=sqrt(38.6) | 345 | 0.8933 | 0.7591 | 0.0523 | 0.00870 | 0.0195 |
| pos 1.0 / neg 0.1 bagging | 627 | 0.8897 | 0.7610 | 0.0678 | 0.00875 | 0.0181 |
| focal (γ=2, α=0.25), custom | 1393 | 0.8912 | 0.7697 | 0.0498 | 0.00913 | 0.0182 |

  Ranking metrics moved by <0.01 AUC/AP across settings. The calibration columns are confounded by `flip_y` label noise and early stopping on AP. Do not cite these as evidence beyond "mechanics work and differences in ranking were small here".

### Inferences
- When using any reweighting, early-stop on a ranking metric (`average_precision` is more sensitive than `auc` under heavy imbalance) and put it first with `first_metric_only=True`. Otherwise the default `binary_logloss`, which the sklearn wrapper always adds, can stop training at a different point.
- If you early-stop on `binary_logloss` while training with `is_unbalance`, the validation logloss is computed on unweighted validation data, so it penalises the inflated probabilities. That is a calibration/ranking trade-off the stopping rule will silently make.
- With a custom objective, built-in metrics receive raw margins (see Q3), so `auc`/`average_precision` remain valid (monotone) but `binary_logloss` would be wrong. Use a `feval` that applies the sigmoid.

### Gaps
- No LightGBM-specific published benchmark comparing early stopping on AP vs AUC vs logloss under imbalance was found.

## Q3. Correct focal-loss custom objective in LightGBM 4.x (lgb.train and LGBMClassifier)

### Takeaway
In 4.x a custom objective is passed as a callable in `params["objective"]` (lgb.train: `f(preds, train_data) -> (grad, hess)`; sklearn: `objective(y_true, y_pred[, weight[, group]]) -> (grad, hess)`). `fobj=` is gone. Preds are raw margins, there is no automatic boost_from_average, and `predict_proba` returns raw scores with a warning, so apply the sigmoid yourself. Widely cited community focal-loss code (jrzaurin) uses the removed `fobj=` API and numerical derivatives. Below is an analytic version I verified numerically. The focal-loss hessian can be negative, so clip it.

### Cited Findings
- `lgb.train` Note: "A custom objective function can be provided for the `objective` parameter. It should accept two parameters: preds, train_data and return (grad, hess). preds ... Predicted values are returned before any transformation, e.g. they are raw margin instead of probability of positive class for binary task." — [engine.py](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/engine.py)
- Internals: `if callable(params["objective"]): fobj = params["objective"]; params["objective"] = "none"`, then `booster.update(fobj=fobj)` each round — [engine.py](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/engine.py). `train()` has no `fobj` argument in 4.x (signature: params, train_set, num_boost_round, valid_sets, valid_names, feval, init_model, keep_training_booster, callbacks) — same source.
- Official example: `def loglikelihood(preds, train_data): labels = train_data.get_label(); preds = 1.0/(1.0+np.exp(-preds)); grad = preds - labels; hess = preds*(1.0-preds); return grad, hess`, passed via "# Pass custom objective function through params"; comment: "when you do customized loss function, the default prediction value is margin. This may make built-in evaluation metric calculate wrong results" — [advanced_example.py](https://raw.githubusercontent.com/microsoft/LightGBM/master/examples/python-guide/advanced_example.py)
- sklearn signature: "`objective(y_true, y_pred) -> grad, hess`, `objective(y_true, y_pred, weight) -> grad, hess` or `objective(y_true, y_pred, weight, group) -> grad, hess`"; y_pred raw margin; for multi-class y_pred and grad/hess are 2-D `[n_samples, n_classes]` — [sklearn.py](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/sklearn.py)
- sklearn `predict_proba` with a callable objective logs "Cannot compute class probabilities or labels due to the usage of customized objective function. Returning raw scores instead." — [sklearn.py](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/sklearn.py). Own check (4.7.0): `predict_proba` returned shape `(3,)` (1-D raw scores, not `(n,2)`) with that warning.
- Community reference implementation (jrzaurin/LightGBM-with-Focal-Loss): `focal_loss_lgb(y_pred, dtrain, alpha, gamma)` with loss `-(a*t + (1-a)*(1-t)) * (1 - (t*p + (1-t)*(1-p)))**g * (t*log p + (1-t)*log(1-p))`, grad/hess via `scipy.misc.derivative` numerical differentiation, used as `lgb.train(params, lgbtrain, valid_sets=[lgbeval], fobj=focal_loss, feval=eval_error)` — [README](https://raw.githubusercontent.com/jrzaurin/LightGBM-with-Focal-Loss/master/README.md). The `fobj=` keyword no longer exists in `lgb.train` ([engine.py](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/engine.py)), and `scipy.misc.derivative` was deprecated/removed in recent SciPy (UNVERIFIED, from memory, not fetched).
- Other implementations: [dahvida/custom_gbm](https://github.com/dahvida/custom_gbm) (JAX autodiff wrapper for focal, LDAM, logit-adjusted and PolyLoss; README says it requires "LightGBM 3.5.5", so it predates 4.x), and [y2019xcj/focalloss-for-lightgbm-xgboost](https://github.com/y2019xcj/focalloss-for-lightgbm-xgboost) (multi-class; not inspected, SNIPPET-ONLY).
- Focal loss definition: Lin et al., "Focal Loss for Dense Object Detection", arXiv:1708.02002 (cited by jrzaurin README, [README](https://raw.githubusercontent.com/jrzaurin/LightGBM-with-Focal-Loss/master/README.md)).
- Own derivation, checked against finite differences on a grid z ∈ [-8,8] for y∈{0,1} (grad and hess both `allclose`; γ=0 reduces to α-weighted logloss hessian). With u = s·z (s=+1 for y=1, -1 for y=0), p_t = σ(u), q = 1-p_t, a_t = α or 1-α:
  - grad = s · a_t · (γ p_t q^γ log p_t − q^(γ+1))
  - hess = a_t · p_t · q^γ · (γ q log p_t − γ² p_t log p_t + (2γ+1) q)
  - Minimum hessian on the grid was negative (−0.035 for y=0, −0.012 for y=1 at γ=2, α=0.25), so clip it, e.g. `np.maximum(hess, 1e-6)`.

  ```python
  import numpy as np, lightgbm as lgb
  from scipy.special import expit
  def focal(y, z, gamma=2.0, alpha=0.25):
      p = expit(z); s = 2*y - 1
      pt = np.where(y == 1, p, 1 - p); q = 1 - pt
      at = np.where(y == 1, alpha, 1 - alpha)
      logpt = np.log(np.clip(pt, 1e-15, None))
      grad = s*at*(gamma*pt*q**gamma*logpt - q**(gamma+1))
      hess = at*pt*q**gamma*(gamma*q*logpt - gamma**2*pt*logpt + (2*gamma+1)*q)
      return grad, np.maximum(hess, 1e-6)
  # lgb.train: params["objective"] = lambda preds, ds: focal(ds.get_label(), preds); params["metric"] = "None"
  #            feval must apply expit() to preds itself; final probs = expit(booster.predict(X))
  # sklearn:   LGBMClassifier(objective=lambda y_true, y_pred: focal(y_true, y_pred)); probs = expit(clf.predict_proba(X))  # returns raw scores
  ```
  (Own check: ran end-to-end with lgb.train + early stopping on an AP feval and with LGBMClassifier + eval_X/eval_y in 4.7.0.)

### Inferences
- With a custom objective there is no `boost_from_average`, so training starts from raw score 0 (p=0.5). Under heavy imbalance, pass `init_score=logit(prevalence)` on the Datasets (and add it back at predict time), or expect many wasted early iterations. Own run: focal took 1393 iterations vs 587 for binary. This is inferred from `params["objective"]="none"` in engine.py, not stated in docs.
- Focal loss with α<0.5 down-weights positives while γ down-weights easy (mostly negative) examples. Its outputs are not calibrated probabilities either. It is a ranking-oriented tool, and in the single own run it did not clearly beat plain `binary` on AUC.
- Multiclass focal: y_pred/grad/hess must be 2-D `[n_samples, n_classes]` in 4.x per the sklearn docstring.

### Gaps
- No official LightGBM focal-loss example exists in microsoft/LightGBM (none found).
- No peer-reviewed tabular benchmark of focal loss in LightGBM was fetched.

## Q4. Empirical evidence: weighting vs resampling (SMOTE, under/oversampling) vs threshold moving, and calibration harm

### Takeaway
Evidence consistently says resampling/reweighting does not improve discrimination (AUC) for strong learners and badly miscalibrates probabilities, while threshold tuning on the natural distribution achieves the same sensitivity/specificity trade-off. Resampling must happen only inside training folds (imblearn pipelines).

### Cited Findings
- van den Goorbergh, van Smeden, Timmerman, Van Calster, "The harm of class imbalance corrections for risk prediction models: illustration and simulation using logistic regression", JAMIA 2022;29(9):1525-1534 — [JAMIA](https://academic.oup.com/jamia/article/29/9/1525/6605096) (page blocked to me; content via search snippets, SNIPPET-ONLY):
  - Random undersampling, random oversampling and SMOTE "yielded poorly calibrated models" in which "the probability to belong to the minority class was strongly overestimated"; they "did not result in higher areas under the ROC curve" vs no correction; correction improved the sensitivity/specificity balance "although similar results were obtained by shifting the probability threshold instead" — [search snippet of JAMIA/PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9382395/) SNIPPET-ONLY
  - Caveat: this study used logistic regression (+ simulation), not boosted trees. arXiv preprint: [arXiv:2202.09101](https://arxiv.org/pdf/2202.09101).
  - A follow-up "The Harms of Class Imbalance Corrections for Machine Learning Based Prediction Models: A Simulation Study" exists — [ResearchGate](https://www.researchgate.net/publication/388849859_The_Harms_of_Class_Imbalance_Corrections_for_Machine_Learning_Based_Prediction_Models_A_Simulation_Study) (UNVERIFIED, title only; content not read)
- Elor & Averbuch-Elor, "To SMOTE, or not to SMOTE?", arXiv:2201.08528 (2022) — [arXiv](https://arxiv.org/pdf/2201.08528) (blocked to me; via search snippets, SNIPPET-ONLY):
  - Experiments with LightGBM, XGBoost, CatBoost (plus weaker learners such as decision tree, SVM, MLP, AdaBoost); "balancing does not improve prediction performance for the strong ones"; balancing "could improve prediction performance for weak classifiers"; for strong classifiers, "optimizing the decision threshold yielded similar prediction quality" to balancing — [search summary](https://arxiv.org/pdf/2201.08528) SNIPPET-ONLY
  - One snippet says oversampling "significantly improved prediction for the weak decision tree and LightGBM, for XGBoost and CatBoost it did not" (SNIPPET-ONLY; this partially conflicts with "balancing does not improve ... strong ones". The likely reconciliation is fixed default threshold vs optimised threshold. UNVERIFIED)
  - 73 datasets; "while oversampling the data is beneficial when using a fixed threshold, it does not improve prediction quality when the threshold is optimized"; "balancing the data results in better logloss for the minority samples, worse logloss for the majority samples and worse logloss overall" — [valeman/smote_is_what_you_dont_need README via search snippet](https://github.com/valeman/smote_is_what_you_dont_need/blob/main/README.md) SNIPPET-ONLY / secondary
- imbalanced-learn "Common pitfalls" (data leakage): resampling the entire dataset before splitting means "the model will not be tested on a dataset with class distribution similar to the real use-case" and leaks information; reported performance "will be over-optimistic". In their adult-census example (made ~98.8%/1.1% imbalanced, HistGradientBoosting): baseline CV balanced accuracy 0.609 ± 0.024 (left-out 0.628 ± 0.009); with resampling done wrongly before CV, CV shows 0.724 ± 0.042 but left-out only 0.698 ± 0.014 — [common_pitfalls.rst](https://raw.githubusercontent.com/scikit-learn-contrib/imbalanced-learn/master/doc/common_pitfalls.rst). (The pairing of these four numbers to the scenarios is from reading the doc's sequential code blocks.)
- imbalanced-learn: SMOTE/ADASYN and most samplers can't handle categorical features ("none of the presented methods (apart of the class `RandomOverSampler`) can deal with the categorical features"); `SMOTENC` handles mixed data via `categorical_features` — [over_sampling.rst](https://raw.githubusercontent.com/scikit-learn-contrib/imbalanced-learn/master/doc/over_sampling.rst)
- A 2026 arXiv preprint, "When Single-Dataset Conclusions Fail: A 45-Task Study of Threshold Tuning and Resampling for Imbalanced Classification", exists — [arXiv:2608.16147](https://arxiv.org/pdf/2608.16147) (UNVERIFIED, title only)
- Another theory/empirics paper: "Do we need rebalancing strategies? A theoretical and empirical study around SMOTE and its variants" — [arXiv:2402.03819](https://arxiv.org/pdf/2402.03819) (UNVERIFIED, title only)

### Inferences
- For LightGBM specifically: SMOTE on raw features interacts poorly with LightGBM's native categorical handling (SMOTE interpolates numerics; categoricals need SMOTENC), and synthetic points add no information that `scale_pos_weight` or threshold tuning doesn't already provide for ranking. Prefer: train unweighted (or mildly weighted), early-stop on AP, then tune the decision threshold on validation data (e.g. sklearn `TunedThresholdClassifierCV`, not fetched here) and, if probabilities matter, calibrate on held-out data.
- If you do resample or weight, probabilities can be mapped back with prior correction. For a positive weight w (or oversampling factor w), odds_true ≈ odds_model / w. This follows from the weighted-likelihood argument and is a standard result, but no source was fetched for it (UNVERIFIED). Note that for `is_unbalance` the initial score is not shifted (see Q1), so the correction is approximate. Isotonic/Platt calibration on an untouched validation set is the documented fallback ([sklearn.py docstring pointing to sklearn calibration](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/sklearn.py)).
- Any resampling must happen inside `imblearn.pipeline.Pipeline` per fold. The early-stopping validation set must keep the natural class distribution.

### Gaps
- Could not access full texts (arxiv.org, academic.oup.com, ncbi.nlm.nih.gov, semanticscholar blocked by proxy), so no calibration-intercept/slope numbers or per-classifier effect sizes from either paper are reported.
- No LightGBM-specific study isolating `is_unbalance` vs `scale_pos_weight` vs balanced bagging vs SMOTE was found.

## Q5. Current LightGBM version and API changes relevant here; multiclass imbalance summary

### Takeaway
The latest PyPI release is 4.7.0 (uploaded 2026-07-18). 4.7.0 adds `eval_X`/`eval_y` to sklearn `fit` and deprecates `eval_set`. Early stopping is via callbacks (or the `early_stopping_round` param), and custom objectives go through `params["objective"]`. For multiclass imbalance, use per-row weights (`class_weight` dict/"balanced") with softmax, or `multiclassova` + `is_unbalance`/`scale_pos_weight`, and evaluate with `multi_logloss`/`auc_mu` (optionally `auc_mu_weights`).

### Cited Findings
- PyPI: latest version `4.7.0`; release upload dates: 4.0.0 2023-07-13, 4.1.0 2023-09-12, 4.2.0 2023-12-22, 4.3.0 2024-01-26, 4.4.0 2024-06-15, 4.5.0 2024-07-26, 4.6.0 2025-02-15, 4.7.0 2026-07-18 — [PyPI JSON](https://pypi.org/pypi/lightgbm/json). Readthedocs "latest" shows "4.7.0.99" (dev) — [docs](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html) SNIPPET-ONLY
- `eval_set`: ".. deprecated:: 4.7.0 A list of (X, y) tuple pairs to use as validation sets. Use `eval_X` and `eval_y` instead."; `_validate_eval_set_Xy` raises "Specify either 'eval_set' or 'eval_X' and 'eval_y', but not both." and "You must specify eval_X and eval_y, not just one of them."; tuples of eval_X/eval_y give multiple validation sets; `eval_set` use emits "The argument 'eval_set' is deprecated, use 'eval_X' and 'eval_y' instead." — [sklearn.py](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/sklearn.py). Verified present in the v4.7.0 tag (28 occurrences of `eval_X`) and absent in v4.6.0 (0) — [v4.7.0 sklearn.py](https://raw.githubusercontent.com/microsoft/LightGBM/v4.7.0/python-package/lightgbm/sklearn.py), [v4.6.0 sklearn.py](https://raw.githubusercontent.com/microsoft/LightGBM/v4.6.0/python-package/lightgbm/sklearn.py). Own check: `fit(X, y, eval_X=Xva, eval_y=yva, eval_metric="average_precision", callbacks=[lgb.early_stopping(50)])` works in 4.7.0.
- polars inputs: ".. versionadded:: 4.7.0 Support for `polars` inputs" — [sklearn.py](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/sklearn.py)
- `best_iteration_`: "The best iteration of fitted model if `early_stopping()` callback has been specified." — [sklearn.py](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/sklearn.py)
- Repository appears to have moved to the `lightgbm-org` GitHub organisation: error messages in sklearn.py point to `https://github.com/lightgbm-org/LightGBM/issues` and Python-Intro links `https://github.com/lightgbm-org/LightGBM/tree/main/python-package` — [sklearn.py](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/sklearn.py), [Python-Intro.rst](https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/Python-Intro.rst); search result titled "Releases · lightgbm-org/LightGBM" at github.com/microsoft/lightgbm/releases (SNIPPET-ONLY). raw.githubusercontent.com/microsoft/LightGBM/master still served files.
- Multiclass: softmax objective uses only per-row weights; `multiclassova` = per-class BinaryLogloss honouring `is_unbalance`/`scale_pos_weight` — [multiclass_objective.hpp](https://raw.githubusercontent.com/microsoft/LightGBM/master/src/objective/multiclass_objective.hpp); `class_weight` docstring explicitly targets multiclass and computes "balanced" as `n_samples / (n_classes * np.bincount(y))` — [sklearn.py](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/sklearn.py); `auc_mu` + `auc_mu_weights` cost matrix — [Parameters.rst](https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/Parameters.rst)

### Inferences
- In multiclass `multiclassova`, `scale_pos_weight` up-weights the "this class" side of every class equally, so it cannot target a single rare class. Per-class dict weights via `class_weight` (softmax) are the precise tool.
- `pos/neg_bagging_fraction` is binary-only per docs, so there is no native balanced bagging for multiclass.
- The removal of `fobj`/`early_stopping_rounds` keyword arguments from `lgb.train` happened in 4.0.0. This is inferred from the current engine.py signature and widely known, but the 4.0.0 changelog was not fetched (UNVERIFIED as to exact version).

### Gaps
- GitHub release notes for 4.7.0 could not be fetched (GitHub API access denied in this session; newreleases.io blocked), so other 4.7.0 changes (e.g. minimum Python/scikit-learn versions) are not listed.
- LightGBM GitHub issues on is_unbalance calibration were not read (API blocked).
