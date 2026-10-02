# Baseline recipes for non-tabular datathon tasks (time series, text, image, anomaly, uplift, spatial CV)

Research date: 2026-10-02. Method: PyPI JSON API plus raw GitHub source/README/docs files, read directly. Function signatures were taken by parsing the current `main`/`master` source files (or the tagged release for scikit-learn 1.9.1) with Python `ast`, so they are the real signatures. I wrote the code marked "assembled recipe" myself by combining quoted pieces. It has NOT been executed in this sandbox, so treat it as UNVERIFIED-at-runtime even though every API call in it was checked against source.

## Q1: For each task, what are the current import paths, signatures and minimal working examples from official sources, and what are the version pitfalls?

### Takeaway
All six baselines can be built with stable, current APIs. The pitfalls that will bite in 2026 are:
- scikit-learn 1.8+ deprecated `LogisticRegression(penalty=...)` in favour of `l1_ratio`/`C`.
- statsforecast takes `df` in `forecast`/`cross_validation`, and pandas month-end frequency is now `'ME'`.
- mlforecast's `lag_transforms` now take classes such as `RollingMean(window_size=...)`.
- causalml `qini_score`/`auuc_score` expect a DataFrame with one column per model score.
- PyOD 3 keeps the classic `fit`/`decision_function` API.

### Cited Findings

#### (1) Time-series forecasting

**statsforecast (Nixtla).** The README quickstart is quoted verbatim below — [statsforecast README](https://github.com/Nixtla/statsforecast/blob/main/README.md):
```python
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsforecast.utils import AirPassengersDF

df = AirPassengersDF
sf = StatsForecast(
    models=[AutoARIMA(season_length=12)],
    freq='ME',
)
sf.fit(df)
sf.predict(h=12, level=[95])
```
Signatures parsed from the source:
- `StatsForecast.__init__(self, models: List[Any], freq: Union[str, int], n_jobs: int=1, fallback_model=None, verbose=False)` — [core.py](https://github.com/Nixtla/statsforecast/blob/main/python/statsforecast/core.py)
- `StatsForecast.forecast(self, h: int, df: DataFrame, X_df=None, level=None, fitted=False, prediction_intervals=None, id_col='unique_id', time_col='ds', target_col='y')` — [core.py](https://github.com/Nixtla/statsforecast/blob/main/python/statsforecast/core.py)
- `StatsForecast.cross_validation(self, h: int, df: DataFrame, n_windows: int=1, step_size: int=1, test_size=None, input_size=None, level=None, fitted=False, refit: Union[bool,int]=True, prediction_intervals=None, id_col='unique_id', time_col='ds', target_col='y')`. This is rolling-origin evaluation built in: `n_windows` cutoffs, `step_size` apart, and `refit=False` or an int lets you avoid refitting at every window. — [core.py](https://github.com/Nixtla/statsforecast/blob/main/python/statsforecast/core.py)
- `SeasonalNaive(season_length: int, alias='SeasonalNaive', prediction_intervals=None)`. Note that `season_length` is required. — [models.py](https://github.com/Nixtla/statsforecast/blob/main/python/statsforecast/models.py)
- `AutoETS(season_length: int=1, model: str='ZZZ', damped=None, phi=None, alias='AutoETS', prediction_intervals=None, distribution='normal')` — [models.py](https://github.com/Nixtla/statsforecast/blob/main/python/statsforecast/models.py)
- The input is a long-format DataFrame with columns `unique_id`, `ds`, `y` (the default `id_col`/`time_col`/`target_col` above). — [core.py](https://github.com/Nixtla/statsforecast/blob/main/python/statsforecast/core.py)

**utilsforecast evaluation.** `utilsforecast.evaluation.evaluate(df, metrics: List[Callable], models=None, train_df=None, level=None, id_col='unique_id', time_col='ds', target_col='y', cutoff_col='cutoff', agg_fn=None, weights=None)`. Because it takes `cutoff_col='cutoff'`, it can score cross-validation output per cutoff. — [utilsforecast evaluation.py](https://github.com/Nixtla/utilsforecast/blob/main/utilsforecast/evaluation.py)

**mlforecast (Nixtla).** The README example is quoted verbatim below — [mlforecast README](https://github.com/Nixtla/mlforecast/blob/main/README.md):
```python
from mlforecast.utils import generate_daily_series
series = generate_daily_series(n_series=20, max_length=100, n_static_features=1,
                               static_as_categorical=False, with_trend=True)
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
models = [lgb.LGBMRegressor(random_state=0, verbosity=-1), LinearRegression()]
from mlforecast import MLForecast
from mlforecast.lag_transforms import ExpandingMean, RollingMean
from mlforecast.target_transforms import Differences
fcst = MLForecast(
    models=models,
    freq='D',
    lags=[7, 14],
    lag_transforms={
        1: [ExpandingMean()],
        7: [RollingMean(window_size=28)]
    },
    date_features=['dayofweek'],
    target_transforms=[Differences([1])],
)
fcst.fit(series)
predictions = fcst.predict(14)
```
The README notes that `predict` "will automatically handle the updates required by the features using a recursive strategy". — [mlforecast README](https://github.com/Nixtla/mlforecast/blob/main/README.md)

mlforecast signatures:
- `MLForecast.fit(self, df, id_col='unique_id', time_col='ds', target_col='y', static_features=None, dropna=True, keep_last_n=None, max_horizon=None, horizons=None, ..., prediction_intervals=None, fitted=False, ...)` — [forecast.py](https://github.com/Nixtla/mlforecast/blob/main/mlforecast/forecast.py)
- `MLForecast.cross_validation(self, df, n_windows: int, h: int, id_col='unique_id', time_col='ds', target_col='y', step_size=None, static_features=None, dropna=True, keep_last_n=None, refit: Union[bool,int]=True, ..., input_size=None, fitted=False, ...)`. Here `n_windows` and `h` are required. — [forecast.py](https://github.com/Nixtla/mlforecast/blob/main/mlforecast/forecast.py)
- `MLForecast.predict(self, h: int, before_predict_callback=None, after_predict_callback=None, new_df=None, level=None, X_df=None, ids=None, transfer_conformal=None)`. Exogenous future values go in `X_df`. — [forecast.py](https://github.com/Nixtla/mlforecast/blob/main/mlforecast/forecast.py)

**Assembled recipe (UNVERIFIED at runtime).** Seasonal naive and AutoETS vs. LightGBM with rolling-origin CV on the same cutoffs:
```python
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive, AutoETS
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean
import lightgbm as lgb
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, rmse   # UNVERIFIED: module name utilsforecast.losses not opened in this session

H, W = 14, 4  # horizon, number of rolling origins
sf = StatsForecast(models=[SeasonalNaive(season_length=7), AutoETS(season_length=7)], freq='D', n_jobs=-1)
cv_sf = sf.cross_validation(h=H, df=df, n_windows=W, step_size=H)
mlf = MLForecast(models=[lgb.LGBMRegressor(verbosity=-1)], freq='D', lags=[7, 14],
                 lag_transforms={7: [RollingMean(window_size=28)]}, date_features=['dayofweek'])
cv_ml = mlf.cross_validation(df=df, n_windows=W, h=H, step_size=H)
cv = cv_sf.merge(cv_ml.drop(columns='y'), on=['unique_id', 'ds', 'cutoff'])
print(evaluate(cv.drop(columns='cutoff'), metrics=[mae, rmse]).groupby('metric').mean(numeric_only=True))
```

**Time-series pitfalls**
- statsforecast `forecast()`/`cross_validation()` take `df` as an argument; the old pattern of passing `df` to the constructor no longer exists. The README itself uses `freq='ME'`, which is the pandas ≥2.2 month-end alias that replaces `'M'`. — [statsforecast README](https://github.com/Nixtla/statsforecast/blob/main/README.md)
- The current mlforecast README uses class-based lag transforms (`mlforecast.lag_transforms.RollingMean(window_size=...)`). Older tutorials used `window_ops` functions such as `rolling_mean`. — [mlforecast README](https://github.com/Nixtla/mlforecast/blob/main/README.md)
- `utilsforecast.evaluate` expects the default `cutoff` column only when evaluating per cutoff. My inference is that you should drop `cutoff` or set `cutoff_col` to get pooled metrics (UNVERIFIED).

#### (2) Text classification

**sentence-transformers.** The README is quoted verbatim below — [sentence-transformers README](https://github.com/huggingface/sentence-transformers/blob/main/README.md):
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
sentences = ["The weather is lovely today.", "It's so sunny outside!", "He drove to the stadium."]
embeddings = model.encode(sentences)
print(embeddings.shape)
# => (3, 384)
```
- The README recommends "Python 3.10+, PyTorch 2.2+, and transformers v5.0+". It now also documents `SparseEncoder`, `CrossEncoder` and `MultiVectorEncoder` with `encode_query`/`encode_document`. — [sentence-transformers README](https://github.com/huggingface/sentence-transformers/blob/main/README.md)

**scikit-learn LogisticRegression (1.9.1).** Signature: `LogisticRegression(penalty='deprecated', *, C=1.0, l1_ratio=0.0, dual=False, tol=1e-4, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, verbose=0, warm_start=False, n_jobs=None)`. — [sklearn _logistic.py @1.9.1](https://github.com/scikit-learn/scikit-learn/blob/1.9.1/sklearn/linear_model/_logistic.py)

The docstring says: "`penalty` was deprecated in version 1.8 and will be removed in 1.10. Use `l1_ratio` and `C` instead. `l1_ratio=0` for `penalty='l2'`, `l1_ratio=1` for `penalty='l1'` ... and `C=np.inf` for `penalty=None`." It also notes that the `l1_ratio` default changed from None to 0.0 in 1.8, and that "`n_jobs` is deprecated in version 1.8 and will be removed in 1.10". — [sklearn _logistic.py @1.9.1](https://github.com/scikit-learn/scikit-learn/blob/1.9.1/sklearn/linear_model/_logistic.py)

`multi_class` no longer appears as a constructor parameter in 1.9.1 (it is absent from the signature above). — [sklearn _logistic.py @1.9.1](https://github.com/scikit-learn/scikit-learn/blob/1.9.1/sklearn/linear_model/_logistic.py)

**Assembled recipes (UNVERIFIED at runtime)**
```python
# TF-IDF + LR
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                    LogisticRegression(C=4.0, max_iter=2000, class_weight='balanced'))  # no penalty= / n_jobs= (deprecated 1.8)
print(cross_val_score(clf, texts, y, cv=5, scoring='f1_macro').mean())

# Sentence-embeddings + linear head
from sentence_transformers import SentenceTransformer
enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
X = enc.encode(list(texts), batch_size=64, normalize_embeddings=True, show_progress_bar=True)  # kwargs: UNVERIFIED in this session
print(cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=5, scoring='f1_macro').mean())
```
Putting TF-IDF inside the Pipeline avoids leaking vocabulary/IDF statistics across CV folds. This follows from how sklearn pipelines refit on each fold; it is my inference, not quoted.

#### (3) Image classification with frozen embeddings

**timm, feature extraction.** Quoted from the official docs — [timm feature_extraction.mdx](https://github.com/huggingface/pytorch-image-models/blob/main/hfdocs/source/feature_extraction.mdx):
```python
>>> m = timm.create_model('resnet50', pretrained=True, num_classes=0)
>>> o = m(torch.randn(2, 3, 224, 224))
>>> print(f'Pooled shape: {o.shape}')
# Pooled shape: torch.Size([2, 2048])
```
`num_classes=0, global_pool=''` gives unpooled features, and `forward_features` / `forward_head` are also documented. — [timm feature_extraction.mdx](https://github.com/huggingface/pytorch-image-models/blob/main/hfdocs/source/feature_extraction.mdx)

**timm, preprocessing.** Quoted from the quickstart — [timm quickstart.mdx](https://github.com/huggingface/pytorch-image-models/blob/main/hfdocs/source/quickstart.mdx):
```python
>>> data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
>>> transform = timm.data.create_transform(**data_cfg)
```
`timm.data` also exports `resolve_model_data_config` (it is in `timm/data/__init__.py` and defined in `config.py`). — [timm data/__init__.py](https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/__init__.py)

**DINOv2 via torch hub.** Quoted from the README — [DINOv2 README](https://github.com/facebookresearch/dinov2/blob/main/README.md):
```python
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dinov2_vits14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
```

**DINOv2 via transformers.** Quoted from the docs — [transformers dinov2.md](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/dinov2.md):
```python
from transformers import AutoImageProcessor, AutoModel
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base', device_map="auto")
inputs = processor(images=image, return_tensors="pt").to(model.device)
```

**Assembled recipe (UNVERIFIED at runtime)**
```python
import torch, timm, numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
m = timm.create_model('resnet50', pretrained=True, num_classes=0).eval()
tf = timm.data.create_transform(**timm.data.resolve_data_config(m.pretrained_cfg))
@torch.inference_mode()
def embed(paths, bs=64):
    out = []
    for i in range(0, len(paths), bs):
        x = torch.stack([tf(Image.open(p).convert('RGB')) for p in paths[i:i+bs]])
        out.append(m(x).cpu().numpy())
    return np.concatenate(out)
Xtr, Xte = embed(train_paths), embed(test_paths)
clf = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr, ytr)
# DINOv2 alternative: m = torch.hub.load('facebookresearch/dinov2','dinov2_vits14').eval(); m(x) returns the CLS embedding (UNVERIFIED: output form not checked)
```
- For transformers DINOv2, taking `outputs.pooler_output` or `last_hidden_state[:,0]` as the image embedding is a common pattern, but I did not confirm it in the docs read (UNVERIFIED).

#### (4) Anomaly detection with PyOD

The README's "Outlier Detection with 5 Lines of Code" is quoted verbatim below — [PyOD README](https://github.com/yzhao062/pyod/blob/master/README.rst):
```python
from pyod.models.iforest import IForest
clf = IForest()
clf.fit(X_train)
y_train_scores = clf.decision_scores_          # training anomaly scores
y_test_scores = clf.decision_function(X_test)   # test anomaly scores
```
- PyOD 3 rebrands as "Agentic Anomaly Detection At Scale". It adds `ADEngine` (`from pyod.utils.ad_engine import ADEngine`) and an `od-expert` skill/MCP server, "while keeping the classic fit/predict API fully backward-compatible". It claims "61 detectors across tabular, time series, graph, text, image, and audio". — [PyOD README](https://github.com/yzhao062/pyod/blob/master/README.rst)
- `ECOD(contamination=0.1, n_jobs=1)` in `pyod.models.ecod`. It is parameter-free apart from contamination. — [ecod.py](https://github.com/yzhao062/pyod/blob/master/pyod/models/ecod.py)
- `IForest(n_estimators=100, max_samples='auto', contamination=0.1, max_features=1.0, bootstrap=False, n_jobs=1, behaviour='old', random_state=None, verbose=0)` in `pyod.models.iforest`. — [iforest.py](https://github.com/yzhao062/pyod/blob/master/pyod/models/iforest.py)
- For evaluation, `sklearn.metrics.average_precision_score(y_true, y_score, *, average='macro', pos_label=1, sample_weight=None)` is the PR-AUC (AP) summary. — [sklearn _ranking.py @1.9.1](https://github.com/scikit-learn/scikit-learn/blob/1.9.1/sklearn/metrics/_ranking.py)

**Assembled recipe (UNVERIFIED at runtime)**
```python
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from sklearn.metrics import average_precision_score, roc_auc_score
for name, det in [('ECOD', ECOD()), ('IForest', IForest(n_estimators=300, random_state=0))]:
    det.fit(X_train)                       # unsupervised: y ignored
    s = det.decision_function(X_test)      # higher = more anomalous
    print(name, 'PR-AUC', average_precision_score(y_test, s), 'base rate', y_test.mean())
```
PR-AUC should be reported next to the positive base rate, because a random scorer's AP is approximately the prevalence. That is a standard fact (my inference); I did not find a source for it in this session.

#### (5) Uplift: T-learner and Qini/AUUC

**causalml T-learner.** In `causalml.inference.meta`:
- `BaseTRegressor(learner=None, control_learner=None, treatment_learner=None, ate_alpha=0.05, control_name=0)` and `BaseTClassifier(...)` have the same constructor. — [tlearner.py](https://github.com/uber/causalml/blob/master/causalml/inference/meta/tlearner.py)
- `fit(X, treatment, y, p=None, ...)` and `predict(X, treatment=None, y=None, p=None, return_components=False, verbose=True, return_ci=False)`. — [tlearner.py](https://github.com/uber/causalml/blob/master/causalml/inference/meta/tlearner.py)
- `fit_predict(X, treatment, y, ...)` also exists. — [tlearner.py](https://github.com/uber/causalml/blob/master/causalml/inference/meta/tlearner.py)

The quickstart is quoted below — [causalml quickstart.rst](https://github.com/uber/causalml/blob/master/docs/quickstart.rst):
```python
from causalml.inference.meta import XGBTRegressor
from causalml.dataset import synthetic_data
y, X, treatment, _, _, e = synthetic_data(mode=1, n=1000, p=5, sigma=1.0)
xg = XGBTRegressor(random_state=42)
te, lb, ub = xg.estimate_ate(X=X, treatment=treatment, y=y)
```

**causalml metrics** (`causalml.metrics.visualize`, also used via `causalml.metrics`; the re-export is UNVERIFIED):
- `qini_score(df, outcome_col='y', treatment_col='w', treatment_effect_col='tau', normalize=True, tmle=False, *args, return_ci=False, n_bootstrap=200, alpha=0.05, random_state=None, **kwarg)` — [visualize.py](https://github.com/uber/causalml/blob/master/causalml/metrics/visualize.py)
- `auuc_score(...)` has the same signature. — [visualize.py](https://github.com/uber/causalml/blob/master/causalml/metrics/visualize.py)
- The docstring describes `df` as "a data frame with model estimates and actual data as columns". It returns "the Qini score for each model estimate column" as a Series, or a DataFrame with SE/CI/p-value if `return_ci=True`. — [visualize.py](https://github.com/uber/causalml/blob/master/causalml/metrics/visualize.py)
- The docstring says the Qini p-value tests H0: Qini=0, "unlike AUUC, whose random baseline is about 0.5", and that "No p-value is reported for AUUC". — [visualize.py](https://github.com/uber/causalml/blob/master/causalml/metrics/visualize.py)
- The `return_ci`/bootstrap arguments are newer additions ("Default False, so existing callers are unaffected"). — [visualize.py](https://github.com/uber/causalml/blob/master/causalml/metrics/visualize.py)

**Assembled recipe: plain sklearn T-learner plus causalml Qini (UNVERIFIED at runtime)**
```python
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from causalml.metrics.visualize import qini_score, auuc_score
m1 = HGB().fit(X_tr[w_tr == 1], y_tr[w_tr == 1])
m0 = HGB().fit(X_tr[w_tr == 0], y_tr[w_tr == 0])
uplift = m1.predict_proba(X_te)[:, 1] - m0.predict_proba(X_te)[:, 1]
df_eval = pd.DataFrame({'y': y_te, 'w': w_te, 't_learner': uplift})   # one column per model score; no 'tau' column on real data
print(qini_score(df_eval, outcome_col='y', treatment_col='w'))
print(auuc_score(df_eval, outcome_col='y', treatment_col='w'))
# causalml-native equivalent: from causalml.inference.meta import BaseTClassifier; BaseTClassifier(learner=HGB()).fit(X_tr, w_tr, y_tr).predict(X_te)
```

**Uplift pitfalls**
- `treatment_effect_col='tau'` is only for synthetic data where the true effect is known. On real data, supply only outcome and treatment columns: "For the latter, both `outcome_col` and `treatment_col` should be provided." — [visualize.py get_cumgain docstring](https://github.com/uber/causalml/blob/master/causalml/metrics/visualize.py)
- causalml's `control_name` defaults to `0`, which assumes 0/1 treatment coding. If treatment is stored as strings such as 'control', pass `control_name='control'`. — [tlearner.py](https://github.com/uber/causalml/blob/master/causalml/inference/meta/tlearner.py)

#### (6) Spatial block cross-validation

- scikit-learn 1.9.1 has `GroupKFold(n_splits=5, *, shuffle=False, random_state=None)`, so `shuffle` is now supported. `StratifiedGroupKFold(n_splits=5, shuffle=False, random_state=None)` and `TimeSeriesSplit(n_splits=5, *, max_train_size=None, test_size=None, gap=0)` are also available. — [sklearn _split.py @1.9.1](https://github.com/scikit-learn/scikit-learn/blob/1.9.1/sklearn/model_selection/_split.py)
- Verde has `BlockKFold(spacing=None, shape=None, n_splits=5, shuffle=False, random_state=None, balance=True)` and `BlockShuffleSplit(spacing=None, shape=None, n_splits=10, test_size=0.1, train_size=None, random_state=None, balancing=10)`. — [verde model_selection.py](https://github.com/fatiando/verde/blob/main/verde/model_selection.py)

The Verde docstring example is quoted below — [verde model_selection.py](https://github.com/fatiando/verde/blob/main/verde/model_selection.py):
```python
>>> X = np.transpose([i.ravel() for i in coords])
>>> kfold = BlockKFold(spacing=1.5, n_splits=4)
>>> for train, test in kfold.split(X):
...     print("Train: {} Test: {}".format(train, test))
```

**Assembled dependency-free recipe (UNVERIFIED at runtime)**
```python
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_score
block = 5_000  # metres if projected CRS; choose >= residual autocorrelation range
gx = np.floor(df.x / block).astype(int); gy = np.floor(df.y / block).astype(int)
groups = gx.astype(str) + '_' + gy.astype(str)
scores = cross_val_score(model, X, y, cv=GroupKFold(n_splits=5), groups=groups)
```
Coordinates must be projected (metres), not lat/lon degrees, for the block size to mean anything. My inference is that you should project with geopandas `to_crs` before blocking; I did not check this against geopandas docs.

### Inferences
- A single "rolling-origin" design is available natively in both Nixtla libraries (`n_windows`, `step_size`, `refit`). Using the same `h`/`n_windows`/`step_size` in both lets you merge their CV frames on (`unique_id`, `ds`, `cutoff`) for a fair comparison against seasonal-naive.
- For text and image tasks, frozen embeddings plus LogisticRegression is the fastest strong baseline. The main 2026 code-break risk is copying old snippets that pass `penalty='l2'`, `n_jobs` or `multi_class` to LogisticRegression. These give FutureWarnings in 1.8–1.9 and will break in 1.10 (or already break, for `multi_class`, which is absent from the 1.9.1 signature).

### Gaps
- I did not execute any assembled recipe; runtime behaviour such as output column names and dtype handling is unverified.
- `utilsforecast.losses` import path, the sentence-transformers `encode` kwargs (`normalize_embeddings`, `batch_size`), and the DINOv2 torch-hub forward output shape were not checked in source this session.
- I did not read the mlforecast CV docs page (nixtlaverse.nixtla.io); signatures come from source only.

## Q2: Which libraries are maintained in 2026, and what are their latest versions?

### Takeaway
Every core library had a release in 2026 except scikit-uplift, whose last release was in Aug 2022. The PyPI project `spatialkfold` returned "Not Found". Prefer causalml for uplift metrics, and sklearn GroupKFold or verde for spatial CV.

### Cited Findings
Latest version, upload date and requires_python for each package, from the PyPI JSON API (`https://pypi.org/pypi/<pkg>/json`, fetched 2026-10-02):

| Package | Latest version | Uploaded | Requires Python | Source |
|---|---|---|---|---|
| statsforecast | 2.1.1 | 2026-07-16 | >=3.10 | [PyPI](https://pypi.org/pypi/statsforecast/json) |
| mlforecast | 1.1.0 | 2026-07-10 | >=3.10 | [PyPI](https://pypi.org/pypi/mlforecast/json) |
| lightgbm | 4.7.0 | 2026-07-18 | >=3.10 | [PyPI](https://pypi.org/pypi/lightgbm/json) |
| scikit-learn | 1.9.1 | 2026-09-10 | >=3.11 | [PyPI](https://pypi.org/pypi/scikit-learn/json) |
| sentence-transformers | 6.1.0 | 2026-09-18 | >=3.10 | [PyPI](https://pypi.org/pypi/sentence-transformers/json) |
| timm | 1.0.30 | 2026-09-22 | not recorded | [PyPI](https://pypi.org/pypi/timm/json) |
| transformers | 5.18.0 | 2026-09-30 | not recorded | [PyPI](https://pypi.org/pypi/transformers/json) |
| torch | 2.14.1 | 2026-09-30 | >=3.10 | [PyPI](https://pypi.org/pypi/torch/json) |
| pyod | 3.6.6 | 2026-09-17 | >=3.9 | [PyPI](https://pypi.org/pypi/pyod/json) |
| causalml | 0.17.0 | 2026-07-04 | >=3.11 | [PyPI](https://pypi.org/pypi/causalml/json) |
| scikit-uplift | 0.5.1 | 2022-08-11 | none declared | [PyPI](https://pypi.org/pypi/scikit-uplift/json) |
| verde | 1.9.0 | 2026-03-19 | >=3.9 | [PyPI](https://pypi.org/pypi/verde/json) |
| geopandas | 1.2.0 | 2026-09-28 | >=3.11 | [PyPI](https://pypi.org/pypi/geopandas/json) |
| spatialkfold | no record ("Not Found") | – | – | [PyPI](https://pypi.org/pypi/spatialkfold/json) |

- The scikit-uplift GitHub README raw fetch returned only 14 bytes, so its repository state was not confirmed. — [scikit-uplift raw README](https://raw.githubusercontent.com/maks-sh/scikit-uplift/master/README.rst)
- The causalml README is active in 2026: it references a KDD 2025 workshop and new benchmark loaders (LaLonde, IHDP, Twins). — [causalml README](https://github.com/uber/causalml/blob/master/README.md)

### Inferences
- scikit-learn 1.9 and causalml 0.17 require Python ≥3.11, so a datathon environment on Python 3.10 will get older versions with different LogisticRegression defaults. Pin versions or check `sklearn.__version__`.
- scikit-uplift (`sklift.metrics.qini_auc_score`) still installs but has had no release since 2022. With sklearn 1.9 / Python 3.11+ it is a compatibility risk (inference, not tested).

### Gaps
- I did not check whether `spatialkfold` exists under a different PyPI name.
- I did not check GitHub commit activity beyond the PyPI release dates.
