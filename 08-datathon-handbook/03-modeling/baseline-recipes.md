# 🧪 Baseline Recipes for Non-Tabular Tasks

<!-- markdownlint-disable MD013 -->

> 📚 Full research report and raw notes: [`technical-datathon-gap-fill-round-two-2026-10-02`](../../_research/technical-datathon-gap-fill-round-two-2026-10-02/research.md).

> Copy-paste baselines for time series, text, image, anomaly, uplift and spatial CV, plus the 2026 API pitfalls. Pick the method first with [`method-selection-guide.md`](method-selection-guide.md). Research round two, 2026-10-02. The sandbox blocked many primary sites, so read the labels: **FULL-TEXT** means the page or file was read in full, usually a GitHub copy or source file. **SNIPPET-ONLY** means the claim comes from a search-engine summary, not the page itself, so do not quote it as verbatim. **UNVERIFIED** means it could not be checked at all, or rests on background knowledge. **RE-CHECK** marks a limit, quota or price the vendor can change at any time. **CONFLICT** marks sources that disagree. "Inference (ours)" is the researchers' own synthesis, not a rule or standard.

## Baseline recipes: six task types, current APIs, and the breaking changes to watch

### What is sourced

The signatures below were parsed with Python `ast` from current `main` source, or from the scikit-learn 1.9.1 tag, so they are the real signatures (**FULL-TEXT** source). README and doc snippets are quoted verbatim from GitHub (**FULL-TEXT**). **Every "assembled recipe" was written by the researcher from those pieces and was not run by us**, so it is UNVERIFIED at runtime even though each call was checked against source. Lines marked UNVERIFIED inside the code were not checked at all.

Library versions from the PyPI JSON API, fetched 2026-10-02 (RE-CHECK before pinning):

| Package | Latest | Uploaded | Requires Python |
|---|---|---|---|
| statsforecast | 2.1.1 | 2026-07-16 | >=3.10 |
| mlforecast | 1.1.0 | 2026-07-10 | >=3.10 |
| lightgbm | 4.7.0 | 2026-07-18 | >=3.10 |
| scikit-learn | 1.9.1 | 2026-09-10 | **>=3.11** |
| sentence-transformers | 6.1.0 | 2026-09-18 | >=3.10 |
| timm | 1.0.30 | 2026-09-22 | not recorded |
| transformers | 5.18.0 | 2026-09-30 | not recorded |
| torch | 2.14.1 | 2026-09-30 | >=3.10 |
| pyod | 3.6.6 | 2026-09-17 | >=3.9 |
| causalml | 0.17.0 | 2026-07-04 | **>=3.11** |
| scikit-uplift | 0.5.1 | **2022-08-11** | none declared |
| verde | 1.9.0 | 2026-03-19 | >=3.9 |
| geopandas | 1.2.0 | 2026-09-28 | >=3.11 |

Sources: [statsforecast](https://pypi.org/pypi/statsforecast/json), [mlforecast](https://pypi.org/pypi/mlforecast/json), [scikit-learn](https://pypi.org/pypi/scikit-learn/json), [sentence-transformers](https://pypi.org/pypi/sentence-transformers/json), [pyod](https://pypi.org/pypi/pyod/json), [causalml](https://pypi.org/pypi/causalml/json), [scikit-uplift](https://pypi.org/pypi/scikit-uplift/json), [verde](https://pypi.org/pypi/verde/json). The PyPI name `spatialkfold` returned "Not Found" ([PyPI](https://pypi.org/pypi/spatialkfold/json)). Whether it exists under another name was not checked.

**Time series.** statsforecast's `forecast` and `cross_validation` take `df` as an argument, the input is long format (`unique_id`, `ds`, `y`), `SeasonalNaive` requires `season_length`, and `cross_validation(h, df, n_windows, step_size, refit, ...)` provides rolling-origin evaluation ([core.py](https://github.com/Nixtla/statsforecast/blob/main/python/statsforecast/core.py); [models.py](https://github.com/Nixtla/statsforecast/blob/main/python/statsforecast/models.py)). The README quickstart uses `freq='ME'` ([statsforecast README](https://github.com/Nixtla/statsforecast/blob/main/README.md)). mlforecast's README uses class-based `lag_transforms` such as `RollingMean(window_size=28)` and predicts with "a recursive strategy" ([mlforecast README](https://github.com/Nixtla/mlforecast/blob/main/README.md)). In `MLForecast.cross_validation`, `n_windows` and `h` are required ([forecast.py](https://github.com/Nixtla/mlforecast/blob/main/mlforecast/forecast.py)). `utilsforecast.evaluation.evaluate` takes `cutoff_col='cutoff'` ([evaluation.py](https://github.com/Nixtla/utilsforecast/blob/main/utilsforecast/evaluation.py)).

**Text.** sentence-transformers recommends "Python 3.10+, PyTorch 2.2+, and transformers v5.0+" ([README](https://github.com/huggingface/sentence-transformers/blob/main/README.md)). In scikit-learn 1.9.1, "`penalty` was deprecated in version 1.8 and will be removed in 1.10. Use `l1_ratio` and `C` instead", `n_jobs` is likewise deprecated for removal in 1.10, the `l1_ratio` default changed to 0.0, and `multi_class` is no longer in the constructor signature ([_logistic.py @1.9.1](https://github.com/scikit-learn/scikit-learn/blob/1.9.1/sklearn/linear_model/_logistic.py)).

**Image.** `timm.create_model(..., num_classes=0)` returns pooled features ([timm feature_extraction](https://github.com/huggingface/pytorch-image-models/blob/main/hfdocs/source/feature_extraction.mdx)). Preprocessing comes from `resolve_data_config` plus `create_transform` ([timm quickstart](https://github.com/huggingface/pytorch-image-models/blob/main/hfdocs/source/quickstart.mdx)). DINOv2 loads through torch hub ([DINOv2 README](https://github.com/facebookresearch/dinov2/blob/main/README.md)) or through `AutoModel.from_pretrained('facebook/dinov2-base')` ([transformers docs](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/dinov2.md)).

**Anomaly.** PyOD 3 rebrands as "Agentic Anomaly Detection At Scale" "while keeping the classic fit/predict API fully backward-compatible" ([PyOD README](https://github.com/yzhao062/pyod/blob/master/README.rst)). `ECOD(contamination=0.1, n_jobs=1)` has no other parameters ([ecod.py](https://github.com/yzhao062/pyod/blob/master/pyod/models/ecod.py)).

**Uplift.** causalml's `qini_score`/`auuc_score(df, outcome_col='y', treatment_col='w', treatment_effect_col='tau', ...)` returns "the Qini score for each model estimate column". `tau` is only for synthetic data with a known true effect. The docstring notes that AUUC's random baseline "is about 0.5" and that "No p-value is reported for AUUC" ([visualize.py](https://github.com/uber/causalml/blob/master/causalml/metrics/visualize.py)). `control_name` defaults to `0` ([tlearner.py](https://github.com/uber/causalml/blob/master/causalml/inference/meta/tlearner.py)).

**Spatial.** scikit-learn 1.9.1's `GroupKFold` now takes `shuffle` and `random_state` ([_split.py @1.9.1](https://github.com/scikit-learn/scikit-learn/blob/1.9.1/sklearn/model_selection/_split.py)). Verde provides `BlockKFold(spacing=..., n_splits=5, ...)` ([verde model_selection.py](https://github.com/fatiando/verde/blob/main/verde/model_selection.py)).

### Inference (ours)

Use the same `h`, `n_windows` and `step_size` in statsforecast and mlforecast. Their cross-validation frames can then be merged on (`unique_id`, `ds`, `cutoff`), which gives a fair comparison against seasonal naive. The fastest strong baseline for text and image is frozen embeddings plus `LogisticRegression`. The 2026 code-break risk is copying old snippets that pass `penalty=`, `n_jobs=` or `multi_class=`. Because scikit-learn 1.9 and causalml 0.17 need Python 3.11+, an environment on Python 3.10 silently installs older versions with different defaults, so check `sklearn.__version__`. scikit-uplift still installs but is stale, so prefer causalml for Qini and AUUC. Report PR-AUC next to the positive base rate, because a random scorer's average precision is about equal to prevalence (a standard fact; no source fetched). For spatial blocks, project coordinates to metres first (geopandas `to_crs`, not checked against docs). Choosing method and block size was covered in the earlier ML-methods report and is not repeated here.

### Ready-to-use code (tested by us, 2026-10-02)

**Test status.** We ran these blocks on synthetic data on Python 3.12.3 with the pinned [`PROJECT_TEMPLATE/requirements.txt`](../PROJECT_TEMPLATE/requirements.txt) plus statsforecast 2.0.1, mlforecast 1.0.2, utilsforecast 0.2.15, pyod 3.6.6, causalml 0.17.0 and verde 1.9.0 (the versions pip resolved alongside the pins; newer releases exist).

| Recipe | Status |
| --- | --- |
| Time series (statsforecast + mlforecast + utilsforecast) | ✅ ran unchanged; the `utilsforecast.losses` import works |
| Text: TF-IDF + logistic regression | ✅ ran unchanged |
| Text: sentence-transformers embeddings | ⏳ not run (needs PyTorch) |
| Image: timm / DINOv2 embeddings | ⏳ not run (needs PyTorch) |
| Anomaly: PyOD ECOD + IForest | ✅ ran unchanged |
| Uplift: T-learner + causalml Qini/AUUC | ✅ ran unchanged |
| Spatial block CV (GroupKFold and Verde) | ✅ ran unchanged |

The synthetic-data scores only prove the code runs; they say nothing about which method wins on your data.

**Time series: seasonal naive and AutoETS against LightGBM on the same rolling origins.** Assumes `df` is long format with columns `unique_id`, `ds`, `y`.
```python
# tested 2026-10-02
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive, AutoETS
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean
import lightgbm as lgb
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, rmse

H, W = 14, 4  # horizon, number of rolling origins
sf = StatsForecast(models=[SeasonalNaive(season_length=7), AutoETS(season_length=7)], freq='D', n_jobs=-1)
cv_sf = sf.cross_validation(h=H, df=df, n_windows=W, step_size=H)
mlf = MLForecast(models=[lgb.LGBMRegressor(verbosity=-1)], freq='D', lags=[7, 14],
                 lag_transforms={7: [RollingMean(window_size=28)]}, date_features=['dayofweek'])
cv_ml = mlf.cross_validation(df=df, n_windows=W, h=H, step_size=H)
cv = cv_sf.merge(cv_ml.drop(columns='y'), on=['unique_id', 'ds', 'cutoff'])
print(evaluate(cv.drop(columns='cutoff'), metrics=[mae, rmse]).groupby('metric').mean(numeric_only=True))
```

**Text: TF-IDF + logistic regression, and sentence embeddings + linear head.** Assumes `texts` (a list of strings) and `y` (labels).
```python
# TF-IDF part tested 2026-10-02; embedding part not run (needs PyTorch)
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                    LogisticRegression(C=4.0, max_iter=2000, class_weight='balanced'))  # no penalty=/n_jobs= (deprecated 1.8)
print(cross_val_score(clf, texts, y, cv=5, scoring='f1_macro').mean())

from sentence_transformers import SentenceTransformer
enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
X = enc.encode(list(texts), batch_size=64, normalize_embeddings=True, show_progress_bar=True)  # kwargs UNVERIFIED
print(cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=5, scoring='f1_macro').mean())
```
Keeping TF-IDF inside the pipeline refits the vocabulary on each fold, so nothing leaks across folds (inference, ours).

**Image: frozen timm embeddings + logistic regression.** Assumes `train_paths`, `test_paths` (lists of image file paths) and `ytr` (training labels).
```python
# not run by us (needs PyTorch)
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
# DINOv2 swap: m = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval()  # output form UNVERIFIED
```

**Anomaly: ECOD and Isolation Forest, scored with PR-AUC against the base rate.** Assumes `X_train`, `X_test`, and binary labels `y_test` (1 = anomaly) used only for scoring.
```python
# tested 2026-10-02
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from sklearn.metrics import average_precision_score
for name, det in [('ECOD', ECOD()), ('IForest', IForest(n_estimators=300, random_state=0))]:
    det.fit(X_train)                      # unsupervised: labels ignored
    s = det.decision_function(X_test)     # higher = more anomalous
    print(name, 'PR-AUC', average_precision_score(y_test, s), 'base rate', y_test.mean())
```

**Uplift: T-learner + causalml Qini/AUUC.** Assumes train/test splits of features `X`, binary treatment `w` and binary outcome `y`, as arrays.
```python
# tested 2026-10-02
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from causalml.metrics.visualize import qini_score, auuc_score
m1 = HGB().fit(X_tr[w_tr == 1], y_tr[w_tr == 1])
m0 = HGB().fit(X_tr[w_tr == 0], y_tr[w_tr == 0])
uplift = m1.predict_proba(X_te)[:, 1] - m0.predict_proba(X_te)[:, 1]
df_eval = pd.DataFrame({'y': y_te, 'w': w_te, 't_learner': uplift})  # one column per model; no 'tau' on real data
print(qini_score(df_eval, outcome_col='y', treatment_col='w'))
print(auuc_score(df_eval, outcome_col='y', treatment_col='w'))
```

**Spatial: block CV without extra dependencies, plus the Verde option.** Assumes `df` has projected coordinates `x`, `y` in metres, and `model`, `X`, `y` are your estimator, features and target.
```python
# tested 2026-10-02
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_score
block = 5_000  # metres in a projected CRS; set >= residual autocorrelation range
groups = (np.floor(df.x / block).astype(int).astype(str) + '_' +
          np.floor(df.y / block).astype(int).astype(str))
scores = cross_val_score(model, X, y, cv=GroupKFold(n_splits=5), groups=groups)
# Verde alternative: from verde import BlockKFold; BlockKFold(spacing=5_000, n_splits=5).split(coords_array)
```

### API pitfalls list

| Library | Old pattern that breaks or warns | Current pattern | Status |
|---|---|---|---|
| scikit-learn ≥1.8 | `LogisticRegression(penalty='l2', n_jobs=..., multi_class=...)` | `l1_ratio` (0 = L2, 1 = L1) and `C` (`np.inf` = no penalty); drop `n_jobs` and `multi_class` | FULL-TEXT source |
| scikit-learn 1.9 | Running on Python 3.10 | Needs Python ≥3.11; check `sklearn.__version__` | PyPI |
| pandas ≥2.2 / statsforecast | `freq='M'` | `freq='ME'` | README |
| statsforecast | `StatsForecast(df=..., ...)` in the constructor | `forecast(h, df)` / `cross_validation(h, df, ...)` | FULL-TEXT source |
| statsforecast | `SeasonalNaive()` | `SeasonalNaive(season_length=7)` (required) | FULL-TEXT source |
| mlforecast | `window_ops` functions such as `rolling_mean` | `mlforecast.lag_transforms.RollingMean(window_size=...)` | README |
| mlforecast | `cross_validation(df)` | `n_windows` and `h` required; future exogenous variables go in `X_df` at predict time | FULL-TEXT source |
| utilsforecast | Pooled metrics while a `cutoff` column is present | Drop `cutoff` before `evaluate` (as in the recipe, which runs) | Tested (ours) |
| sentence-transformers 6 | Old torch or transformers versions | Python 3.10+, PyTorch 2.2+, transformers v5+ | README |
| causalml | Passing `tau` on real data | `tau` only for synthetic data; give `outcome_col` and `treatment_col` | FULL-TEXT docstring |
| causalml | String treatment labels | `control_name='control'` (default is `0`) | FULL-TEXT source |
| causalml | Expecting an AUUC p-value | Only Qini has a p-value; AUUC's random baseline is about 0.5 | FULL-TEXT docstring |
| scikit-uplift | `sklift` as the default | Last release Aug 2022; prefer causalml | PyPI |
| Spatial CV | Blocking on lat/lon degrees | Project to metres first | Inference (ours) |
