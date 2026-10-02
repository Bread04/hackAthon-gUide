# Read the rules, pin the APIs, guard the data

A datathon guide can now fill its five remaining gaps with mostly checkable material, but the evidence is uneven, and the labels below say how far to trust each part. **Kaggle-style competitions run on one standard rules template.** Its main terms are one account per person, no private sharing of code or data outside your team, external data allowed by default only if it is "reasonably accessible to all" at "minimal cost", and ranking decided solely by the Private Leaderboard. Winners must hand over reproducible training and inference code under a licence that varies by competition. In-person datathons such as ASA DataFest and Citadel/Correlation One run on NDAs or public-data-only terms and are judged on a report, not a leaderboard. All six non-tabular baselines (time series, text, image, anomaly, uplift, spatial CV) can be built on libraries released in 2026. The 2026 pitfalls are concrete: scikit-learn 1.8+ deprecates `LogisticRegression(penalty=...)`, pandas month-end is `'ME'`, mlforecast now uses class-based lag transforms, causalml's Qini wants one column per model, and scikit-uplift has had no release since 2022. For free compute, Kaggle is the most predictable free GPU. Two options that older guides still recommend have ended or reportedly ended: DigitalOcean's Student Pack credit (confirmed) and SageMaker Studio Lab (single snippet). On privacy, pseudonymised data is still personal data under GDPR, and a credentialed health dataset such as MIMIC forbids pasting rows into ChatGPT. Hardware prizes are mostly judged on general criteria with a "does the physical interaction work" emphasis, solo entry is usually allowed, and university rules converge on "no code before kickoff, AI allowed if disclosed, code must stay public". **Evidence caveat:** the sandbox blocked kaggle.com, Colab, Hugging Face, Devpost, HHS, GDPR sites and most vendor pages. Primary text reached us only through GitHub copies, source files and PyPI. Every limit and price is **RE-CHECK**, section 4 is **not legal advice**, and every recipe in section 2 is **not run by us**.

**How to read the labels.** **FULL-TEXT** means the page or file was read in full, usually a GitHub copy or source file. **SNIPPET-ONLY** means the claim comes from a search-engine summary, not the page itself, so do not quote it as verbatim. **UNVERIFIED** means it could not be checked at all, or rests on background knowledge. **RE-CHECK** marks a limit, quota or price the vendor can change at any time. **CONFLICT** marks sources that disagree. "Inference (ours)" is the researchers' own synthesis, not a rule or standard. This report does not repeat the earlier gap-fill reports on AI disclosure, IP clauses, hardware build tips, accessibility, or method selection. It adds datathon rules, runnable-shaped code, compute, PII handling, and hardware judging and winners.

## 1. Datathon rules: Kaggle's template allows external data by default but bans private sharing outright

### What is sourced

The Kaggle text below comes from two 2026 verbatim copies of official rules pages that participants committed to GitHub: the NVIDIA Nemotron 3 Reasoning Challenge and Playground Series S6. They match almost word for word, which is good evidence that this is the current template. They are still third-party copies, so treat them as reliable but not primary (**FULL-TEXT** of the copies) ([Nemotron rules copy](https://github.com/ahb-sjsu/agi-hpc/blob/HEAD/benchmarks/nemotron-rules.txt); [Playground S6 copy](https://github.com/leechanwoo-kor/kaggle-playground-series-s6e3/blob/HEAD/docs/rules.md)).

**Accounts and teams.** Submitting "through more than one Kaggle account" means disqualification (§3.5a). Each person may join only one team, and the team cap is set per competition: 5 for Nemotron and 3 for Playground S6. A merged team must have "a total Submission count less than or equal to the maximum allowed as of the Team Merger Deadline", where the maximum is the daily cap multiplied by the number of days the competition has run (§3.5c) ([Nemotron](https://github.com/ahb-sjsu/agi-hpc/blob/HEAD/benchmarks/nemotron-rules.txt)).

**Submissions and leaderboards.** Both 2026 copies allow **5 submissions per day and 2 final selections**. If you do not select, Kaggle picks the final submissions automatically. "The potential winner(s) are determined solely by the leaderboard ranking on the Private Leaderboard". Participants are not told which test rows are public and which are private, and ties go to the submission entered first ([Nemotron §3.7, §3.18](https://github.com/ahb-sjsu/agi-hpc/blob/HEAD/benchmarks/nemotron-rules.txt)).

**CONFLICT: caps and selection rules vary.** One team's notes record a live cap of 1 per day in a 2026 agent competition (SNIPPET-ONLY) ([dalloliogm notes](https://github.com/dalloliogm/kaggle_competitions/blob/HEAD/competitions/autonomous-agent-prediction-beta/AGENTS.md)), and WiDS 2020 allowed 10 per day (SNIPPET-ONLY). Kaggle's 2026 "kaggriculture" simulation competition overrode "select up to two" with "only the latest 2 submissions are tracked", and ranked teams with a post-deadline Bradley-Terry tournament ([participant notes quoting official pages](https://github.com/romansvet/kaggriculture/blob/HEAD/docs/strategy/2026-09-26-rules1.md)).

**Code, data and labelling.**

- Hand-labelling or human prediction of validation or test records is banned outside Hackathon-format competitions (§3.4b).
- "Privately sharing code or data outside of Teams is not permitted" (§3.5d). Public sharing is allowed only on that competition's Kaggle forum or notebooks, and doing so licenses the code under an OSI-approved licence "that in no event limits commercial use" (§3.6b).
- Any open-source code inside your model must also carry a licence that does not limit commercial use (§3.6c).
- AutoML is allowed if you hold an appropriate licence for it (§2.6c).

([Nemotron](https://github.com/ahb-sjsu/agi-hpc/blob/HEAD/benchmarks/nemotron-rules.txt))

**External data and models.** These are "acceptable unless specifically prohibited by the Host", provided they are "reasonably accessible to all" and of "minimal cost". The template's own examples: a small LLM subscription is reasonable, but a proprietary dataset licence costing more than a prize is not (§2.6a/b) ([Nemotron](https://github.com/ahb-sjsu/agi-hpc/blob/HEAD/benchmarks/nemotron-rules.txt); [Playground S6](https://github.com/leechanwoo-kor/kaggle-playground-series-s6e3/blob/HEAD/docs/rules.md)).

**CONFLICT: older default.** WiDS 2020 used a stricter default: "you may not use data other than the Competition Data" unless the competition website says otherwise. The current WiDS rules could not be read, so the WiDS position for 2025/26 is UNVERIFIED (SNIPPET-ONLY, kaggle.com/c/widsdatathon2020/rules). A data-security clause in both copies forbids giving Competition Data to non-participants, even when the data is CC BY 4.0 ([Playground S6](https://github.com/leechanwoo-kor/kaggle-playground-series-s6e3/blob/HEAD/docs/rules.md)).

**Winner obligations.** Winners must deliver training code, inference code and an environment description, plus a reproducible write-up, and must license the code under the competition's winner licence. Observed winner licences:

- CC BY 4.0 for Nemotron and "None" for Playground S6 (both from the FULL-TEXT copies).
- Apache 2.0 and MIT in other competitions (SNIPPET-ONLY).

Data or models with incompatible licences are exempt from the open-source grant. Winners must answer the winner notification within 1 week and return prize documents within 2 weeks or forfeit. Team prize money is split evenly unless the team unanimously agrees otherwise (§2.5, §2.8, §3.8–3.9) ([Nemotron](https://github.com/ahb-sjsu/agi-hpc/blob/HEAD/benchmarks/nemotron-rules.txt)).

**Dataset licences** (licence texts are FULL-TEXT via SPDX; the CC deed pages were blocked):

- **CC BY 4.0** allows any use with attribution ([SPDX CC-BY-4.0](https://github.com/spdx/license-list-data/blob/main/text/CC-BY-4.0.txt)).
- **CC BY-NC** bars use "primarily intended for or directed towards commercial advantage or monetary compensation" ([SPDX CC-BY-NC-4.0](https://github.com/spdx/license-list-data/blob/main/text/CC-BY-NC-4.0.txt)).
- **CC BY-SA** adds share-alike to adapted material ([SPDX CC-BY-SA-4.0](https://github.com/spdx/license-list-data/blob/main/text/CC-BY-SA-4.0.txt)).
- **ODbL** distinguishes two cases. A "Produced Work" (a chart, model output or dashboard) does not create a Derivative Database, but publishing it requires a notice. A publicly used Derivative Database must stay under ODbL or a compatible licence, and you must offer it, or a file of the alterations, in machine-readable form (§4.3–4.6) ([SPDX ODbL-1.0](https://github.com/spdx/license-list-data/blob/main/text/ODbL-1.0.txt)).

**In-person datathons** (all SNIPPET-ONLY; pages blocked):

- **ASA DataFest:** sign an NDA, store the data securely, "delete all data from thumb drives, hard drives, etc." at the end, and do not name the data source in publicity until all DataFest events are complete ([Penn State DataFest FAQ](https://datafest.psu.edu/faqs/); [ASA DataFest in a box](https://ww2.amstat.org/education/datafest/datafestinabox.cfm)).
- **Citadel datathons:** only "publicly-available data sources" plus data Citadel provides, with no material non-public information or former-employer data ([Citadel terms PDF](https://www.citadel.com/wp-content/uploads/2026/03/Citadel-High-School-Terminals-Programs-Certification-and-Terms.pdf)). The Women's and Summer Invitationals are open to undergraduates aged 18+ at US or Canadian universities, graduating between December 2026 and June 2028 ([Citadel datathons](https://www.citadel.com/careers/programs-and-events/datathons/)). The deliverable is reported as about a 15-page report plus code (secondary source).
- **WiDS:** teams of up to 4, with at least half identifying as women ([WiDS](https://www.widsworldwide.org/learn/datathon/)).

**Disqualification precedent.** The best-documented case is PetFinder.my (announced January 2020). The first-place team hid about 3,500 scraped samples with leaked test labels inside an "external" Pixabay dataset, and swapped in the leaked labels for only about 1 in 10 pets to avoid suspicion. It was caught when a later participant got access to the winning code to put it into production. The team was removed from the leaderboard, and the key member, a Grandmaster, was banned permanently (SNIPPET-ONLY) ([TDS/Medium](https://medium.com/data-science/kaggle-1st-place-winner-cheated-10-000-prize-declared-irrecoverable-bb7e1b639365); [Vice](https://www.vice.com/en/article/kaggle-data-science-community-rocked-by-pet-adoption-contest-cheating-scandal/); [The Register](https://www.theregister.com/2020/01/21/ai_kaggle_contest_cheat/)). The rules allow disqualification for "cheating, deception, or other unfair playing practices", along with removal from the leaderboard and loss of points or medals (§3.8d–e) ([Nemotron](https://github.com/ahb-sjsu/agi-hpc/blob/HEAD/benchmarks/nemotron-rules.txt)). **We found no accessible, named case of disqualification specifically for multiple accounts or private sharing.** Any claim that these are "common" is UNVERIFIED.

### Inference (ours)

The competition-specific rules override the general template, so the daily cap, the final-selection mechanism and the external-data default must be read for each competition, not assumed. The merger cap means a late merge between two teams that have both used most of their submissions can be blocked, so merge early. Pick your final submissions yourself; we could not confirm what Kaggle's automatic pick chooses. PetFinder shows that hosts inspect winning code after the competition, so declare every external dataset publicly in the forum. A model under a non-commercial licence may clash with §3.6c unless the host allows it; check the forum for host clarifications. Whether model weights trained on CC BY-NC data count as "Adapted Material" is legally unsettled and was not researched. For DataFest-style NDA events, a portfolio repo should hold code only, never data.

### Ready-to-use checklist

- [ ] Read the competition-specific rules section first; it overrides the template.
- [ ] Record these terms:
  - team cap and merger deadline
  - daily submission cap
  - number of final selections and how they are chosen (selected by you, or "latest N")
  - data-use tier (Competition Use / Non-Commercial / Commercial)
  - winner licence
  - external-data and pretrained-model policy
- [ ] Use one Kaggle account per person, ever, and only one team per person.
- [ ] Share code only publicly on the competition's forum or notebooks, never privately with another team.
- [ ] Do not hand-label or manually predict test rows.
- [ ] Post every external dataset or model in the forum. Confirm it is free or cheap and accessible to all.
- [ ] Keep dependencies under licences that do not restrict commercial use (no GPL or non-commercial code unless the host allows it).
- [ ] Select your final submissions manually before the deadline: one CV-best and one diverse.
- [ ] From day 1, keep a reproducible repo with training code, inference code, an environment file and a write-up skeleton.
- [ ] For winners: reply to the winner notification within 1 week and return prize documents within 2 weeks.
- [ ] Note the data licence's consequences: CC BY means attribute; CC BY-NC means no product built on it; CC BY-SA or ODbL means published cleaned data keeps the same licence; an ODbL chart or model output needs a notice; a DUA or NDA means code only and delete the data afterwards.
- [ ] Never repost raw Competition Data anywhere during the competition.

## 2. Baseline recipes: six task types, current APIs, and the breaking changes to watch

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

### Ready-to-use code (assembled recipes, not run by us, UNVERIFIED at runtime)

**Time series: seasonal naive and AutoETS against LightGBM on the same rolling origins.** Assumes `df` is long format with columns `unique_id`, `ds`, `y`.
```python
# not run by us
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive, AutoETS
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean
import lightgbm as lgb
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, rmse   # UNVERIFIED import path

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
# not run by us
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
# not run by us
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
# not run by us
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
# not run by us
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
# not run by us
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
| utilsforecast | Pooled metrics while a `cutoff` column is present | Drop `cutoff` or set `cutoff_col` | UNVERIFIED (ours) |
| sentence-transformers 6 | Old torch or transformers versions | Python 3.10+, PyTorch 2.2+, transformers v5+ | README |
| causalml | Passing `tau` on real data | `tau` only for synthetic data; give `outcome_col` and `treatment_col` | FULL-TEXT docstring |
| causalml | String treatment labels | `control_name='control'` (default is `0`) | FULL-TEXT source |
| causalml | Expecting an AUUC p-value | Only Qini has a p-value; AUUC's random baseline is about 0.5 | FULL-TEXT docstring |
| scikit-uplift | `sklift` as the default | Last release Aug 2022; prefer causalml | PyPI |
| Spatial CV | Blocking on lat/lon degrees | Project to metres first | Inference (ours) |

## 3. Free compute: Kaggle is the dependable free GPU, and two well-known options have closed

### What is sourced

Every vendor page was blocked, so this section is **SNIPPET-ONLY and every figure is RE-CHECK** unless marked otherwise.

**Kaggle Notebooks.** About **30 GPU-hours a week** (a "floating" quota that can go higher depending on demand) and 20 TPU-hours, with sessions of up to 12 h (9 h for TPU). The GPU is either 1× P100 (16 GB) or 2× T4. Machines have 4 CPU cores and 29–30 GB RAM. Up to **20 GB** of output persists in `/kaggle/working` ([Kaggle Notebooks docs](https://www.kaggle.com/docs/notebooks); [floating quota thread](https://www.kaggle.com/product-feedback/173129)). New notebooks default to Private, and **making a notebook Public is permanent** ([Kaggle privacy launch](https://www.kaggle.com/product-feedback/34719)). A Secrets add-on holds API keys, and forks of a public notebook do not inherit them ([Kaggle secrets thread](https://www.kaggle.com/general/414523)). Competition scoring kernels often run without internet; the usual workaround is to chain an internet-on training notebook into an offline inference notebook ([Kaggle competitions setup](https://www.kaggle.com/docs/competitions-setup); [TDS chaining kernels](https://towardsdatascience.com/easy-kaggle-offline-submission-with-chaining-kernels-30bba5ea5c4d/)). The private dataset quota is UNVERIFIED.

**Google Colab (free).** Google's own FAQ says limits are "dynamic", with no guaranteed resources. VMs are private to your account and are deleted after idling or at a maximum lifetime, and files are lost when the session ends. Mounting Drive gives **any code in the notebook access to your whole Drive** ([Colab FAQ](https://research.google.com/colaboratory/faq.html)). Third-party reports describe a T4, sessions of about 12 h, an idle disconnect of about 90 minutes, and roughly 15–30 GPU-hours a week (not official) ([Thunder Compute, Sept 2026](https://www.thundercompute.com/blog/colab-alternatives-for-cheap-deep-learning-in-2025)).

**CONFLICT: Colab Pro for Education.** Free Pro for students at US institutions: one snippet says new signups are closed, while Swarthmore IT advertised it as available in January 2026 ([Google blog](https://blog.google/outreach-initiatives/education/colab-higher-education/); [Swarthmore ITS](https://blogs.swarthmore.edu/its/2026/01/13/google-colab-free-for-students-and-faculty/)).

**Hugging Face ZeroGPU.** **CONFLICT:** sources disagree on the free quota, giving 3.5 minutes a day in one and 5 minutes in another. PRO costs about $9 a month for about 40 minutes a day ([HF ZeroGPU docs](https://huggingface.co/docs/hub/en/spaces-zerogpu); [HF forum](https://discuss.huggingface.co/t/zero-gpu-daily-quota/168376)). It is for hosting demos, not training.

**Lightning AI.** The free tier gives 15 credits a month and one Studio that restarts every 4 h. **CONFLICT:** one source says 80 interruptible GPU-hours a month, another about 22 T4-hours ([aicreditmart](https://aicreditmart.com/ai-credits-providers/lightning-ai-free-plan-22-gpu-hours-month-guide-2026/); [Lightning](https://lightning.ai/on-demand-gpus)).

**GitHub Codespaces.** The free plan gives **120 core-hours and 15 GB a month**, CPU only ([GitHub community FAQ](https://github.com/orgs/community/discussions/38697)).

**Closed or uncertain options.**

- **Paperspace:** being absorbed into DigitalOcean (the Gradient API was deprecated in July 2024), and its free tier is UNVERIFIED ([DO docs](https://docs.digitalocean.com/products/paperspace/notebooks/)).
- **SageMaker Studio Lab:** reportedly closed on 30 July 2026. This rests on a **single snippet**, so it is UNVERIFIED ([Thunder Compute](https://www.thundercompute.com/blog/colab-alternatives-for-cheap-deep-learning-in-2025)).
- **DigitalOcean Student Pack credit (FULL-TEXT, confirmed):** the $200 offer closed for redemption on **31 July 2026**, and all credits expired on **1 August 2026** ([GitHub community discussion #201240](https://github.com/orgs/community/discussions/201240)).

### Inference (ours)

Kaggle's documented 12-hour sessions and weekly quota make it the safest default for GPU work over a weekend. A team of 3–4 has 90–120 GPU-hours if each member runs on their own account, but check that the event rules allow it. Colab suits fast prototyping, not critical overnight runs. Codespaces or a Lightning CPU Studio is a reproducible shared dev box, and most tabular datathons need no GPU. ZeroGPU fits a final Gradio demo. Guides that still point students to DigitalOcean credits or Studio Lab are out of date. If the data is under an NDA or DUA, or contains PII or health data, treat every one of these platforms as a third-party processor and use only what the organisers approve. The vendors' terms of service on training and data residency were not retrieved.

### Ready-to-use checklist

- [ ] Before the event, re-check each platform's current quota page; every number here is RE-CHECK.
- [ ] Check whether the data's licence or DUA allows uploads to Kaggle, Colab, HF or Lightning at all.
- [ ] Code lives in a shared Git repo from minute one. Data and checkpoints live in persistent storage (`/kaggle/working`, Drive, HF Hub). Nothing important lives only in a runtime.
- [ ] Pin `requirements.txt` with `==` versions. Record the Python, CUDA and key library versions in the README.
- [ ] Checkpoint every N minutes or epochs, and make training resumable.
- [ ] Run long Kaggle jobs as "Save & Run All" versions, not interactive sessions.
- [ ] For offline competitions: download wheels and weights in an internet-on notebook, save them as a dataset, and attach that dataset to the offline inference notebook.
- [ ] Keep API keys in Kaggle Secrets or environment variables, never in cells. Keep notebooks Private, since publishing is irreversible.
- [ ] Do not mount Drive in notebooks you did not write.
- [ ] Add raw-data paths to `.gitignore`. Never push data to a public repo.

## 4. PII handling: pseudonymised is still personal, and a dataset agreement can rule out hosted LLMs

**Not legal advice.** GDPR applies in the EU/EEA, and UK GDPR is near-identical. HIPAA applies only to US covered entities and their business associates, but its Safe Harbor list is a useful practical checklist. Most legal and policy text below is SNIPPET-ONLY because the primary sites were blocked. Policies change, so date-stamp this section and check current terms.

### What is sourced

GDPR Art. 4(1) defines personal data as information relating to a person identifiable "directly or indirectly", including by "location data" or "an online identifier" (SNIPPET-ONLY) ([gdpr-text.com Art. 4](https://gdpr-text.com/read/article-4/)). Recital 26 says pseudonymised data that can be re-attributed "should be considered to be information on an identifiable natural person". Only data rendered anonymous so that the person "is not or no longer identifiable" falls outside GDPR (SNIPPET-ONLY) ([Recital 26](https://gdpr-info.eu/recitals/no-26/)). The ICO calls pseudonymisation "effectively only a security measure" (SNIPPET-ONLY; the exact attribution is uncertain) ([ICO](https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/personal-information-what-is-it/what-is-personal-data/what-is-personal-data/)). The Art. 9 special categories, such as health, biometrics and ethnicity, need an extra legal basis (UNVERIFIED; page blocked).

HIPAA allows de-identification by two routes, Safe Harbor or Expert Determination (UNVERIFIED; [HHS guidance](https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html) was blocked). Safe Harbor removes **18 identifier types**. The less obvious rules are:

- ZIP codes may keep only the first 3 digits, and only if that 3-digit area has more than 20,000 people; otherwise use "000".
- All date elements except the year are removed.
- Ages over 89 are grouped as "90 or older".
- IP addresses, URLs, device serial numbers, full-face photos, and "any other unique identifying number, characteristic or code" are all on the list.

Safe Harbor also requires that the covered entity have no actual knowledge that the remaining data could identify someone (SNIPPET-ONLY; items 9–18 from secondary lists) ([CASRAI](https://casrai.org/guides/18-hipaa-identifiers); [AccountableHQ](https://www.accountablehq.com/post/hipaa-s-18-identifiers-the-phi-safe-harbor-list-explained)).

**Why removing names is not enough.**

- Sweeney estimated that **87%** of the US population (1990 Census) was likely unique on {5-digit ZIP, sex, full date of birth}, and re-identified Governor Weld's hospital records ([Sweeney](https://www.researchgate.net/publication/267716853_Simple_Demographics_Often_Identify_People_Uniquely); [EFF](https://www.eff.org/deeplinks/2009/09/what-information-personally-identifiable)). **CONFLICT:** Golle's re-analysis of the 2000 Census gives a lower figure, about 63% (UNVERIFIED) ([Golle](https://www.researchgate.net/publication/221342213_Revisiting_the_Uniqueness_of_Simple_Demographics_in_the_US_Population)).
- Netflix Prize: with "8 movie ratings (of which 2 may be completely wrong) and dates that may have a 14-day error, 99% of records can be uniquely identified" ([Narayanan and Shmatikov](https://www.cs.cornell.edu/~shmat/shmat_oak08netflix.pdf)).
- AOL: the New York Times identified searcher No. 4417749 from her search terms alone ([NYT PDF copy](https://www2.hawaii.edu/~strev/ICS614/materials/NYT%20-%20confidentiality%20-%20A%20Face%20is%20Exposed%20for%20AOL%20Searcher%20%202006-08-24.pdf)).

All three are SNIPPET-ONLY. A 2026 paper on LLM-driven deanonymisation exists, but only its title was seen ([arXiv 2602.16800](https://arxiv.org/html/2602.16800), UNVERIFIED findings).

**Tools** (versions from PyPI, FULL-TEXT; READMEs read directly):

- **Presidio 2.2.364** (July 2026, MIT) is moving from Microsoft to the community-run "Data Privacy Stack" organisation, and its Docker images move to `ghcr.io/data-privacy-stack/presidio-*` ([transition doc](https://github.com/microsoft/presidio/blob/main/docs/project_transition.md)). Its README warns "there is no guarantee that Presidio will find all sensitive information" ([Presidio README](https://github.com/microsoft/presidio/blob/main/README.MD)).
- **scrubadub 2.0.1** dates from September 2023 ([PyPI](https://pypi.org/pypi/scrubadub/json)).
- **Faker 40.40.0** (September 2026, MIT) can be seeded so it returns the same values each run ([Faker README](https://github.com/joke2k/faker/blob/master/README.rst)).
- **SDV 1.38.5** is under the **Business Source License**, not OSI open source. You may use it except "for a Synthetic Data Service", and each release converts to MIT after four years ([SDV LICENSE](https://github.com/sdv-dev/SDV/blob/main/LICENSE)).

**LLM data-sharing** (all SNIPPET-ONLY, provider pages blocked; RE-CHECK):

- **OpenAI:** API data has not been used for training by default since 1 March 2023, but may be retained for up to 30 days for abuse monitoring. Zero Data Retention exists for eligible use cases ([OpenAI data controls](https://developers.openai.com/api/docs/guides/your-data)).
- **Anthropic:** its 2025 consumer-terms update made Free, Pro and Max chats eligible for training unless users opt out, with 5-year retention if opted in and 30 days if not. This does not apply to the API or other commercial-terms services ([Anthropic](https://www.anthropic.com/news/updates-to-our-consumer-terms)).
- **Gemini:** on the free API tier, prompts may be used for product improvement and read by human reviewers. The paid tier and Vertex AI are excluded ([Gemini API terms](https://ai.google.dev/gemini-api/terms)).
- **PhysioNet (MIMIC):** the credentialed DUA "explicitly prohibits... sending it through APIs provided by companies like OpenAI, or using it in online platforms like ChatGPT", and strongly recommends local LLMs. It names conditional exceptions: Azure OpenAI with human review opted out, Gemini via Vertex AI, and Claude. Check the live page for the exact list ([PhysioNet LLM guidance](https://physionet.org/news/post/llm-responsible-use/)).

### Inference (ours)

The workable ladder for a team, in order:

1. Drop columns you do not need.
2. Generalise quasi-identifiers: date of birth to year or age band, ZIP to 3 digits or region, dates to month or relative offsets.
3. Suppress rare combinations. k = 5 is a common rule of thumb, not a sourced standard.
4. Replace direct IDs with random or salted IDs. This is pseudonymisation, so the data is still personal data, and the key stays out of the repo.
5. Use Faker or SDV output for demos.

Hashing emails is not anonymisation: the same input always gives the same hash, so the result is linkable and often brute-forceable. Free text such as clinical notes, chats and support tickets is where PII hides, so run Presidio and then spot-check by hand. Ratings, locations, purchases and search queries act as fingerprints even with no names attached. SDV output trained on real records is not automatically anonymous, because outliers can be memorised, and the BSL rules out building a commercial synthetic-data service on it. For LLMs: "not training" is different from "not retaining", consumer chat apps and free API tiers are off-limits for real personal data, and a hosted API is acceptable only when the DUA allows it, on a paid or commercial tier, after de-identification, with a log of what was sent. Consent for user testing should be explicit and opt-in, covering what is recorded, why, who sees it, how long it is kept and how to withdraw. GDPR consent conditions are UNVERIFIED here, and no consent-form template was sourced.

### Ready-to-use checklist and code

- [ ] Read the dataset's licence or DUA **before** loading it anywhere. If it restricts sharing: no hosted LLM, no Colab, Kaggle or HF upload, no public repo, no chat apps.
- [ ] List direct identifiers (against the 18 Safe Harbor types) and quasi-identifiers (ZIP, date of birth, sex, rare categories, location traces, free text).
- [ ] Drop, generalise, suppress, then pseudonymise, in that order. Keep any re-identification key off the repo and off shared drives.
- [ ] Check the minimum group size on the quasi-identifiers before sharing any derived table.
- [ ] Run Presidio over free-text columns, then spot-check a sample by hand.
- [ ] For demos, screenshots and videos, use Faker personas or synthetic data, never real rows.
- [ ] LLM use: paid or commercial tier only, de-identified input only, provider named in the README, prompts logged. For credentialed health data, use a local model unless the DUA lists the provider.
- [ ] Delete data at the end if the agreement requires it, and record that you did.

```python
# Presidio quickstart, quoted from the official getting-started doc (FULL-TEXT source; not run by us)
# pip install presidio-analyzer presidio-anonymizer && python -m spacy download en_core_web_lg
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
text = "My phone number is 212-555-5555"
results = AnalyzerEngine().analyze(text=text, entities=["PHONE_NUMBER"], language='en')
print(AnonymizerEngine().anonymize(text=text, analyzer_results=results))

# k-anonymity spot check (assembled by us; not run; k=5 is a rule of thumb, UNVERIFIED as a standard)
QIS = ['zip3', 'age_band', 'sex']
assert df.groupby(QIS).size().min() >= 5, "rare quasi-identifier combinations: generalise or suppress"

# Faker demo personas (README pattern; not run by us)
from faker import Faker
Faker.seed(4321); fake = Faker()
demo_users = [{'name': fake.name(), 'address': fake.address()} for _ in range(50)]
```
Source for the Presidio code: [Presidio getting started](https://github.com/microsoft/presidio/blob/main/docs/getting_started/getting_started_text.md). Source for the Faker pattern: [Faker README](https://github.com/joke2k/faker/blob/master/README.rst).

## 5. Hardware judging, solo entry and university rules: a working physical loop wins, and so do solo hackers who cut scope

### What is sourced

**Hardware judging.** MLH's standard rules weight four criteria equally: Technology, Design, Completion and Learning. For hardware, Design "might be more about how good the human-computer interaction is". The criteria explicitly exclude pitch quality, code quality and idea novelty, and "judges are free to make decisions based on their gut feeling" (**FULL-TEXT**) ([MLH standard rules](https://github.com/MLH/mlh-policies/blob/main/standard-hackathon-rules.md)). Other events (all SNIPPET-ONLY):

- **StarkHacks** (Purdue, 36 h, billed as the largest hardware hackathon) judges Functionality and Execution, Technical Complexity, and Problem Relevance and Impact ([StarkHacks Devpost](https://starkhacks.devpost.com/)).
- **TreeHacks 2025** awarded its hardware prize for "integrating hardware components... from IoT devices to robotics" ([TreeHacks 2025](https://treehacks-2025.devpost.com/)).
- **Hackaday Prize 2023**, the latest rules found, scores Concept, Design, Production (reproducibility and manufacturing), Benchmark and Communication (documentation). This is the opposite emphasis from a weekend hackathon ([Hackaday Prize 2023 rules](https://cdn.hackaday.io/files/1901528135463168/Hackaday%20Prize%202023%20Official%20Rules.pdf)).

**Hardware winners** (winners' own READMEs, **FULL-TEXT**):

- **WiGhost**, Best Hardware Hack at Hack&Roll 2026: several ESP32 nodes that make Wi-Fi interference "visible, physical, and interactive" ([repo](https://github.com/soham131345/wighostviz)).
- **InkSight**, 1st Best Hardware Hack at GenAI Genesis (year UNVERIFIED): "RAG meets robotics" to read and chat with your notebook ([repo](https://github.com/AryanK1511/InkSight)).
- **OPTimism**, Hack the 6ix: a gyroscope for posture warnings and an ultrasonic sensor for screen distance, plus an AI chatbot and a sponsor API (Circle) ([repo](https://github.com/ricsign/OPTimism)).
- **SleepStop/CarSafety**, 3rd Overall plus Best Hardware Hack at Hack the Valley V: a webcam and OpenCV detect drowsy drivers ([repo](https://github.com/EmreCenk/car-safety)).
- **BrainViz**, MLH Best Hardware Hack at CodeJam: an EEG headset drives a Unity game, pitched for accessibility ([repo](https://github.com/zephirl/BrainViz)).
- **BrailleWare**, PennApps XII (2015) Grand Prize plus Best Hardware Hack ([repo](https://github.com/edhyah/BrailleWare)).
- **Axolotl**, a McHacks 9 hardware prize winner, was an 8-bit computer *simulation*, i.e. software only ([repo](https://github.com/Eerohne/Axolotl)).

Official 2024–26 hardware winners at TreeHacks, HackMIT and Hack the North could not be confirmed.

**Solo participation** (SNIPPET-ONLY unless noted):

- HackMIT 2023: "Teams can be made up of anywhere from 1 to 4 hackers" ([HackMIT rules](https://hack-mit-2023.devpost.com/rules)).
- Cal Hacks: "You can come with a team or solo" ([Cal Hacks](https://www.calhacks.io/)).
- HackHarvard 2025's judging prompt asks solo projects whether the hacker learned something new ([HackHarvard](https://hackharvard-2025.devpost.com/)).
- Hack the North 2025 caps teams at 4 and states no minimum ([HTN rules](https://hackthenorth2025.devpost.com/rules)).
- **CONFLICT:** PennApps caps teams at 4, but one snippet says "Team required: 2 to 4 members" ([PennApps XXIV](https://pennapps-xxiv.devpost.com/)).
- MLH sets no minimum team size and requires "All team members should actively participate" (FULL-TEXT) ([MLH](https://github.com/MLH/mlh-policies/blob/main/standard-hackathon-rules.md)).
- One solo-winner blog advises reusing familiar code, going for "low hanging fruit", and "If there are API sponsors at the event, utilize their software!" ([solo tips blog](https://jcacperalta.github.io/blog/winning-a-hackathon/)).
- A secondary post claims a solo developer won an Anthropic hackathon in 8 hours (UNVERIFIED) ([Medium](https://medium.com/@CodeCoup/this-solo-developer-won-the-anthropic-hackathon-in-8-hours-then-open-sourced-the-entire-ai-coding-85c555ccf6ac)).

**University rules** (three rule sets read in **FULL-TEXT**):

- **MLH standard rules:** "students" includes bootcamp attendees and people who graduated within the last 12 months. You may reuse an idea but not code. Open-sourcing a project early just to use it at the event is not allowed. Only "a few lines of code" of bug fixes are allowed at demo time. Code "must remain public post event to be eligible for prizes". AI is allowed with disclosure ([MLH](https://github.com/MLH/mlh-policies/blob/main/standard-hackathon-rules.md)).
- **TreeHacks Terms and Conditions** (Stanford, updated 27 August 2024): a zero-tolerance cheating policy ("No code may be written prior to the official start of hacking"), enforced by an internal detection system, manual review of all winners and a prize holding period. Penalties include a permanent ban and "notification to our sponsors and company recruiters". Organisers may share registration details and GitHub/LinkedIn profiles with sponsors. The document has no AI, IP or team-size clause, and it oddly cites "the laws of the Province of Ontario" ([TreeHacks T&C](https://github.com/TreeHacks/policies/blob/master/terms-and-conditions.md)).
- **UH CodeRED Astra 2025** (University of Houston) is an MLH fork that adds an 18+ age requirement and a precise 24-hour window ([CodeRED rules](https://github.com/UHCodeRED/rules)).
- **HackMIT 2023** (SNIPPET-ONLY) lets teams plan in advance but not code, and requires cited open source ([HackMIT](https://hack-mit-2023.devpost.com/rules)).

### Inference (ours)

Completion carries equal weight under MLH. One reliable sensing-to-feedback loop therefore beats an ambitious rig that fails at the demo, while the physical interaction collects extra Technology and Design credit. The winner pattern is cheap sensors or microcontrollers (ESP32, ultrasonic, gyroscope, webcam), one human use case (safety, health, accessibility or education), and an AI or sponsor-API layer. Several of these projects stacked the hardware prize with an overall placing. "Hardware" prizes are sometimes awarded loosely, as Axolotl shows, so read the prize text. Solo hackers trade parallel work for decision speed. Sponsor-API tracks and the MLH-style Learning criterion are their most realistic routes to a prize, and solo plus hardware is the hardest combination. None of the full-text university rule sets assigns IP to organisers, so the team presumably keeps ownership while the code stays public. Sponsor-prize terms still need a separate read. No aggregated data on solo win rates exists in the notes.

### Ready-to-use checklist

- [ ] **Hardware team:** find the rubric. If it is MLH-style, plan for a working demo (Completion) and a clear physical interaction (Design/HCI) rather than a polished pitch.
- [ ] **Hardware team:** scope to one sensor → logic (with AI or a sponsor API) → feedback loop around a named human problem, and open with one sourced statistic.
- [ ] **Hardware team:** read the hardware prize's own wording, and enter the general track as well, since prizes stack.
- [ ] **Solo:** confirm the event's minimum team size (it is 1 at HackMIT and Cal Hacks; PennApps is in conflict).
- [ ] **Solo:** use a stack you already know, aim at a sponsor-API prize, and cut scope to a single demo path.
- [ ] **Solo:** avoid combining solo entry with complex hardware.
- [ ] **University rules:** plan before kickoff but write no code before it. Start a fresh repo at the official start time.
- [ ] **University rules:** cite every library or template. Do not pre-publish a project "for the event".
- [ ] **University rules:** disclose AI tools in the submission and the README.
- [ ] **University rules:** keep the repo public after the event, or you may lose the prize.
- [ ] **University rules:** check age and eligibility (18+ at CodeRED; recent graduates count as students under MLH) and expect your profile to be shared with sponsors.
- [ ] **University rules:** read sponsor-prize terms separately for IP and licence grants.

## Conclusion

Across all five gaps, the decisive information sits in the specific event's own documents, not in general norms: the competition-specific rules, the data licence or DUA, the vendor's current quota page, the event's prize wording. The guide's most useful contribution is therefore a short list of fields to look up, plus defaults for when a field is silent. The silent-field defaults differ by format. Kaggle's template now allows external data and pretrained models by default, older or sponsored datathons often do not, and NDA events forbid even keeping the data. A team that carries habits from one format into another is the one most likely to break a rule without noticing.

The second lesson is how quickly this material goes stale. Within one year, scikit-learn deprecated the most-copied `LogisticRegression` argument, a well-known student cloud credit ended, a free notebook service reportedly closed, and Presidio changed maintainers. The guide should date-stamp sections 2–4, pin library versions in every recipe, and give each recipe a smoke test against the pinned environment before the event. The recipes here are checked against source but not executed, and the compute and policy figures rest mostly on snippets, so both deserve a re-check pass shortly before each event rather than permanent trust.
