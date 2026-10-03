# 🔍 Error Analysis & Experiment Tracking

<!-- markdownlint-disable MD013 -->

> Find out *where* your model fails and turn it into features or honest pitch points; then log every run so each pitch number points to one. Part A is error analysis, part B is tracking.
>
> 📚 Source research: [`technical-ml-gaps-2026-10-03`](../../_research/technical-ml-gaps-2026-10-03/research.md). **Label key used throughout.** **SNIPPET-ONLY** means the claim comes from a search-result snippet because the primary page could not be fetched; check the source before relying on it. **UNVERIFIED** means it comes from general knowledge or an unconfirmed attribution. **RE-CHECK** marks prices, quotas and limits that change often; confirm them on the day. **Vendor-reported** or **author-reported** marks numbers from the people selling or publishing the tool. Every code block is labelled **not run by us**: each is either quoted from the cited source or assembled from cited calls, and none was executed. All version and licence tables are dated **2026-10-03** and were taken from PyPI or npm JSON on that date. Nothing here is legal advice.

## A. Error analysis: tag 100 mistakes before you add a feature

### Plain-English explanation

Error analysis means looking at the specific rows your model gets wrong and asking why. The canonical method comes from Andrew Ng's *Machine Learning Yearning*, chapters 13–19 (all **SNIPPET-ONLY**; the PDF was blocked, [MLY PDF](https://home-wordpress.deeplearning.ai/wp-content/uploads/2022/03/andrew-ng-machine-learning-yearning.pdf), [KDnuggets summary](https://www.kdnuggets.com/2018/05/7-useful-suggestions-machine-learning-yearning.html)):

1. Gather about **100 misclassified validation examples**.
2. Tag each one in a spreadsheet. Rows are examples, columns are error categories that emerge as you look, and a comments column holds anything else.
3. Count the categories. Each category's share is the **ceiling** on what fixing it could gain: if only 5% of errors are blurry photos, a perfect blur fix gains at most 5% of the errors.

Ng also suggests splitting the dev set into an "eyeball" set you inspect by hand and a "blackbox" set you only score. At a 5% error rate, you need about 2,000 eyeball examples to get 100 errors.

Google's *Rules of ML* adds the next step. Rule #26 says "Look for patterns in the measured errors, and create new features" (**SNIPPET-ONLY**, [Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml)).

Tools automate the "find the bad group" step, but they do not replace looking:

- **Slicing** computes your metric per segment (region, device, age band).
- **Slice finding** searches for segment combinations you did not think of.
- **Label-noise detection** flags rows whose labels are probably wrong. Real benchmark datasets contain such errors ([labelerrors.com](https://labelerrors.com), linked from the [cleanlab README](https://raw.githubusercontent.com/cleanlab/cleanlab/master/README.md)).

Error analysis also gives you an honest **limitations slide**, which judges tend to reward. That last point is our inference.

### Decision table: which error-analysis tool for which question

| Question | Tool | Effort | Output |
|---|---|---|---|
| "Is the model worse for some known group?" | pandas `groupby` or fairlearn `MetricFrame` (any column works as `sensitive_features`) | Minutes | Per-group metric table with counts |
| "Which group did I not think to check?" | `sliceline` (Slicefinder) on binned features plus a 0/1 or absolute-residual error vector | ~15 min | Top-k worst slices |
| "Are some labels wrong?" | `cleanlab` `find_label_issues` or `Datalab` on out-of-fold `pred_probs` | ~15 min | Ranked suspect rows |
| "Does test look different from train?" | Evidently `DataDriftPreset`, or adversarial validation (see the guide's CV page) | ~15 min | HTML drift report |
| "One-click evaluation report for the deck" | Deepchecks `model_evaluation()` suite (**AGPL**) | 15–30 min if it installs | HTML report |
| "Interactive error tree and heatmap" | Microsoft `raiwidgets` `ErrorAnalysisDashboard` | 30+ min; stale releases | Notebook widget |
| "Why are these 100 rows wrong?" | Spreadsheet plus your eyes (Ng) | 1 hour | Error-category counts |

Sources: [fairlearn assessment docs](https://raw.githubusercontent.com/fairlearn/fairlearn/main/docs/user_guide/assessment/perform_fairness_assessment.rst), [sliceline README](https://raw.githubusercontent.com/DataDome/sliceline/main/README.rst), [cleanlab README](https://raw.githubusercontent.com/cleanlab/cleanlab/master/README.md), [Evidently README](https://raw.githubusercontent.com/evidentlyai/evidently/main/README.md), [deepchecks README](https://raw.githubusercontent.com/deepchecks/deepchecks/main/README.md) and the [RAI Error Analysis README](https://raw.githubusercontent.com/microsoft/responsible-ai-toolbox/main/docs/erroranalysis-dashboard-README.md). The effort column is our estimate.

### A 10-step 48-hour error-analysis routine (our synthesis; not a published checklist)

1. **Score on out-of-fold predictions.** Use OOF predictions, so every training row has an honest prediction (see the guide's CV page).
2. **Compare segments.** Build a per-segment metric table and show counts next to every metric, so small slices are not over-read.
3. **Find slices you missed.** Run Sliceline on binned features.
4. **Confusion by slice.** Compute a confusion matrix for each top slice.
5. **Read the worst errors.** Sort by loss (confident-wrong classifications, largest residuals), then hand-tag 50–100 of them (Ng).
6. **Check regression residuals.** Plot residuals against the prediction, key features and time:
   - A fan shape suggests a log target.
   - Bias within one slice suggests a missing feature (Rule #26).
7. **Check labels.** Run cleanlab on OOF `pred_probs`. Clean only training rows, never test or leaderboard data.
8. **Check drift.** Look for drift between train and test.
9. **Act on each top category.** Turn it into a feature, a segment-specific threshold, a data fix, or a decision to "accept and disclose".
10. **Put it on a slide.** For example: "X% worse on segment S (n=…), cause Y, fix Z gave +Δ."

The source for the data-centric loop in steps 7–9 is cleanlab's own suggested workflow: train an initial model, use it to diagnose data issues, retrain the same model on the improved data, and only then try other modelling techniques ([cleanlab README](https://raw.githubusercontent.com/cleanlab/cleanlab/master/README.md)).

### Minimal code (tested by us unless marked)

Per-segment table, quoted from the [fairlearn docs](https://raw.githubusercontent.com/fairlearn/fairlearn/main/docs/user_guide/assessment/perform_fairness_assessment.rst) with the doctest prompts removed. **Tested 2026-10-03** on the sample data (runs).

```python
# tested 2026-10-03 (fairlearn 0.14.0) — quoted from fairlearn docs
from fairlearn.metrics import MetricFrame, count, false_positive_rate, selection_rate
from sklearn.metrics import recall_score
my_metrics = {'tpr': recall_score, 'fpr': false_positive_rate, 'sel': selection_rate, 'count': count}
mf = MetricFrame(metrics=my_metrics, y_true=y_true, y_pred=y_pred, sensitive_features=sf_data)
mf.by_group; mf.difference(method='to_overall')
```

Automatic slices, from the [sliceline README](https://raw.githubusercontent.com/DataDome/sliceline/main/README.rst). **Tested 2026-10-03 (sliceline 0.3.0):** features must be binned to categories, and with default settings it returned no slices on our sample data; setting `min_sup` (minimum slice size), `k` and `max_l` made it work.

```python
# tested 2026-10-03 (sliceline 0.3.0) on the sample data
import pandas as pd
from sliceline.slicefinder import Slicefinder
X_binned = pd.DataFrame({                      # every column must be categorical
    "age": pd.qcut(X_val["age"], 4).astype(str),
    "discharge_to": X_val["discharge_to"].astype(str),
})
errors = (y_val != y_pred).astype(int)          # or abs(y_val - y_pred) for regression
sf = Slicefinder(alpha=0.95, k=3, max_l=2, min_sup=20).fit(X_binned, errors)
print(sf.top_slices_)                            # each row is a slice; None = "any value"
```

Label issues on out-of-fold probabilities. This is assembled from the cleanlab README and `filter.py`. The `return_indices_ranked_by="self_confidence"` call runs (tested). The requirement that `pred_probs` be out-of-sample; the latter matches the confident-learning setup but its docs page was blocked. **Tested 2026-10-03** on the sample data (runs).

```python
# tested 2026-10-03 (cleanlab 2.9.0): the ranked_by value below works
from sklearn.model_selection import cross_val_predict
from cleanlab.filter import find_label_issues
pred_probs = cross_val_predict(model, X, y, cv=5, method="predict_proba")
idx = find_label_issues(labels=y, pred_probs=pred_probs,
                        return_indices_ranked_by="self_confidence")
print(idx[:50])  # hand-review these rows first
# one-call audit alternative (README): lab = cleanlab.Datalab(data=df, label="label");
# lab.find_issues(features=emb, pred_probs=pred_probs); lab.report()
```

Drift report, quoted from the [Evidently README](https://raw.githubusercontent.com/evidentlyai/evidently/main/README.md). **Tested 2026-10-03** on the sample data (runs).

```python
# tested 2026-10-03 (evidently 0.7.23) — quoted from Evidently README
from evidently import Report
from evidently.presets import DataDriftPreset
report = Report([DataDriftPreset(method="psi")], include_tests="True")
my_eval = report.run(train_df, test_df)
my_eval.save_html("drift.html")
```

### Pitfalls

**Error analysis on training predictions.** It finds nothing, because the model has memorised those rows. Use OOF or validation predictions.

**Over-reading tiny slices.** A segment with n=12 and a 40% error rate is noise until shown otherwise. Always print counts.

**"Cleaning" the test set.** This is leakage dressed up as data quality. The cleanlab loop applies to training data only.

**Deepchecks licence and staleness.** Deepchecks is **AGPL-3.0**, and its premium `ee` features need a commercial licence. Its last PyPI release was December 2024 ([deepchecks README](https://raw.githubusercontent.com/deepchecks/deepchecks/main/README.md); [PyPI](https://pypi.org/pypi/deepchecks/json)). The AGPL is unlikely to matter in a hackathon notebook, but it matters if the team later ships a hosted product (our inference).

**Microsoft's Error Analysis widgets are stale.** The last releases were raiwidgets 0.36.0 (July 2024) and erroranalysis 0.5.5 (January 2025) ([PyPI](https://pypi.org/pypi/raiwidgets/json)). That their older dependency pins may clash with current stacks is **UNVERIFIED**.

**Unverified source details.** Google's rule numbers other than #26, and Ng's advice on mislabelled dev examples, were not checked against the sources (**UNVERIFIED**).

### Versions and licences (as of 2026-10-03)

| Package | Version | Released | Licence | Python |
|---|---|---|---|---|
| cleanlab | 2.9.0 | 2026-01-13 | Apache-2.0 | ≥3.10 |
| sliceline | 0.3.0 | 2026-02-03 | BSD-3-Clause | ≥3.10,<4 |
| fairlearn | 0.14.0 | 2026-06-07 | MIT | ≥3.10 |
| evidently | 0.7.23 | 2026-09-11 | Apache-2.0 | ≥3.10 |
| deepchecks | 0.19.1 | 2024-12-15 | **AGPLv3+** | n/a |
| raiwidgets / responsibleai | 0.36.0 | 2024-07-08 | MIT | ≥3.7 |
| erroranalysis | 0.5.5 | 2025-01-27 | MIT | ≥3.7 |

All versions are from PyPI JSON: [cleanlab](https://pypi.org/pypi/cleanlab/json), [sliceline](https://pypi.org/pypi/sliceline/json), [fairlearn](https://pypi.org/pypi/fairlearn/json), [evidently](https://pypi.org/pypi/evidently/json), [deepchecks](https://pypi.org/pypi/deepchecks/json), [raiwidgets](https://pypi.org/pypi/raiwidgets/json) and [erroranalysis](https://pypi.org/pypi/erroranalysis/json).

### Learning resources

**Method.**

- Andrew Ng, *Machine Learning Yearning*, chapters 13–19 on basic error analysis ([PDF](https://home-wordpress.deeplearning.ai/wp-content/uploads/2022/03/andrew-ng-machine-learning-yearning.pdf), **SNIPPET-ONLY**).
- Zinkevich, *Rules of ML*, the "Human Analysis of the System" section ([PDF](https://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf), **SNIPPET-ONLY**).

**Papers behind the tools.**

- SliceLine (SIGMOD 2021, [PDF](https://mboehm7.github.io/resources/sigmod2021b_sliceline.pdf)).
- Confident Learning (JAIR 2021, [arXiv 1911.00068](https://arxiv.org/abs/1911.00068)).

**Hands-on.** The Microsoft Error Analysis dashboard notebooks ([GitHub](https://github.com/microsoft/responsible-ai-toolbox/blob/main/notebooks/individual-dashboards/erroranalysis-dashboard/)).

### See it on real numbers

[`07-worked-example/evaluate_and_explain.py`](../07-worked-example/evaluate_and_explain.py) slices out-of-fold predictions by discharge destination, age band and missing lab, ranks slices by lift over their base rate, saves the 20 worst errors to read by hand, and flags when missed cases look low-risk on every recorded feature (a sign the driver is something the data does not measure).

---

## B. Lightweight experiment tracking: one JSONL line per run beats an unshared dashboard

### Plain-English explanation

Experiment tracking means writing down, for every model you train, exactly what you did and what score you got. The goal is that at 3 a.m. you can still answer three questions: which run was best, what code produced it, and was the gain real or noise?

A 2–4 person team with no infrastructure does not need a platform. The guide's toolbox already recommends a `feature_log.csv` plus a config file in Git. This section adds what to log, how to share it, and when a real tracker is worth it. Our recommended default, an inference from the sources, is:

- An **append-only JSONL or CSV file committed to Git**. It has zero dependencies, merges trivially, and loads straight into pandas for the leaderboard slide.
- Add **Trackio** when you want live curves or a shared dashboard. It is "local-first ... you shouldn't need to make an account", stores runs in SQLite, and is API-compatible with W&B, so `import trackio as wandb` works ([Trackio README](https://raw.githubusercontent.com/gradio-app/trackio/main/README.md)).

### Decision table: which tracker

| Option | Account needed | Shared across laptops | Setup | Best when | Caveat |
|---|---|---|---|---|---|
| JSONL/CSV in Git (our default) | No | Yes, via Git | 2 min | Any team | No curves; discipline needed |
| **Trackio** | No (local); HF token with write permission for a shared Space | Yes, via `space_id="user/space"` (auto-deploys an HF Space) or a self-hosted `server_url` | 5 min | Want W&B-style curves for free | README says hosting on HF is free; Space limits **RE-CHECK** in light of the Spaces changes in section 4 |
| **MLflow** (local) | No | Only if someone runs a reachable server (`mlflow server --host 0.0.0.0`, venue LAN **UNVERIFIED**) | 5–10 min | One person running many tabular runs; want model registry | New projects default to **SQLite since 3.7.0**; the file store is deprecated |
| **W&B** | Yes (every member) | Yes | 5 min | Team already uses W&B | Free-tier limits disputed, **RE-CHECK** |
| **DVC experiments** (dvclive) | No | Via Git | 20+ min | Team already uses DVC | Needs a Git repo; `dvc exp run` needs a DVC pipeline |
| **Aim** | No | Self-hosted (`aim up`) | 10 min | Want local queries over runs | Last release May 2025 |

Sources: [Trackio README](https://raw.githubusercontent.com/gradio-app/trackio/main/README.md), [MLflow README](https://raw.githubusercontent.com/mlflow/mlflow/master/README.md), [MLflow CHANGELOG](https://raw.githubusercontent.com/mlflow/mlflow/master/CHANGELOG.md), [W&B README](https://raw.githubusercontent.com/wandb/wandb/main/README.md), [DVC experiment tracking](https://raw.githubusercontent.com/iterative/dvc.org/main/content/docs/start/experiments/experiment-tracking.md), [DVC running experiments](https://raw.githubusercontent.com/iterative/dvc.org/main/content/docs/user-guide/experiment-management/running-experiments.md) and the [Aim README](https://raw.githubusercontent.com/aimhubio/aim/main/README.md).

**MLflow documentation conflict.** The tracking docs still say MLflow "logs data to the local `mlruns` directory" by default ([tracking docs](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/classic-ml/tracking/index.mdx)). The 3.7.0 changelog (2025-12-05) says new projects default to "SQLite as Default Backend ... unless existing mlruns data is detected". It also deprecates the file store in favour of a `mlflow migrate-filestore` command ([CHANGELOG](https://raw.githubusercontent.com/mlflow/mlflow/master/CHANGELOG.md)). Trust the changelog for new installs. MLflow records the Git commit in the tag `mlflow.source.git.commit` ([mlflow_tags.py](https://raw.githubusercontent.com/mlflow/mlflow/master/mlflow/utils/mlflow_tags.py)).

**W&B free tier: sources conflict (all RE-CHECK).**

- One source says up to 5 model seats, 5 GB of storage a month and 1 GB of Weave ingestion, with Pro at $60 a month (**SNIPPET-ONLY**, [ZenML](https://www.zenml.io/blog/wandb-pricing)).
- Another says a 100 GB limit with single-user-only access (**SNIPPET-ONLY**, [Spheron](https://www.spheron.network/blog/weights-biases-pricing-vs-self-hosted-mlflow-2026/)).
- An academic plan is described with 100 GB of free storage (**SNIPPET-ONLY**, [wandb pricing](https://wandb.ai/site/pricing/)).
- The pricing page title appeared as "CoreWeave Forge Pricing" in search results, which suggests a rebrand (**UNVERIFIED**).

### What to log for every run (our synthesis)

| Field | Why |
|---|---|
| run_id, timestamp, author | Who did what, when |
| `git rev-parse HEAD` + dirty flag | Reproduce the exact code |
| Data version: file names + hash or row counts of train/test; feature-set name | Catch "same code, different data" |
| Full config / hyperparameters | Reproduce the exact model |
| Random seed(s), including the CV split seed | Separate signal from seed noise |
| CV scheme (n_folds, group or time key) | Make scores comparable |
| **Metric per fold**, mean ± std | A gain smaller than the fold std is noise |
| Public leaderboard score, if submitted | Track CV vs leaderboard agreement |
| Runtime | Budget the remaining hours |
| Path to OOF predictions and submission file | Enables error analysis (section 2) and ensembling without retraining |
| Free-text note: what changed and why | The pitch narrative writes itself |

### Minimal code (the JSONL logger is tested; tracker snippets are not run by us)

A dependency-free logger. Illustrative code, not from a source; **tested** (runs, and `pd.read_json(..., lines=True)` reads it back). The runnable example writes the same kind of log to `outputs/experiments.jsonl`.

```python
# tested 2026-10-03 — illustrative
import json, subprocess, time, statistics
def log_run(cfg, fold_scores, note="", path="runs.jsonl"):
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"]).strip())
    rec = {"ts": time.strftime("%F %T"), "commit": commit, "dirty": dirty, **cfg,
           "folds": fold_scores, "cv_mean": statistics.mean(fold_scores),
           "cv_std": statistics.pstdev(fold_scores), "note": note}
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
# leaderboard slide: pd.read_json("runs.jsonl", lines=True).sort_values("cv_mean")
```

Trackio as a drop-in for W&B. Calls are from the [Trackio README](https://raw.githubusercontent.com/gradio-app/trackio/main/README.md) and the [W&B README](https://raw.githubusercontent.com/wandb/wandb/main/README.md). Not run by us.

```python
# not run by us — assembled from README calls
import trackio as wandb
wandb.init(project="hackathon", config={"lr": 3e-4, "model": "lgbm"},
           space_id="username/hackathon-dash")   # optional: shared dashboard on an HF Space
for epoch, (tr, va) in enumerate(history):
    wandb.log({"train_loss": tr, "val_loss": va})
wandb.finish()
# locally: `trackio show` ; query: trackio query project --project hackathon --sql "SELECT ..."
```

MLflow, local. The tracking URI comes from the [MLflow README](https://raw.githubusercontent.com/mlflow/mlflow/master/README.md). The `log_params` and `log_metric` calls are standard API from background knowledge, not re-checked this session (**UNVERIFIED**). Not run by us.

```python
# not run by us
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")   # after: uvx mlflow server
mlflow.set_experiment("hackathon")
with mlflow.start_run():
    mlflow.log_params(cfg)
    mlflow.log_metric("cv_mean", cv_mean)
```

DVCLive, quoted from the [DVC get-started docs](https://raw.githubusercontent.com/iterative/dvc.org/main/content/docs/start/experiments/experiment-tracking.md). `live.next_step()` is **UNVERIFIED** because the quoted snippet was truncated. Not run by us.

```python
# not run by us
from dvclive import Live
with Live() as live:
    live.log_param("epochs", NUM_EPOCHS)
    live.log_metric("val_f1", score); live.next_step()
# then: dvc exp show --md --sort-by val_f1
```

### Pitfalls

**Reporting seed noise as progress.** If a "gain" is smaller than the standard deviation across folds or seeds, do not put it in the pitch.

**Trackers nobody else can see.** A local MLflow on one laptop is invisible to teammates. Decide on the shared location in hour one.

**Committing large files.** Never commit large OOF or prediction files to Git. Log their paths, and keep the files in a shared drive or Kaggle dataset (our inference).

**Account friction mid-event.** W&B needs every member to have an account. A shared HF Space for Trackio needs an HF token with write permission, plus whatever the current Spaces plan allows (**RE-CHECK**).

**Stale trackers.** Aim's release pace has slowed: the last release was 3.29.1 in May 2025 ([PyPI](https://pypi.org/pypi/aim/json)).

**Not evaluated here.** Comet, Neptune and ClearML were out of scope.

### Versions and licences (as of 2026-10-03)

| Package | Version | Released | Licence | Python |
|---|---|---|---|---|
| trackio | 0.40.0 | 2026-09-30 | MIT | ≥3.10 |
| mlflow | 3.16.1 | 2026-09-16 | Apache-2.0 | ≥3.10 |
| wandb | 0.30.0 | 2026-09-09 | MIT (client library) | ≥3.10 |
| dvc | 3.67.1 | 2026-03-31 | Apache-2.0 | ≥3.9 |
| aim | 3.29.1 | 2025-05-08 | Apache-2.0 | ≥3.7 |

All versions are from PyPI JSON: [trackio](https://pypi.org/pypi/trackio/json), [mlflow](https://pypi.org/pypi/mlflow/json), [wandb](https://pypi.org/pypi/wandb/json), [dvc](https://pypi.org/pypi/dvc/json) and [aim](https://pypi.org/pypi/aim/json). The W&B hosted service's terms are separate from the MIT-licensed client.

### Learning resources

**Trackers.**

- [Trackio README](https://raw.githubusercontent.com/gradio-app/trackio/main/README.md).
- [MLflow README](https://raw.githubusercontent.com/mlflow/mlflow/master/README.md) and [tracking docs](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/classic-ml/tracking/index.mdx). Read these alongside the 3.7.0 changelog entry.
- [W&B README](https://raw.githubusercontent.com/wandb/wandb/main/README.md).
- [Aim README](https://raw.githubusercontent.com/aimhubio/aim/main/README.md).

**DVC.** [DVC experiment tracking](https://raw.githubusercontent.com/iterative/dvc.org/main/content/docs/start/experiments/experiment-tracking.md) and the [`dvc exp show` reference](https://raw.githubusercontent.com/iterative/dvc.org/main/content/docs/command-reference/exp/show.md).
