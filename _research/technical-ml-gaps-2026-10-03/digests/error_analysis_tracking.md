# Error Analysis and Lightweight Experiment Tracking for 24-48h ML Hackathons/Datathons

Research date: 2026-10-03. Access note: the egress proxy blocked developers.google.com, fairlearn.org, wandb.ai, docs.wandb.ai, docs.cleanlab.ai, huggingface.co, martin.zinkevich.org and deeplearning.ai. Primary evidence therefore comes from PyPI JSON (fetched directly) and GitHub READMEs/doc sources via raw.githubusercontent.com. Anything from search snippets only is marked SNIPPET-ONLY.

## Q1. Which tools/libraries help? Versions, licences, minimal usage

### Takeaway
All ten tools are pip-installable and free. Every one has a permissive licence except Deepchecks, which is AGPL-3.0+. For a 48h event the useful core is: **sliceline** or **fairlearn MetricFrame** to slice performance by segment, **cleanlab** to find label noise, and **one** tracker. Deepchecks and Evidently produce one-call HTML reports that are handy as pitch-deck evidence. The Microsoft Error Analysis widgets work but have not had a release since 2024/early 2025.

### Cited Findings

**Version/licence table (from PyPI JSON `https://pypi.org/pypi/<pkg>/json`, fetched 2026-10-03)**

| Package | Latest version | Upload date | Licence (PyPI metadata) | Python |
|---|---|---|---|---|
| cleanlab | 2.9.0 | 2026-01-13 | Apache-2.0 | >=3.10 |
| deepchecks | 0.19.1 | 2024-12-15 | AGPLv3+ | n/a |
| evidently | 0.7.23 | 2026-09-11 | Apache-2.0 | >=3.10 |
| sliceline | 0.3.0 | 2026-02-03 | BSD-3-Clause | >=3.10,<4 |
| fairlearn | 0.14.0 | 2026-06-07 | MIT | >=3.10 |
| mlflow | 3.16.1 | 2026-09-16 | Apache-2.0 | >=3.10 |
| wandb | 0.30.0 | 2026-09-09 | MIT (client library) | >=3.10 |
| trackio | 0.40.0 | 2026-09-30 | MIT | >=3.10 |
| aim | 3.29.1 | 2025-05-08 | Apache-2.0 | >=3.7 |
| dvc | 3.67.1 | 2026-03-31 | Apache-2.0 | >=3.9 |
| raiwidgets / responsibleai | 0.36.0 | 2024-07-08 | MIT | >=3.7 |
| erroranalysis | 0.5.5 | 2025-01-27 | MIT | >=3.7 |

Sources: [cleanlab](https://pypi.org/pypi/cleanlab/json), [deepchecks](https://pypi.org/pypi/deepchecks/json), [evidently](https://pypi.org/pypi/evidently/json), [sliceline](https://pypi.org/pypi/sliceline/json), [fairlearn](https://pypi.org/pypi/fairlearn/json), [mlflow](https://pypi.org/pypi/mlflow/json), [wandb](https://pypi.org/pypi/wandb/json), [trackio](https://pypi.org/pypi/trackio/json), [aim](https://pypi.org/pypi/aim/json), [dvc](https://pypi.org/pypi/dvc/json), [raiwidgets](https://pypi.org/pypi/raiwidgets/json), [erroranalysis](https://pypi.org/pypi/erroranalysis/json)

**cleanlab (label noise and data issues)**
- The README's one-call audit with Datalab:
  ```python
  lab = cleanlab.Datalab(data=dataset, label="column_name_for_labels")
  # Fit any ML model, get its feature_embeddings & pred_probs for your data
  lab.find_issues(features=feature_embeddings, pred_probs=pred_probs)
  lab.report()
  ```
  The README says it detects "outliers, duplicates, label errors, etc" in text, audio, image and tabular data. — [cleanlab README](https://raw.githubusercontent.com/cleanlab/cleanlab/master/README.md)
- `CleanLearning` wrapper, quoted from the README: `cl = cleanlab.classification.CleanLearning(sklearn.YourFavoriteClassifier())`; `label_issues = cl.find_label_issues(data, labels)`; `cl.fit(data, labels)`; `cleanlab.dataset.health_summary(labels, confident_joint=cl.confident_joint)`. — [cleanlab README](https://raw.githubusercontent.com/cleanlab/cleanlab/master/README.md)
- The README claims it works "with any dataset and any model" and lists dedicated support for binary/multi-class, multi-label, token classification, **regression**, segmentation, object detection and multi-annotator data. — [cleanlab README](https://raw.githubusercontent.com/cleanlab/cleanlab/master/README.md)
- The README's suggested workflow: (1) train an initial model, (2) use it to diagnose data issues, (3) retrain the same model on the improved data, (4) then try other modelling techniques. — [cleanlab README](https://raw.githubusercontent.com/cleanlab/cleanlab/master/README.md)
- The lower-level API is `cleanlab.filter.find_label_issues(labels, pred_probs, *, return_indices_ranked_by=None, filter_by="prune_by_noise_rate", ...)`. Its docstring says it "Identifies potentially bad labels in a classification dataset using confident learning" and returns a boolean mask. Passing `return_indices_ranked_by` returns indices sorted by severity instead. — [cleanlab/filter.py](https://raw.githubusercontent.com/cleanlab/cleanlab/master/cleanlab/filter.py)
- The method comes from Confident Learning, JAIR 2021, by Northcutt, Jiang and Chuang. — [arXiv 1911.00068](https://arxiv.org/abs/1911.00068) (cited in the README)

**sliceline (automatic slice finding)**
- The README says: "Given an input dataset X and a model error vector errors, SliceLine finds the top slices in X that identify where a ML model performs significantly worse."
  ```python
  from sliceline.slicefinder import Slicefinder
  slice_finder = Slicefinder()
  slice_finder.fit(X, errors)
  print(slice_finder.top_slices_)
  X_trans = slice_finder.transform(X)
  ```
  The demo notebooks cover Titanic (classification) and California housing (regression). The optional install `pip install sliceline[optimized]` adds Numba. — [sliceline README](https://raw.githubusercontent.com/DataDome/sliceline/main/README.rst)
- It is based on the SIGMOD 2021 paper by Sagadeeva and Boehm. — [paper PDF](https://mboehm7.github.io/resources/sigmod2021b_sliceline.pdf)
- UNVERIFIED (from general knowledge, not checked in the docs this session): X should be categorical or binned, so numeric features need binning first.

**fairlearn MetricFrame (metrics per segment)**
- Doc example (verbatim, doctest prompts removed):
  ```python
  from fairlearn.metrics import MetricFrame, count, false_positive_rate, selection_rate
  from sklearn.metrics import recall_score
  my_metrics = {'tpr': recall_score, 'fpr': false_positive_rate, 'sel': selection_rate, 'count': count}
  mf = MetricFrame(metrics=my_metrics, y_true=y_true, y_pred=y_pred, sensitive_features=sf_data)
  mf.overall; mf.by_group; mf.group_min(); mf.group_max(); mf.difference(); mf.difference(method='to_overall')
  ```
  The docs also describe a `control_features` parameter. — [fairlearn perform_fairness_assessment.rst](https://raw.githubusercontent.com/fairlearn/fairlearn/main/docs/user_guide/assessment/perform_fairness_assessment.rst)
- Inference: "sensitive_features" can be **any** segment column (region, device, store, age bin), so MetricFrame works as a general per-slice metrics table. Any sklearn-style metric works, including MAE or RMSE for regression.

**Deepchecks**
- README snippet:
  ```python
  from deepchecks.tabular.suites import model_evaluation
  suite = model_evaluation()
  suite_result = suite.run(train_dataset=train_dataset, test_dataset=test_dataset, model=model)
  suite_result.save_as_html()  # or suite_result.show()
  ```
  — [deepchecks README](https://raw.githubusercontent.com/deepchecks/deepchecks/main/README.md)
- Licence: AGPL 3.0. Premium monitoring features in `ee` need a commercial licence. The last PyPI release (0.19.1) is from Dec 2024, so maintenance pace is a caveat. — [deepchecks README](https://raw.githubusercontent.com/deepchecks/deepchecks/main/README.md), [PyPI](https://pypi.org/pypi/deepchecks/json)

**Evidently**
- README snippet for drift between train and test (or between time periods):
  ```python
  from evidently import Report
  from evidently.presets import DataDriftPreset
  report = Report([DataDriftPreset(method="psi")], include_tests="True")
  my_eval = report.run(iris_frame.iloc[:60], iris_frame.iloc[60:])
  my_eval.save_html("file.html")
  ```
  It also has `TextEvals` and descriptors (`Sentiment`, `TextLength`, `Contains`) for evaluating text/LLM outputs. — [evidently README](https://raw.githubusercontent.com/evidentlyai/evidently/main/README.md)

**Microsoft Error Analysis (Responsible AI Toolbox)**
- It identifies cohorts with high error rates through a **decision tree** and an **error heatmap**. Diagnosis tools are data exploration, global and local explanations, and what-if analysis. Minimal call:
  ```python
  from raiwidgets import ErrorAnalysisDashboard
  ErrorAnalysisDashboard(global_explanation, dashboard_pipeline, dataset=X_test_original,
                         true_y=y_test, categorical_features=categorical_features)
  ```
  — [erroranalysis-dashboard-README](https://raw.githubusercontent.com/microsoft/responsible-ai-toolbox/main/docs/erroranalysis-dashboard-README.md); notebooks: [GitHub](https://github.com/microsoft/responsible-ai-toolbox/blob/main/notebooks/individual-dashboards/erroranalysis-dashboard/)
- Last releases: raiwidgets 0.36.0 (2024-07) and erroranalysis 0.5.5 (2025-01). — [PyPI raiwidgets](https://pypi.org/pypi/raiwidgets/json)

**Trackers (minimal code from READMEs)**
- **MLflow**: by default runs log to local storage with no server. The tracking docs say "By default, without any particular server/database configuration, MLflow Tracking logs data to the local `mlruns` directory." — [MLflow tracking docs source](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/classic-ml/tracking/index.mdx). This is **contradicted for new projects** by the changelog. MLflow 3.7.0 (2025-12-05) made "SQLite as Default Backend ... instead of file-based storage, unless existing mlruns data is detected", and the file store is deprecated, with a `mlflow migrate-filestore` command. — [MLflow CHANGELOG](https://raw.githubusercontent.com/mlflow/mlflow/master/CHANGELOG.md). The README quick start is `uvx mlflow server`, then `mlflow.set_tracking_uri("http://localhost:5000")`. — [MLflow README](https://raw.githubusercontent.com/mlflow/mlflow/master/README.md). MLflow has a built-in tag `mlflow.source.git.commit` for the git commit. — [mlflow_tags.py](https://raw.githubusercontent.com/mlflow/mlflow/master/mlflow/utils/mlflow_tags.py)
- **W&B**:
  ```python
  import wandb
  with wandb.init(project="my-awesome-project", config={"epochs": 1337, "lr": 3e-4}) as run:
      run.log({"accuracy": 0.9, "loss": 0.1})
  ```
  Results are viewed at wandb.ai/home, so an account is needed. — [wandb README](https://raw.githubusercontent.com/wandb/wandb/main/README.md)
- **Trackio** (Hugging Face): "lightweight, free", "local-first ... you shouldn't need to make an account". Logs go to SQLite and can be frozen to Parquet. It is API-compatible with `wandb.init/log/finish`, so `import trackio as wandb` works. Dashboard: `trackio show`. A shared team dashboard comes from `trackio.init(project=..., space_id="username/space_id")`, which auto-deploys a free HF Space (needs an HF token with write permission). It also has a self-hosted server through `server_url=...`, and a CLI `trackio query project --project <name> --sql "SELECT ..."`. The README says: "Everything here, including hosting on Hugging Face, is free!" — [trackio README](https://raw.githubusercontent.com/gradio-app/trackio/main/README.md)
- **Aim**:
  ```python
  from aim import Run
  run = Run(); run["hparams"] = {"learning_rate": 0.001, "batch_size": 32}
  run.track(value, name='loss', step=i, context={"subset": "train"})
  ```
  Then `aim up` starts the UI. The Python API supports queries via `Repo(...).query_metrics("metric.name == 'loss'")`. The last release, 3.29.1, is from May 2025. — [aim README](https://raw.githubusercontent.com/aimhubio/aim/main/README.md), [PyPI](https://pypi.org/pypi/aim/json)
- **DVC experiments**: `pip install dvclive`, then:
  ```python
  from dvclive import Live
  with Live() as live:
      live.log_param("epochs", NUM_EPOCHS)
      live.log_metric(name, value); live.next_step()
  ```
  There are callbacks for Lightning, Hugging Face and Keras. — [DVC Get Started: Experiment Tracking](https://raw.githubusercontent.com/iterative/dvc.org/main/content/docs/start/experiments/experiment-tracking.md). `dvc exp show` prints a table of experiments with metrics, params and deps, and supports `--csv/--md/--json` and `--sort-by`. — [dvc exp show docs](https://raw.githubusercontent.com/iterative/dvc.org/main/content/docs/command-reference/exp/show.md). DVC needs a Git repo.

### Inferences
- Order for a hackathon: (1) per-segment table with MetricFrame or a pandas groupby, (2) Sliceline for slices you did not think of, (3) cleanlab on out-of-fold `pred_probs` for label noise, (4) optional Evidently drift report between train and test. Deepchecks and RAI are "nice if it installs cleanly". RAI's older dependency pins are a plausible install risk; this is UNVERIFIED.
- Deepchecks' AGPL is unlikely to matter for a hackathon notebook. It could matter if the team ships a hosted product afterwards.
- cleanlab needs **out-of-sample** `pred_probs`, typically from cross_val_predict. This matches the confident-learning setup, but this session did not get it verbatim from the README; it is in docs.cleanlab.ai, which was blocked.

### Gaps
- Could not read docs.cleanlab.ai, fairlearn.org or the Sliceline readthedocs pages (proxy blocked), so the exact parameter guidance for those is not quoted (e.g. the requirement for out-of-sample pred_probs, Sliceline's `alpha`, `k` and `min_sup`).
- Did not verify Deepchecks' `data_integrity` / `train_test_validation` suites verbatim; only `model_evaluation` was quoted.
- DVC's `log_metric` loop in the quoted snippet was truncated. `live.next_step()` comes from general DVCLive knowledge and is UNVERIFIED.

## Q2. Published guidance and checklists on error analysis

### Takeaway
The canonical guidance amounts to one practice: hand-inspect around 100 errors, tag them into categories in a spreadsheet, and let the counts set priorities (Ng). Then turn the patterns into new features (Google Rule #26). Tools such as the Microsoft Error Analysis tree, Sliceline and MetricFrame automate the "find the bad cohort" step.

### Cited Findings
- **Andrew Ng, *Machine Learning Yearning***, chapters 13-19 ("Basic Error Analysis"). Each point below is SNIPPET-ONLY because the original PDF was proxy-blocked:
  - Ch. 14: gather about 100 misclassified dev examples and count the main error categories. This estimates the "ceiling" on improvement from fixing each category. — [MLY PDF (deeplearning.ai)](https://home-wordpress.deeplearning.ai/wp-content/uploads/2022/03/andrew-ng-machine-learning-yearning.pdf) SNIPPET-ONLY; summary in [KDnuggets](https://www.kdnuggets.com/2018/05/7-useful-suggestions-machine-learning-yearning.html)
  - Use a spreadsheet while reviewing errors: rows are examples, columns are error categories, plus a comments column. Categories emerge during the review. — same source, SNIPPET-ONLY
  - Split the dev set into an "Eyeball dev set" (inspected by hand) and a "Blackbox dev set". Example: at a 5% error rate you need an Eyeball set of about 2,000 examples to get about 100 errors ("0.05*2,000 = 100"). — same source, SNIPPET-ONLY
  - UNVERIFIED (general knowledge, not checked against the PDF this session): mislabeled dev/test examples are handled as their own error category.
- **Google "Rules of Machine Learning" (Zinkevich)**:
  - Rule #26: "Look for patterns in the measured errors, and create new features." Examples the model got wrong are ones it "would like to fix if given the opportunity". Look for trends outside the current feature set; the rule's own example is adding post length when longer posts are demoted. — [Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml) (page blocked; content via search snippet, SNIPPET-ONLY); [PDF](https://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf)
  - Rule #26 sits in the "Human Analysis of the System" section of Phase II: Feature Engineering. — SNIPPET-ONLY, same source
  - Rule numbers of other relevant rules (e.g. #23 "You are not a typical end user", #24 "Measure the delta between models", #27 "Try to quantify observed undesirable behavior") are UNVERIFIED this session because the page was blocked.
- **Microsoft Error Analysis**: a decision tree and heatmap to "identify cohorts with high error rates", followed by diagnosis through data exploration, explanations and what-if analysis. — [erroranalysis-dashboard-README](https://raw.githubusercontent.com/microsoft/responsible-ai-toolbox/main/docs/erroranalysis-dashboard-README.md)
- **SliceLine paper** (SIGMOD 2021) is the academic basis for automated slice finding. — [PDF](https://mboehm7.github.io/resources/sigmod2021b_sliceline.pdf)
- **Confident Learning paper** (JAIR 2021) is the basis for label-noise detection. — [arXiv](https://arxiv.org/abs/1911.00068); examples of label errors in real datasets: [labelerrors.com](https://labelerrors.com) (linked from the [cleanlab README](https://raw.githubusercontent.com/cleanlab/cleanlab/master/README.md))
- The cleanlab README describes the data-centric loop: train, diagnose the data, fix, retrain the same model, then iterate on the model. — [cleanlab README](https://raw.githubusercontent.com/cleanlab/cleanlab/master/README.md)

### Inferences (a practical 48h checklist synthesised from the sources above; not itself a published checklist)
1. **Get honest errors.** Use out-of-fold predictions (CV) for every training row, so errors and cleanlab inputs are not overfit.
2. **Slice.** Build a per-segment metric table (MetricFrame or `df.groupby(seg).apply(metric)`) over the obvious categorical columns, time periods and bins of key numeric columns. Show counts next to metrics so small slices are not over-read.
3. **Auto-slice.** Run Sliceline on binned features with `errors` set to 0/1 misclassification or to absolute residual for regression.
4. **Confusion by segment.** Compute a confusion matrix per top slice to see which class pairs fail where.
5. **Worst errors.** Sort by loss: confident-wrong classifications (high predicted probability on the wrong class) or the largest absolute residuals. Inspect 50-100 by hand and tag them in a spreadsheet (Ng).
6. **Regression residuals.** Plot residual vs prediction, vs each key feature and vs time. Look for heteroscedasticity, which suggests a log target. Look for systematic bias in a slice, which suggests a missing feature or interaction (Rule #26). Check residuals for target outliers.
7. **Label noise.** Run cleanlab `find_label_issues` / Datalab on OOF pred_probs, then review the top-ranked rows. Only drop or relabel training rows; never "clean" the test/leaderboard data.
8. **Leakage and train-test drift.** Run an Evidently DataDriftPreset or adversarial validation on train vs test.
9. **Convert findings.** For each top error category: write a new feature, a segment-specific model or threshold, a data fix, or "accept and disclose".
10. **Pitch points.** Judges respond to slides like "model is X% worse on segment S (n=…), cause Y, fix Z gave +Δ". Error analysis gives an honest limitations slide and a story about the next iteration.

### Gaps
- I could not fetch the original ML Yearning PDF or Google's Rules of ML page, so the quotes above rest on search snippets and secondary summaries.
- I found no published checklist aimed at hackathons or datathons; the checklist above is a synthesis.

## Q3. Which tracking option fits a 2-4 person team with no infra, and what to log; free-tier limits

### Takeaway
With no infra, the lowest-friction options are (a) a shared **CSV/JSONL results log** committed to git, or (b) **Trackio**, which needs no account locally and gives a shared dashboard on a free HF Space through `space_id`. W&B's free tier is the most polished hosted option but needs accounts, and the limits are disputed (RE-CHECK). Local MLflow is solid but is single-machine unless someone hosts a server. DVC experiments fit only teams already comfortable with DVC and git.

### Cited Findings
- **Trackio**: local-first with no account, SQLite storage, wandb-compatible API, an optional free HF Space dashboard shared by the team, and a self-hosted server mode. The README says hosting on HF is free. — [trackio README](https://raw.githubusercontent.com/gradio-app/trackio/main/README.md). RE-CHECK: HF Space free hardware and persistence limits were not verified (huggingface.co was blocked).
- **W&B free tier**: sources conflict.
  - "Free plan covers up to 5 model seats, 5GB of storage per month, and 1GB of Weave data ingestion per month"; Pro is "$60/month for up to 10 model seats and 100GB". — [ZenML blog](https://www.zenml.io/blog/wandb-pricing), [wandb.ai/site/pricing](https://wandb.ai/site/pricing/) (SNIPPET-ONLY, RE-CHECK)
  - This is contradicted by "100GB limit per account ... single-user-only access" from [Spheron blog](https://www.spheron.network/blog/weights-biases-pricing-vs-self-hosted-mlflow-2026/) (SNIPPET-ONLY, RE-CHECK).
  - An academic plan is described as unlimited tracking hours, teams and projects with 100GB of free cloud storage. — SNIPPET-ONLY via [wandb pricing](https://wandb.ai/site/pricing/), RE-CHECK
  - The pricing page title in search results reads "CoreWeave Forge Pricing", which suggests W&B is now presented under CoreWeave branding. UNVERIFIED.
- **MLflow local**: needs no account and no server. New projects default to SQLite since 3.7.0, and the UI comes from `mlflow server`/`mlflow ui`. The git commit is recorded as the tag `mlflow.source.git.commit`. — [CHANGELOG](https://raw.githubusercontent.com/mlflow/mlflow/master/CHANGELOG.md), [README](https://raw.githubusercontent.com/mlflow/mlflow/master/README.md), [mlflow_tags.py](https://raw.githubusercontent.com/mlflow/mlflow/master/mlflow/utils/mlflow_tags.py). Sharing across laptops needs a reachable server or a database. Inference: on a hackathon LAN, one teammate could run `mlflow server --host 0.0.0.0`, but this is UNVERIFIED for venue networks.
- **Aim**: self-hosted only, via `aim up`. The release cadence slowed (last release 2025-05). — [aim README](https://raw.githubusercontent.com/aimhubio/aim/main/README.md), [PyPI](https://pypi.org/pypi/aim/json)
- **DVC**: experiments are tied to git commits, and `dvc exp show --csv/--md` exports a comparison table. It requires a Git repo, plus DVC pipeline setup for `dvc exp run` (per the running-experiments docs: "This page is not applicable if you are [saving experiments] without a pipeline"). — [dvc exp show](https://raw.githubusercontent.com/iterative/dvc.org/main/content/docs/command-reference/exp/show.md), [running experiments](https://raw.githubusercontent.com/iterative/dvc.org/main/content/docs/user-guide/experiment-management/running-experiments.md)

### Inferences
- **Recommended default**: a one-function JSONL/CSV logger plus `results.csv` in git. It has zero dependencies, merges trivially, and pandas reads it for the leaderboard slide. Add Trackio (`import trackio as wandb`) when you want curves or a shared live dashboard. If the team already uses W&B, keep W&B, since Trackio can stand in for it later.
- **What to log per run** (synthesis): run_id, timestamp, author, `git rev-parse HEAD` plus a dirty flag, data version (filename plus hash or row count of train/test, feature-set name), full config/hyperparameters, random seed(s), CV scheme (n_folds, split seed, group/time key), **metric per fold**, mean ± std, public leaderboard score if a submission was made, runtime, path to the OOF predictions, path to the submission file, and a free-text note ("what changed / why"). Saving the OOF predictions lets you do error analysis and ensembling later without retraining.
- Minimal logger pattern (illustrative code, not from a source):
  ```python
  import json, subprocess, time, hashlib
  def log_run(cfg, fold_scores, note="", path="runs.jsonl"):
      commit = subprocess.check_output(["git","rev-parse","--short","HEAD"]).decode().strip()
      rec = {"ts": time.strftime("%F %T"), "commit": commit, **cfg,
             "folds": fold_scores, "cv_mean": sum(fold_scores)/len(fold_scores), "note": note}
      open(path, "a").write(json.dumps(rec) + "\n")
  ```
- Seed control: log and fix numpy, random and framework seeds, and the CV split seed. Results that differ only by seed are noise and should not be reported as gains in the pitch. Compare against the fold std.

### Gaps
- Exact W&B free-tier limits (seats, storage, hours) could not be confirmed from the primary page (wandb.ai blocked); sources conflict. **RE-CHECK.**
- HF Spaces free-tier limits for Trackio dashboards (sleep after inactivity, storage persistence) were not verified. **RE-CHECK.**
- Comet, Neptune and ClearML were out of scope and not evaluated.
