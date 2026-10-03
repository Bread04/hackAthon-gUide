# Tabular foundation models and AI-assisted data science for 24-48h datathons (2026)

Provenance note: arxiv.org, auto.gluon.ai and substack were egress-blocked for WebFetch. Items marked WEBFETCH were read from a fetched GitHub page (summarised by the fetch tool's small model). Items marked SNIPPET-ONLY come from search-result snippets only. Nothing here was verified against the primary paper PDFs. Several arXiv IDs returned by search (26xx.xxxxx) are beyond my training knowledge and I could not open them.

## 1. Tabular foundation models (TabPFN, TabICL, TabDPT, Mitra, CARTE): limits, speed, licence, hardware, benchmark standing, AutoGluon integration

### Takeaway
TabPFN (v2.5 and newer), TabICLv2, TabDPT and Mitra are now usable via pip and in AutoGluon 1.5+. They are strong on small to medium datasets, with TabICLv2 and TabPFN-2.5 reported at or above tuned GBDTs and AutoGluon on TabArena. Verdict: worthwhile as an extra ensemble member or quick baseline. Check licence terms (TabPFN weights are non-commercial) and use a GPU. Do not replace GBDTs wholesale, and confirm results on your own CV.

### Cited Findings
- TabPFN repo lists TabPFN-2, 2.5, 3, 3.5 (current default, released Sept 2026) and a 3.5-Fast variant. Limits for 3.5 / 3.5-Fast: up to 1,000,000 rows and 20,000 features (class limits vary by checkpoint). CPU use is capped at 5,000 samples by default (env-var override). GPU recommended (~8GB VRAM older GPUs work). Python 3.10+. Weights auto-download. — [PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN) (WEBFETCH; the fetch summary cited tech report arXiv 2609.17895, which I could not open, so UNVERIFIED)
- TabPFN licensing: code Apache-2.0, but model weights are under non-commercial licences; TabPFN-3 / 3.5 require accepting terms through a browser login at first use; a paid enterprise edition exists. — [PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN) (WEBFETCH). Practical datathon risk: the login/terms step needs internet and an account on the competition machine; do it before the event.
- TabPFN-2.5 (arXiv Nov 2025): described as leading TabArena (datasets up to 100k training points), substantially outperforming tuned tree models and matching AutoGluon 1.4 (4h tuned ensemble). — [arXiv 2511.08667](https://arxiv.org/pdf/2511.08667) (SNIPPET-ONLY; this is the authors' own claim)
- A TabPFN-3 technical report exists at arXiv 2605.13986. — [link](https://arxiv.org/pdf/2605.13986) (SNIPPET-ONLY title, contents not read, UNVERIFIED)
- TabICL repo: TabICLv2 (ICML 2026, classification and regression), v1.1 (May 2025), v1 (ICML 2025). Pre-trained on 300-48K samples, generalises up to ~600K rows; designed for 2-100 columns, degrades beyond 100 features. GPU recommended, CPU supported, CPU/disk offloading; H100 does 50k x 100 in under 10 s; KV caching; claims ~10x faster than TabPFN-2.5 on large data and "outperforms heavily tuned XGBoost/CatBoost/LightGBM on TabArena on ~80% of datasets". Licence listed as "permissive" (exact licence not confirmed). — [soda-inria/tabicl](https://github.com/soda-inria/tabicl) (WEBFETCH; authors' own claims)
- TabICLv2 paper: on TabArena/TALENT, untuned TabICLv2 has best average rank 4.66 vs RealTabPFN-2.5 5.11 and TabPFN-2.5 5.45; another comparison has TabICLv2 default 4.82 vs AutoGluon 1.4 extreme (4h) 5.24 vs RealTabPFN-2.5 (tuned+ensembled) 5.88. — [arXiv 2602.11139](https://arxiv.org/html/2602.11139v1) (SNIPPET-ONLY; authors' own benchmark framing)
- AutoGluon 1.5: adds RealTabPFN-2, RealTabPFN-2.5, TabDPT as foundation models in AutoGluon-Tabular, plus Mitra default weights; TabDPT, TabICL, TabPFN, Mitra install through `pip install autogluon.tabular[tabarena]`; RealTabPFN-2.5 called the strongest individual model on TabArena. A "Tabular Foundational Models" tutorial exists in docs for 1.6.1. — [AutoGluon 1.5 release notes](https://auto.gluon.ai/stable/whats_new/v1.5.0.html), [tutorial](https://auto.gluon.ai/stable/tutorials/tabular/tabular-foundational-models.html), search listing [AutoGluon what's new](https://auto.gluon.ai/dev/whats_new/index.html) (all SNIPPET-ONLY; pages blocked). Answer to "is AutoGluon integrating them": yes, SNIPPET-ONLY.
- Mitra is AutoGluon's own foundation model with open weights (NeurIPS 2025 "Mixed Synthetic Priors"). — [autogluon/autogluon](https://github.com/autogluon/autogluon) (WEBFETCH confirms the Mitra publication reference only)
- TabArena is a "living" benchmark (NeurIPS 2025 D&B track). — [paper](https://papers.neurips.cc/paper_files/paper/2025/file/1697e3fb412da11dc9488249f9e7bbc9-Paper-Datasets_and_Benchmarks_Track.pdf) (SNIPPET-ONLY). I could not read its leaderboard or its GBDT-vs-foundation-model details.
- Overview blog "The state of Tabular Foundation Models (2026)" exists. — [link](https://mindfulmodeler.substack.com/p/the-state-of-tabular-foundation-models) (blocked, title only)
- A further paper "TabFM: A Zero-Shot Foundation Model for Tabular Data" dated 2026-09-29 appeared in search. — [link](https://arxiv.org/pdf/2609.37959) (SNIPPET-ONLY, UNVERIFIED, not assessed)

### Inferences
- Rows/features limits for datathon data (typically under 100k rows) fit TabPFN-2.5+/TabICLv2 comfortably; wide tables (>100 features for TabICL) need feature selection first.
- Because the benchmark rank claims come from model authors (TabICL, PriorLabs) and tuned-GBDT baselines vary, treat "beats LightGBM" as likely but not guaranteed on a specific dataset; run 5-fold CV against a LightGBM/CatBoost baseline.
- Safest route for a team: AutoGluon `presets` with foundation models enabled (inherits stacking and validation), rather than hand-rolled TabPFN code.
- Non-commercial weights are probably fine for a hackathon, but check the event's rules if the prize involves a commercial entity (unconfirmed).

### Gaps
- TabDPT and CARTE specifics (limits, licence, speed) not retrieved. CARTE (graph model for strings/relational-ish tables) not researched at all.
- TALENT results beyond TabICLv2 snippet; TabArena live leaderboard numbers; foundation model performance on high-cardinality categoricals, text columns and time-dependent splits not found.
- Exact AutoGluon preset names/flags and whether foundation models are on by default in `best_quality` (docs blocked).
- Real GPU memory/time numbers for TabPFN-3.5 at 100k rows (only vendor-level statements).

## 2. LLM coding agents for data science: benchmark reliability and documented failure modes

### Takeaway
Agent benchmarks show meaningful but far-from-reliable performance, with documented validation overfitting, invalid submissions, benchmark leakage/contamination and silent state errors. Verdict: worthwhile as a productivity tool with human verification, not safe as an unsupervised pipeline.

### Cited Findings
- MLE-bench leaderboard (repo): top entries are Famou-Agent 2.0 at 64.44% (Gemini-3-Pro-Preview) and AIBuildAI at 63.11% (Claude-Opus-4.6); new submissions are paused "while we develop an improved process for ensuring submissions are fair and comparable". The repo documents 14+ competition-specific issues, including label leakage via public test files, validation-script failures, test set preparation errors, and a split built from a public labelled corpus agents may find. A plagiarism detector exists in extras. — [openai/mle-bench](https://github.com/openai/mle-bench) (WEBFETCH). The metric here is medal rate over Kaggle competitions with large compute/time budgets, not 24-48h datathon conditions.
- Original MLE-bench: o1-preview with AIDE reached a 16.9% medal rate. — [arXiv 2410.07095](https://arxiv.org/pdf/2410.07095) (SNIPPET-ONLY)
- "AI Research Agents for MLE-bench": best agent raised MLE-bench lite medal rate from 39.6% to 47.7%; choosing the final solution by test instead of validation score would add 9-13 points absolute, indicating systematic validation overfitting; agents often produced invalid submissions even with a validation server and did not always use it; pass@6 roughly double pass@1, i.e. high run-to-run variance. — [arXiv 2507.02554](https://arxiv.org/html/2507.02554) (SNIPPET-ONLY)
- Same snippet-level claim: top agents "routinely miss pipeline errors, struggle with long-horizon code dependencies, and overfit to validation metrics". — search result summary of MLE-bench literature (SNIPPET-ONLY, attribution to specific paper unclear)
- DSEval (as reported in a search summary): up to 27% of failures are silent state-integrity violations (e.g. in-place dataframe mutation) even when the final output is correct. — [LLM-Based Data Science Agents survey](https://arxiv.org/pdf/2510.04023) (SNIPPET-ONLY, attribution of the 27% figure to DSEval unverified)
- ELT-Bench: 57% success on data loading vs 3.9% on transformation. — same search summary (SNIPPET-ONLY, source paper not identified)
- Silent-failure patterns listed: code runs but groups by the wrong key, omits an exclusion filter, rationale claims exclusion that did not happen, truncated code returning "no output". — [Trace Integrity for LLM Data Agents](https://arxiv.org/pdf/2608.26036) (SNIPPET-ONLY, vision paper)
- Other benchmarks surfaced but not read: AgenticDataBench [2607.01647](https://arxiv.org/pdf/2607.01647), ML2B [2509.22768](https://arxiv.org/pdf/2509.22768), MLE-STAR [2506.15692](https://arxiv.org/pdf/2506.15692) (titles only).

### Inferences
- MLE-bench scores are inflated or noisy by leakage/contamination in some tasks and by large budgets, so they overstate datathon reliability; the more relevant lesson is the validation-overfitting and invalid-submission findings.
- "Hallucinated columns" and "wrong CV" are plausible per the above silent-failure classes but I found no source quantifying them specifically.
- Agents that run many iterations on a small validation set will overfit it; datathon teams doing this with a public leaderboard face the same issue.

### Gaps
- No primary-source read of Data Interpreter, AutoKaggle, Jupyter AI, Claude Code or Cursor evaluations; no data on agent-specific rate of target leakage / wrong CV split. Not found, not invented.
- Survey 2510.04023 and DSEval details not opened.

## 3. Verifying agent-written ML code: assertions, baselines, checklists

### Takeaway
No agent-specific published checklist was found. General leakage checklists apply directly and are cheap to run. Verdict: always apply them to agent-written code.

### Cited Findings
- General leakage checklist items: preprocess after splitting; never `.fit()` on unsplit data; impute using training statistics only; use `sklearn.pipeline.Pipeline` inside CV; apply SMOTE/oversampling only after the split; use time-based splits for temporal data; use group splits when entities have multiple rows; vet very-high-correlation features; check for duplicate rows across train/test. — [Kaggle leakage guide](https://www.kaggle.com/general/423738), [Kaggle discussion](https://www.kaggle.com/discussions/general/532001), [Wikipedia: Leakage](https://en.wikipedia.org/wiki/Leakage_(machine_learning)) (SNIPPET-ONLY summaries)
- `leakcheck`, a linter for train/test leakage in ML datasets, exists. — [Sofienguy1/leakcheck](https://github.com/Sofienguy1/leakcheck) (SNIPPET-ONLY; maturity/quality unknown, UNVERIFIED, vet before use)
- MLE-bench findings imply: use the provided validation/submission checker, and do not pick final models by repeated peeking at one validation score. — [arXiv 2507.02554](https://arxiv.org/html/2507.02554) (SNIPPET-ONLY)

### Inferences (my own suggested checks, not from a source)
- Assertions to require: no ID/row overlap between train and validation folds; target absent from feature list; submission row count, column names and ID order match the sample submission; predictions not constant; CV metric matches the competition metric.
- Baseline sanity: compare against a dummy model and a default LightGBM/CatBoost run; a score far above the baseline on first try suggests leakage; verify the agent's CV split strategy (group/time) matches how the test set was built.
- Have a human re-read data loading, split and feature code every time; pin seeds; keep a held-out slice untouched until the end.

### Gaps
- No published agent-specific verification checklist or tool (e.g. Claude Code/Cursor guidance) found; leakcheck effectiveness untested.
