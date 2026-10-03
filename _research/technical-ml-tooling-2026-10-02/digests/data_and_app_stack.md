# Data-wrangling, explainability and app stack for datathons (state as of Oct 2026)

Reliability warning: WebFetch summaries of GitHub/PyPI pages frequently printed wrong years (e.g. "2024") next to current versions. Version numbers below are what the pages showed; exact release dates are only given when they looked consistent. GitHub API (gh) was blocked and docs.pola.rs, pandas.pydata.org, gradio.app, huggingface.co, docs.streamlit.io and render.com were egress-blocked, so vendor-doc claims come via search snippets (SNIPPET-ONLY). Re-check all prices and limits.

## 1. Latest stable versions and what breaks 2024 copy-paste code

### Takeaway
Versions seen (Oct 2026): Polars 1.44.2 (2.0.0-rc.2 pre-release exists), DuckDB 1.5.6, Streamlit 1.64.0, Gradio 6.29.0, marimo 0.25.1, pandas 3.0.6, pandera 0.33.1, SHAP 0.52.0 on PyPI (0.53.0rc0 on GitHub), ydata-profiling 4.18.4 on PyPI vs 4.20.0 on GitHub (conflict). The biggest 2024-code breakers are pandas 3.0 (string dtype, Copy-on-Write), Gradio 6 (launch/theme/Chatbot changes), Streamlit removal of old st.cache, SHAP Python floor, and the upcoming Polars 2.0 row-order change.

### Cited Findings
- Polars: latest stable Python Polars 1.44.2; pre-release 2.0.0-rc.2. Dates shown on the page were garbled so are not reported. — [GitHub releases](https://github.com/pola-rs/polars/releases); [PyPI](https://pypi.org/pypi/polars/json) shows 1.44.2 as info.version, Python 3.10+.
- Polars 2.0-rc breaking changes listed: Parquet ENUM read as pl.String, new Map dtype, cut()/qcut() deprecated, zero-width input behavior changes. — [GitHub releases](https://github.com/pola-rs/polars/releases) (page summary; verify against release notes)
- Polars 2.0 pre-release: engine="auto" for LazyFrame.collect now resolves to the streaming engine; streaming engine does not guarantee row order for join, group_by, unpivot; fix with explicit sort or maintain_order=True; claimed ~5x faster. SNIPPET-ONLY from [Polars pre-release post](https://pola.rs/posts/announcing-polars-2/), [upgrade guide](https://docs.pola.rs/releases/upgrade/2/), and secondary [runtimewire](https://runtimewire.com/article/polars-2-0-rc-streaming-default-ritchie-vink).
- Polars GPU: GitHub releases page summary said GPU support "remains limited"; no usable source found on cudf/GPU engine status. UNVERIFIED.
- DuckDB: latest bugfix release v1.5.6 (page dated Sep 28 2026); v1.5.5 dated Jul 22 2026; v1.5.0 "Variegata" was the major release. Backward compatible within 1.5.x. — [GitHub releases](https://github.com/duckdb/duckdb/releases). PyPI summary showed 1.5.6 but with inconsistent dates (ignore) — [PyPI](https://pypi.org/pypi/duckdb/json).
- Streamlit: latest 1.64.0 (PyPI shows 1.64.0, previous 1.63.1, 1.63.0, 1.62.2; Python 3.10-3.14). New: st.echarts_chart, async-aware st.cache_data/cache_resource, on_change="ignore" modes, live param on st.text_input, st.pagination. v1.62.0 removed the old st.cache API and global-figure support in st.pyplot, deprecated savefig kwargs. — [GitHub releases](https://github.com/streamlit/streamlit/releases); [PyPI](https://pypi.org/pypi/streamlit/json)
- Gradio: latest 6.29.0, Python 3.10-3.13 on PyPI. — [GitHub releases](https://github.com/gradio-app/gradio/releases); [PyPI](https://pypi.org/pypi/gradio/json)
- Gradio 6 breaking changes (released around Nov 2025 per [AlternativeTo](https://alternativeto.net/news/2025/11/gradio-6-released-with-faster-performance-for-creating-machine-learning-apps-in-python)): theme/css/js/head moved from gr.Blocks to launch(); show_api replaced by footer_links; event-listener show_api removed and api_name=False replaced by api_visibility; Chatbot tuple format removed; Sketch deprecated. Upgrade to 5.50 first to see deprecation warnings; only 6.x gets support. SNIPPET-ONLY from [Gradio 6 migration guide](https://gradio.app/main/guides/gradio-6-migration-guide) via search.
- marimo: latest 0.25.1 (Oct 1 2026), 0.25.0 (Sep 23 2026: Pixi sandbox backend, WASM/single-file HTML export), 0.24.x (HF Hub dataset integration, slide export). Python 3.10-3.14. — [GitHub releases](https://github.com/marimo-team/marimo/releases); [PyPI](https://pypi.org/pypi/marimo/json)
- ydata-profiling: GitHub page showed 4.20.0 (adds Python 3.14) and 4.19.1 renaming the package to "data-profiling"; PyPI json for ydata-profiling showed 4.18.4 (Apr 22 2026), 4.18.1 (Jan 13 2026), 4.18.0 (Nov 21 2025). CONFLICT: likely the project moved to a new PyPI name "data-profiling"; the old name appears stale. Verify with `pip index versions data-profiling`. — [GitHub](https://github.com/ydataai/ydata-profiling/releases); [PyPI](https://pypi.org/pypi/ydata-profiling/json)
- SHAP: PyPI showed 0.52.0 as latest (requires Python >=3.12 per that page); GitHub showed v0.53.0rc0 as newest tag. Since 0.50.0 Python 3.9/3.10 dropped (min 3.11 per GitHub summary; conflicts with ">=3.12" in the PyPI summary, so check the exact floor); numba/llvmlite no longer required; build moved to scikit-build-core + nanobind. — [GitHub releases](https://github.com/shap/shap/releases); [PyPI](https://pypi.org/pypi/shap/json)
- pandas: 3.0.0 shipped Jan 21 2026 (SNIPPET-ONLY from [InfoQ/others](https://www.infoq.com/news/2026/02/pandas-library)); latest 3.0.6, Python 3.11+; 3.1.0rc0 exists. — [GitHub releases](https://github.com/pandas-dev/pandas/releases); official post [Pandas 3.0 Released](https://pandas.pydata.org/community/blog/pandas-3.0.html) (not fetched)
- pandas 3.0 breakers: default str dtype instead of object (checks like dtype == "object" break; use pd.api.types.is_string_dtype); Copy-on-Write is the only mode; chained assignment such as df[df.A>0]["B"]=1 now raises/does nothing useful instead of warning; SettingWithCopyWarning removed; new default datetime resolution; initial pd.col syntax; removed everything deprecated in 2.x (upgrade to 2.3 first). — [GitHub releases](https://github.com/pandas-dev/pandas/releases); SNIPPET-ONLY details from search of [migration guides](https://medium.com/@yogeshkrishnanseeniraj/pandas-3-0-migration-guide-what-actually-breaks-why-and-how-to-fix-it-bebe95fbd053)
- pandera: latest 0.33.1 (PyPI date shown as Aug 15 but year garbled); 0.33.0 added CLI and PyArrow table support; 0.32.0 narwhals backend (pa.config.set_config(use_narwhals_backend=True)); pandas 3.0 compatibility in 0.30.0. Use `import pandera.pandas as pa` / `pandera.polars` rather than top-level `import pandera as pa` for DataFrameModel. — [GitHub releases](https://github.com/unionai-oss/pandera/releases); [PyPI](https://pypi.org/pypi/pandera/json). Note: top-level import deprecation inferred from the module layout; verify.

### Inferences
- A guide pinning pandas<3 or teaching `.loc` assignments avoids most breakage; string-dtype and chained-assignment lines are the highest-risk 2024 snippets.
- Polars users should pin polars<2 or add explicit sorts, since 2.0 will arrive as breaking for order-dependent code.
- Python 3.12+ is the safe baseline for SHAP, pandas 3, and everything above.

### Gaps
- Exact release dates for Polars 1.44.2, Streamlit 1.64.0, Gradio 6.29.0, SHAP 0.52.0 (fetch summaries garbled or omitted dates).
- Polars GPU engine status in 2026; Polars 1.x streaming engine stabilisation details.
- Whether Polars 2.0 stable has shipped (only rc.2 seen).
- ydata-profiling vs data-profiling naming and which is current.

## 2. New tools worth recommending to beginners

### Takeaway
Search found little primary evidence on 2025-26 adoption; only generic comparisons. Streamlit, Gradio and marimo remain the beginner picks; Evidently is useful for drift/data-quality reports; others are optional.

### Cited Findings
- Dash 3 shipped Pages, background callbacks and AG Grid in March 2025; Dash 4 redesigned core components in early 2026; Streamlit was acquired by Snowflake; Streamlit ships in 1-3 days vs Dash/Panel 1-2 weeks. — [Fastero 2026 roundup](https://fastero.com/blog/best-data-visualization-tools-2026), [Databrain guide](https://www.usedatabrain.com/how-to/create-python-dashboard) (SNIPPET-ONLY, secondary blogs)
- Evidently: open-source ML and LLM observability framework, 100+ metrics, data drift/quality reports, test suites, dashboards. — [GitHub](https://github.com/evidentlyai/evidently)
- marimo: reactive, git-friendly notebooks deployable as apps, WASM export, HF dataset integration (see above).
- Vizro, Taipy, Panel, Hex, Great Tables: only listed as existing in search; no versions, status or adoption evidence found. — [Planeks overview](https://www.planeks.net/python-dashboard-development-framework/)

### Inferences
- Beginner recommendation (my judgement): Streamlit or Gradio for demos; marimo as optional notebook; Evidently only if drift/monitoring matters; Panel/Dash/Vizro/Taipy/Hex are not needed for a 24-48h datathon; Great Tables fine for polished tables but low priority.

### Gaps
- Trackio, W&B, MLflow status; Hex free tier; Vizro/Taipy current releases; Great Tables version. Not researched or found.

## 3. Free hosting for demos (all RE-CHECK; SNIPPET-ONLY from search results unless noted)

### Takeaway
Free hosting got tighter: Hugging Face now restricts creating Gradio/Docker Spaces on free accounts, Fly.io has no free tier for new customers, Railway is a one-time trial. Streamlit Community Cloud and Render free remain usable but sleep.

### Cited Findings
- Streamlit Community Cloud: apps with no traffic for 12 hours go to sleep; wake by visiting (a click-to-wake); limits CPU 0.078-2 cores, memory 690MB-2.7GB, storage up to 50GB. — [Streamlit docs: Manage your app](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app); [Fastero explainer](https://fastero.com/blog/why-your-streamlit-app-keeps-sleeping); commits may no longer reliably prevent sleep per [forum](https://discuss.streamlit.io/t/how-to-prevent-the-app-enter-the-sleep-mode/87959)
- Hugging Face Spaces: CPU Basic free = 2 vCPU, 16GB RAM; sleeps after 48h inactivity. Static Spaces free; Gradio and Docker Spaces require a paid plan (PRO $9/month) to create; free accounts in good standing can host up to 2 ZeroGPU Gradio Spaces; free ZeroGPU ~5 min/day (PRO 40 min). Docker restriction began around July 2026. — [HF Spaces docs](https://huggingface.co/docs/hub/en/spaces-overview), [GPU docs](https://huggingface.co/docs/hub/en/spaces-gpus), [HF forum: new free accounts cannot create CPU Basic Gradio Spaces](https://discuss.huggingface.co/t/new-free-accounts-cannot-create-cpu-basic-gradio-spaces-only-zerogpu-available/177629), [forum: Docker SDK marked Paid](https://discuss.huggingface.co/t/docker-sdk-now-marked-as-paid-when-creating-a-new-space/177580), [eesel pricing](https://www.eesel.ai/blog/hugging-face-pricing)
- Render: free web services spin down after 15 min without inbound traffic, ~1 min cold start; 750 free instance hours per workspace per month, suspended after exhaustion. — [Render free-tier article](https://render.com/articles/platforms-with-a-real-free-tier-for-developers-in-2026), [DEV post](https://dev.to/sanjaysah/750-free-hours-a-month-but-a-month-is-730-3-free-tier-mistakes-that-took-down-my-app-17g8)
- Fly.io: no free tier for new accounts (since 2024); trial of 2 hours compute or 7 days; pay-as-you-go (shared-cpu-1x 256MB ~ $2.02/month). Legacy plans keep 3 small VMs. — [Costbench](https://costbench.com/software/developer-tools/flyio/free-plan/), [agentdeals](https://agentdeals.dev/vendor/fly-io) (third-party; check fly.io/docs/about/pricing)
- Railway: 30-day trial with $5 credit, credit card required per one source; afterwards a free plan at ~$1/month usage with up to 1 vCPU/0.5GB; Hobby $5/month, Pro $20/month. Sources conflict on card requirement/free-plan wording (one says "No Credit Card" in a title). — [bex.co](https://bex.co/blog/2026/07/29/railway-free-tier-trial-credit), [saaspricepulse](https://www.saaspricepulse.com/tools/railway), [srvrlss](https://www.srvrlss.io/provider/railway/)

### Inferences
- For a datathon: Streamlit Community Cloud (GitHub repo, free) is the simplest free demo; wake the app before judging. Gradio demos on HF need PRO or a static/ZeroGPU route; Render free is the fallback with a ~1 min cold start.

### Gaps
- Official pricing pages not fetched (blocked); whether Streamlit SDK Spaces still free on HF; Render free Postgres expiry terms; Railway exact current terms.
