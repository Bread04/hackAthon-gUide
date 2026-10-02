# Research evidence

<!-- markdownlint-disable MD013 -->

> Where every claim in this toolkit comes from. Folder names follow the BMad Deep Recon convention, so `/bmad-deep-recon` → **Refresh** can find and update them. Don't rename them.

| Folder / file | What it is | Key output |
| --- | --- | --- |
| [`technical-hackathon-toolkit-2026-09-21/`](technical-hackathon-toolkit-2026-09-21/research.md) | Research run 1: hackathon strategy, frontend, backend, AI/RAG, MCP (56 sources) | `research.md`, which feeds folders 01–05 (and their `docs/evidence.md`) |
| [`technical-hackathon-github-repos-2026-09-21/`](technical-hackathon-github-repos-2026-09-21/research.md) | Research run 2: high-star GitHub repos, with 278 repos API-verified (34 sources) | `research.md` + `repo-tables.md`, which feed `06-repo-catalog/` and every folder's `docs/repos.md` |
| [`technical-agents-and-fullstack-practices-2026-09-23/`](technical-agents-and-fullstack-practices-2026-09-23/research.md) | Research run 3: AI agents and workflows, plus a backend/frontend currency check (123 sources) | `research.md`, which feeds `04-ai-and-rag/docs/agents-and-tool-use.md`, `skills/hackathon-ai`, the backend and frontend skills, and folders 02–05 `docs/evidence.md` (labels **A**) |
| [`technical-datathons-and-ml-solutions-2026-09-23/`](technical-datathons-and-ml-solutions-2026-09-23/research.md) | Research run 4: Datathons & ML hackathons, real-world data pipelines, rapid modeling, and last-mile solutions (14 key sources) | `research.md`, which feeds `08-datathon-handbook/` and `skills/hackathon-datathon` |
| [`technical-datathon-judging-2026-10-02/`](technical-datathon-judging-2026-10-02/research.md) | Research run 5: datathon judging rubrics, winner post-mortems, logistics and rules. **Evidence is thin:** most event sites were unreachable, so many items are snippet-only or UNVERIFIED | `research.md` (full report) + `digests/`; condensed into `08-datathon-handbook/01-playbook/judging-and-winning-evidence.md` |
| [`technical-hackathon-gap-fill-2026-10-02/`](technical-hackathon-gap-fill-2026-10-02/research.md) | Research run 6: gap fill, covering AI-use/IP rules, hardware track, accessibility/privacy, remote demos. **Evidence is thin** (mostly snippet-only or UNVERIFIED) | `research.md` (full report) + `digests/`; condensed into `01-hackathon-playbook/docs/rules-hardware-a11y-remote.md` |
| [`technical-ml-tooling-2026-10-02/`](technical-ml-tooling-2026-10-02/research.md) | Research run 7: ML tooling currency (versions, benchmarks, breaking changes, hosting, AI-agent verification). Versions from PyPI; the rest mostly snippet-only | `research.md` (full report) + `digests/`; condensed into `08-datathon-handbook/03-modeling/tooling-2026-update.md` |
| [`technical-ml-methods-2026-10-02/`](technical-ml-methods-2026-10-02/research.md) | Research run 8: which ML method per datathon scenario, plus imbalance/calibration/ensembling/validation. Almost entirely snippet-only | `research.md` (full report) + `digests/`; condensed into `08-datathon-handbook/03-modeling/method-selection-guide.md` |
| [`technical-datathon-gap-fill-round-two-2026-10-02/`](technical-datathon-gap-fill-round-two-2026-10-02/research.md) | Research run 9: datathon rules and licences, baseline recipes, free compute, PII, hardware/solo/university rules. Mixed evidence (Kaggle rules via GitHub copies; much snippet-only) | `research.md` + `digests/` + `imports/recipe-tests/` (the scripts we ran); feeds `datathon-rules-and-licences.md`, `baseline-recipes.md`, `tooling-2026-update.md` §6, `privacy-and-pii.md`, `rules-hardware-a11y-remote.md` §5 |
| [`technical-multi-agent-llm-rag-2026-10-02/`](technical-multi-agent-llm-rag-2026-10-02/research.md) | Research run 10: LLM mechanics, multi-agent systems, RAG variants, agents/RAG at hackathons (GitHub-first) | `research.md` + `digests/`; feeds `04-ai-and-rag/docs/how-llms-work.md`, `multi-agent-systems.md`, `rag-architecture.md` (RAG variants) |
| [`distribute.py`](distribute.py) | Splits both reports and `repo-tables.md` into each folder's `docs/evidence.md` and `docs/repos.md`. Run it from the toolkit root after a Refresh | Regenerated files |
| [`help-me-papi-import.md`](help-me-papi-import.md) | What was adapted from maxi-cmyk/help-me-papi, what was corrected, and what was left out | Import log |

Inside each run folder:

- `research.md`: the cited report
- `digests/`: raw findings from each researcher
- `imports/`: verified data, such as `github-metrics.json`
- `repo-tables.md` (run 2 only): the category tables, the source for every `docs/repos.md`
- `.memlog.md`: the append-only decision and claims log (runs 1-4)
- `imports/recipe-tests/` (run 9 only): the scripts used to test code in the guide

Every Claude research run is kept here in full (report + raw researcher notes), even when a guide page carries a condensed copy.
