# Roadmap

<!-- markdownlint-disable MD013 -->

What is next for this guide, in rough priority order. Contributions welcome (see [`CONTRIBUTING.md`](CONTRIBUTING.md)).

## Needs a decision from the maintainer

- [ ] **Add a `LICENSE`.** Without one, others cannot legally reuse the guide.

## Needs verification (open web access required)

The full, line-by-line list is generated in [`VERIFICATION.md`](VERIFICATION.md). Summary by page:

Research runs 5-8 and round two were limited by a blocked sandbox, so many claims are SNIPPET-ONLY or UNVERIFIED. Re-check them at the primary source:

- Datathon rubrics and AI-use policies ([`judging-and-winning-evidence.md`](08-datathon-handbook/01-playbook/judging-and-winning-evidence.md))
- Kaggle rules (read from GitHub copies), WiDS 2025/26 rules, DataFest NDA ([`datathon-rules-and-licences.md`](08-datathon-handbook/01-playbook/datathon-rules-and-licences.md))
- Hackathon rules on AI, pre-existing code, IP; hardware winners; HackMIT/PennApps/Cal Hacks rules ([`rules-hardware-a11y-remote.md`](01-hackathon-playbook/docs/rules-hardware-a11y-remote.md))
- Free-compute quotas and hosting limits ([`tooling-2026-update.md`](08-datathon-handbook/03-modeling/tooling-2026-update.md))
- LLM provider data-retention terms, HIPAA/GDPR wording ([`privacy-and-pii.md`](01-hackathon-playbook/docs/privacy-and-pii.md))
- Benchmark claims behind [`method-selection-guide.md`](08-datathon-handbook/03-modeling/method-selection-guide.md)

## Planned content

- [ ] Run the deep-learning code in [`deep-learning-quickstart.md`](08-datathon-handbook/03-modeling/deep-learning-quickstart.md) and the untested snippets in [`ml-in-your-app.md`](04-ai-and-rag/docs/ml-in-your-app.md) (PyTorch, transformers, transformers.js) on a normal machine
- [ ] Run the image and sentence-embedding recipes and mark them tested in [`baseline-recipes.md`](08-datathon-handbook/03-modeling/baseline-recipes.md). Blocked in our sandbox: Hugging Face and the PyTorch wheel index were unreachable on 2026-10-02, so this needs a normal machine

## Re-check cadence

Prices, quotas and tools: next re-check due **2026-10-22** (see the root README). A GitHub Actions workflow (`monthly-recheck.yml`) opens a re-check issue on the 1st of each month with the PyPI version drift and the count of flagged claims. Re-run `python tools/smoke_test.py` after changing any pin.

## Done

- Datathon handbook with runbook, prompts, method selection and tested baseline recipes
- Reorganized handbook (`00-start-here`, `08-ai-agent-kit`)
- `WORKFLOW.md`: one phase map for both tracks
- De-duplication: single `SKILLS.md`, datathon prompts merged into `DT01`-`DT20`
- CI: link and duplicate-file check, plus a smoke test of every handbook script on pinned Python 3.12 dependencies
- All Claude research runs (1-10) archived in full in `_research/` (report + raw notes)
- Pinned `PROJECT_TEMPLATE/requirements.txt` (tested together)
- Fixed: quickstart artifact path after the reorganization; AutoGluon script no longer prints a made-up leaderboard
- From Hackathon Starter Pack (MIT, credited in `THIRD_PARTY_NOTICES.md`): finding hackathons, sponsor tracks, judge Q&A and story templates, demo-day testing, non-coder guide, staying well, CV/LinkedIn formulas
- Datathon rules and licences, privacy/PII guide, free compute, hardware judging, solo entry and university rules
- Runnable end-to-end datathon example on synthetic data (`07-worked-example/run_end_to_end.py`), in the smoke test
- `VERIFICATION.md` generated checklist, `tools/check_versions.py`, markdown lint and generated-page checks in CI, monthly re-check issue
- Shorter README; reference material moved to `FAQ.md`; printable cheat sheets for both tracks
- Issue templates: outdated info, missing topic, event report
- ML gap fill (run 11): fundamentals and metrics, error analysis and tracking, deep-learning quickstart, ML in apps, getting data; tabular, ONNX/FastAPI, fairlearn, cleanlab, sliceline and Evidently code tested on the sample data
