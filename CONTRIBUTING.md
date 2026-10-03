# Contributing

<!-- markdownlint-disable MD013 -->

Thanks for helping keep this guide accurate. Tools, prices and rules change fast, so small corrections are the most valuable contributions.

## Good contributions

- A broken link, outdated version, or changed price (include the source URL and date)
- A fix or addition to a checklist that you used at a real event (or open an **Event report** issue)
- Clearing items from [`VERIFICATION.md`](VERIFICATION.md): check a flagged claim at its primary source and replace the label with the link
- A better prompt, template or worksheet
- Replacing an UNVERIFIED or SNIPPET-ONLY claim with one you checked at the primary source

## Rules of the house

1. **Label evidence.** Cite a source, or mark the claim `our suggestion`, `SNIPPET-ONLY` (seen only in a search summary) or `UNVERIFIED`.
2. **Mark volatile facts.** Prices, free-tier limits and version numbers get a date and a "re-check" note.
3. **One page, one job.** Link new pages from the README of their folder, and from the root `README.md` if they answer an "I want to..." question.
4. **Keep numbering.** Folders are numbered in reading order; do not renumber without updating links.
5. **Credit adapted work.** If you adapt content from another project, check its licence, credit it at the bottom of the page, and add its notice to `THIRD_PARTY_NOTICES.md`.
6. **No secrets or real personal data** in examples. Use synthetic data.

## Before you open a pull request

```bash
python tools/check_links.py   # must print "0 broken links, 0 duplicate groups"
# If you touched any .py file or a pin (in a Python 3.12 env built from
# 08-datathon-handbook/PROJECT_TEMPLATE/requirements.txt):
python tools/smoke_test.py
```

Other helpers:

```bash
python tools/list_unverified.py   # regenerate VERIFICATION.md after editing a page with flagged claims
python tools/check_versions.py    # pinned versions vs PyPI
python _research/distribute.py    # regenerate docs/evidence.md and docs/repos.md from _research
npx markdownlint-cli2 "**/*.md"   # markdown lint (rules in .markdownlint-cli2.jsonc)
```

All of these run in CI. Research notes live under `_research/` (see its README); do not edit them by hand, add a new run folder instead.
