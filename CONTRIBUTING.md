# Contributing

<!-- markdownlint-disable MD013 -->

Thanks for helping keep this guide accurate. Tools, prices and rules change fast, so small corrections are the most valuable contributions.

## Good contributions

- A broken link, outdated version, or changed price (include the source URL and date)
- A fix or addition to a checklist that you used at a real event
- A better prompt, template or worksheet
- Replacing an UNVERIFIED or SNIPPET-ONLY claim with one you checked at the primary source

## Rules of the house

1. **Label evidence.** Cite a source, or mark the claim `our suggestion`, `SNIPPET-ONLY` (seen only in a search summary) or `UNVERIFIED`.
2. **Mark volatile facts.** Prices, free-tier limits and version numbers get a date and a "re-check" note.
3. **One page, one job.** Link new pages from the README of their folder, and from the root `README.md` if they answer an "I want to..." question.
4. **Keep numbering.** Folders are numbered in reading order; do not renumber without updating links.
5. **No secrets or real personal data** in examples. Use synthetic data.

## Before you open a pull request

```bash
python tools/check_links.py   # must print "0 broken links"
```

The same check runs in CI. Research notes live under `_research/` (see its README); do not edit them by hand, add a new run folder instead.
