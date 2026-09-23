---
name: hackathon-frontend
description: Frontend toolkit for hackathon UI covering clarity-first design, Tailwind v4 tokens, shadcn, WCAG 2.2 AA, Core Web Vitals budgets, anti-AI-slop rules, and the Stitch/21st/impeccable workflow. Use when building or reviewing UI during a hackathon.
---

# Frontend Toolkit

<!-- markdownlint-disable MD013 -->

> Condensed rules. Full detail: `../../02-frontend/docs/` · macros in `../../02-frontend/PROMPTS.md`. Verified 2026-09-21.

## Before any UI code

Write a brief: product/audience/job · 4–6 named colour tokens with contrast ratios · 1–2 type roles · 4 px spacing scale · ASCII wireframe · **one bold moment**.

## Standards (quality gates)

| Area | Gate |
| --- | --- |
| Performance | **LCP ≤ 2.5 s · INP ≤ 200 ms · CLS ≤ 0.1** (p75, mobile). JS ≤ ~170 KB and Lighthouse ≥ 80 are *heuristics* (2018 source) |
| Accessibility | **WCAG 2.2 AA**: text 4.5:1 (large 3:1); UI and focus 3:1; `:focus-visible` always; focus not hidden by sticky UI; targets ≥ 24×24 px; no drag-only; allow paste in logins |
| Semantics | Landmarks; one h1; no skipped levels; `<button>` for actions, `<a>` for navigation; labelled inputs |
| Tokens | Tailwind v4 `@theme` primitives → `@theme inline` semantics → components use semantics only. No raw hex or px |
| Whitespace | 8 px system with half steps: 4/8/12/16/24/32/48/64/80. Section gaps ≥ 48 px; text ≤ 65ch |

## Anti-slop (Impeccable + Anthropic frontend-design)

Avoid: default Inter/Arial · purple-to-blue gradients · glows on dark · identical or nested card grids · bounce easing · all-caps eyebrows and middle-dot metadata · pure #000 or grey · grey text on colour · cramped padding.

## Hackathon priorities

- **Hero focus:** about 70% of visual effort goes on the first viewport the judge sees.
- **One deliberate aesthetic** (`../../02-frontend/docs/aesthetics-directory.md`), recorded in `design.md`.
- **Reduction Framework:** Tier 1 visible, Tier 2 subdued, Tier 3 hidden. The judge's path only touches Tier 1.
- **Fake data first:** build the UI on hardcoded JSON, then swap in the API when it's ready.
- **Animate only transform and opacity** (60 fps). Toasts, not crashes.

## Tool flow

```text
Stitch (directions) → Stitch MCP → DESIGN.md → @theme tokens
→ shadcn MCP (base components) → 21st MCP (1 distinctive component)
→ Taste Skill (generate) → Claude Code assembles → /impeccable audit → polish
→ npx impeccable detect <demo-url> → Playwright screenshots
```

## Key facts (2026-09)

- Impeccable (pbakaus): 1 skill, 24 commands, 61 rules. `npx impeccable install` → `/impeccable init`.
- shadcn: Base UI is the default primitive layer (Jul 2026); `cn` package (Sep 2026); MCP via `mcp init --client claude`.
- DTCG tokens spec stable (2025.10). Style Dictionary converts it.
- WCAG 3 is still a draft (not before ~2028), so build to 2.2.
- Stitch sync is one-way (Stitch → code).

## Sources

[web.dev vitals](https://web.dev/articles/vitals) · [WCAG 2.2](https://www.w3.org/TR/WCAG22/) · [Tailwind theme](https://tailwindcss.com/docs/theme) · [Atlassian spacing](https://atlassian.design/foundations/spacing) · [Impeccable](https://github.com/pbakaus/impeccable) · [Anthropic frontend-design](https://github.com/anthropics/skills/blob/main/skills/frontend-design/SKILL.md) · [shadcn changelog](https://ui.shadcn.com/docs/changelog)
