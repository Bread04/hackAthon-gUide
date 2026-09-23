---
name: hackathon-backend
description: Backend toolkit for hackathons covering Next.js 16 (proxy.ts) + Supabase (RLS, migrations) + Clerk (Third-Party Auth), RFC 9457 errors, service layers, OWASP API 2023, and AI/RAG on pgvector. Use when designing, scaffolding or reviewing a hackathon backend.
---

# Backend Toolkit

<!-- markdownlint-disable MD013 -->

> Condensed rules. Full detail: `../../03-backend/docs/` and `../../04-ai-and-rag/docs/` · macros in `../../03-backend/PROMPTS.md`. Verified 2026-09-21.

## Stack defaults

Next.js 16 · Supabase (Postgres + pgvector) · Clerk via **Supabase Third-Party Auth** · Zod · Vercel AI SDK · Upstash rate limiting.

## ⚠️ Don't use outdated patterns

| Outdated | Current |
| --- | --- |
| Clerk "Supabase JWT template" | **Deprecated since 2025-04-01.** Use Third-Party Auth: `role: authenticated` claim + supabase-js `accessToken()` |
| `middleware.ts` | **`proxy.ts`** on Next.js 16+ (and it's not your only auth check) |
| RFC 7807 | **RFC 9457** `application/problem+json` |
| Assistant prefill for JSON (Claude) | Returns 400 on 4.6+; use structured outputs or tools |
| Supabase Auth `getUser()` / `getSession()` on the server | **`getClaims()`** (verifies the signature on every call) |
| `railway add --plugin`, `supabase db seed`, "Vercel 10 s timeout" | `railway add --database`, `db reset` / `db push --include-seed`, Hobby 300 s (`../hackathon-deployment/SKILL.md`) |
| Relying on automatic table grants | **Explicit `GRANT` per table.** Supabase made exposure opt-in for new projects on 2026-05-30 and enforces it on all projects on **2026-10-30** |

## Speed modes

⚡ JavaScript + small validators + `{data}/{error}`, **or** 🛡️ TypeScript + Zod + RFC 9457. Pick one per project (`../../03-backend/docs/hackathon-patterns.md`). Use `DEMO_MODE` for synthetic data, keep LLM calls server-side with a pinned model, and always have a fallback.

## Architecture

- **Reads:** Server Components → server-only DAL → DTOs.
- **Mutations:** Server Actions (thin: `await auth()` → Zod → service). *Server Actions are public endpoints.*
- **REST:** Route Handlers only for webhooks and external callers.
- **Layers:** `app/` → `server/services` (authorisation, rules) → `server/repositories` (Supabase only).

## Database

RLS on every exposed table · one policy per operation · `to authenticated` · `(select auth.jwt()->>'sub')` with a `text` `user_id` for Clerk · index policy columns · migrations only (`migration new` → `db reset` → `db push`) · pgTAP RLS tests.

## Security (OWASP API 2023)

Ownership checks (API1) · DTOs (API3) · rate limits with **429**, especially on LLM calls (API4/6) · allowlist fetched URLs (API7) · no `NEXT_PUBLIC_` server keys (API8) · delete test routes (API9) · treat LLM and third-party output as untrusted (API10).

## AI and RAG

Sonnet 5 ($2/$10) as workhorse · Haiku 4.5 ($1/$5) for bulk · Opus 5 ($5/$25) for the judged step. RAG: ~800-token chunks + contextual prefix → pgvector **HNSW** + tsvector **hybrid (RRF k=50)** → rerank 150→20 → documents first, query last.

## Free tiers

Supabase: 2 projects, 500 MB, **pauses after 7 days idle**. Clerk Hobby: 50k MRU, **fixed 7-day sessions**.

## Related skills

Deploying: `../hackathon-deployment/SKILL.md` · something broke: `../hackathon-troubleshooting/SKILL.md` · team repo and conflicts: `../hackathon-git-teamwork/SKILL.md`.

## Sources

[RFC 9457](https://www.rfc-editor.org/rfc/rfc9457.html) · [Supabase × Clerk](https://supabase.com/docs/guides/auth/third-party/clerk) · [Clerk quickstart](https://clerk.com/docs/nextjs/getting-started/quickstart) · [Next.js data security](https://nextjs.org/docs/app/guides/data-security) · [Supabase RLS](https://supabase.com/docs/guides/database/postgres/row-level-security) · [OWASP API 2023](https://api-security.owasp.org/editions/2023/en/0x11-t10) · [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing) · [Supabase hybrid search](https://supabase.com/docs/guides/ai/hybrid-search)
