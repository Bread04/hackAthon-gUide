# 🔎 RAG Architecture Standard

<!-- markdownlint-disable MD013 -->

> Contextual Retrieval + hybrid search + reranking, all inside Supabase Postgres. Verified 2026-09-21.

---

## Reference pipeline

```text
INGEST                                   QUERY
──────                                   ─────
docs → chunk (~800 tok)                  user question
     → contextualize each chunk            → embed query
       (cheap model + cached full doc)     → hybrid search: BM25-ish (tsvector) + vector (HNSW)
     → prepend 50–100 tok context             fused with RRF (k=50), ~150 candidates
     → embed  +  tsvector                  → rerank → top 20
     → Postgres (pgvector)                 → LLM with long-context prompt shape (docs first, query last)
```

### Why this shape

Anthropic's **Contextual Retrieval** results (retrieval failure rate, top-20) ([Anthropic, 2024-09](https://www.anthropic.com/news/contextual-retrieval)):

| Setup | Failure rate | Reduction |
| --- | --- | --- |
| Baseline embeddings | 5.7% | — |
| + Contextual embeddings | 3.7% | −35% |
| + Contextual BM25 | 2.9% | −49% |
| + Reranking | 1.9% | **−67%** |

> [!NOTE]
> These are **single-vendor figures from 2024**, so the model choices may be dated. The *pattern* is sound and widely copied. Measure on your own data.

**Contextualisation prompt (key sentence verbatim from Anthropic):**

```text
<document>
{{WHOLE_DOCUMENT}}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{{CHUNK_CONTENT}}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval. Answer only with the succinct context and nothing else.
```

The key sentence ("Please give a short succinct context…") is verbatim from the source. The `<document>`/`<chunk>` wrapper and the last sentence are reconstructed and weren't verified in this research, so compare them with the Anthropic post or the cookbook notebook before relying on them. Run it on **Haiku 4.5** with **prompt caching** of the full document and the **Batch API**.

---

## Defaults

| Parameter | Default | Source / confidence |
| --- | --- | --- |
| Chunk size | ~800 tokens | Anthropic (high). Chroma's chunking report (SNIPPET-ONLY) found ~200-token recursive chunks consistently good, so **test 200 vs 800 on your golden set** |
| Chunk overlap | 10–15% | help-me-papi practitioner guidance (not separately verified) |
| Context prefix | 50–100 tokens | Anthropic (high) |
| Candidates before rerank | ~150 | Anthropic (high) |
| Chunks passed to the LLM | 20 | Anthropic (high) |
| Vector index | **HNSW** ("should be your default choice") | [Supabase](https://supabase.com/docs/guides/ai/vector-indexes/hnsw-indexes) (high) |
| Fusion | RRF, `rrf_k = 50`, full-text weight = semantic weight = 1 | [Supabase](https://supabase.com/docs/guides/ai/hybrid-search) (high) |
| Embedding | text-embedding-3-small / voyage-4-lite ($0.02/M) | see `model-selection.md` |
| Reranker | Voyage rerank-3-lite ($0.02/M) | [Voyage](https://docs.voyageai.com/docs/pricing) |

---

## Supabase schema (pgvector + hybrid)

```sql
create extension if not exists vector with schema extensions;

create table documents (
  id bigint primary key generated always as identity,
  source text not null,
  content text not null,                                  -- context prefix + chunk
  fts tsvector generated always as (to_tsvector('english', content)) stored,
  embedding extensions.vector(1536)                       -- match your model's dimensions
);

create index on documents using gin (fts);
create index on documents using hnsw (embedding vector_cosine_ops);
-- 3072-dim models: index a halfvec cast with halfvec_cosine_ops
```

**Opclasses:** `vector_cosine_ops` (`<=>`) · `vector_ip_ops` (`<#>`, for normalised embeddings) · `vector_l2_ops` (`<->`). **Keep the operator in your query consistent with the opclass** ([Supabase HNSW](https://supabase.com/docs/guides/ai/vector-indexes/hnsw-indexes)).

### Hybrid search function (RRF)

Adapted from the [Supabase hybrid search guide](https://supabase.com/docs/guides/ai/hybrid-search):

```sql
create or replace function hybrid_search(
  query_text text,
  query_embedding extensions.vector(1536),
  match_count int,
  full_text_weight float = 1,
  semantic_weight float = 1,
  rrf_k int = 50
) returns setof documents language sql as $$
with full_text as (
  select id, row_number() over (order by ts_rank_cd(fts, websearch_to_tsquery(query_text)) desc) as rank_ix
  from documents
  where fts @@ websearch_to_tsquery(query_text)
  order by rank_ix
  limit least(match_count, 30) * 2
),
semantic as (
  select id, row_number() over (order by embedding <=> query_embedding) as rank_ix
  from documents
  order by rank_ix
  limit least(match_count, 30) * 2
)
select documents.*
from full_text
full outer join semantic on full_text.id = semantic.id
join documents on coalesce(full_text.id, semantic.id) = documents.id
order by
  coalesce(1.0 / (rrf_k + full_text.rank_ix), 0.0) * full_text_weight +
  coalesce(1.0 / (rrf_k + semantic.rank_ix), 0.0) * semantic_weight
  desc
limit least(match_count, 30);
$$;
```

> Supabase's own example uses `vector(512)` with `vector_ip_ops` and `<#>`. The version above uses cosine (`<=>`) to match the index. Check the guide if you change the opclass.

Add RLS to `documents` if the corpus is per-user (see `../../03-backend/docs/database-supabase.md`).

---

## Vector store alternatives (only if not on Supabase)

| Store | Free tier (official pricing pages, 2026-09-22) | Notes |
| --- | --- | --- |
| [Pinecone](https://www.pinecone.io/pricing/) Starter | 2 GB storage, 2M write + 1M read units/mo, 5 indexes, 1 project, AWS us-east-1 only | Builder $20/mo flat |
| [Qdrant Cloud](https://qdrant.tech/pricing/) free cluster | 0.5 vCPU, 1 GB RAM, 4 GB disk, single node, free cloud inference on selected models | Card/inactivity terms not stated |
| [Chroma Cloud](https://www.trychroma.com/pricing) Starter | $0 + usage, **$5 free credits/month** | Or run Chroma embedded via pip (Apache-2.0) |
| [Weaviate Cloud](https://weaviate.io/pricing) Free | Always free: 100K objects, 1 GB memory, 10 GB disk, 1 collection | Embeddings 2K req/day |

**Recommendation:** stay on Supabase pgvector. It's one less service and one less set of keys, and RLS applies to your embeddings too.

---

## Retrieval refinements

From help-me-papi `AI/skills/chunking.md` and `retrieval.md`. These are practitioner guidance, not separately verified.

| Technique | When |
| --- | --- |
| **Semantic chunking** (split on headings or paragraphs, then cap the size) | Default over naive fixed-size splits |
| **Code:** chunk by function or class (AST-aware), never by line count | Code-search features |
| **Tables:** repeat the header row in every chunk | Spreadsheets, CSVs, PDFs with tables |
| **Chunk metadata:** source, section heading, position (N of M), `parent_id`, timestamp | Always. It enables citations, reranking and freshness filters |
| **Context expansion:** retrieve small chunks, then pull neighbouring chunks by `parent_id` | When answers need more surrounding context |
| **Metadata filters inside the SQL** (owner, tenant, date) *before* similarity search | Always, for per-user data. Never filter after retrieval |
| **Query rewriting:** the LLM turns the chat turn into 2–3 standalone queries | Conversational interfaces (`../PROMPTS-RAG.md` R10) |

> **Top-K difference:** help-me-papi reranks 20–50 candidates down to **3–8**, while Anthropic's Contextual Retrieval passes the **top 20** to the model. Start with 20 for recall, and cut it if latency or cost hurts.

## Failure triage: classify before you fix

| Failure class | Symptom | Fix |
| --- | --- | --- |
| **Retrieval miss** | The right chunk never appears in the candidates | Better chunking, contextual prefixes, hybrid search, query rewriting |
| **Ranking miss** | The right chunk is retrieved but buried below the cut | Better reranker; adjust RRF weights |
| **Generation miss** | The right chunks are in the prompt but the answer is still wrong | Tighter grounding prompt, citations, guardrails (`../PROMPTS-RAG.md` R9) |

Don't reach for a bigger model as the default fix for every failure class. Add real failures to the golden set so they can't come back.

## Evaluation (keep it light)

1. Write **20–50 golden questions** with the chunk that should answer each one.
2. Track **retrieval** (is the right chunk in the top-K?) separately from **answer quality**.
3. Change one thing at a time: contextual chunks → hybrid → rerank.
4. Track **cost and p50/p95 latency** next to quality. A 2% quality gain that doubles latency isn't automatically a win.
5. [Ragas](https://github.com/vibrantlabsai/ragas) (moved from explodinggradients; ⭐ 15.8k; last push 2026-02, slowing) works for metrics if you pin the version.

---

## Starter repos

| Repo | Why | Status |
| --- | --- | --- |
| [supabase-community/chatgpt-your-files](https://github.com/supabase-community/chatgpt-your-files) | Upload → chunk → embed → chat with RLS on Supabase | ⭐ 516 · pushed 2026-05 |
| [vercel/chatbot](https://github.com/vercel/chatbot) (formerly ai-chatbot) | Fork-and-go Next.js chat UI | ⭐ 21k · pushed 2026-07 |
| [vercel/ai](https://github.com/vercel/ai) | Streaming, tools, structured output | ⭐ 26.9k · pushed 2026-09 |
| [anthropics/claude-cookbooks](https://github.com/anthropics/claude-cookbooks) | Contextual retrieval and RAG notebooks | ⭐ 52.9k · pushed 2026-09 |
| [pgvector/pgvector](https://github.com/pgvector/pgvector) | Extension docs (HNSW `m`, `ef_construction`) | ⭐ 23k · pushed 2026-09 |

---

## RAG variants: when each is worth it

> 📚 Full research report and raw notes: [`technical-multi-agent-llm-rag-2026-10-02`](../../_research/technical-multi-agent-llm-rag-2026-10-02/research.md).

> Added 2026-10-02 from a GitHub-first research run; see the label key at the top of [`multi-agent-systems.md`](multi-agent-systems.md). Headline: Most RAG variants lose to three cheap upgrades

The guide's `rag-architecture.md` already prescribes contextual retrieval, hybrid search and reranking in Supabase. What it lacks is a view of the alternatives, with honest evidence grades. The rough order of value per hour of hackathon work is: **long context (if the corpus is small) > naive RAG > hybrid + rerank > contextual retrieval > agentic loop > parent-child > query rewriting > CRAG-style grading > LightRAG > ColPali > GraphRAG > Self-RAG training**. That ranking is an inference from the evidence below, not a measured result. One discrepancy needs flagging. The guide's default chunk size is about 800 tokens, following Anthropic. Chroma's chunking report found RecursiveCharacterTextSplitter at **200 tokens with no overlap** "performed consistently well across all metrics", with 85.4-89.5% recall, while LLMSemanticChunker reached 91.9% recall (SNIPPET-ONLY, [Chroma research](https://www.trychroma.com/research/evaluating-chunking); code at [chunking_evaluation](https://github.com/brandonstarxel/chunking_evaluation)). The two sources measured different things, so treat chunk size as a parameter to test on your golden set rather than a settled default.

| Variant | How it works | Cost / latency / complexity | Sourced evidence | Hackathon verdict |
|---|---|---|---|---|
| **Long context, no RAG** | Put the whole corpus in a cached prompt | Zero infrastructure; higher cost per query | "Knowledge bases smaller than 200,000 tokens can simply be included directly in the prompt" (vendor-reported, [Anthropic](https://www.anthropic.com/news/contextual-retrieval)). "When resourced sufficiently, LC consistently outperforms RAG", but RAG uses about 1.5k tokens against 10k-100k (SNIPPET-ONLY, [Li et al., EMNLP 2024](https://aclanthology.org/2024.emnlp-industry.66/)). Counterpoints exist ([NVIDIA 2409.01666](https://arxiv.org/pdf/2409.01666), [2501.01880](https://arxiv.org/pdf/2501.01880); content UNVERIFIED) | **Default for small corpora.** Ship on day 1 |
| **Naive RAG** | Chunk → embed → top-k → stuff into the prompt | Hours to build | Baseline top-20 retrieval failure of 5.7% (vendor-reported, [Anthropic](https://www.anthropic.com/news/contextual-retrieval)) | Build first; measure every add-on against it |
| **Hybrid BM25 + vectors, plus reranker** | Fuse lexical and dense rankings (RRF), then a cross-encoder rescoring step | Low effort; a reranker "typically introduces a 1–2 second delay" ([LightRAG](https://github.com/HKUDS/LightRAG)) | With contextual chunks: 5.7% → 2.9% without rerank, → 1.9% with rerank (vendor-reported, [Anthropic](https://www.anthropic.com/news/contextual-retrieval)) | **Biggest single upgrade.** Already the guide's standard |
| **Contextual retrieval** | An LLM writes 50-100 tokens of context per chunk before embedding and BM25 indexing | One ingestion pass; "$1.02 per million document tokens" with caching (vendor-reported, 2024, re-check) | −35% failure from contextual embeddings alone ([Anthropic](https://www.anthropic.com/news/contextual-retrieval)); reference notebook and eval set in [claude-cookbooks](https://github.com/anthropics/claude-cookbooks/tree/main/capabilities/contextual-embeddings) | Strong; no new infrastructure |
| **Agentic RAG** | Retrieval exposed as a tool in an agent loop; the model decides when, what and how often to search | Several LLM calls per query (no sourced latency figure) | Self-Route answered 82% of queries at the RAG step and cut cost 65% (Gemini-1.5-Pro) and 39% (GPT-4o) at comparable quality (SNIPPET-ONLY, [Li et al.](https://aclanthology.org/2024.emnlp-industry.66/)) | **The 2026 default for multi-hop questions.** Cap iterations |
| **Parent-child / sentence window** | Match on small chunks, return the larger parent section | Config change in LlamaIndex or LangChain (class names UNVERIFIED) | No primary benchmark found | Use for contracts and manuals |
| **Query rewriting / multi-query / HyDE** | The LLM rephrases or writes a hypothetical answer, which is embedded for search | +1 LLM call before retrieval | HyDE beat zero-shot Contriever on BEIR (SNIPPET-ONLY, [arXiv 2212.10496](https://arxiv.org/abs/2212.10496)). In LightRAG's LLM-judged tests HyDE won only 24.8% against LightRAG, near NaiveRAG (vendor-reported, [LightRAG](https://github.com/HKUDS/LightRAG)) | Multi-query for vague chat turns; HyDE is low priority |
| **CRAG / Self-RAG** | CRAG: an evaluator grades retrieved docs and falls back to web search. Self-RAG: a fine-tuned LM emits reflection tokens | Self-RAG needs training; CRAG-style grading is +1 call | [Self-RAG](https://github.com/AkariAsai/self-rag) (ICLR 2024); [CRAG](https://github.com/HuskyInSalt/CRAG) ([arXiv 2401.15884](https://arxiv.org/pdf/2401.15884.pdf)) | Don't train. Re-create the grade → rewrite → web-fallback loop in LangGraph as a "self-correcting" demo story |
| **LightRAG** | Knowledge graph plus vectors, dual-level retrieval, incremental updates | pip install with a web UI; default stores "not for production", Postgres recommended | Overall win rates vs NaiveRAG 60.0-84.8%, roughly even with GraphRAG (49.6-54.8%); self-reported LLM pairwise judgments, not accuracy ([LightRAG](https://github.com/HKUDS/LightRAG)) | Best graph option if cross-document entities matter to the demo |
| **GraphRAG (Microsoft)** | LLM-extracted knowledge graph, Leiden communities, pre-summarised; global map-reduce or local fan-out | "Indexing can be an expensive operation… start small" ([graphrag](https://github.com/microsoft/graphrag)) | 72-83% comprehensiveness win rate on global sensemaking over ~1M-token datasets; root summaries used 9-43x fewer tokens per query (SNIPPET-ONLY, LLM-judged, [arXiv 2404.16130](https://arxiv.org/abs/2404.16130)) | Only for "summarise themes across the whole corpus"; now frozen |
| **Multimodal (ColPali)** | A VLM embeds page images into ColBERT-style multi-vectors; no OCR step | Needs a GPU and a multi-vector store (store support UNVERIFIED); ~256KB per page | ViDoRe nDCG@5 81.3 vs 67.0 for the best OCR pipeline; >10x faster indexing (SNIPPET-ONLY, [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/99e9e141aafc314f76b0ca3dd66898b3-Paper-Conference.pdf)) | Worth it for slides and scanned or chart-heavy PDFs; [RAGFlow](https://github.com/infiniflow/ragflow) is the parsing alternative |

Two cautions follow. The graph approaches are scored with **LLM pairwise "comprehensiveness/diversity" win rates**, not exact-match accuracy, so their benefit is broad summarisation, not factoid lookup. And the projects that maintain the variants tell the same story as the evidence: research pipelines are freezing while agentic loops in general frameworks take over.

### Repo status changes (checked 2026-10-02)

Each change below was checked against LICENSE files, shallow clones and PyPI on 2026-10-02. Where `repos.md` currently shows a ✅, the status needs updating.

| Repo | Current guide status | New finding (2026-10-02) | Action |
|---|---|---|---|
| [microsoft/graphrag](https://github.com/microsoft/graphrag) | ✅ in `repos.md` | README: "largely in maintenance mode, and won't be accepting new PRs or implementing new features"; PyPI 3.2.0 (2026-09-23) | Mark ⚠️ maintenance |
| [weaviate/Verba](https://github.com/weaviate/Verba) | not listed | Archived, "Project Discontinued"; v2.1.3 | Add to the avoid list |
| [NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques) | not listed | Excellent catalogue of 30+ technique notebooks, but a **custom non-commercial licence** | Learning only; do not copy code into a product |
| [AkariAsai/self-rag](https://github.com/AkariAsai/self-rag), [HuskyInSalt/CRAG](https://github.com/HuskyInSalt/CRAG) | not listed | Last commits 2024-03-19 and 2024-10-08; CRAG has no LICENSE file at the root | Dormant, research-grade; borrow the ideas only |
| [stanford-futuredata/ARES](https://github.com/stanford-futuredata/ARES) | not listed | `ares-ai` 0.6.6 (2024-07); last commit 2025-03; needs ≥50 human-labelled examples | Stale; prefer DeepEval |
| [vibrantlabsai/ragas](https://github.com/vibrantlabsai/ragas) | ⚠️ slowing | 0.4.3 (2026-01-13); last commit 2026-02-24 | Confirmed; pin the version |
| [confident-ai/deepeval](https://github.com/confident-ai/deepeval), [truera/trulens](https://github.com/truera/trulens) | DeepEval ✅ | DeepEval 4.2.7 (2026-09-29), pytest-style RAG metrics. TruLens 2.14.0 (2026-09-03), "RAG triad"; self-cites groundedness F1 0.81 (vendor-reported) | Recommended eval options |
| [pgvector/pgvector](https://github.com/pgvector/pgvector) | ✅ | Tag v0.8.7, commit 2026-10-01; Python client 0.5.0 | No change |
| [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) | ✅ | `lightrag-hku` 1.5.7 (2026-09-02); default query mode is mixed with reranker since 2025.08 | No change |
| [illuin-tech/colpali](https://github.com/illuin-tech/colpali) | not listed | `colpali-engine` 0.3.18 (2026-08-22), MIT; loads via Sentence Transformers v6 | Add for visual-PDF RAG |
| [microsoft/autogen](https://github.com/microsoft/autogen) / [ag2ai/ag2](https://github.com/ag2ai/ag2) | AutoGen flagged; AG2 ✅ | AG2 1.x removed the `autogen` import name | Warn that AutoGen 0.2 tutorials won't run on `pip install ag2` |
| [geekan/MetaGPT](https://github.com/geekan/MetaGPT) | not listed | PyPI idle since 2025-03-09 | Avoid for new builds |
| [langgraph-supervisor-py](https://github.com/langchain-ai/langgraph-supervisor-py) | not listed | LangChain recommends supervisor-via-tools instead | Use tool-calling |
| [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT), [karpathy/LLM101n](https://github.com/karpathy/LLM101n) | not listed | nanoGPT "very old and deprecated" (Nov 2025) → nanochat; LLM101n "does not yet exist" and is archived | Point learners to nanochat |

Haystack's licence was not checked in this research. `repos.md` lists it as Apache-2.0.
