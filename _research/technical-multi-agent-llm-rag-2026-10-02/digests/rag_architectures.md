# RAG Architectures and Variants (as of 2026-10-02)

Method notes: arxiv.org, aclanthology.org and research.trychroma.com were blocked by the egress proxy, and so was the GitHub REST API. Repo facts (licence, last commit, tags) come from shallow `git clone` / `git ls-remote` of each repo on 2026-10-02. Package versions come from the PyPI JSON API on 2026-10-02. Paper numbers marked SNIPPET-ONLY come from search-result summaries, not the paper text. Star counts are not given because the API was blocked.

## Q1: For each RAG variant — mechanism, cost/latency/complexity, evidence, hackathon verdict

### Takeaway
For a hackathon the best-value stack is: naive RAG with sensible recursive chunking, then hybrid BM25 + vectors, a reranker, and Anthropic-style contextual chunk headers. If the corpus is under ~200k tokens, skip RAG and put it all in the prompt. GraphRAG/LightRAG, Self-RAG/CRAG and ColPali are worth it only when the demo depends on their specific strength: global "sensemaking" questions, a self-correction story, or visually rich PDFs. Agentic RAG (retrieval as a tool in a loop) is the cheapest way to get iterative/multi-hop behaviour in 2026.

### Cited Findings

**Naive RAG (chunk -> embed -> top-k -> stuff into prompt)**
- How it works: split documents into chunks, embed them, retrieve the top-k chunks by vector similarity, and put them in the prompt. It is the baseline that every variant below is compared against (e.g. "NaiveRAG" is LightRAG's baseline) — [LightRAG README](https://github.com/HKUDS/LightRAG)
- Anthropic's contextual-retrieval baseline (standard embeddings) had a 5.7% top-20 retrieval failure rate on their eval sets — [Anthropic: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- Verdict: build it first, in hours. Everything else is an add-on that you measure against it.

**Chunking strategies (fixed/recursive, semantic, LLM-based)**
- Chroma's technical report compared chunkers on token-level recall/precision/IoU. RecursiveCharacterTextSplitter at 200 tokens with no overlap "performed consistently well across all metrics". ClusterSemanticChunker at 200 tokens had the best precision (8.0%) and IoU (8.0%). LLMSemanticChunker had the highest recall (91.9%), ClusterSemanticChunker@400 was next (91.3%), and RecursiveCharacterTextSplitter got 85.4–89.5% recall. The report used separators `["\n\n","\n",".","?","!"," ",""]` rather than the defaults — [Chroma research](https://www.trychroma.com/research/evaluating-chunking) (SNIPPET-ONLY; page blocked); code at [brandonstarxel/chunking_evaluation](https://github.com/brandonstarxel/chunking_evaluation)
- Anthropic's contextual retrieval adds 50–100 tokens of context per chunk, and top-20 chunks worked best in their tests — [Anthropic](https://www.anthropic.com/news/contextual-retrieval)
- Verdict: use recursive splitting at about 200–400 tokens with sentence-aware separators. Semantic or LLM chunking gives small gains for extra cost, so it is low priority.

**Hybrid search (BM25 + dense vectors) and reranking**
- How it works: run a lexical (BM25) index alongside the vector index and fuse the ranked lists (commonly reciprocal rank fusion). Then a cross-encoder reranker rescores the top-N candidates (Anthropic retrieved 150 and kept the top 20 — UNVERIFIED in this session; the "top-20" figure is confirmed) — [Anthropic](https://www.anthropic.com/news/contextual-retrieval)
- Evidence: contextual embeddings plus contextual BM25 cut top-20 retrieval failure from 5.7% to 2.9% (−49%). Adding reranking brought it to 1.9% (−67%) — [Anthropic](https://www.anthropic.com/news/contextual-retrieval)
- Latency: LightRAG's docs say enabling a reranker "typically introduces a 1–2 second delay". They recommend running it locally and suggest `BAAI/bge-reranker-v2-m3`. Since 2025.08 LightRAG's default query mode is mixed with a reranker — [LightRAG README](https://github.com/HKUDS/LightRAG)
- pgvector (Postgres extension, latest tag v0.8.7, commit 2026-10-01) lets one Postgres instance hold the vectors, with Postgres full-text search providing the lexical side — [pgvector/pgvector](https://github.com/pgvector/pgvector) (hybrid-in-Postgres pattern: UNVERIFIED in this session)
- Verdict: high value and low effort. Hybrid plus a reranker is the single biggest quality upgrade over naive RAG.

**Query rewriting / HyDE / multi-query**
- HyDE: an instruction-following LLM writes a hypothetical answer document. That document is embedded (originally with Contriever), and real documents near it are retrieved; the encoder's "dense bottleneck" filters out the hallucinated details. It beat zero-shot Contriever on BEIR (ACL 2023, Gao, Ma, Lin, Callan) — [arXiv 2212.10496](https://arxiv.org/abs/2212.10496) / [ACL Anthology](https://aclanthology.org/2023.acl-long.99/) (SNIPPET-ONLY)
- Multi-query / rewriting: the LLM writes several rephrasings or sub-questions and the results are unioned or fused. NirDiamant/RAG_Techniques has a whole "Query Enhancement" section of notebooks — [RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques)
- Evidence caution: in LightRAG's own LLM-judged comparisons, HyDE and RQ-RAG (query rewriting) baselines lost to LightRAG by a margin similar to NaiveRAG (e.g. HyDE won 24.8% of comparisons vs LightRAG's 75.2% overall on Agriculture). In that setup they were not much better than naive RAG — [LightRAG README](https://github.com/HKUDS/LightRAG)
- Cost: each rewrite adds one extra LLM call before retrieval, which adds latency (no sourced number found).
- Verdict: cheap to add as one LangChain/LlamaIndex option. Use multi-query for vague user questions. HyDE helps mostly when there is no labelled data or domain-tuned embedder.

**Contextual Retrieval (Anthropic, Sept 2024)**
- How it works: before indexing, an LLM writes a short context (50–100 tokens) for each chunk explaining where it sits in the whole document. That context is prepended before both embedding and BM25 indexing. Prompt caching makes the full document cheap to re-send for each chunk — [Anthropic](https://www.anthropic.com/news/contextual-retrieval)
- Numbers: contextual embeddings −35% failure (5.7%→3.7%); plus contextual BM25 −49% (→2.9%); plus reranking −67% (→1.9%). One-time cost is "$1.02 per million document tokens" with prompt caching. Gemini Text 004 and Voyage embeddings did especially well — [Anthropic](https://www.anthropic.com/news/contextual-retrieval)
- "Knowledge bases smaller than 200,000 tokens can simply be included directly in the prompt" — [Anthropic](https://www.anthropic.com/news/contextual-retrieval)
- Reference code: `capabilities/contextual-embeddings/guide.ipynb` (with eval set `data/evaluation_set.jsonl`, Promptfoo evaluation, an AWS Lambda variant) in [anthropics/claude-cookbooks](https://github.com/anthropics/claude-cookbooks/tree/main/capabilities/contextual-embeddings) (MIT, last commit 2026-09-28)
- Verdict: strong. It is one ingestion-time LLM pass with no new infrastructure, and it has the best-documented gains of any variant here.

**Parent-child / small-to-big / sentence-window retrieval**
- How it works: index small chunks (sentences or ~100-token pieces) for precise matching, but return the larger parent section/window to the LLM for context. LlamaIndex (AutoMergingRetriever, sentence window) and LangChain (ParentDocumentRetriever) implement this; RAG_Techniques covers it under "Context and Content Enrichment" — [RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques); [llama_index](https://github.com/run-llama/llama_index); [langchain](https://github.com/langchain-ai/langchain) (class names from general knowledge — UNVERIFIED this session)
- Evidence: I found no primary quantitative benchmark this session (gap).
- Verdict: a cheap config change in the frameworks. Use it when answers need surrounding context (contracts, manuals).

**GraphRAG (Microsoft)**
- How it works: an LLM extracts entities and relationships into a knowledge graph. Leiden hierarchical community detection groups the graph into levels C0–C3, and an LLM pre-summarises each community. "Global" queries map-reduce over community summaries; "local" queries fan out from entities — [MSR publication page](https://www.microsoft.com/en-us/research/publication/from-local-to-global-a-graph-rag-approach-to-query-focused-summarization/); [arXiv 2404.16130](https://arxiv.org/abs/2404.16130)
- Evidence: on global sensemaking questions over two ~1M-token datasets it beat vector RAG with 72–83% comprehensiveness and 62–71% diversity win rates (LLM-judged). Root-level (C0) summaries needed 9x–43x fewer tokens per query and still won 72% on comprehensiveness — [arXiv 2404.16130](https://arxiv.org/pdf/2404.16130) (SNIPPET-ONLY; via search summaries incl. [beancount.io summary](https://beancount.io/bean-labs/research-logs/2026/06/04/graphrag-local-to-global-query-focused-summarization))
- Cost: README warns "GraphRAG indexing can be an expensive operation… start small" and strongly recommends prompt tuning for your data — [microsoft/graphrag](https://github.com/microsoft/graphrag)
- Status (important): the README now says the project "is largely in maintenance mode, and won't be accepting new PRs or implementing new features"; only bug fixes and CVE/dependency updates continue — [microsoft/graphrag](https://github.com/microsoft/graphrag). PyPI `graphrag` 3.2.0 (2026-09-23), MIT.
- Verdict: only if the demo is "summarise themes across a whole corpus". Indexing cost and setup time are high, and the project is in maintenance mode.

**LightRAG (HKUDS)**
- How it works: a graph-plus-vector design that extracts entities and relations into a KG and also keeps vector embeddings. It uses dual-level (low-level entity / high-level theme) retrieval, supports incremental updates, and is pitched as a lighter alternative to GraphRAG. It works with a 30B open-source LLM — [LightRAG README](https://github.com/HKUDS/LightRAG); paper [arXiv 2410.05779](https://arxiv.org/abs/2410.05779)
- Evidence (authors' own LLM-judged win rates, "Overall"): vs NaiveRAG LightRAG wins 67.6% (Agriculture), 61.2% (CS), 84.8% (Legal), 60.0% (Mix). Vs GraphRAG it is roughly even: 54.8%, 52.0%, 52.8%, and 49.6% (loses on Mix) — [LightRAG README](https://github.com/HKUDS/LightRAG). These are self-reported, pairwise LLM judgments, not accuracy.
- Ops: the default in-memory/file stores are "not for production"; PostgreSQL is the recommended all-in-one backend (Neo4j, Milvus, Qdrant, MongoDB, OpenSearch also supported). The embedding model cannot be changed after indexing without re-embedding everything. It has a Docker setup wizard (2026.03). Multimodal support comes via the sister project RAG-Anything — [LightRAG README](https://github.com/HKUDS/LightRAG)
- Verdict: the best graph-RAG option for a hackathon. It installs with pip, is actively developed and has a web UI. Use it if relationships or entities across documents matter to the demo.

**Agentic RAG (retrieval as a tool, iterative)**
- How it works: retrieval (vector search, web search, SQL, etc.) is exposed as tools to an LLM agent. The agent decides whether, what and how many times to retrieve, rewrites queries, and stops when it has enough. Implemented in LangGraph (langgraph 1.2.12 on PyPI, 2026-09-21) and LlamaIndex agents; RAG_Techniques has "Iterative and Adaptive Techniques" — [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph); [RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques)
- Long-context study finding supporting routing: Self-Route first tries RAG and lets the model say whether the query is answerable from retrieved chunks, falling back to full long context only if not. 82% of queries were resolved at the RAG step for Gemini-1.5-Pro; cost fell 65% (Gemini-1.5-Pro) and 39% (GPT-4o) at comparable performance — [Li et al., EMNLP 2024 Industry](https://aclanthology.org/2024.emnlp-industry.66/) (SNIPPET-ONLY)
- Cost: several LLM calls per query, so latency grows with the number of tool turns (no sourced number).
- Verdict: the default "modern" pattern. With a tool-calling model it is about the same code as naive RAG and handles multi-hop questions. Cap the number of iterations.

**Corrective RAG (CRAG) and Self-RAG**
- Self-RAG (Asai et al., ICLR 2024 Oral): the LM is fine-tuned to emit reflection tokens. These decide when to retrieve (multiple times or not at all) and critique relevance and support, with segment-wise beam search. Released 7B/13B Llama-2 checkpoints — [AkariAsai/self-rag](https://github.com/AkariAsai/self-rag) (MIT, last commit 2024-03-19, i.e. dormant)
- CRAG (Yan et al. 2024): a lightweight retrieval evaluator grades the retrieved docs as correct, incorrect or ambiguous. It triggers knowledge refinement or a web-search fallback — [HuskyInSalt/CRAG](https://github.com/HuskyInSalt/CRAG) (last commit 2024-10-08; no LICENSE file found at repo root); paper [arXiv 2401.15884](https://arxiv.org/pdf/2401.15884.pdf)
- Verdict: don't train Self-RAG at a hackathon. Do borrow the ideas as a LangGraph loop: an LLM grades the retrieved docs, then rewrites the query or falls back to web search. This makes a good "self-correcting" demo story.

**Multimodal RAG (ColPali / document images)**
- How it works: ColPali embeds page images directly with a VLM (PaliGemma-3B ViT patches projected into ColBERT-style multi-vectors) and uses late-interaction matching. This skips OCR and layout parsing. Later variants include ColQwen2/2.5. As of the latest README, ColPali models also load through Sentence Transformers v6 `MultiVectorEncoder` — [illuin-tech/colpali](https://github.com/illuin-tech/colpali) (MIT, last commit 2026-08-24; `colpali-engine` 0.3.18 on PyPI, 2026-08-22)
- Evidence: ViDoRe nDCG@5 81.3 vs 67.0 for the best text+OCR+captioning pipeline. Indexing is >10x faster than OCR pipelines; storage is ~256KB per page — [ColPali paper, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/99e9e141aafc314f76b0ca3dd66898b3-Paper-Conference.pdf) / [emergentmind summary](https://www.emergentmind.com/topics/colpali-methodology) (SNIPPET-ONLY)
- Alternatives: RAGFlow (deep document understanding/OCR parsing) and HKUDS RAG-Anything (text, images, tables, equations) — [infiniflow/ragflow](https://github.com/infiniflow/ragflow); [LightRAG README](https://github.com/HKUDS/LightRAG)
- Verdict: worth it when the source is slide decks, scanned or chart-heavy PDFs. It needs a GPU and a multi-vector store such as Vespa or Qdrant (store support: UNVERIFIED).

**Long context instead of RAG**
- Anthropic: corpora under 200k tokens can go straight into the prompt; prompt caching makes this cheap — [Anthropic](https://www.anthropic.com/news/contextual-retrieval)
- Li et al. (EMNLP 2024 Industry, Gemini-1.5/GPT-4o): "when resourced sufficiently, LC consistently outperforms RAG in terms of average performance". RAG remains far cheaper (~1.5k retrieved tokens vs 10k–100k full context), which motivates Self-Route — [ACL Anthology](https://aclanthology.org/2024.emnlp-industry.66/) (SNIPPET-ONLY)
- Counterpoint papers exist: [In Defense of RAG in the Era of Long-Context LMs (NVIDIA, 2409.01666)](https://arxiv.org/pdf/2409.01666) and [Long Context vs. RAG: An Evaluation and Revisits (2501.01880)](https://arxiv.org/pdf/2501.01880) (titles only; content UNVERIFIED)
- Verdict: for small hackathon corpora this is often the right answer. Ship it on day 1 with caching, and add RAG only if the corpus outgrows the window or the cost per query matters.

### Inferences
- Rough order of value per hour of hackathon work: long-context (if small) > naive RAG > hybrid + rerank > contextual retrieval > agentic loop > parent-child > query rewriting > CRAG-style grading > LightRAG > ColPali > GraphRAG > Self-RAG (training).
- The graph approaches are judged with LLM pairwise "comprehensiveness/diversity" win rates, not exact-match accuracy. Their benefit is for broad summarisation questions, not factoid lookup.
- GraphRAG going into maintenance mode, Verba being archived, and Self-RAG/CRAG being dormant all suggest the field is moving towards agentic loops in general frameworks rather than standalone research pipelines.

### Gaps
- No primary-source numbers for parent-child retrieval, multi-query or RRF fusion alone.
- GraphRAG, ColPali, HyDE and Li et al. numbers are SNIPPET-ONLY because arXiv and ACL were blocked.
- Exact latency numbers for agentic/iterative RAG: none found.

## Q2: Best GitHub reference implementations and starter repos

### Takeaway
Actively maintained in late 2026: LightRAG, LlamaIndex, LangChain/LangGraph, RAGFlow, pgvector, DeepEval, TruLens, Haystack, ColPali and the Claude cookbooks. Maintenance-only or stale: microsoft/graphrag (maintenance mode), RAGAS (repo moved to vibrantlabsai/ragas; last commit Feb 2026), ARES, Self-RAG and CRAG. weaviate/Verba is archived. NirDiamant/RAG_Techniques is a great tutorial set, but its custom licence is non-commercial.

### Cited Findings
Each licence comes from the repo's LICENSE file and each "last commit" from a shallow clone, both on 2026-10-02. Versions come from PyPI.

| Repo | Purpose | Licence | Latest version / activity |
|---|---|---|---|
| [microsoft/graphrag](https://github.com/microsoft/graphrag) | Reference GraphRAG pipeline (index + local/global search) | MIT | PyPI `graphrag` 3.2.0 (2026-09-23); last commit 2026-09-23; **maintenance mode, no new features** |
| [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) | Lightweight graph+vector RAG, server + web UI, many storage backends | MIT | PyPI `lightrag-hku` 1.5.7 (2026-09-02); last commit 2026-09-26 |
| [NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques) | Notebook catalogue of 30+ techniques (query enhancement, context enrichment, iterative, eval, graph, etc.) | **Custom non-commercial licence** (attribution required; commercial use needs written permission) | last commit 2026-09-21 |
| [run-llama/llama_index](https://github.com/run-llama/llama_index) | Data framework: loaders, indexes, retrievers (auto-merging, hybrid), agents | MIT | PyPI `llama-index` 0.14.25 (2026-09-21); last commit 2026-10-01 |
| [langchain-ai/langchain](https://github.com/langchain-ai/langchain) / [langgraph](https://github.com/langchain-ai/langgraph) | Retrievers and text splitters; LangGraph for agentic / corrective RAG graphs | MIT | `langchain` 1.4.3 (2026-09-28), `langgraph` 1.2.12 (2026-09-21); last commit 2026-10-01 |
| [vibrantlabsai/ragas](https://github.com/vibrantlabsai/ragas) (old: explodinggradients/ragas) | RAG/LLM eval metrics plus synthetic test-set generation | Apache-2.0 | PyPI `ragas` 0.4.3 (2026-01-13); last commit 2026-02-24 (slowing) |
| [anthropics/claude-cookbooks](https://github.com/anthropics/claude-cookbooks) `capabilities/contextual-embeddings` | Contextual retrieval notebook + eval set + Promptfoo eval + Lambda | MIT | last commit 2026-09-28 |
| [infiniflow/ragflow](https://github.com/infiniflow/ragflow) | Full RAG app/engine with deep document parsing (OCR/layout), agent workflows | Apache-2.0 | tags v0.27.2 and v1.0.0-rc1; last commit 2026-10-01 |
| [weaviate/Verba](https://github.com/weaviate/Verba) | "Golden RAGtriever" end-to-end RAG UI on Weaviate | BSD-style (Weaviate B.V.) | v2.1.3; **archived, "Project Discontinued"**; last commit 2026-06-08 |
| [pgvector/pgvector](https://github.com/pgvector/pgvector) | Vector similarity search inside Postgres (HNSW/IVFFlat) | PostgreSQL licence | tag v0.8.7; last commit 2026-10-01; Python client `pgvector` 0.5.0 (2026-07-06) |
| [stanford-futuredata/ARES](https://github.com/stanford-futuredata/ARES) | Automated RAG eval: synthetic queries + fine-tuned judges + Prediction-Powered Inference | Apache-2.0 | PyPI `ares-ai` 0.6.6 (2024-07-11); last commit 2025-03-28 (stale) |
| [truera/trulens](https://github.com/truera/trulens) | Tracing + feedback functions; "RAG triad" (context relevance, groundedness, answer relevance) | MIT | PyPI `trulens` 2.14.0 (2026-09-03); last commit 2026-10-01 |
| [confident-ai/deepeval](https://github.com/confident-ai/deepeval) | Pytest-style LLM unit tests; RAG metrics + G-Eval | Apache-2.0 | PyPI `deepeval` 4.2.7 (2026-09-29); last commit 2026-10-01 |
| [illuin-tech/colpali](https://github.com/illuin-tech/colpali) | ColPali/ColQwen visual document retrievers | MIT | `colpali-engine` 0.3.18 (2026-08-22); last commit 2026-08-24 |
| [AkariAsai/self-rag](https://github.com/AkariAsai/self-rag) | Self-RAG training/inference | MIT | last commit 2024-03-19 (dormant) |
| [HuskyInSalt/CRAG](https://github.com/HuskyInSalt/CRAG) | Corrective RAG reference code | No LICENSE file at root | last commit 2024-10-08 (dormant) |
| [deepset-ai/haystack](https://github.com/deepset-ai/haystack) | Pipeline framework for RAG/agents | (licence not checked) | PyPI `haystack-ai` 3.3.0 (2026-10-01) |

**Evaluation tool specifics**
- RAGAS: offers "objective metrics" (LLM-based plus traditional), custom `DiscreteMetric` aspect critiques, and test-set generation "if you don't have a test dataset ready" — [ragas README](https://github.com/vibrantlabsai/ragas)
- DeepEval: "similar to Pytest but specialized for unit testing LLM apps". RAG metrics are Answer Relevancy, Faithfulness, Contextual Recall, Contextual Precision, Contextual Relevancy, and a RAGAS composite (the average of those four), plus G-Eval custom criteria — [deepeval README](https://github.com/confident-ai/deepeval)
- TruLens: built around the RAG triad. Its README cites groundedness F1 0.81 on LLM-AggreFact, ahead of Bespoke-MiniCheck-7B (per a Snowflake benchmark); it has Snowflake Cortex provider support — [trulens README](https://github.com/truera/trulens); [Snowflake blog](https://www.snowflake.com/en/engineering-blog/benchmarking-LLM-as-a-judge-RAG-triad-metrics/)
- ARES: scores context relevance, answer faithfulness and answer relevance using fine-tuned classifiers plus PPI for confidence intervals. It needs a human-annotated validation set of "at least 50 examples but several hundred examples is ideal" — [ARES README](https://github.com/stanford-futuredata/ARES)
- Golden question sets: Anthropic's cookbook ships `data/evaluation_set.jsonl` (questions with gold chunks) and measures retrieval with Pass@k-style metrics. This makes a good template for a hand-built 20–50 question golden set — [claude-cookbooks contextual-embeddings](https://github.com/anthropics/claude-cookbooks/tree/main/capabilities/contextual-embeddings) (Pass@k metric naming: UNVERIFIED this session)

### Inferences
- Hackathon starter picks: LlamaIndex or LangChain/LangGraph plus pgvector (or Chroma) for the core; the Anthropic contextual-embeddings notebook as the retrieval-quality recipe; LightRAG if you need a graph; DeepEval (pytest-style, actively maintained) or RAGAS for scoring a 20–50 question golden set.
- Avoid building on Verba (archived) or Self-RAG/CRAG code (dormant, research-grade). Treat GraphRAG as stable but frozen.
- RAG_Techniques is excellent for learning, but don't copy its code into a project you might commercialise, because of the non-commercial licence.

### Gaps
- GitHub star counts and release dates were unavailable (API blocked).
- Haystack licence not checked. LlamaIndex/LangGraph PyPI licence fields were empty; their repo LICENSE files are MIT (LangGraph's LICENSE not checked separately).
- Benchmarks comparing RAGAS, DeepEval, TruLens and ARES against human judgments were not found, apart from TruLens's self-cited groundedness figure.
