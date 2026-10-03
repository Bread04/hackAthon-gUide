# Multi-agent LLM and RAG architectures: hackathon implementation (24-48h)

Research date: 2026-10-02. Method note: GitHub search API was unavailable in this session (403), and several sites were blocked by the egress proxy (huggingface.co, microsoft.github.io, aka.ms, angelhack.com, multiappagenthackathon.com). Repo READMEs were read directly from raw.githubusercontent.com. Claims seen only in search-engine summaries are marked SNIPPET-ONLY.

## Q1. Real hackathon projects (2024-2026) using multi-agent systems or RAG

### Takeaway
Winning or well-documented agent/RAG hackathon projects tend to have a narrow, visible "wow" moment (two voice agents switching to a sound protocol; a research report with citations; an agent acting across apps). They also pin the architecture down: a fixed pipeline or state machine, a small number of named specialist agents, Postgres/pgvector or SQL for memory, and a reproducible mock or sandbox mode for judges. Several of the best-documented "winners" come from multi-week virtual hackathons, not 24-48h events, so their scope overstates what a weekend team can build.

### Cited Findings
**GibberLink (1st place, ElevenLabs x a16z international hackathon, Feb 2025). README says it won.**
- What it is: two independent ElevenLabs conversational agents, one a hotel caller and one a receptionist. Both are prompted to switch to the ggwave data-over-sound protocol once they identify that the other side is an AI, and to keep speaking English otherwise. The repo provides the API that lets the agents use the protocol. — [GibberLink README](https://github.com/PennyroyalTea/gibberlink)
- The README says it "won first place on 11labs x a16z international hackathon" ([Devpost link in README](https://devpost.com/software/gibber-link)) and went viral, with coverage in Forbes and TechCrunch. — [GibberLink README](https://github.com/PennyroyalTea/gibberlink)
- What made the demo work: a simple two-agent setup (prompting plus one protocol tool), shown on two physical devices in a short video. Viewers can check the result themselves by decoding the sound with the ggwave web demo. — [GibberLink README](https://github.com/PennyroyalTea/gibberlink)

**AI Agents Hackathon 2025 (Microsoft; virtual, 3 weeks, Apr 8-30 2025; 18,000+ registrants, 570 submissions).** Judging criteria: "innovation, impact, usability, solution quality, and category alignment". — [winners.md](https://raw.githubusercontent.com/microsoft/AI_Agents_Hackathon/main/docs/winners.md)
- **RiskWise (Best Overall)**: supply-chain risk analysis built with Python, React/Next.js, Azure AI Agent Service, Semantic Kernel and SQL. Caveat: the organizer's own write-up hedges about the architecture ("It likely uses Semantic Kernel to plan…"), so its details are UNVERIFIED. Project: [issue #526](https://github.com/microsoft/AI_Agents_Hackathon/issues/526). — [winners.md](https://raw.githubusercontent.com/microsoft/AI_Agents_Hackathon/main/docs/winners.md)
- **Apollo, Deep Research Meta Agent (Best C#)**: a coordinator agent manages two specialist agents, "Athena" (research) and "Hermes" (analysis), in a Semantic Kernel group chat. It uses "self-reflective RAG", iterating on retrieval and gap-checking against PostgreSQL/pgvector vector memory, with web retrieval through Bing/Exa search. It writes each section separately, then synthesizes a final report with citations. A state machine and an async event pipeline run the flow. Stack: ASP.NET Core, React, Azure AI Agent Service, GPT-4. Project: [issue #681](https://github.com/microsoft/AI_Agents_Hackathon/issues/681). — [winners.md](https://raw.githubusercontent.com/microsoft/AI_Agents_Hackathon/main/docs/winners.md)
- **TARIFFED! (Best use of Azure AI Agent Service)**: an agent with Bing Search grounding plus private SQL Server data, built on .NET 9/Aspire with Blazor and Docker. Project: [issue #349](https://github.com/microsoft/AI_Agents_Hackathon/issues/349). — [winners.md](https://raw.githubusercontent.com/microsoft/AI_Agents_Hackathon/main/docs/winners.md)
- **Konveyor (Best Python)**: a knowledge-transfer agent that uses Semantic Kernel planners and memory plus a vector DB over docs, wikis and chat logs. Project: [issue #645](https://github.com/microsoft/AI_Agents_Hackathon/issues/645). — [winners.md](https://raw.githubusercontent.com/microsoft/AI_Agents_Hackathon/main/docs/winners.md)
- **Bits2Brain (Best Java)**: built with LangChain4j, Neo4j knowledge graph, Azure Video Indexer and Computer Vision. **ModelProof (Best JS/TS)**: a dual-LLM "sentinel" moderation chat. **WorkWizee (Best Copilot)**: Copilot Studio with the Jira and Bitbucket APIs. Projects: [#638](https://github.com/microsoft/AI_Agents_Hackathon/issues/638), [#517](https://github.com/microsoft/AI_Agents_Hackathon/issues/517), [#587](https://github.com/microsoft/AI_Agents_Hackathon/issues/587). — [winners.md](https://raw.githubusercontent.com/microsoft/AI_Agents_Hackathon/main/docs/winners.md)

**RAGHack 2024 (Microsoft; winners announced Oct 7 2024).** Winners by category, names only, since the discussion post gives no architecture details:
- Best Overall: DocAssistant.Charty. Best in PostgreSQL: Football-Analysis-Copilot. Best in Azure SQL: UniChatbot. Best in Cosmos DB: Discord Community Agent. Best in Azure AI Search: Manufacturing Support Bot. Best in Python: StoryWeave. Best in .NET: ContosoTravelAgency. Best in Java: Paper Mentor AI. Best in JS/TS: Learning Path Certifications Builder. Best in Azure AI Studio: MyFitnessBuddy. — [RAG_Hack discussion #171](https://github.com/microsoft/RAG_Hack/discussions/171)
- The detailed winners blog (aka.ms/raghack/winners-blog) was blocked, so architectures are a gap.

**OpenStoa (1st Place, The Synthesis Hackathon, track "Agents That Keep Secrets"). README badge says it won.**
- A community where humans and AI agents both participate, with zero-knowledge (ZK) proof sign-in. Built on the Next.js 15 App Router with Drizzle/SQL. It ships an SDK, a CLI and a stdio MCP server so agents can join. It is an agent-integration project, not RAG. — [OpenStoa README](https://github.com/zkproofport/openstoa)

**Life OS (Multi-App AI Agent Hackathon, Lemma + Comma Capital, Sept 13 2026; a one-day build of about 6.5 hours). Not marked as a winner: the README makes no winner claim.**
- It is "a LangGraph state machine, not a free-form chat agent". The fixed pipeline runs intake → priority → planner → scheduler → executor → critic → (human approval) → auditor, across Gmail, Calendar, Notion, Slack and Sheets. Irreversible actions such as sending email are staged as drafts. — [Life OS README](https://github.com/SaurabhVelankar/multi-app-ai-agent-hackathon)
- Demo reliability trick: "connectors use a deterministic **sandbox** (`LIFE_OS_USE_MOCK_CONNECTORS=1`) so judges can reproduce without OAuth". — [Life OS README](https://github.com/SaurabhVelankar/multi-app-ai-agent-hackathon)
- Evals: four golden JSON fixtures (meeting email, simple goal, ambiguous email, newsletter). Each runs through the real graph in mock mode, then the test asserts which apps were written, whether human-in-the-loop (HITL) triggered, and that an audit row exists. The pytest evals mock the LLM entirely. — [Life OS EVALS.md](https://raw.githubusercontent.com/SaurabhVelankar/multi-app-ai-agent-hackathon/main/EVALS.md)

**Other leads (not confirmed):**
- iot-smart-home-ai-ops: "3rd Prize @ SEAL Hackathon 2026". A 5-layer multi-agent IoT ops platform using Gemini 2.5, LangGraph, Qdrant and TimescaleDB. SNIPPET-ONLY; repo URL not found. — [search summary of GitHub topic page](https://github.com/topics/hackathon-winner?l=typescript&o=desc&s=stars)
- LLMGameHub (later Immersia) won the Gradio Agents & MCP Hackathon (June 2025), Agentic Demo Showcase track. SNIPPET-ONLY: huggingface.co was blocked. — [HF blog](https://huggingface.co/blog/kikikita/immersia-ai-games)
- MCP's 1st Birthday Hackathon winners (Nov 2025) included Portfolio Intelligence Platform (Best Consumer), GCP - Game Context Protocol (Best Creative) and Anim Lab AI (Community Choice). SNIPPET-ONLY. — [gradio.app winners](https://gradio.app/mcp-birthday-winners)
- HackWinnerDB is an open, sourced YAML database of hackathon winners, filterable by technology (e.g. `?technology=gemini`). It is useful for finding more examples. — [hackwinnerdb README](https://github.com/notsointresting/hackwinnerdb)

### Inferences
- The recurring pattern among winners is one coordinator with 2-3 named specialists, or a fixed pipeline, plus retrieval into Postgres/pgvector or SQL. None of the confirmed examples uses a large, open-ended agent swarm.
- Apollo and RiskWise come from a 3-week event. A 24-48h team should copy the visible output (a cited report, a decision dashboard) and simplify the orchestration.
- Life OS shows a mock or sandbox mode, golden-fixture evals and a "for judges" README section. These map directly to rubrics that weight reliability.

### Gaps
- Architecture details for the RAGHack 2024 winners (the blog was blocked; the per-project GitHub issues were not read).
- No confirmed 24-48h winner that used a classic document-QA RAG with a public repo was found. Repo-wide GitHub search was unavailable.
- Devpost pages were not fetched.

## Q2. Starter templates and quickstarts for a 24-48h build

### Takeaway
Pick a starter by output shape. For a chat UI with tools, use Vercel `ai-chatbot` or `create-llama`. For a graph-based agent you can inspect, use the LangGraph templates plus `agent-chat-ui`. For handoff-based multi-agent, use the OpenAI Agents SDK and `openai-cs-agents-demo`. For a subagent research system, use `claude-agent-sdk-demos`. For RAG over Postgres with auth, use Supabase `chatgpt-your-files`. For prebuilt crews, use `crewAI-examples`.

### Cited Findings
- **vercel/ai-chatbot** ("Chatbot"): Next.js App Router, AI SDK (text, structured objects, tool calls, chat hooks), shadcn/ui, Neon Postgres chat history, Vercel Blob, Auth.js, and one-click Vercel deploy. Models route through the Vercel AI Gateway, and you can switch to a direct provider. — [README](https://github.com/vercel/ai-chatbot)
- **langchain-ai/react-agent**: a LangGraph ReAct agent template for LangGraph Studio. Tavily search is the default tool, tools are added in `tools.py`, and the default model is `claude-sonnet-4-5-20250929` (switchable to OpenAI). — [README](https://github.com/langchain-ai/react-agent)
- **langchain-ai/rag-research-agent-template**: three graphs (index, retrieval, researcher subgraph). The router either plans research, asks a clarifying question, or declines. It generates multiple queries and retrieves in parallel. The default retriever is local Elasticsearch, with other providers configurable. It ships sample docs. — [README](https://github.com/langchain-ai/rag-research-agent-template)
- **langchain-ai/agent-chat-ui**: a Next.js chat UI for any LangGraph server with a `messages` key. Install with `npx create-agent-chat-app`, or use the hosted copy at agentchat.vercel.app. — [README](https://github.com/langchain-ai/agent-chat-ui)
- **openai/openai-agents-python**: a lightweight multi-agent SDK with agents, handoffs/agents-as-tools, guardrails, HITL, sessions, built-in tracing, and realtime/voice agents. It is provider-agnostic. Install with `pip install openai-agents`. — [README](https://github.com/openai/openai-agents-python)
- **openai/openai-cs-agents-demo**: a Python backend (Agents SDK customer-service example) plus a Next.js UI that visualizes agent orchestration and handoffs, using ChatKit. — [README](https://github.com/openai/openai-cs-agents-demo)
- **anthropics/claude-agent-sdk-demos**: demos include an email agent (agentic search over IMAP), Excel, hello-world, and a **Research Agent**. The research agent splits a request into subtopics, spawns parallel researcher subagents, synthesizes a report, and tracks subagent activity. There is also a React + Express chat app and a .docx resume generator. Warning in the README: "intended for local development only". — [README](https://github.com/anthropics/claude-agent-sdk-demos)
- **run-llama/create-llama**: run `npx create-llama@latest` to choose a use case (Agentic RAG, Data Analysis, Report Generation) and a backend (Next.js with LlamaIndex.TS, or Python FastAPI). It comes with a shadcn chat UI. Drop files in `data/` and run `npm run generate` to index them. Defaults are `gpt-4.1` and `text-embedding-3-large`. — [README](https://github.com/run-llama/create-llama)
- **supabase-community/chatgpt-your-files**: a "pgvector to Prod in 2 hours" workshop. Git-tag checkpoints (step-1…) cover storage, documents, embeddings and chat, with row-level security and 3rd-party auth. Prerequisites are Docker and Node 18+. — [README](https://github.com/supabase-community/chatgpt-your-files)
- **crewAIInc/crewAI-examples**: full apps (pinned to CrewAI 0.152.0 with uv). Includes Flows (Self Evaluation Loop, Lead Score with HITL, Content Creator), Crews (Trip Planner, Stock Analysis with SEC data, Meta Quest Knowledge PDF Q&A, Match Profile to Positions with vector search) and a Starter Template. — [README](https://github.com/crewAIInc/crewAI-examples)

### Inferences
- The fastest path to a deployed URL for judges is Vercel `ai-chatbot` or a `create-llama` Next.js app. LangGraph Studio templates are strong for showing the graph during judging, but you need a separate UI (agent-chat-ui).
- Supabase's tagged checkpoints suit teams new to RAG, because a broken step can be reset to a known-good tag.

### Gaps
- Actual setup times for each template were not measured.
- Mastra, Google ADK and Pydantic AI templates were not checked.

## Q3. Demo reliability: step caps, fallbacks, tracing, quick evals

### Takeaway
Every major framework has a step cap, but the defaults vary a lot. LangGraph's current default is very high (10007), so teams should set caps explicitly. Pair the cap with a controlled fallback output, a deterministic mock mode for external tools, and one tracing tool. A small golden-fixture eval suite, like Life OS's four fixtures, is a realistic "how we know it works" artifact for a hackathon.

### Cited Findings
- Anthropic advises "stopping conditions (such as a maximum number of iterations) to maintain control" and "extensive testing in sandboxed environments, along with the appropriate guardrails". — [Building effective agents (Dec 19 2024)](https://www.anthropic.com/engineering/building-effective-agents)
- OpenAI Agents SDK: going over `max_turns` raises `MaxTurnsExceeded`. `Runner` accepts `error_handlers` keyed by `"max_turns"`, `"model_refusal"` and `"invalid_final_output"`, which let a run "return a controlled final output instead of ending the run with the corresponding error". — [running_agents.md](https://raw.githubusercontent.com/openai/openai-agents-python/main/docs/running_agents.md)
- LangGraph: `DEFAULT_RECURSION_LIMIT = int(getenv("LANGGRAPH_DEFAULT_RECURSION_LIMIT", "10007"))` on current main. `GraphRecursionError` is "Raised when the graph has exhausted the maximum number of steps. This prevents infinite loops." Older LangGraph versions defaulted to 25 (from memory, UNVERIFIED in this session). — [_config.py](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/langgraph/_internal/_config.py), [errors.py](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/langgraph/errors.py)
- CrewAI agents expose `max_iter` ("Maximum iterations for an agent to execute a task"), `max_rpm` (requests per minute), and tool-result caching that turns on when the crew sets `cache=True`. — [base_agent.py](https://raw.githubusercontent.com/crewAIInc/crewAI/main/lib/crewai/src/crewai/agents/agent_builder/base_agent.py)
- Claude Agent SDK: the README examples use `max_turns` and an `allowed_tools` permission allowlist. Hooks can return `permissionDecision: "deny"`. — [claude-agent-sdk-python README](https://github.com/anthropics/claude-agent-sdk-python)
- Deterministic fallback in practice: Life OS mock connectors return fake `external_id`s, and its pytest evals mock the LLM, so the demo and tests run without OAuth or API keys. — [Life OS EVALS.md](https://raw.githubusercontent.com/SaurabhVelankar/multi-app-ai-agent-hackathon/main/EVALS.md)
- **Langfuse**: open source (MIT), "self-hosted in minutes". It covers tracing (LLM calls, retrieval, embeddings, agent actions), evaluations (LLM-as-a-judge, code evaluators, user feedback, manual labels) and datasets. — [Langfuse README](https://github.com/langfuse/langfuse)
- **Arize Phoenix**: open source and self-hosted, with OpenTelemetry-based tracing, LLM and retrieval evals, datasets and experiments, a playground, and a remote MCP server. It has out-of-the-box integrations for the OpenAI Agents SDK, Claude Agent SDK and LangGraph. — [Phoenix README](https://github.com/Arize-ai/phoenix)
- **LangSmith** is tied to LangGraph tooling. agent-chat-ui asks for a LangSmith API key for deployed LangGraph servers. — [agent-chat-ui README](https://github.com/langchain-ai/agent-chat-ui)
- The OpenAI Agents SDK has built-in tracing of agent runs. — [openai-agents-python README](https://github.com/openai/openai-agents-python)

### Inferences
- A practical demo recipe:
  1. Set an explicit step cap (e.g. `max_turns`/`recursion_limit` around 10-15) with a graceful fallback message.
  2. Add a `USE_MOCKS=1` flag for external APIs.
  3. Pre-record a backup video.
  4. Pre-run the exact demo queries and cache their outputs.
  5. Run 3-5 golden fixtures as pytest, asserting on structured outputs or side effects rather than exact text.
  6. Turn on one tracing tool (the SDK's built-in tracing, Langfuse or Phoenix) so you can show the trace to judges.
- Langfuse's and Phoenix's self-hosting needs Docker. A hosted free tier may be faster for a hackathon (not verified here).

### Gaps
- No sourced latency budgets (e.g. "keep each demo step under N seconds") for hackathon demos were found.
- Provider prompt-caching docs and cost-cap mechanisms (spend limits) were not fetched.
- No published "30-minute eval" guide was found. The Life OS fixtures are the closest concrete example.

## Q4. Decision criteria: single call or workflow vs agent; RAG vs long context; judges and "multi-agent"

### Takeaway
Start with a single call plus retrieval. Move to a fixed workflow when the steps are known, and use an agent only when the path truly depends on the model's decisions. Long context beats RAG on quality when resources allow, but RAG is much cheaper, and routing between the two keeps most of the quality. Agent-focused rubrics now reward reliability evidence and real actions, not the "multi-agent" label itself.

### Cited Findings
- "optimizing single LLM calls with retrieval and in-context examples is usually enough"; "find[] the simplest solution possible, and only increas[e] complexity when needed". — [Anthropic, Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
- "workflows offer predictability and consistency for well-defined tasks, whereas agents are the better option when flexibility and model-driven decision-making are needed"; "Agentic systems often trade latency and cost for better task performance". — [Anthropic](https://www.anthropic.com/engineering/building-effective-agents)
- On frameworks: they "often create extra layers of abstraction that can obscure the underlying prompts and responses, making them harder to debug". The workflow patterns named in the article are prompt chaining, routing, parallelization, orchestrator-workers and evaluator-optimizer. — [Anthropic](https://www.anthropic.com/engineering/building-effective-agents)
- RAG vs long context (Li et al., EMNLP 2024 Industry): "when resourced sufficiently, LC consistently outperforms RAG in terms of average performance", but RAG's lower cost remains an advantage. LC and RAG predictions were identical for over 60% of queries. SELF-ROUTE routes a query to RAG when the model judges the retrieved chunks sufficient and to LC otherwise, which cut cost by 65% (Gemini-1.5-Pro) and 39% (GPT-4o) with performance comparable to LC. SNIPPET-ONLY: from search summary, PDF not opened. — [arXiv 2407.16833](https://arxiv.org/pdf/2407.16833), [ACL Anthology](https://aclanthology.org/2024.emnlp-industry.66/)
- Judging weights for an agent hackathon (Multi-App AI Agent Hackathon, Sept 2026), as copied by a participant:
  - 30% Technical execution ("Real multi-step agent; ≥3 apps; actions, not just reads")
  - 25% Reliability & evaluation ("Retries, failure modes, traces, evals")
  - 20% Usefulness
  - 15% Originality ("not 'generic assistant'")
  - 10% Demo clarity

  The brief says "Not: a chatbot that only talks." The official site was blocked, so this is the participant's transcription. — [Life OS Hackathon.md](https://raw.githubusercontent.com/SaurabhVelankar/multi-app-ai-agent-hackathon/main/Hackathon.md)
- AngelHack's 2026 agent-hackathon playbook (SNIPPET-ONLY, page blocked) reportedly says judges should evaluate "task completion rate, tool-use accuracy, cost per run, and latency, not just demo quality". It also warns that teams "default to flashy demos that wow the room but don't survive a real workflow test" and that without consistent criteria "judges score videos and vibes". — [AngelHack blog](https://angelhack.com/blog/ai-agent-hackathon/)
- Microsoft's AI Agents Hackathon 2025 judged on "innovation, impact, usability, solution quality, and category alignment". "Multi-agent" was not a criterion in itself, though winners like Apollo described their multi-agent orchestration explicitly. — [winners.md](https://raw.githubusercontent.com/microsoft/AI_Agents_Hackathon/main/docs/winners.md)

### Inferences
- Rules of thumb for a 24-48h build:
  1. If the corpus fits in context (one to a few documents), skip RAG and stuff the context, using caching if available.
  2. Use RAG when the corpus is large or changing, when cost per query matters, or when per-source citations are the demo's selling point.
  3. Use a fixed workflow (chain or router) when you can draw the steps in advance. Life OS deliberately chose "a LangGraph state machine, not a free-form chat agent".
  4. Claim "multi-agent" only if each agent has a distinct role you can show in a trace or UI.
- Judges on agent-specific rubrics appear to penalize chatbot wrappers and reward evidence such as evals, traces and HITL for irreversible actions.

### Gaps
- No first-hand judge quotes about reactions to "multi-agent" claims were found (Devpost/LinkedIn/blogs not reachable or not found).
- The full text of the RAG vs long-context paper was not read. Newer 2025-2026 comparisons with 1M+ context models were not found.
