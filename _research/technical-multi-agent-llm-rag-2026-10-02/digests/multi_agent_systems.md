# Multi-Agent LLM Systems: Architectures, Communication, Frameworks, Failure Modes (as of 2026-10-02)

Access notes: anthropic.com pages and raw.githubusercontent.com READMEs were fetched in full. PyPI JSON API was queried directly on 2026-10-02 for every version below. arxiv.org, cdn.openai.com (OpenAI "A practical guide to building agents" PDF), cognition.ai, huggingface.co and the LangGraph docs site were blocked by the egress proxy, so claims from those come only from search snippets and are marked SNIPPET-ONLY. Claims from my background knowledge that I could not confirm are marked UNVERIFIED.

## Core patterns: single agent with tools vs workflows vs multi-agent, and what Anthropic, OpenAI and the frameworks recommend

### Takeaway
Everyone agrees on the order of escalation. Start with one optimized LLM call. Then use deterministic workflows (chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer). Then use a single tool-using agent loop. Go multi-agent only when the task is high-value, breadth-first and parallelizable, or needs more context than one window holds. Anthropic's production multi-agent system is an orchestrator-worker setup in which a lead agent spawns subagents with isolated context windows. OpenAI's SDK reduces multi-agent design to two primitives: "agents as tools" (a manager keeps control) and "handoffs" (control transfers to the specialist).

### Cited Findings
**Anthropic, "Building effective agents" (Dec 19, 2024; Erik S. and Barry Zhang)**
- Definitions: "Workflows are systems where LLMs and tools are orchestrated through predefined code paths. Agents ... are systems where LLMs dynamically direct their own processes and tool usage." Both are grouped as "agentic systems". — [Anthropic](https://www.anthropic.com/research/building-effective-agents)
- "Find the simplest solution possible, and only increase complexity when needed. This might mean not building agentic systems at all." Agentic systems "trade latency and cost for better task performance". For many applications, "optimizing single LLM calls with retrieval and in-context examples is usually enough." — [Anthropic](https://www.anthropic.com/research/building-effective-agents)
- The building block is the "augmented LLM": an LLM plus retrieval, tools and memory. MCP is mentioned as one way to implement it. — [Anthropic](https://www.anthropic.com/research/building-effective-agents)
- The five workflow patterns:
  - **Prompt chaining**: sequential calls with programmatic "gates". Use it when a task decomposes cleanly into fixed subtasks; it trades latency for accuracy.
  - **Routing**: classify the input, then send it to a specialized prompt or model, e.g. easy queries to Haiku and hard ones to Sonnet.
  - **Parallelization**: either *sectioning* (independent subtasks) or *voting* (the same task run many times).
  - **Orchestrator-workers**: a central LLM breaks the task down dynamically, delegates and synthesizes. It differs from parallelization because "subtasks aren't pre-defined, but determined by the orchestrator".
  - **Evaluator-optimizer**: a generator and an evaluator run in a loop. Use it when evaluation criteria are clear and iterating gives measurable gains.
  — [Anthropic](https://www.anthropic.com/research/building-effective-agents)
- Agents "are typically just LLMs using tools based on environmental feedback in a loop". Use them for open-ended problems where the number of steps can't be predicted. "Higher costs, and the potential for compounding errors." Recommends stopping conditions such as max iterations. — [Anthropic](https://www.anthropic.com/research/building-effective-agents)
- On frameworks: they "often create extra layers of abstraction that can obscure the underlying prompts and responses, making them harder to debug". "Start by using LLM APIs directly." Three principles: simplicity, transparency (show planning steps), and a carefully designed agent-computer interface (ACI). On SWE-bench, "we actually spent more time optimizing our tools than the overall prompt." — [Anthropic](https://www.anthropic.com/research/building-effective-agents)
- The page now carries a note: "Much of the tooling landscape described in this post has changed since December 2024", and it points readers to "Claude Managed Agents". — [Anthropic](https://www.anthropic.com/research/building-effective-agents)

**Anthropic, "How we built our multi-agent research system" (Jun 13, 2025)**
- Architecture is orchestrator-worker. A LeadResearcher saves its plan to Memory, because context over 200,000 tokens gets truncated. It spawns parallel Subagents, each with its own context window, which search and return condensed findings. A separate CitationAgent then attributes claims to sources. — [Anthropic](https://www.anthropic.com/engineering/built-multi-agent-research-system)
- Rationale: "The essence of search is compression." Subagents compress information in parallel and give "separation of concerns", which "reduces path dependency". — [Anthropic](https://www.anthropic.com/engineering/built-multi-agent-research-system)
- Result: Opus 4 as lead with Sonnet 4 subagents "outperformed single-agent Claude Opus 4 by 90.2% on our internal research eval". The gain was largest on breadth-first queries, e.g. listing the board members of every IT company in the S&P 500. — [Anthropic](https://www.anthropic.com/engineering/built-multi-agent-research-system)
- "Multi-agent systems work mainly because they help spend enough tokens to solve the problem." On BrowseComp, three factors explained 95% of performance variance, and token usage alone explained 80%. Upgrading to Sonnet 4 helped more than doubling the token budget on Sonnet 3.7. — [Anthropic](https://www.anthropic.com/engineering/built-multi-agent-research-system)
- Good fit: "valuable tasks that involve heavy parallelization, information that exceeds single context windows, and interfacing with numerous complex tools". Poor fit: domains needing shared context or many inter-agent dependencies. "Most coding tasks involve fewer truly parallelizable tasks than research, and LLM agents are not yet great at coordinating and delegating to other agents in real time." — [Anthropic](https://www.anthropic.com/engineering/built-multi-agent-research-system)
- Effort-scaling rules embedded in the prompt:
  - Simple fact-finding: 1 agent with 3-10 tool calls.
  - Comparisons: 2-4 subagents with 10-15 calls each.
  - Complex research: more than 10 subagents.
  — [Anthropic](https://www.anthropic.com/engineering/built-multi-agent-research-system)
- Parallelization: the lead spawns 3-5 subagents at once, and each subagent makes 3+ tool calls in parallel. This "cut research time by up to 90% for complex queries". A tool-testing agent that rewrote MCP tool descriptions gave a "40% decrease in task completion time". — [Anthropic](https://www.anthropic.com/engineering/built-multi-agent-research-system)
- Each subagent's delegation brief needs "an objective, an output format, guidance on the tools and sources to use, and clear task boundaries". — [Anthropic](https://www.anthropic.com/engineering/built-multi-agent-research-system)

**OpenAI Agents SDK docs (agent orchestration page)**
- There are two ways to orchestrate: let the LLM decide (agents with instructions, tools and handoffs) or orchestrate in code. The two can be mixed. — [openai-agents-python docs/multi_agent.md](https://raw.githubusercontent.com/openai/openai-agents-python/main/docs/multi_agent.md)
- **Agents as tools**: "A manager agent keeps control of the conversation and calls specialist agents through `Agent.as_tool()`." Use it when one agent should own the final answer or apply shared guardrails. — [same](https://raw.githubusercontent.com/openai/openai-agents-python/main/docs/multi_agent.md)
- **Handoffs**: "A triage agent routes the conversation to a specialist, and that specialist becomes the active agent for the rest of the turn." The two primitives can be combined. — [same](https://raw.githubusercontent.com/openai/openai-agents-python/main/docs/multi_agent.md)
- Code orchestration is "more deterministic and predictable, in terms of speed, cost and performance". Patterns: structured-output routing, chaining, a `while` loop with an evaluator agent, and `asyncio.gather` for parallel runs. Advice: "Have specialized agents that excel in one task" and "Invest in evals". — [same](https://raw.githubusercontent.com/openai/openai-agents-python/main/docs/multi_agent.md)
- OpenAI's "A practical guide to building agents" PDF could not be fetched (proxy 403). UNVERIFIED from memory: it advises maximizing a single agent's capabilities first, and describes "manager" (agents as tools) and "decentralized" (handoff) multi-agent patterns. — [OpenAI PDF, blocked](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)

**Framework pattern terminology**
- LangGraph **supervisor (hierarchical)**: "specialized agents are coordinated by a central supervisor agent. The supervisor controls all communication flow and task delegation." LangChain now recommends "using the supervisor pattern directly via tools rather than this library for most use cases" because tool calling "gives you more control over context engineering". — [langgraph-supervisor-py README](https://raw.githubusercontent.com/langchain-ai/langgraph-supervisor-py/main/README.md)
- LangGraph **swarm**: "agents dynamically hand off control to one another based on their specializations. The system remembers which agent was last active." — [langgraph-swarm-py README](https://raw.githubusercontent.com/langchain-ai/langgraph-swarm-py/main/README.md)
- Microsoft Agent Framework orchestration patterns: "sequential, concurrent, handoff, and group collaboration patterns; includes checkpointing, streaming, human-in-the-loop, and time-travel". — [microsoft/agent-framework README](https://raw.githubusercontent.com/microsoft/agent-framework/main/README.md)
- CrewAI splits the same choice in two: "Crews" for autonomy and role-based collaboration, and "Flows" for "event-driven" deterministic control. — [crewAI README](https://raw.githubusercontent.com/crewAIInc/crewAI/main/README.md)
- MetaGPT models a team on a software company. It has PM, architect, project manager and engineer roles and "carefully orchestrated SOPs" (its tagline is "Code = SOP(Team)"). — [MetaGPT README](https://raw.githubusercontent.com/geekan/MetaGPT/main/README.md)
- Cognition, "Don't Build Multi-Agents" (Walden Yan, Jun 2025): "share context, and share full agent traces, not just individual messages", and "Actions carry implicit decisions, and conflicting decisions carry bad results". A reported April 2026 follow-up, "Multi-Agents: What's Actually Working", says a narrow class of setups does work. Snippets also say Devin's coding and review agents work best *without* shared context. SNIPPET-ONLY: cognition.ai is blocked. — [search snippets via fortegrp/x.com](https://fortegrp.com/insights/designing-effective-agent-architectures-principles-for-enterprise-ai-systems), [X post](https://x.com/walden_yan/status/2047054401341370639)
- **Debate**: multiple agents propose answers and critique each other over several rounds (Du et al., "Improving Factuality and Reasoning in Language Models through Multiagent Debate", arXiv 2305.14325). UNVERIFIED: not fetched, arXiv is blocked.
- **Blackboard**: agents read and write a shared state store instead of messaging each other directly. No primary source was fetched. The closest confirmed analogues are LangGraph's shared graph state and Anthropic's "subagent output to a filesystem" artifact pattern (see the next section). UNVERIFIED as a named pattern in the sources I checked.

### Inferences
- The decision ladder: single call → workflow → single agent with tools → orchestrator plus isolated-context subagents. Go up a rung only when evals show a gain.
- "Agents as tools" (OpenAI) is the same pattern as "supervisor via tool calling" (LangChain), as Anthropic's lead-agent-spawns-subagents design, and as Claude Agent SDK subagents. This is the convergent default in 2026.
- Handoff/swarm suits conversational routing such as customer-support triage. Supervisor/agents-as-tools suits tasks whose output must be synthesized.

### Gaps
- Could not read the full OpenAI practical guide PDF, the Cognition post, or the debate and blackboard papers; all were blocked.
- The "Claude Managed Agents" post now referenced by Anthropic was not fetched.

## How agents communicate and share state: message passing, shared memory, handoffs, MCP and A2A

### Takeaway
Within one process, agents communicate in three ways: as tool calls whose results come back to the caller, as handoffs that transfer the conversation, or through a shared state or memory store. The lesson in production is to pass references to artifacts rather than copy large outputs through the coordinator. Between systems, MCP standardizes agent-to-tool and agent-to-data connections, and A2A standardizes peer communication between opaque agents (JSON-RPC 2.0 over HTTP(S), Agent Cards). The two are complementary.

### Cited Findings
- Anthropic's research system:
  - Lead and subagents communicate synchronously: the lead "waits for each set of subagents to complete". Consequences: "the lead agent can't steer subagents, subagents can't coordinate, and the entire system can be blocked while waiting for a single subagent". Going asynchronous would add "challenges in result coordination, state consistency, and error propagation".
  - Shared external memory: the plan is saved to Memory. Agents "summarize completed work phases and store essential information in external memory" and "spawn fresh subagents with clean contexts while maintaining continuity through careful handoffs".
  - Artifact pattern: "Subagents call tools to store their work in external systems, then pass lightweight references back to the coordinator". This avoids a "game of telephone" and reduces "token overhead from copying large outputs through conversation history".
  — [Anthropic](https://www.anthropic.com/engineering/built-multi-agent-research-system)
- OpenAI SDK: with agents-as-tools the manager keeps the conversation. With handoffs the specialist "becomes the active agent for the rest of the turn". Code chaining transforms one agent's output into the next agent's input. — [OpenAI Agents SDK docs](https://raw.githubusercontent.com/openai/openai-agents-python/main/docs/multi_agent.md)
- LangGraph: the supervisor uses a "tool-based agent handoff mechanism". The swarm uses "customizable handoff tools" and remembers the last active agent. LangGraph itself provides "durable execution", "human-in-the-loop" state inspection and modification, and "short-term working memory ... and long-term persistent memory across sessions". — [supervisor](https://raw.githubusercontent.com/langchain-ai/langgraph-supervisor-py/main/README.md), [swarm](https://raw.githubusercontent.com/langchain-ai/langgraph-swarm-py/main/README.md), [LangGraph README](https://raw.githubusercontent.com/langchain-ai/langgraph/main/README.md)
- **MCP**: Anthropic describes the Model Context Protocol as a way to "integrate with a growing ecosystem of third-party tools with a simple client implementation". With MCP servers, "agents encounter unseen tools with descriptions of wildly varying quality". The Python SDK `mcp` is at **2.2.0** (released 2026-09-07). — [Anthropic BEA](https://www.anthropic.com/research/building-effective-agents), [Anthropic MARS](https://www.anthropic.com/engineering/built-multi-agent-research-system), [PyPI mcp](https://pypi.org/pypi/mcp/json), [repo](https://github.com/modelcontextprotocol/python-sdk)
- **A2A (Agent2Agent)**:
  - What it is: "An open protocol enabling communication and interoperability between opaque agentic applications". It lets agents "communicate and collaborate ... as agents, not just as tools".
  - Features: capability discovery via "Agent Cards"; negotiation of modalities; long-running tasks; "without exposing their internal state, memory, or tools".
  - Transport: "JSON-RPC 2.0 over HTTP(S)" with sync, streaming (SSE) and async push notifications.
  - Governance: Linux Foundation project contributed by Google, Apache 2.0 licence. It "complements MCP".
  - `a2a-sdk` is at **1.2.1** (2026-09-30).
  — [A2A README](https://raw.githubusercontent.com/a2aproject/A2A/main/README.md), [PyPI a2a-sdk](https://pypi.org/pypi/a2a-sdk/json)
- Framework support for the protocols:
  - Microsoft Agent Framework 1.0 advertises "cross-runtime interoperability via A2A and MCP". — [autogen README](https://raw.githubusercontent.com/microsoft/autogen/main/README.md)
  - Google ADK 2.0 lists a "Task API: Structured agent-to-agent delegation". — [adk-python README](https://raw.githubusercontent.com/google/adk-python/main/README.md)
  - The A2A course shows exposing ADK, LangGraph or BeeAI agents as A2A servers. — [A2A README](https://raw.githubusercontent.com/a2aproject/A2A/main/README.md)
- UNVERIFIED: MCP was donated to the Linux Foundation's Agentic AI Foundation in Dec 2025. Not confirmed in this session.

### Inferences
- In a hackathon-scale system, the most robust communication design is a coordinator calling subagents as tools. Subagents write large outputs to files or a store and return paths or summaries.
- A2A only matters when agents cross process or organization boundaries. Inside one app, plain function or tool calls are simpler.
- MCP is the de facto tool interface. Tool-description quality is a measured performance lever (the 40% time reduction above).

### Gaps
- Did not read the MCP or A2A spec pages themselves for message schemas (task lifecycle states, etc.).
- No benchmark comparing message-passing and shared-state designs was found.

## Frameworks in 2026: GitHub URLs, latest versions and fit

### Takeaway
The field has consolidated:
- AutoGen is in maintenance mode, succeeded by Microsoft Agent Framework 1.x. AG2 continues the community fork.
- LangGraph (1.2.x) is the low-level stateful-graph choice. CrewAI (1.15.x) is the role-based "crew" choice. The OpenAI Agents SDK (0.22.x) and Claude Agent SDK (0.2.x) are lightweight vendor harnesses. Google ADK is at 2.x.
- LlamaIndex has shifted company focus to document parsing. MetaGPT's PyPI package has been idle since March 2025.

### Cited Findings
All versions are from PyPI JSON, queried 2026-10-02 ([pattern](https://pypi.org/pypi/langgraph/json)).

| Framework (PyPI pkg) | Latest version (upload date) | GitHub | One-line fit (sourced) |
|---|---|---|---|
| LangGraph (`langgraph`) | 1.2.12 (2026-09-21) | https://github.com/langchain-ai/langgraph | "Low-level orchestration framework for building, managing, and deploying long-running, stateful agents"; durable execution, HITL, memory. Supervisor/swarm add-ons exist, but LangChain now recommends supervisor-via-tools. [README](https://raw.githubusercontent.com/langchain-ai/langgraph/main/README.md) |
| CrewAI (`crewai`) | 1.15.23 (2026-09-28) | https://github.com/crewAIInc/crewAI | Role-based "Crews" for autonomous collaboration plus event-driven "Flows" for precise control. [README](https://raw.githubusercontent.com/crewAIInc/crewAI/main/README.md) |
| AutoGen (`autogen-agentchat`) | 0.7.5 (2025-09-30) | https://github.com/microsoft/autogen | "Maintenance mode. It will not receive new features"; new users should start with Microsoft Agent Framework. [README](https://raw.githubusercontent.com/microsoft/autogen/main/README.md) |
| Microsoft Agent Framework (`agent-framework`) | 1.19.0 (2026-09-18) | https://github.com/microsoft/agent-framework | "Enterprise-ready successor to AutoGen", for .NET/Python (README also mentions Go); graph workflows with sequential/concurrent/handoff/group patterns, checkpointing, A2A and MCP. [README](https://raw.githubusercontent.com/microsoft/agent-framework/main/README.md) |
| AG2 (`ag2`) | 1.1.1 (2026-09-29) | https://github.com/ag2ai/ag2 | Community continuation of AutoGen, now billed as an "Open-Source AgentOS". Classic `ConversableAgent`/`GroupChat`/`import autogen` moved to "AG2 Classic"; the `ag2` package "no longer ships the `autogen` import name". [README](https://raw.githubusercontent.com/ag2ai/ag2/main/README.md) |
| OpenAI Agents SDK (`openai-agents`) | 0.22.3 (2026-09-17) | https://github.com/openai/openai-agents-python | Minimal primitives (agents, tools, handoffs, guardrails); agents-as-tools vs handoffs. [docs](https://raw.githubusercontent.com/openai/openai-agents-python/main/docs/multi_agent.md) |
| Claude Agent SDK (`claude-agent-sdk`) | 0.2.163 (2026-09-30) | https://github.com/anthropics/claude-agent-sdk-python | Claude Code harness as a library (bundles the CLI); full Read/Write/Edit/Bash toolset, permissions, MCP; README mentions "programmatic subagents and session forking". [README](https://raw.githubusercontent.com/anthropics/claude-agent-sdk-python/main/README.md) |
| Google ADK (`google-adk`) | 2.11.0 (2026-10-02) | https://github.com/google/adk-python | "Code-first Python framework"; "optimized for Gemini, ADK is model-agnostic"; "multiple specialized agents into flexible hierarchies"; Task API for agent-to-agent delegation. [README](https://raw.githubusercontent.com/google/adk-python/main/README.md) |
| Pydantic AI (`pydantic-ai`) | 2.53.0 (2026-10-02) | https://github.com/pydantic/pydantic-ai | Typed, model-agnostic agent loop with validated outputs; Pydantic Graph and Evals are separate packages; claims coverage "to complex, long-running multi-agent collaboration". [README](https://raw.githubusercontent.com/pydantic/pydantic-ai/main/README.md) |
| smolagents (`smolagents`) | 1.26.0 (2026-05-29) | https://github.com/huggingface/smolagents | "Agents that think in code!"; core logic "fits in ~1,000 lines of code"; minimal abstractions. [README](https://raw.githubusercontent.com/huggingface/smolagents/main/README.md) |
| LlamaIndex (`llama-index-core`) | 0.14.25 (2026-09-21) | https://github.com/run-llama/llama_index | OSS framework for RAG and agent apps; the company says its "primary focus has shifted towards LlamaParse" (document parsing). Best when agents are document- or RAG-heavy. [README](https://raw.githubusercontent.com/run-llama/llama_index/main/README.md) |
| MetaGPT (`metagpt`) | 0.8.2 (2025-03-09) | https://github.com/geekan/MetaGPT | SOP-driven "software company" of role agents; the team's attention moved to the MGX product (launched Feb 2025). [README](https://raw.githubusercontent.com/geekan/MetaGPT/main/README.md) |
| CAMEL (`camel-ai`) | 0.2.90 (2026-03-22) | https://github.com/camel-ai/camel | Role-playing / "society" multi-agent research framework. UNVERIFIED description: README grep and GitHub API returned nothing usable this session. |

### Inferences
- For a 2026 hackathon:
  - Claude Agent SDK or OpenAI Agents SDK: fastest path to an orchestrator-plus-subagents setup with one vendor.
  - LangGraph: when explicit state, checkpointing or human-in-the-loop matter.
  - CrewAI: quick role-based demos.
  - Pydantic AI: typed outputs across providers.
  - Avoid starting new work on AutoGen 0.x (maintenance mode) or MetaGPT (stale on PyPI).
- AG2's 1.x break (classic `autogen` namespace removed) means old AutoGen tutorials will not run as-is on `pip install ag2`.
- Pre-1.0 version numbers (openai-agents 0.22, claude-agent-sdk 0.2) signal API churn. Pin versions.

### Gaps
- GitHub star counts were not collected; the GitHub API was not authorized for these repos.
- CAMEL's README positioning was not confirmed.
- Microsoft Agent Framework's PyPI 1.19.0 vs the README's "1.0" wording reflects minor-version increments; I did not check the exact 1.0 GA date.

## Documented failure modes and costs, and when a single agent beats multi-agent

### Takeaway
Multi-agent systems are expensive: about 15× the tokens of chat and about 3.75× those of a single agent, per Anthropic. They fail mostly through specification and system design, then inter-agent misalignment, then weak verification (MAST). Single agents or workflows win when subtasks are tightly coupled or need shared context (most coding), when the task isn't valuable enough to pay for the tokens, or when one context window suffices.

### Cited Findings
**Costs**
- "Agents typically use about 4× more tokens than chat interactions, and multi-agent systems use about 15× more tokens than chats." "Multi-agent systems require tasks where the value of the task is high enough to pay for the increased performance." — [Anthropic](https://www.anthropic.com/engineering/built-multi-agent-research-system)
- Coordination complexity "grows rapidly". Early failures included "spawning 50 subagents for simple queries, scouring the web endlessly for nonexistent sources, and distracting each other with excessive updates". Other failures: duplicated work across subagents caused by vague delegation; agents "continuing when they already had sufficient results"; and a preference for "SEO-optimized content farms" over authoritative sources. — [Anthropic](https://www.anthropic.com/engineering/built-multi-agent-research-system)
- "Agents are stateful and errors compound". "Small changes to the lead agent can unpredictably change how subagents behave" (emergent behaviour). Mitigations: resume from checkpoints rather than restart, retry logic, full production tracing, and "rainbow deployments" so running agents are not broken mid-task. — [Anthropic](https://www.anthropic.com/engineering/built-multi-agent-research-system)
- Infinite loops and runaway agents: Anthropic recommends "stopping conditions (such as a maximum number of iterations)". The research system added "explicit guardrails to prevent the agents from spiraling out of control" and explicit effort budgets. — [BEA](https://www.anthropic.com/research/building-effective-agents), [MARS](https://www.anthropic.com/engineering/built-multi-agent-research-system)

**MAST: "Why Do Multi-Agent LLM Systems Fail?"**
- Authors and venue: Cemri, Pan, Yang, Agrawal, Chopra, Tiwari, Keutzer, Parameswaran, Klein, Ramchandran, et al. (UC Berkeley), arXiv 2503.13657. NeurIPS 2025 Datasets & Benchmarks Track (a proceedings PDF exists per search). The repo has an annotated trace dataset on HF (`mcemri/MAD`, also `mcemri/MAST-Data`). — [MAST repo README](https://raw.githubusercontent.com/multi-agent-systems-failure-taxonomy/MAST/main/README.md), [NeurIPS PDF (search result)](https://proceedings.neurips.cc/paper_files/paper/2025/file/b1041e52d3be19f0a9bc491657488e4a-Paper-Datasets_and_Benchmarks_Track.pdf)
- 14 failure modes in 3 categories: (i) specification / system design, (ii) inter-agent misalignment, (iii) task verification / termination. Built from 150 traces with expert annotators, inter-annotator κ = 0.88. MAST-Data has 1,600+ (1,642) annotated traces from 7 MAS frameworks. SNIPPET-ONLY (arXiv blocked). — [arXiv HTML v3 (snippet)](https://arxiv.org/html/2503.13657v3), [HF dataset (snippet)](https://huggingface.co/datasets/mcemri/MAST-Data)
  - Version conflict: another snippet says "five open-source frameworks across 150 tasks", and the repo README says "over 1K annotated MAS traces". These likely reflect paper v1/v2 vs v3. — [themoonlight review snippet](https://www.themoonlight.io/en/review/why-do-multi-agent-llm-systems-fail), [MAST README](https://raw.githubusercontent.com/multi-agent-systems-failure-taxonomy/MAST/main/README.md)
- Category shares of 41.8% (specification), 36.9% (inter-agent misalignment) and 21.3% (verification). SNIPPET-ONLY, and from a secondary aggregator. — [futureagi substack (snippet)](https://futureagi.substack.com/p/why-do-multi-agent-llm-systems-fail)
- Mode names:
  - **FC1, specification / system design**:
    - FM-1.1 Disobey task specification (≈11.8% per snippet)
    - FM-1.2 Disobey role specification (UNVERIFIED)
    - FM-1.3 Step repetition (among the most prevalent)
    - FM-1.4 Loss of conversation history
    - FM-1.5 Unaware of termination conditions
  - **FC2, inter-agent misalignment**:
    - FM-2.1 Conversation reset
    - FM-2.2 Fail to ask for clarification (UNVERIFIED)
    - FM-2.3 Task derailment (UNVERIFIED)
    - FM-2.4 Information withholding
    - FM-2.5 Ignored other agent's input
    - FM-2.6 Reasoning-action mismatch
  - **FC3, task verification**:
    - FM-3.1 Premature termination
    - FM-3.2 No or incomplete verification
    - FM-3.3 Incorrect verification (UNVERIFIED)
  
  Names without the UNVERIFIED tag appear in search snippets (SNIPPET-ONLY). "Unaware of termination conditions" and "information withholding" were described as appearing "almost exclusively in failed runs". — [search snippets incl. HF IBM/Berkeley blog](https://huggingface.co/blog/ibm-research/itbenchandmast), [arXiv PDF (snippet)](https://arxiv.org/pdf/2503.13657)
- A GitHub issue notes that the human-labelled dataset "spans three taxonomy versions with renumbered codes". Mode numbering differs across paper versions. — [MAST issue #18 (search title)](https://github.com/multi-agent-systems-failure-taxonomy/MAST/issues/18)
- Snippet takeaway: failures stem from "system design decisions and poor or ambiguous prompt specifications", not only from model capability. Recommends multi-level verification. SNIPPET-ONLY. — [arXiv PDF snippet](https://arxiv.org/pdf/2503.13657), [futureagi](https://futureagi.substack.com/p/why-do-multi-agent-llm-systems-fail)

**When a single agent wins**
- Anthropic: multi-agent is a poor fit for domains "that require all agents to share the same context or involve many dependencies between agents", e.g. most coding. For many applications a single optimized call with retrieval "is usually enough". — [MARS](https://www.anthropic.com/engineering/built-multi-agent-research-system), [BEA](https://www.anthropic.com/research/building-effective-agents)
- Cognition: splitting work across agents without shared full traces leads to conflicting implicit decisions. SNIPPET-ONLY. — [fortegrp summary](https://fortegrp.com/insights/designing-effective-agent-architectures-principles-for-enterprise-ai-systems)
- OpenAI SDK: code orchestration is more predictable in "speed, cost and performance" than LLM-driven orchestration. — [OpenAI Agents SDK docs](https://raw.githubusercontent.com/openai/openai-agents-python/main/docs/multi_agent.md)

### Inferences
- Since token spend explains about 80% of BrowseComp variance, much of the "multi-agent advantage" is a compute-budget effect. Compare against a single agent given an equal token budget before concluding that the architecture helps.
- About 15× chat cost (≈3.75× a single agent) means multi-agent is only worth it when a task's value clearly exceeds that marginal spend.
- Most MAST categories (specification, termination, verification) are addressable with engineering: explicit role and termination specs, iteration caps, and an independent verifier or evaluator-optimizer step. These double as a design checklist.

### Gaps
- Could not read the MAST paper body (arXiv blocked), so I could not confirm per-mode percentages, the full list of frameworks studied (e.g. ChatDev, MetaGPT, HyperAgent, AppWorld, AG2, Magentic-One, OpenManus: UNVERIFIED), or intervention results.
- No independent benchmark numbers on single-agent vs multi-agent token multipliers were found beyond Anthropic's figures.
- No quantitative data on infinite-loop frequency was found, beyond MAST's step-repetition and termination modes.
