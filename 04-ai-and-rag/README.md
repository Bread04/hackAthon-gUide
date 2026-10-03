# 🤖 04 · AI & RAG

<!-- markdownlint-disable MD013 -->

> **In plain English:** adding AI features to *your app*: a chatbot, an agent that uses tools, summarising, answering questions about your own documents (that's "RAG"), or talking out loud (voice). Only need AI for *writing code*? That's in [`05-tools-and-mcp/`](../05-tools-and-mcp/README.md) instead.

## 🟢 Start here

| File | What it's for |
| --- | --- |
| [`docs/how-llms-work.md`](docs/how-llms-work.md) | **How LLMs work, for builders:** tokens, context, caching, tool loops, structured output, cost; plus learning repos |
| ⭐ [`docs/model-selection.md`](docs/model-selection.md) | Which AI model to use and what it costs, including **free options** (Gemini free tier, Ollama) |
| [`docs/prompt-engineering.md`](docs/prompt-engineering.md) | How to write instructions that get good, reliable answers |
| [`PROMPTS-ML.md`](PROMPTS-ML.md) | Copy-paste prompts: pick a model (ML1), build a chat feature (ML5), **make the AI demo-proof (ML7)** |

## 🟡 When you need it

| File | Read it when… |
| --- | --- |
| ⭐ [`docs/agents-and-tool-use.md`](docs/agents-and-tool-use.md) + [`PROMPTS-ML.md`](PROMPTS-ML.md) ML12–14 | Your AI needs to **do things**: call tools, search, act in steps (agents, MCP, safety, demo-proofing) |
| [`docs/ml-in-your-app.md`](docs/ml-in-your-app.md) | **Putting a model in your app:** API vs pretrained vs train your own, Hugging Face in 2026, serving with FastAPI/ONNX (tested), in-browser models, fine-tuning vs prompting vs RAG |
| [`docs/multi-agent-systems.md`](docs/multi-agent-systems.md) | **Thinking about multiple agents?** How they work, what they cost (~15x tokens), how they fail, frameworks, and a decision tree for agents vs workflow vs RAG at a hackathon |
| [`docs/rag-architecture.md`](docs/rag-architecture.md) + [`PROMPTS-RAG.md`](PROMPTS-RAG.md) | Your app answers questions using your own documents; RAG variants (hybrid, rerank, contextual, GraphRAG, agentic, ColPali, long context) and when each is worth it |
| [`docs/voice-and-realtime.md`](docs/voice-and-realtime.md) | Your app talks or listens |
| [`docs/multimodal-pipelines.md`](docs/multimodal-pipelines.md) | Your app analyses video, audio or documents |

## 📚 Reference

| File | What it's for |
| --- | --- |
| [`docs/repos.md`](docs/repos.md) | Recommended AI and data libraries |
| [`docs/evidence.md`](docs/evidence.md) | Sources behind this folder (you can skip this) |

**Next folder:** [`05-tools-and-mcp/`](../05-tools-and-mcp/README.md) · Words you don't know? See [`GLOSSARY.md`](../GLOSSARY.md).
