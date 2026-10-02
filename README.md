# 🧰 Hackathon Toolkit

<!-- markdownlint-disable MD013 -->

> **A guide to building something impressive at a hackathon with AI coding tools**, even if you're new to this.
> It runs the whole event on the **BMad Method**: step-by-step skills for Claude Code that take you from idea → plan → build → check → pitch. Every step also has a copy-paste prompt if you don't have Claude Code.

---

## 👋 New here? Start with these steps

| Step | Open this | Time |
| --- | --- | --- |
| **1. Understand the event** | [`01-hackathon-playbook/first-hackathon.md`](01-hackathon-playbook/first-hackathon.md): your first hackathon in plain English, and how BMad guides each step | 10 min |
| **2. Set up BMad** | [`01-hackathon-playbook/docs/setup-bmad.md`](01-hackathon-playbook/docs/setup-bmad.md): install Claude Code and practise once, the week before | 45 min |
| **3. See it done** | [`01-hackathon-playbook/worked-example.md`](01-hackathon-playbook/worked-example.md): one team's whole hackathon, skill by skill | 10 min |
| **📊 Entering a Datathon / ML Track?** | [`08-datathon-handbook/00-start-here/first-datathon.md`](08-datathon-handbook/00-start-here/first-datathon.md): The beginner's guide to data hackathons, 1-click starter & ML solutions | 10 min |
| **📊 Datathon day-of** | [`08-datathon-handbook/00-start-here/runbook.md`](08-datathon-handbook/00-start-here/runbook.md): 8 gates with exit checks and decision tables | 10 min |

Not sure where you are in the event? [`WORKFLOW.md`](WORKFLOW.md) maps every phase. Words you don't know? [`GLOSSARY.md`](GLOSSARY.md). On the day, follow [`battle-plan.md`](01-hackathon-playbook/battle-plan.md) hour by hour.

---

## ⚠️ Three things that trip beginners up

1. **Get your app online in the first hour**, even if it only says "hello". Leaving deployment to the end is the most common way projects die. See [`03-backend/docs/deploy-step-by-step.md`](03-backend/docs/deploy-step-by-step.md).
2. **Never put passwords or API keys in your code.** Keep them in a `.env` file that isn't uploaded to GitHub. See [`GLOSSARY.md`](GLOSSARY.md) → "API key".
3. **Stop adding features about 3 hours before the deadline.** Use that time to fix bugs, record a backup video and practise the pitch.

The full list of "things that changed recently" (tools that were renamed, retired or changed price) is in [`06-repo-catalog/`](06-repo-catalog/README.md) → "What changed in 2025–26".

---

---

## 🛤️ Pick a learning path

| You are… | Read in this order | Time |
| --- | --- | --- |
| **A complete beginner** | [`first-hackathon.md`](01-hackathon-playbook/first-hackathon.md) → [`GLOSSARY.md`](GLOSSARY.md) → [`worked-example.md`](01-hackathon-playbook/worked-example.md) → [`setup-your-laptop.md`](01-hackathon-playbook/docs/setup-your-laptop.md) | ~1 h |
| **Here to win a software hackathon** | [`rules-hardware-a11y-remote.md`](01-hackathon-playbook/docs/rules-hardware-a11y-remote.md) → [`problem-selection.md`](01-hackathon-playbook/templates/problem-selection.md) → [`battle-plan.md`](01-hackathon-playbook/battle-plan.md) → [`pitch-and-demo.md`](01-hackathon-playbook/docs/pitch-and-demo.md) → [`mvp-and-demo-checklist.md`](01-hackathon-playbook/templates/mvp-and-demo-checklist.md) | ~2 h |
| **Entering a datathon / ML track** | [`first-datathon.md`](08-datathon-handbook/00-start-here/first-datathon.md) → [`runbook.md`](08-datathon-handbook/00-start-here/runbook.md) → [`method-selection-guide.md`](08-datathon-handbook/03-modeling/method-selection-guide.md) | ~1.5 h |
| **Not a coder** | [`first-hackathon.md`](01-hackathon-playbook/first-hackathon.md) → [`non-coder-guide.md`](01-hackathon-playbook/docs/non-coder-guide.md) → [`problem-selection.md`](01-hackathon-playbook/templates/problem-selection.md) → [`pitch-and-demo.md`](01-hackathon-playbook/docs/pitch-and-demo.md) → [`project-readme.md`](01-hackathon-playbook/templates/project-readme.md): non-coders can own research, the idea, the README and the pitch | ~1 h |

**Where am I in the event?** [`WORKFLOW.md`](WORKFLOW.md) maps all 8 phases for both tracks, with exit checks. Printable one-pagers: [`CHEATSHEET.md`](CHEATSHEET.md) (software) · [`08-datathon-handbook/00-start-here/cheatsheet.md`](08-datathon-handbook/00-start-here/cheatsheet.md) (datathon).

---

## 🗺️ What's in each folder

The folders are numbered in the order you'll need them.

| Folder | In plain English | Level |
| --- | --- | --- |
| 📘 [`01-hackathon-playbook/`](01-hackathon-playbook/README.md) | **The game plan.** The BMad workflow hour by hour: setup, idea, plan, build, pitch, after | 🟢 Start here |
| 🎨 [`02-frontend/`](02-frontend/README.md) | **What people see.** Making the app look good and work on phones | 🟢/🟡 |
| ⚙️ [`03-backend/`](03-backend/README.md) | **The engine behind the scenes.** Database, logins, keeping it secure, putting it online | 🟡 |
| 🤖 [`04-ai-and-rag/`](04-ai-and-rag/README.md) | **Adding AI to your app.** Choosing a model, writing prompts, agents and tools, chatbots over your own documents, voice | 🟡 |
| 🔧 [`05-tools-and-mcp/`](05-tools-and-mcp/README.md) | **Supercharging your AI assistant.** Plug-ins (MCP servers), free tools, staying safe | 🟡 |
| ⭐ [`06-repo-catalog/`](06-repo-catalog/README.md) | **The shopping list.** Recommended free libraries, and ones to avoid | 🟢 |
| 🧠 [`07-bmad-workflow/`](07-bmad-workflow/README.md) | **The BMad engine room.** The config pack that wires this toolkit into BMad, a verdict on every BMad skill, and how the wiring works. Open it when setup asks you to, or to customise | 🟡 |
| 📊 [`08-datathon-handbook/`](08-datathon-handbook/README.md) | **The Datathon Playbook.** For data science & ML competitions: dirty data wrangling (Polars/DuckDB), rapid ML baselines (LightGBM/CatBoost/AutoGluon), leak-free CV, What-If simulators, and winning pitch decks | 🟢/🟡 |
| 📄 [`skills/`](skills/README.md) | **One-page cheat sheets** you can load into Claude Code | 🟢 |
| 🔬 [`_research/`](_research/README.md) | **The proof.** Where every fact came from. You never need to open this | 📚 Reference |

**Level key:** 🟢 beginner-friendly · 🟡 read it when you need it · 🔴 advanced · 📚 reference only

---

## 🔗 Everything else

| Need | Open |
| --- | --- |
| A task-based index ("I want to…"), the FAQ, price tags and how the folders fit together | [`FAQ.md`](FAQ.md) |
| Words you don't know | [`GLOSSARY.md`](GLOSSARY.md) |
| What is still unverified, and how to help | [`VERIFICATION.md`](VERIFICATION.md) · [`ROADMAP.md`](ROADMAP.md) · [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| Where every fact came from | [`_research/`](_research/README.md) |

---

_Credits: parts adapted from MIT-licensed works, see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md) · Plans: [`ROADMAP.md`](ROADMAP.md)._

_Built from cited research (checked 2026-09-21 to 2026-10-02; see [`VERIFICATION.md`](VERIFICATION.md) for what is still unconfirmed). Prices and tools change often; the next re-check is due **2026-10-22**. See [`_research/README.md`](_research/README.md) for how to refresh it._
