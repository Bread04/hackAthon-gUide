# Hackathon rules on AI coding assistants, originality, pre-existing work, IP

Access note: ethglobal.com, guide.mlh.com, microsoft.github.io and all *.devpost.com pages were blocked by the egress proxy. Only GitHub-hosted pages were fetched in full (marked FETCHED). Everything else is SNIPPET-ONLY (WebSearch summary text, not the page itself). Quotes in SNIPPET-ONLY items are search-tool paraphrases, not verified verbatim.

## 1. What major organizers say about AI assistants, vibe coding, disclosure, disqualification

### Takeaway
Across organizers the pattern is "AI allowed, disclosure mandatory, human authorship/judgement expected". Non-disclosure is the disqualification trigger, not AI use. No source found uses the term "vibe coding" in a rule; MLH's wording is "created by humans".

### Cited Findings
- MLH Standard Hackathon Rules (FETCHED): "Teams may use AI to assist them while coding, utilizing tools such as code completion, code generation, image generation, or other similar tools." Teams "should be honest and transparent about the AI code tools they used"; list AI tools in submissions and answer judge/organizer questions. Disqualification is at organizers' discretion (breaking rules, CoC, "unsporting behaviour"). — [MLH standard-hackathon-rules.md](https://github.com/MLH/mlh-policies/blob/main/standard-hackathon-rules.md)
- MLH (SNIPPET-ONLY): Member Events must allow generative AI/LLMs; MLH "strongly believes that the best projects will leverage AI, but ultimately be created by humans"; recommends crediting tools, detailed README separating what was built vs used, project should not be a reskin of an existing AI tool; uncredited AI use leads to disqualification and a report to cheating@mlh.io. — [MLH member-event-guidelines.md](https://github.com/MLH/mlh-policies/blob/main/member-event-guidelines.md) and [MLH guide](https://guide.mlh.com/general-information/judging-and-submissions/rules-for-your-hackathon) (search result text, UNVERIFIED verbatim)
- ETHGlobal (SNIPPET-ONLY): AI tools (ChatGPT, Claude Code, Copilot, Cursor) "generally permitted"; submission must document where and how AI was used, including which code, files or assets were AI-generated/assisted. Undisclosed pre-existing work or material misrepresentation of what was built may mean disqualification, prizes revoked, ban from future events. — [ethglobal.com/rules](https://ethglobal.com/rules); also [New York 2025 details](https://ethglobal.com/events/newyork2025/info/details)
- Devpost "Agents for Humans" (AWS Strands) (SNIPPET-ONLY): may use "frameworks, libraries, starter templates, and AI coding assistants" but must disclose any other pre-existing code or work. Projects must be newly created in the Submission Period. — [rules](https://agentsforhumans.devpost.com/rules)
- USAII Global AI Hackathon 2026 (SNIPPET-ONLY): AI coding assistants allowed but must be disclosed. — [rules](https://usaii-global-ai-hackathon-2026.devpost.com/rules)
- First Commit Hackathon 2026 (SNIPPET-ONLY): write-up must name AI coding tools used; non-disclosure is a violation. — [repo](https://github.com/WaliMedugu/first-commit-hacathon-2026); an AWS First Commit team repo issue asks all members for AI tools disclosure before submission: [issue #13](https://github.com/dhruvh6/aws_firstcommit_hackathon/issues/13)
- ImpactHack 2026 (SNIPPET-ONLY): no penalty for AI-assisted development; AI allowed throughout. — [page](https://impacthack26.devpost.com/)
- Microsoft Agents League 2026 (FETCHED): no specific AI-tool restriction; entry must be "your own original work"; Demo Video "solely your own work"; disqualification for multiple accounts, automated participation, cheating/fraud. — [OFFICIAL RULES.md](https://github.com/microsoft/Agents-League-AISF-Regulations/blob/main/OFFICIAL%20RULES.md)
- Microsoft AI Dev Days Hackathon (FETCHED): disqualifies "cheating, hacking, creating a bot or other automated program, or...committing fraud". — [OFFICIAL_RULES.md](https://github.com/Azure/AI-Dev-Days-Hackathon/blob/main/OFFICIAL_RULES.md)

### Inferences
- Safe default for any event: keep a log (tool, purpose, files) from hour one; put it in README/submission even if the rules are silent.
- "Reskin of an AI tool" (MLH guidance) is a risk for thin LLM-wrapper projects.

### Gaps
- No Google/AWS rule found that explicitly uses the phrase "vibe coding". Corporate rules (Microsoft Agents League) are silent on AI tools, so team should ask organizers.
- Could not fetch ETHGlobal, Devpost, or MLH guide pages directly; no university hackathon (e.g. HackTX 2025, UH CodeRED) text retrieved beyond search links.

## 2. Pre-event code, templates/boilerplate, open source, sponsor APIs

### Takeaway
Standard rule: start new work at kickoff; libraries/frameworks/open source/starter kits are fine; reuse of your own earlier project code is not (MLH) unless an event has a continuity/extension track with disclosure.

### Cited Findings
- MLH (FETCHED): teams may work on a prior idea "as long as they do not re-use code or other project materials"; "Teams can use libraries, frameworks, or open-source code"; open-sourcing a project pre-event just to use it at the event is "against the spirit of the rules and is not allowed." — [MLH rules](https://github.com/MLH/mlh-policies/blob/main/standard-hackathon-rules.md)
- ETHGlobal (SNIPPET-ONLY): Classic "From Scratch" track bans pre-existing project-specific code, designs, assets; open-source libraries and starter kits OK. Continuity tracks ("Extend Open Source", "Ship a Feature") allow existing codebase if pre-event work is documented and substantive new work is shown. Pre-built projects may enter but not qualify for partner prizes or Finalist. Pre-existing work must be disclosed in writing to ETHGlobal and in the submission. — [ethglobal.com/rules](https://ethglobal.com/rules)
- Microsoft AI Dev Days 2026 (FETCHED): "Projects must be newly created by the Entrant after the start of the Hackathon Submission Period" (Feb 10 - Mar 15, 2026); may incorporate open-source if they "comply with applicable open source licenses" and build upon it; ineligible if developed with sponsor financial or preferential support; third-party integration permissions must be obtained. — [OFFICIAL_RULES.md](https://github.com/Azure/AI-Dev-Days-Hackathon/blob/main/OFFICIAL_RULES.md)
- OpenAI Build Week on Devpost (SNIPPET-ONLY): new, or pre-existing project "meaningfully extended" with Codex/GPT-5.6 after start; pre-existing projects judged only on added work; must document prior vs new work with evidence of tool use in the period. — [rules](https://openai.devpost.com/rules)
- WebMCP Challenge (SNIPPET-ONLY): same extended-pre-existing model, judged on work added. — [rules](https://webmcp.devpost.com/rules)
- Devpost eligibility pattern (SNIPPET-ONLY): projects that received sponsor/administrator funding, were developed under contract, or got a commercial license from the Sponsor before the end of submission are ineligible. — search result across Devpost rules, e.g. [Agents for Humans](https://agentsforhumans.devpost.com/rules)

### Inferences
- Check each event: "from scratch" vs "meaningfully extended" vs continuity. Commit history timestamps are what judges inspect; make a clean repo created at kickoff and list dependencies/templates in the README.
- Sponsor API usage: no rule found restricting API use itself; sponsor-tech use is usually a prize eligibility requirement (UNVERIFIED, not read in a rules page).

### Gaps
- No verbatim sponsor API terms retrieved (key sharing, rate limits, ToS).

## 3. IP / ownership terms and license grants

### Takeaway
Entrants keep ownership; sponsors take a license. Scope varies from narrow (judging only) to broad perpetual marketing licenses (Microsoft). Warranty of originality plus no third-party IP is universal.

### Cited Findings
- Microsoft AI Dev Days (FETCHED): submission must be "solely owned by you, your Team, your Organization with no other person or entity having any right or interest in it"; grants Microsoft "a worldwide, perpetual, irrevocable, royalty free, fully paid up, non-exclusive right and license" for promotional use. — [OFFICIAL_RULES.md](https://github.com/Azure/AI-Dev-Days-Hackathon/blob/main/OFFICIAL_RULES.md)
- Microsoft AI Agents Hackathon 2025 (SNIPPET-ONLY): Microsoft does not claim ownership, but receives an irrevocable, royalty-free, worldwide license to use the entry in any media for commercial or non-commercial purposes including marketing. — [rules](https://microsoft.github.io/AI_Agents_Hackathon/rules/)
- Microsoft Agents League 2026 (FETCHED): warrants not copied without permission and no violation of others' IP; may use Microsoft trademarks under limited license for submission only. — [OFFICIAL RULES.md](https://github.com/microsoft/Agents-League-AISF-Regulations/blob/main/OFFICIAL%20RULES.md)
- Google AI Hackathon (Devpost) (SNIPPET-ONLY): entrants warrant ownership of IP in the Code; grant Google a perpetual, irrevocable, worldwide, royalty-free, non-exclusive license to use, reproduce, perform, display and create derivatives for testing, evaluation, promotion. — [rules](https://googleai.devpost.com/rules)
- Gemini 3 Hackathon (SNIPPET-ONLY): entrant retains ownership of IP including moral rights; submissions remain entrant IP. — [rules](https://gemini3.devpost.com/rules)
- Build with Gemini XPRIZE (SNIPPET-ONLY): submissions must be original, solely owned by Entrant; sponsor gets non-exclusive license for judging. — [rules](https://www.geminixprize.com/rules)
- AWS Code with Kiro / AWS AI Agent Global Hackathon (SNIPPET-ONLY): submissions remain entrants' IP; Sponsor gets fully paid, non-exclusive license for judging. — [Kiro](https://kiro.devpost.com/rules), [AWS Agent](https://aws-agent-hackathon.devpost.com/rules)
- MLH standard rules (FETCHED): no IP ownership or sponsor API terms stated. — [MLH rules](https://github.com/MLH/mlh-policies/blob/main/standard-hackathon-rules.md)

### Inferences
- Solely-owned warranty can conflict with GPL/copyleft dependencies, employer-owned code, or AI output with uncertain ownership; check team members' employment/university IP policies before entering.
- Broad perpetual marketing licenses mean anything in the repo/video (including pre-existing code) may be reused by the sponsor.

### Gaps
- No rule found addressing ownership of AI-generated code specifically (UNVERIFIED either way).
- University hackathon IP terms not retrieved.

## 4. Named events with URL and rule (summary table)
1. MLH standard rules (FETCHED) https://github.com/MLH/mlh-policies/blob/main/standard-hackathon-rules.md : AI allowed, be transparent; no reuse of own code; libraries/open source OK.
2. ETHGlobal (SNIPPET-ONLY) https://ethglobal.com/rules : AI allowed with file-level documentation; From Scratch vs Continuity tracks; undisclosed pre-existing work leads to DQ/ban.
3. Microsoft AI Dev Days 2026 (FETCHED) https://github.com/Azure/AI-Dev-Days-Hackathon/blob/main/OFFICIAL_RULES.md : new project only, open-source with license compliance, perpetual license to Microsoft.
4. Microsoft Agents League 2026 (FETCHED) https://github.com/microsoft/Agents-League-AISF-Regulations/blob/main/OFFICIAL%20RULES.md : original work, no AI restriction, entry period May 19-Jun 14, 2026.
5. OpenAI Build Week (SNIPPET-ONLY) https://openai.devpost.com/rules : pre-existing allowed if meaningfully extended with the sponsor tool and documented.
6. Agents for Humans / AWS Strands (SNIPPET-ONLY) https://agentsforhumans.devpost.com/rules : AI assistants and templates OK; disclose other pre-existing code.
7. Google AI Hackathon (SNIPPET-ONLY) https://googleai.devpost.com/rules : perpetual non-exclusive license to Google.
8. USAII Global AI Hackathon 2026 (SNIPPET-ONLY) https://usaii-global-ai-hackathon-2026.devpost.com/rules : AI assistants allowed if disclosed.
