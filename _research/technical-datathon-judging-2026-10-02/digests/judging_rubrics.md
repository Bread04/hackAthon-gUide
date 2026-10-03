# How real datathons and data-science hackathons are judged

METHOD CAVEAT: The sandbox egress proxy blocked WebFetch for nearly every event site (widsworldwide.org, citadel.com, devpost, kaggle, rit.edu, medium, etc.). Only github.com fetched. Everything below from official sites is therefore taken from WebSearch result summaries (search-engine extracts of the cited URLs), not read in full. Numeric weights were NOT found for most events; they are marked UNVERIFIED rather than guessed. Report-writer should treat this as a partial evidence base.

## What are the exact criteria and weights used by at least 6 named real events?

### Takeaway
Criteria are published for several events, but explicit percentage weights are rare. Only the RIT DataFest 2026 rubric (1-5 scales, 25 points) gives numeric scoring. Most events list unweighted criteria; WiDS main track is a pure leaderboard metric.

### Cited Findings
- Citadel/Correlation One Data Open (finals): judges evaluate "technical rigor, investigative depth, creativity, and real-world impact"; finalists give written reports plus live presentations. No weights found (UNVERIFIED). Correlation One 2021 and 2020 championship posts, years 2020-2021 (older) — [C1 2021](https://www.correlation-one.com/blog/the-data-open-championship-2021), [C1 2020](https://www.correlation-one.com/blog/imperial-college-hec-paris-team-wins-100-000-at-2020-data-open-championship)
- Citadel Data Open (press coverage, old, ~2018): a panel of ten judges selected by Citadel and Correlation One; reported to examine code and questions students; assessment of creativity, rigour, technical ability, team collaboration — [Hedge Fund Journal](https://thehedgefundjournal.com/citadel-s-data-open-showcases-innovative-recruitment-approach/) (secondary, old)
- WiDS Datathon 2025 (Kaggle track): metric is F1 score (ADHD diagnosis from fMRI); prizes go to top 5 on the global Kaggle leaderboard and top undergraduate teams, all determined by leaderboard rank; teams up to 4, at least half women-identifying — [WiDS datathon page](https://www.widsworldwide.org/learn/datathon/) (search extract; 2026 deadline text appeared in the same extract, so exact year mapping UNVERIFIED)
- WiDS Datathon Excellence in Research Award: papers judged on real-world impact potential, scientific rigor, clarity of communication; no weights found — [WiDS datathon archive](https://www.widsworldwide.org/topics/datathon/) (search extract)
- WiDS Datathon++ University Edition: rubric not retrievable; UNVERIFIED — [page](https://www.widsworldwide.org/learn/datathon/datathon-university-edition/)
- TAMU Datathon (2019 and 2022 Devpost, older): seven criteria, no weights found: Purpose (clear understanding of the problem), Framework (mapped task to a data-science problem), Data Use (used data, acquired additional data), Models & Analytics, Validation (assessed quality of solutions and models), Impact, Presentation (effectiveness, engagement, team performance) — [TAMU 2022](https://tamudatathon2022.devpost.com/), [TAMU 2019](https://tamudatathon2019.devpost.com/)
- ASA DataFest (general): awards Best Insight, Best Visualization, Best Use of External Data; judged on insight, visualization, external data, communication; teams get 5 minutes and about 3 slides (other sites say 2 slides, max 10 min); feedback per team: a reason selected, one outstanding comment, one area to improve — [UD DataFest 2025 / search extract](https://dsi.udel.edu/ud-datafest-2025/), [ASA DataFest in a Box](https://ww2.amstat.org/education/datafest/datafestinabox.cfm), [UCLA DataFest](https://newsroom.ucla.edu/stories/classroom-community-hundreds-compete-15th-datafest-hackathon)
- RIT DataFest 2026 (only numeric rubric found): Best Insight = 5 criteria each 1-5 (Significance of Insights, Data-Driven Rigor, Novelty, Actionability, Methodology) = 25 points total; Best Visualization = 5 criteria on 1-5 Likert (Clarity of Message, Aesthetic Design, Appropriateness of Visualization Type, Creativity/Innovation, Technical Execution) — [RIT DataFest](https://www.rit.edu/science/datafest) (search extract; equal weighting by construction)
- NUS Datathon 2024 (Singlife track): judged by NUS Statistics and Data Science Society, a data scientist and a senior ML engineer from Singlife; three evaluation metrics, F1 emphasised on an imbalanced dataset; second-place team's deliverables were an EDA notebook, main notebook, code, methodology docs; it stressed class-imbalance handling, Optuna tuning (~10% F1 gain) and LIME interpretability for business stakeholders — [GitHub writeup](https://github.com/reibena/NUS-Datathon-2024) (fetched; team's own account, not official rubric). Official Devpost criteria not retrievable: UNVERIFIED — [Cat B](https://nus-datathon-2024-cat-b.devpost.com/)
- iNTUition (NTU general hackathon, not data-specific): Technical Difficulty, Creativity, Polish, General Impressiveness; no weights — [iNTUition v5](https://intuitionv5.devpost.com/) (search extract)
- Claremont Graduate Univ. Ethical AI Hackathon: equal 20% weights shown for Socially Responsible Impact, Ease of Use/Transparency/Explainability, Innovative Technology and Intelligent Adaptation (other criteria not captured; extract showed only these three) — [CGU rubric](https://research.cgu.edu/hackathon/home/judging-rubric/)
- Generic (non-event) templates with weights, for contrast: Technical 25 / Innovation 20 / Impact 20 / Design 15 / Completeness 10 / Presentation 10; another: Problem 25 / Working solution 30 / Engineering 20 / Communication 15 / Viability 10 — [DEV Community template](https://dev.to/pranjulrathour/a-hackathon-judging-rubric-organisers-can-copy-13ig) (author's suggestion, NOT a real event)
- Berkeley I School 2012 judging form uses Technology, Design, Business, Presentation, Inspiration — [form](https://blogs.ischool.berkeley.edu/idh2012/files/2012/10/Judging-Form.pdf) (OLD, 2012)

### Inferences
- Real datathon rubrics converge on: problem understanding, method/validation rigor, insight/impact, communication. Weights are mostly undisclosed or equal.
- Leaderboard events (WiDS, Kaggle-hosted) are decided entirely by metric; narrative-judged events (DataFest, TAMU, Data Open finals) are decided by qualitative criteria.

### Gaps
- No weights for Citadel Data Open, WiDS Univ. Edition, TAMU, NUS, UCLA/Berkeley/Georgia Tech datathons (pages blocked or none published).
- Singapore NTU datathon (data-specific) and Georgia Tech Hacklytics current rubrics not found.

## What do judges say they reward and penalize?

### Takeaway
Judges consistently reward understanding the business problem and a clear story supported by sound technique; leakage and bending data to fit a narrative are called out as failures. Evidence is mostly from secondary pieces.

### Cited Findings
- A judge takeaway: really understand the business problem and tell the story along with good techniques; many students jump to a solution without first understanding the problem — [Michigan Daily on Ross Business+Tech datathon](https://www.michigandaily.com/news/campus-life/ross-businesstech-hosts-annual-datathon/) (search extract)
- Presentations must mix technical showcase with a simple solution a layperson understands; judged on theme relevance — same search extract set (UNVERIFIED which source; see [U-M Ross datathon](https://businesstech.bus.umich.edu/datathon/))
- Leakage example from WiDS participant spotlight: distance-based features acted as proxies for future information, creating leakage; advice not to compromise data ethics and to update the message if data contradicts the idea — [WiDS Datathon Spotlight](https://www.widsworldwide.org/get-inspired/blog/datathon-spotlight-applying-judicial-scrutiny-to-code-with-valentina-torres-da-silva/) (participant voice, not judge)
- TAMU rubric explicitly scores Validation and Framework, i.e., penalizes unvalidated models or unmapped problems — [TAMU 2022](https://tamudatathon2022.devpost.com/)
- RIT rubric scores "Data-Driven Rigor" and "Actionability" — [RIT](https://www.rit.edu/science/datafest)
- NUS winners stress interpretability (LIME) for stakeholders — [GitHub](https://github.com/reibena/NUS-Datathon-2024)
- Medical-ML datathon paper on raising awareness of bias (ethics as a datathon theme) — [PMC12250157](https://pmc.ncbi.nlm.nih.gov/articles/PMC12250157) (not read in full)
- Eugene Yan "How to Win a Data Hackathon (Hacklytics 2021)" and a UCLA DataFest tips post exist but could not be read — [Eugene Yan](https://eugeneyan.com/writing/how-to-win-data-hackathon/) (OLD, 2021), [Medium DataFest tips](https://medium.com/@rikesh.data/10-steps-to-winning-asa-datafest-b782c886aa0d)

### Inferences
- "No interactivity" penalty: NOT found in any source; do not assert it. Interactivity is not a stated criterion in any rubric located.
- Leakage is penalized implicitly via validation/rigor criteria, rarely named in rubrics.

### Gaps
- No direct judge interviews found with explicit penalty lists; ethics/bias criteria appear only in CGU Ethical AI rubric.

## How do submission formats differ?

### Takeaway
Formats range from pure Kaggle leaderboard (WiDS) to 2-3 slide pitches (DataFest) to written report plus live presentation (Data Open finals) to notebooks plus code (NUS).

### Cited Findings
- WiDS: Kaggle leaderboard submission (F1) — [WiDS](https://www.widsworldwide.org/learn/datathon/)
- DataFest: 2-3 slides, 5-10 minutes — [UCLA](https://newsroom.ucla.edu/stories/classroom-community-hundreds-compete-15th-datafest-hackathon), [RIT](https://www.rit.edu/science/datafest)
- Data Open: written report and live presentation to judges — [C1 2021](https://www.correlation-one.com/blog/the-data-open-championship-2021)
- NUS Datathon: notebooks (EDA and main), code, methodology — [GitHub](https://github.com/reibena/NUS-Datathon-2024)
- Kaggle competition winners must deliver reproducible code and documentation to sponsor — [Kaggle rules example](https://www.kaggle.com/competitions/data-science-and-ai/rules)

### Inferences
- Live demo/interactive app is more typical of general hackathons (iNTUition: Polish, Technical Difficulty) than datathons.

### Gaps
- Per-event format details for UCLA/Berkeley/Georgia Tech datathons not retrieved.

## What changed in 2025-2026 (AI-assisted work rules, disclosure, LLM policy)?

### Takeaway
2025 hackathon rules generally permit AI tools but require disclosure, with disqualification risk for non-disclosure; I found no datathon-specific (WiDS, Citadel, DataFest) AI policy text.

### Cited Findings
- Hackathon rules in the search extract: all AI tools (LLMs, code generators, no-code platforms) must be disclosed; failure to disclose is grounds for disqualification; best projects should leverage AI but be human-created — extract covering [HackTX 2025 rules](https://hacktx2025.devpost.com/rules), [AWS GenAI Hackathon 2025](https://startups.aws.com/events/aws-generative-ai-hackathon-challenge-2025), [NeSy 2025](https://hackathon.optimas.ai/rules) (quotes not verified per page; which event says which is UNVERIFIED)
- Kaggle discourages using AI alone to judge hackathon submissions, citing adversarial-input vulnerability and bias — [Kaggle docs](https://www.kaggle.com/docs/competitions-setup) (search extract)
- March 2026 Kaggle Playground churn competition reportedly won by LLM agents (600K lines of code, 850 experiments) — [Blockchain.News](https://blockchain.news/news/llm-agents-win-kaggle-competition) (secondary), related [NVIDIA blog](https://developer.nvidia.com/blog/winning-a-kaggle-competition-with-generative-ai-assisted-coding/)
- WiDS 2026 global datathon winners announced — [WiDS blog](https://www.widsworldwide.org/get-inspired/blog/2026-wids-worldwide-global-datathon-winners-announced/) (content not read)

### Inferences
- With AI-built solutions winning leaderboards, narrative-judged events likely shift weight toward problem framing, validation and explanation; this is inference, no source states it.

### Gaps
- No LLM-use policy found for WiDS, Citadel Data Open, DataFest, NUS, TAMU 2025 (TAMU 25: https://td25.devpost.com/ not read). All UNVERIFIED.
