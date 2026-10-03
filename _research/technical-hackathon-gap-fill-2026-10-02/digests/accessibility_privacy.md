# Accessibility, privacy and responsible-AI expectations for hackathon projects

Method note: sandbox blocked w3.org, webaim.org, hack4sdg.com, devpost subdomains, developers.openai.com and resolvehackathon.open-conf.gr on fetch. Almost everything below is therefore SNIPPET-ONLY (from WebSearch result summaries, not full-page reads). Rubric text is paraphrased from snippets, not verbatim quotes. Verify before quoting.

## Which hackathons or rubrics explicitly score accessibility, inclusion, ethics or privacy?

### Takeaway
Several 2025-2026 hackathons score these explicitly, usually as a 5-30% category; MLH's standard four criteria (Technology, Design, Completion, Learning) do not. Teams should check the specific event rubric, and cheap evidence (a11y scan screenshot, short ethics section) covers most of it.

### Cited Findings
- SNIPPET-ONLY: Open Hackathon (Resolve 2025) has a dedicated "Ethics, Security & Inclusion" category: whether solutions adhere to ethics, transparency, data privacy and accessibility and empower diverse users; "Excellent" needs strong ethical framework, robust security, full inclusion and privacy-conscious design — [Judging 2025](https://resolvehackathon.open-conf.gr/judging-2025/)
- SNIPPET-ONLY: Hack4SDG 2025: sustainability and ethics = 20% of judging; judges look at bias, privacy, accessibility and unintended consequences — [Judging Criteria](https://www.hack4sdg.com/2025-event/judging-criteria/)
- SNIPPET-ONLY: Civic Hacks 2026 (Devpost) scores "Inclusion & Accessibility" (reaching underserved communities, remaining barriers); best overall hack needs top social impact and technical scores plus a minimum ethics score — [Civic Hacks 2026](https://civic-hacks-2026.devpost.com/)
- SNIPPET-ONLY: GenAI Hackathon by Impetus & AWS: "Responsible AI" = 5% of judging — [Devpost](https://impetusawsgenaihackathon.devpost.com/)
- SNIPPET-ONLY: 2025 Code For Change AI Hackathon: "responsible AI and ethical tech principles" within Social Good & Relevance (30% of score) — [Devpost](https://code-in-vision.devpost.com/)
- SNIPPET-ONLY: SANS AI Cybersecurity Hackathon: "Ethical & Responsible AI (15%)" incl. "Data Handling & Bias Mitigation" (how user data is collected, processed, stored) — [Rules](https://ai-cybersecurity-hackathon.devpost.com/rules)
- SNIPPET-ONLY: Bellingcat Hackathon rubric (2023, older than preferred range) scores accessibility incl. usability for people who don't know "git clone" — [PDF](https://www.bellingcat.com/app/uploads/2023/03/Bellingcat-Hackathon-Grading-Mar23.pdf)
- SNIPPET-ONLY: MLH judging = Technology, Design, Completion, Learning, weighted equally; no explicit ethics/accessibility project criterion found in MLH docs, but MLH Community Values require events open/accessible to all (disability, etc.) — [MLH guide](https://guide.mlh.com/general-information/judging-and-submissions/rules-for-your-hackathon), [MLH Community Values](https://mlh.io/community-values)
- SNIPPET-ONLY: Qloo LLM Hackathon rules allow post-deadline modification of submissions that disclose personally identifiable information — [Rules](https://qloo-hackathon.devpost.com/rules)
- Devpost's general advice on judging criteria — [Devpost blog](https://info.devpost.com/blog/understanding-hackathon-submission-and-judging-criteria) (not read; title only)

### Inferences
- Weight ranges (5% to 30%) suggest ethics/accessibility rarely decides a win but can break ties or gate prizes (Civic Hacks minimum ethics score).
- Design score under MLH can plausibly absorb accessibility evidence (inference).

### Gaps
- No verbatim rubric text obtained (fetch blocked). Devpost's own rubric templates and Google/Microsoft/Hugging Face event rubrics not checked.

## Minimal accessibility checklist verifiable with free tools

### Takeaway
Target WCAG 2.2 AA essentials: contrast, keyboard operability, visible focus, alt text, labels, captions, 24px targets. Run axe/Lighthouse/WAVE, then do 5 manual checks, since automation covers only part.

### Cited Findings
- SNIPPET-ONLY: WCAG 2.2 SC 2.5.8 Target Size (Minimum), AA: pointer targets at least 24x24 CSS px, or spacing so a 24px circle around the target contains no other target — [Deque WCAG 2.2](https://dequeuniversity.com/resources/wcag-2.2/) and other summaries from search
- SNIPPET-ONLY: 2.4.11 Focus Not Obscured (Minimum), AA: focused element not entirely hidden (e.g. by sticky headers/cookie banners); 2.5.7 Dragging Movements, AA: provide single-pointer alternative to drag; 3.3.8 Accessible Authentication (Minimum), AA: no cognitive function test (e.g. memorising/transcribing) unless alternative — [TetraLogical](https://tetralogical.com/blog/2023/10/05/whats-new-wcag-2.2/), [Vispero](https://vispero.com/resources/new-success-criteria-in-wcag22/)
- Lighthouse a11y score = weighted average of axe-core audits (pass/fail, weighted by axe user impact); manual audits listed separately do not affect score — [Lighthouse scoring](https://developer.chrome.com/docs/lighthouse/accessibility/scoring) (snippet)
- SNIPPET-ONLY: Lighthouse/axe catch only a fraction of issues: Deque claims ~57% of issues by volume (vendor claim, [Deque axe](https://www.deque.com/axe/devtools/extension/edge/)); other sources say Lighthouse 30-50% ([BOIA](https://www.boia.org/blog/googles-lighthouse-accessibility-tests-are-helpful-but-not-perfect)); a score of 100 is not compliance ([Curb Cut](https://www.curbcutaccessibility.com/blog/lighthouse-accessibility-score-100/)). Percentages are secondary/vendor, treat as UNVERIFIED.
- Older WCAG criteria (UNVERIFIED this session, from my background knowledge, not fetched): 1.4.3 text contrast 4.5:1 (3:1 large text), 1.4.11 non-text contrast 3:1, 1.1.1 alt text, 2.1.1 keyboard, 2.4.7 focus visible, 1.3.1/3.3.2/4.1.2 labels and names, 1.2.2 captions for prerecorded video, 1.4.4 resize to 200%, 2.3.1 no more than 3 flashes/s. Confirm at w3.org/WAI/WCAG22/quickref.

### Inferences (suggested 15-minute check)
1. Run axe DevTools + Lighthouse + WAVE on each demo screen; fix critical/serious.
2. Tab through the whole demo flow with no mouse: everything reachable, focus visible, no trap.
3. Check contrast of text and UI controls (axe covers text); no info by colour alone.
4. Alt text on meaningful images, empty alt on decorative; form inputs labelled; one h1 and logical headings; `lang` set.
5. Captions/transcript on demo video; zoom to 200%, test at phone width; respect prefers-reduced-motion.
6. Disclose known gaps in README ("Accessibility statement: tested X, not tested Y").

### Gaps
- w3.org quickref and WebAIM checklist unreachable; exact SC wording unverified. WAVE-specific docs not retrieved.

## Privacy and data-handling practices for demos

### Takeaway
Use synthetic or explicitly consented data, keep secrets out of the repo, minimise collection, and read the LLM provider's data terms. GDPR minimum basics: lawful basis, data minimisation, retention limits.

### Cited Findings
- GDPR Art. 5(1)(c) data minimisation: personal data "adequate, relevant and limited to what is necessary"; retention limited under Art. 5(1)(e); Art. 6 requires a lawful basis (consent, contract, etc.) — [GDPR Art. 5](https://gdpr-text.com/read/article-5/), [Strac summary](https://www.strac.io/blog/gdpr-data-minimization)
- SNIPPET-ONLY: OpenAI API: data not used for training by default since March 2023 unless opted in; processed in US, DPA available (third-party summaries, treat cautiously) — [OpenAI data controls](https://developers.openai.com/api/docs/guides/your-data), [aipolicydesk](https://www.aipolicydesk.com/blog/privacy-first-ai-api-no-training-gdpr-ccpa-2026). Consumer chat apps differ from API terms (UNVERIFIED here; check each provider).
- SNIPPET-ONLY: Qloo hackathon rules allow modification of submissions disclosing PII — [Rules](https://qloo-hackathon.devpost.com/rules)
- NIST AI 600-1 lists "Data Privacy" and "Information Security" among 12 GenAI risks — [witness.ai summary](https://witness.ai/blog/nist-ai-600-1-generative-ai-profile/)

### Inferences (minimal standard)
- Synthetic/public-domain data or documented consent; no real PII, health, student or children's data in repo, demo video or prompts.
- Secrets: env vars, `.env` in .gitignore, run secret scanning (GitHub secret scanning) before making repo public; rotate any leaked key.
- No third-party analytics/cookies unless needed; if used, say so. Don't send user inputs to an LLM provider without telling users; state provider and retention.
- Delete test data after judging; document data sources and licenses.

### Gaps
- No primary-source fetch of Anthropic/Google/OpenAI retention terms; no hackathon-specific privacy rule beyond Qloo snippet; cookie-law (ePrivacy) not researched.

## Responsible-AI items judges raise, and one-page template

### Takeaway
Judges mention bias, privacy, hallucination/confabulation, human oversight and transparency. A one-page "AI Use Card" adapted from Model Cards plus NIST GenAI risks covers them.

### Cited Findings
- NIST AI 600-1 (final July 26 2024) is a voluntary GenAI profile of AI RMF 1.0 organised by GOVERN/MAP/MEASURE/MANAGE; 12 risks: CBRN, Confabulation, Dangerous/Violent/Hateful Content, Data Privacy, Environmental Impacts, Harmful Bias and Homogenization, Human-AI Configuration, Information Integrity, Information Security, Intellectual Property, Obscene/Degrading/Abusive Content, Value Chain — [witness.ai](https://witness.ai/blog/nist-ai-600-1-generative-ai-profile/); [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework)
- Model Cards (Mitchell et al.) have nine sections: model details, intended use, factors, metrics, evaluation data, training data, quantitative analyses, ethical considerations, caveats and recommendations — [arXiv 1810.03993](https://arxiv.org/pdf/1810.03993)
- Rubrics naming bias/data handling: SANS (15%), Hack4SDG (20% ethics incl. bias, unintended consequences), Impetus/AWS (5% Responsible AI) — see first section.

### Inferences: one-page template (my synthesis)
1. Purpose and intended users; out-of-scope uses.
2. Models/APIs used (name, version), what data goes to them, provider retention.
3. Data: source, consent/synthetic, no PII statement.
4. Known failure modes: hallucination rate or spot-check results (e.g. "checked 20 outputs, N wrong"), disclosure label on AI output.
5. Bias check: tested across 2-3 relevant groups/inputs; results and limits.
6. Human-in-the-loop: where a person reviews/overrides; no automated high-stakes decisions.
7. Safety: abuse cases, content filters, rate limits.
8. Accessibility statement and known gaps.
9. Open risks and what you'd do next.

### Gaps
- No judge-interview or primary rubric text on hallucination disclosure found; hackathon-specific model-card requirements not found. Template is not an established standard.
