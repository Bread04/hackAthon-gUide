# Datathon logistics: datasets/tracks, rules (AI, external data, DUA), prep checklists, ethics expectations

Method note: WebFetch was egress-blocked for widsworldwide.org, turing.ac.uk, cdc.cs.unc.edu, datasciencesociety.net, so findings come from WebSearch result summaries only (no full-page reads). Exact official rule text (AI policy, external data) was NOT found for any named datathon; treat those as gaps.

## 1. Datasets and challenge tracks (6+ named examples, size/format)

### Takeaway
Recent datathons use sponsor-donated, domain-specific tabular (and sometimes imaging-derived) data, most often healthcare, with some civic, space/defense, and multi-track university events. Sizes are usually modest (thousands to tens of thousands of rows), fitting laptop/Kaggle-notebook work.

### Cited Findings
- WiDS Datathon 2025 (health; brain): predict ADHD diagnosis and sex from fMRI functional connectomes plus socio-demographic/emotion/parenting data; 3 datasets (categorical, quantitative, connectomes) on 1213 participants; connectomes are 200x200 matrices. Data from Healthy Brain Network (Child Mind Institute) and Reproducible Brain Charts; sponsor Ann S. Bowers Women's Brain Health Initiative with Cornell/UCSB. Hosted on Kaggle (widsdatathon2025). — [Gender-Aware ADHD paper](https://doi.org/10.3390/cmsf2025012006); [WiDS 8th datathon blog](https://www.widsworldwide.org/get-inspired/blog/8th-annual-wids-datathon-challenges-unraveling-the-mysteries-of-the-female-brain/); [mary-diana repo](https://github.com/mary-diana/wids-datathon)
- WiDS Datathon 2024 (health equity): Gilead-sponsored; predict whether metastatic breast cancer patients got diagnosis within 90 days of screening; ~39,000 records (train/test), features: demographics, diagnosis codes, treatment, insurance, socio-economic and environmental data; patients diagnosed 2015-2018; open Jan 9 to Mar 1. — [WiDS 2024 GitHub write-up](https://github.com/rebrinehart/WiDS-2024-Datathon); [Lafayette](https://dss.lafayette.edu/wids-datathon-2024/); [MathWorks](https://blogs.mathworks.com/student-lounge/2024/01/01/predicting-timely-diagnosis-of-metastatic-breast-cancer-for-the-wids-datathon-2024/)
- WiDS Datathon++ University Edition 2025: sponsored by WBHI and CMI (same brain-health theme, university track). — [WiDS](https://www.widsworldwide.org/get-inspired/blog/wids-datathon-university-edition-2025-winners/)
- WiDS rule of note: teams must have at least 50% women; datasets are described as well-curated real-world data not readily in the public domain. — [WiDS keyword page](https://www.widsworldwide.org/keywords/datathon/)
- Women in Data 2025 Datathon "Space Aware" (defense/space; with US Space Force): explore open space data on space situational awareness; judged on depth of analysis, practical application, presentation quality, originality/innovation. Data size/format not found. — [Women in Data](https://www.womenindata.org/blog/datathon-2025)
- Carolina Data Challenge 2025 (UNC; multi-track): five tracks (Business, Health Sciences, Social Sciences, Natural Sciences, Pop Culture), each with hand-picked datasets. Sizes not found. — [CDC](https://cdc.cs.unc.edu/)
- MIT Critical Datathon 2023 (clinical): MIMIC-IV-derived dataset for pulse oximetry correction models, released on PhysioNet under credentialed access. (2023, slightly outside preferred window; example of clinical datathon format.) — [PhysioNet](https://physionet.org/content/mit-critical-datathon-2023/1.0.0/)
- Correlation One / Merck-MSD datathons: substance-abuse treatment discharge records across the US plus American Community Survey and state-level treatment center data; another C1 datathon used "smart cities"/urban environment data. Years not confirmed (likely pre-2024). — [Correlation One blog](https://www.correlation-one.com/blog/merck-msd-datathons)
- Qlik/C40 Cities climate resiliency datathon: public datasets, Qlik Sense, teams of max 2; four city-level challenges (adaptation, air quality, resiliency) plus a free choice. Dated 2020 (outside window; climate/civic example only). — [Qlik press release](https://www.qlik.com/us/news/company/press-room/press-releases/qlik-launches-global-datathon-challenge-to-develop-data-driven-solutions-for-climate-resiliency)

### Inferences
- Expect tabular CSV/Parquet-scale data and a Kaggle-style or presentation-style deliverable; large raster/imaging data is usually pre-reduced (e.g., connectome matrices).
- Finance, logistics and sports tracks: no 2024-2026 named, sourced example found (see Gaps).

### Gaps
- No sourced recent finance, logistics, or sports datathon examples; Citadel/Correlation One datathon page (citadel.com/careers/programs-and-events/datathons/) surfaced in search but was not read. UNVERIFIED.
- Exact file sizes for WiDS 2025 (check Kaggle page) not found.

## 2. Rules: external data, pretrained models, AI assistants, IP, privacy/DUA, deadlines, formats

### Takeaway
Only the data-access/DUA side is well sourced (PhysioNet). Official AI-assistant and external-data rules for specific 2024-2026 datathons were not retrievable; teams must read each event's rules and ask organizers.

### Cited Findings
- PhysioNet-hosted clinical datathon data requires credentialed access: CITI training completion and signing the Credentialed Health Data Use Agreement (v1.5.0) before files can be downloaded. — [PhysioNet datathon dataset](https://physionet.org/content/mit-critical-datathon-2023/1.0.0/); [MIMIC getting started](https://mimic.mit.edu/docs/gettingstarted/)
- The credentialed DUA prohibits sharing data with third parties, including sending it through APIs or online services; PhysioNet publishes specific guidance on using MIMIC with LLMs/online services (e.g., GPT) — in practice only privately hosted/approved models. — [PhysioNet: LLM responsible use](https://physionet.org/news/post/llm-responsible-use/); [PhysioNet: GPT responsible use](https://physionet.org/news/post/gpt-responsible-use/)
- MIMIC-IV is described as fully de-identified but still used under the DUA. — [PhysioNet](https://physionet.org/content/mit-critical-datathon-2023/1.0.0/)
- WiDS 2024 ran Jan 9 to Mar 1 on Kaggle (submission = predictions on test set, scored by leaderboard); WiDS 2025 also on Kaggle. — [Lafayette](https://dss.lafayette.edu/wids-datathon-2024/); [GitHub repo](https://github.com/saidataanalytics/wids_2025)
- Generic 2025 trend: organizational AI-use policies are becoming more permissive but expect transparency about how AI was used and output verified; third-party AI tools may retain/train on inputs. — [Morgan Lewis](https://www.morganlewis.com/blogs/sourcingatmorganlewis/2025/12/ai-usage-policies-revisited-structure-trends-and-transparency); [UT Austin](https://security.utexas.edu/ai-tools) (general, not datathon-specific)

### Inferences
- If data is under a DUA, pasting rows into ChatGPT/Claude-type web tools is likely a violation; use code-only prompts with schema/synthetic samples.
- Kaggle-hosted WiDS events typically follow Kaggle competition rule norms (external data usually must be public/reasonably accessible; team size caps; code sharing) — this is from general knowledge, UNVERIFIED for 2025 specifics.

### Gaps
- No datathon's official text on LLM/coding-assistant policy found. UNVERIFIED.
- IP/ownership terms, pretrained-model rules, and submission file formats (beyond Kaggle predictions CSV) not found.

## 3. Pre-event preparation checklists

### Takeaway
Past participants/organizers converge on: pre-research the domain, set up shared repo/environment, prewrite wrangling code, pick roles, and define the problem before modeling.

### Cited Findings
- Teams of 4-6 with mixed skills (analyst, data scientist, UX/front-end, PM/business). — [Sogeti Labs](https://labs.sogeti.com/5-tips-to-participate-and-succeed-in-your-datathons/)
- Research challenge topic in advance; check data providers' GitHub pages before extraction. — [Sogeti Labs](https://labs.sogeti.com/5-tips-to-participate-and-succeed-in-your-datathons/) (via search summary)
- Prepare list of data wrangling/pipeline code; ensure laptop/software ready; create common repo with shared templates. — search summary of [Sogeti Labs](https://labs.sogeti.com/5-tips-to-participate-and-succeed-in-your-datathons/), [Hex](https://hex.tech/blog/the-modern-datathon/), [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2016/09/how-to-prepare-for-your-first-data-science-hackathon-in-less-than-2-weeks/) (attribution to individual items imprecise; 2016 source dated)
- Define the problem with domain experts/affected people, set goal and schedule; review all cases when released; join online community; do not fit data to your solution. — [Datathon participant guidelines, Data Science Society](https://www.datasciencesociety.net/datathon-participants-guidelines/) (search summary only)
- NeurIPS'23 "How to Data in Datathons" (Alan Turing Institute) is the organizer-side guide on dataset selection/prep for datathons; full text not read. — [Turing PDF](https://www.turing.ac.uk/sites/default/files/2024-06/neurips-2023-how-to-data-in-datathons-paper-datasets_and_benchmarks.pdf)

### Inferences
- One-week plan: day 1-2 complete data access/DUA/credentialing (can take days), day 3 environment + repo + baseline template, day 4-5 read rules/past winners, day 6 roles + presentation template, day 7 dry run.

### Gaps
- Details on organizer-recommended checklists from WiDS/Correlation One/MLH not retrieved.

## 4. Ethics, fairness, privacy expectations from judges

### Takeaway
No datathon-specific judging rubric mentioning ethics was found; generic data-ethics principles (privacy/consent, bias, transparency, harm prevention, accountability) apply, and the WiDS framing itself centers equity/sex differences.

### Cited Findings
- Women in Data 2025 rubric lists depth of analysis, practical application, presentation, originality; no explicit ethics criterion in the summary found. — [Women in Data](https://www.womenindata.org/blog/datathon-2025)
- WiDS 2024 theme was explicitly "Equity in Healthcare"; WiDS 2025 required modeling sex differences in ADHD, so subgroup performance reporting is topical. — [WiDS 2024 repo](https://github.com/rebrinehart/WiDS-2024-Datathon); [ADHD paper](https://doi.org/10.3390/cmsf2025012006)
- Generic data-ethics principles: privacy/consent, fairness and bias mitigation, transparency, harm prevention, accountability. — [Snowflake data ethics](https://www.snowflake.com/en/data-governance/policy/data-ethics/); [Improvado](https://improvado.io/blog/ethical-data-management) (generic, not judges' statements)
- Turing/NeurIPS guide advises "don't compromise on data ethics; maintain integrity of data" (via search summary). — [Turing PDF](https://www.turing.ac.uk/sites/default/files/2024-06/neurips-2023-how-to-data-in-datathons-paper-datasets_and_benchmarks.pdf)

### Inferences
- Safe practice: no re-identification attempts, no uploading restricted data to external services, report subgroup metrics and limitations, state AI-tool usage.

### Gaps
- No primary-source judge quotes on PII/de-identification/bias checks. UNVERIFIED.
