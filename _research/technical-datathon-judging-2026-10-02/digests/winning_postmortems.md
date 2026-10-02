# Winning datathon / data-science hackathon teams: what they did, why others lost

Method caveat: many primary pages (eugeneyan.com, medium.com, dev.to, odsr.illinois.edu, widsworldwide.org, kaggle.com, wharton, jetbrains blog, mbs.edu) were blocked by the egress proxy, so those items rest on web-search snippets only and are flagged "snippet-only". Fully fetched pages: Rice 2024 repo, IndoML 2024 repo, UIUC 2024 repo, WiDS 2025 (ghazalehran) repo.

## What common workflow, time allocation, team roles and tooling do winning teams report?

### Takeaway
Winners favor ready-made clean data, pretrained models or simple methods, and a deployed simple demo, over ambitious scraping or novel architectures. No source I could reach gives hard hour-by-hour time splits or formal role assignments; those are gaps.

### Cited Findings
- Hacklytics 2021 (Georgia Tech 36h datathon; judge Eugene Yan; older than the 2023-26 window): top teams used available datasets/APIs (scraping and HTML parsing is a time sink), ML libraries and pretrained models (a fake-news team fine-tuned pretrained BERT for 3 epochs), simple UIs, and deployed prototypes judges could try. (snippet-only) — [Eugene Yan](https://eugeneyan.com/writing/how-to-win-data-hackathon/)
- Rice Datathon 2024 Chevron track 1st place: pipeline of cleaning (unknown to NaN, drop low-utility columns), KNN imputation (k=10), ordinal encoding, engineered spatial feature (KNN on neighboring-well performance), AdaBoost over HistGradientBoosting; test RMSE 78.698 averaged over five train/test splits vs train RMSE 21.264; feature importance showed the engineered spatial metric "by far the most important". — [ajholzbach/Datathon_2024](https://github.com/ajholzbach/Datathon_2024)
- IndoML 2024 Datathon (NielsenIQ), Team PRMAS winners: ByT5-small, Llama-flanT5 and MultiModal-ByT5 combined by hard voting with hierarchy compliance, plus label-correction preprocessing and deliberate data splitting. — [iamansinha/Datathon-IndoML-2024](https://github.com/iamansinha/Datathon-IndoML-2024)
- Illinois Statistics Datathon 2026 (~600 students, 200+ teams; Synchrony/Databricks/AWS sponsors): winning Team 225 (4 members: two sophomores, one MS CS, one senior Statistics) used no ML or AI tools (task asked for none) and beat teams with ~10x more submissions; formula Interval CV = Daily CV x Shape x Bias. Same search summary says they ran an OLS regression on leaderboard data to infer score weights (only in the secondary search summary; not verified on the page). — [Illinois ODSR](https://odsr.illinois.edu/2026-datathon-winners-predict-call-center-activity-with-no-ai/) (snippet-only)
- Women in Data Datathon 2022 winners attribute win to a novel self-built dataset, a project showcasing each member's strength, ML, visuals/storytelling, and extensibility after the event (self-report; 2022). — [Medium](https://medium.com/womenintechnology/how-my-team-won-my-first-hackathon-404627379bd4) (snippet-only)

### Inferences
- Tooling is mainstream (sklearn/XGBoost-style GBMs, pretrained transformers, simple UI); the edge is in choice of features and framing, not exotic models.
- Small teams (3-5) are the norm across the examples (Illinois 4; Wharton datathon 3-5).

### Gaps
- No verified hour-by-hour time allocation or role split (e.g., who does storytelling) from any 2023-26 winner; Eugene Yan page and Hacklytics lessons blog could not be fetched.

## What were the concrete differentiators (framing, validation discipline, domain features, interactive demo, storytelling)?

### Takeaway
Differentiators found: domain-informed engineered features (Rice spatial neighbor feature), reverse-engineering the metric/leaderboard rather than adding model complexity (Illinois), business reframing with a novel metric (Wharton RFM variant), validation across repeated splits, and a live/deployed demo (Hacklytics).

### Cited Findings
- Rice 2024: hypothesis "neighboring well performance drives individual well outcomes" turned into a feature and confirmed by importance ranking; validated across five splits before final fit. — [Rice repo](https://github.com/ajholzbach/Datathon_2024)
- Illinois 2026: asymmetric scoring punished underprediction; winners built a simple decomposition tuned to the metric (see above). — [Illinois ODSR](https://odsr.illinois.edu/2026-datathon-winners-predict-call-center-activity-with-no-ai/) (snippet-only)
- Wharton Customer Analytics datathon (15 teams of 3-5; 1st $1,500, 2nd $500; judges led by faculty director Raghuram Iyengar): winning presentation "Modeling Consumer Retail Preferences for an International Consumer Brand" created an RFM variation and found high-churn customers contributed ~74% of revenue, a business-facing finding. Event date not confirmed (appears older than 2023). — [Knowledge at Wharton](https://knowledge.wharton.upenn.edu/article/datathon-challenge-boost-sales-global-retailer/) (snippet-only)
- Hacklytics 2021: deployed prototypes/simple UI so judges could try them. — [Eugene Yan](https://eugeneyan.com/writing/how-to-win-data-hackathon/) (snippet-only)
- Judge-side writing: judges want a functional, impactful prototype not messy backend; a winning team "came prepared with numbers" and withstood probing from multiple angles; always state clearly what problem is solved. (search-summary of several dev.to/LinkedIn pieces; single-author opinion, not verified) — [Search hits incl. dev.to](https://dev.to/kurbaitaev/what-judges-actually-score-notes-from-a-year-of-hackathon-judging-3p4l)
- Kaggle AES 2.0 (2024) 1st place jumped from 619th public to 1st private, attributed to examining train/test distribution shift, two-stage pseudo-labeling, simple average of 4 variants, thresholds optimized to avoid overfitting (snippet-only) — [Hippocampus Garden](https://hippocampus-garden.com/kaggle_aes2/)

### Inferences
- A feature or framing grounded in the domain mechanism, plus a number that translates to business value, appears to be the repeatable edge.
- Evidence for interactive demo as a differentiator is limited to Hacklytics 2021; unverified for 2023-26.

### Gaps
- No rubric-level judge comments from 2023-26 datathons retrieved; Q&A performance evidence absent.

## What recurring failure modes cost teams?

### Takeaway
Reported failure modes: public-leaderboard overfitting, ignoring train/test shift, over-investing in ideation or tech and under-investing in insight documentation, unclear problem statement, and complexity for its own sake.

### Cited Findings
- Public-LB overfitting: "shake-up" metrics exist because fine-tuning to the public split hurts the private result; community mantra "Trust your CV". — [davidthaler/shakeup](https://github.com/davidthaler/shakeup); [Kaggle Handbook (Medium)](https://medium.com/global-maksimum-data-information-technologies/kaggle-handbook-fundamentals-to-survive-a-kaggle-shake-up-3dec0c085bc8)
- Kaggle CMI Problematic Internet Use 1st-place write-up titled "...Or How I Won the Lottery" (title suggests large luck/shake-up component; contents not read) — [Kaggle](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/writeups/lennart-haupts-first-place-write-up-or-how-i-won-t) (snippet-only)
- Data-science hackathon retrospectives: ideation consumed too much time; little time left for insights/documentation; trade-off between answering everything vs doing a few things very well; people skip stating the problem. (LinkedIn/Medium opinion pieces) — [Search hits](https://www.linkedin.com/pulse/meta-lessons-from-failed-data-science-hackathon-abdulmajedraja-rs); [Eugene Yan on evaluating ideas](https://eugeneyan.com/writing/evaluating-ideas-at-a-hackathon/)
- Illinois 2026: teams with ~10x more submissions lost to a simple metric-aware approach. — [Illinois ODSR](https://odsr.illinois.edu/2026-datathon-winners-predict-call-center-activity-with-no-ai/) (snippet-only)
- WiDS 2025 repo (not a winner; no score disclosed) shows unfinished notebooks "(in progress)/(planned)" with deep-learning extensions planned, an example of scope outrunning delivery (my reading, not the author's statement) — [ghazalehran/WiDS-Datathon-2025](https://github.com/ghazalehran/WiDS-Datathon-2025)

### Inferences
- Leakage, broken demos and weak Q&A are widely asserted in advice pieces but I found no sourced concrete 2023-26 case; do not present them as evidenced.

### Gaps
- No named team post-mortem with confirmed leakage or demo failure found; most post-mortem pages blocked.

## Named examples (with lessons)

### Takeaway
Eleven examples; verification level noted.

### Cited Findings
1. Rice Datathon 2024, Chevron track, 1st (Holzbach, Burudgunte, Uzowihe, Stegall) — [repo](https://github.com/ajholzbach/Datathon_2024). Lessons: encode a domain hypothesis as a feature; report multi-split validation. (fetched)
2. IndoML 2024 Datathon (NielsenIQ), Team PRMAS, winner — [repo](https://github.com/iamansinha/Datathon-IndoML-2024). Lessons: ensemble heterogeneous models; enforce label hierarchy consistency in post-processing. (fetched)
3. UIUC Datathon 2024 (Synchrony IVR), 4th of 345, F1 0.90 — [repo](https://github.com/shengzhuyin/uiuc-datathon-24). Lessons: EDA plus feature importance for interpretability; tested BERT before choosing GPT-3.5 for sentiment; a podium-adjacent finish with a wide tool stack. (fetched)
4. Illinois Statistics Datathon 2026, Team 225, 1st — [article](https://odsr.illinois.edu/2026-datathon-winners-predict-call-center-activity-with-no-ai/). Lessons: understand the scoring metric; simple decomposable model beat volume of submissions. (snippet-only)
5. WiDS Global Datathon 2025, Team Jenna, 1st (fMRI ADHD/sex prediction) — [WiDS](https://www.widsworldwide.org/get-inspired/blog/wids-global-datathon-2025-winners/). Lesson: team credits understanding of multidimensional connectivity data (domain data knowledge) for Sex_F gains; another team (TUIASI_FML_AISynergy) cited stacking for less overfitting. (snippet-only)
6. Kaggle Automated Essay Scoring 2.0 (2024), 1st place, 619th public to 1st private — [Hippocampus Garden](https://hippocampus-garden.com/kaggle_aes2/). Lessons: analyze distribution shift; simple averaging and non-overfit thresholds. (snippet-only)
7. Kaggle CMI Problematic Internet Use, Lennart Haupt, 1st — [write-up](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/writeups/lennart-haupts-first-place-write-up-or-how-i-won-t). Lesson: private-LB volatility can be large; do not trust public rank. (title only; UNVERIFIED details)
8. Wharton Customer Analytics datathon winners (RFM variant, 74% revenue from high-churn customers) — [article](https://knowledge.wharton.upenn.edu/article/datathon-challenge-boost-sales-global-retailer/). Lesson: lead with a business-quantified insight. (snippet-only; date unverified)
9. Women in Data Datathon 2022 winners — [Medium](https://medium.com/womenintechnology/how-my-team-won-my-first-hackathon-404627379bd4). Lesson: original dataset plus storytelling plus play to member strengths. (snippet-only; 2022)
10. Hacklytics 2021 winners as observed by Eugene Yan — [post](https://eugeneyan.com/writing/how-to-win-data-hackathon/). Lesson: clean public data, pretrained models, deployed simple UI. (snippet-only; outside date range)
11. Quantium team, MBS Analytics Datathon (predict wickets) — [MBS](https://mbs.edu/news/howzat-team-quantium-wins-analytics-datathon-with-plan-to-predict-wickets). Only headline seen; details UNVERIFIED.

### Gaps
- Only examples 1-3 were read in full; others need re-verification when egress allows. Hackathon DevPost pages and interviews with runners-up not reached.
