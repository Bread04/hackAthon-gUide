# 📜 Datathon Rules, Leaderboards & Dataset Licences

<!-- markdownlint-disable MD013 -->

> 📚 Full research report and raw notes: [`technical-datathon-gap-fill-round-two-2026-10-02`](../../_research/technical-datathon-gap-fill-round-two-2026-10-02/research.md).

> What competition rules usually say about accounts, sharing, external data, leaderboards, winner obligations and dataset licences. Research round two, 2026-10-02. The sandbox blocked many primary sites, so read the labels: **FULL-TEXT** means the page or file was read in full, usually a GitHub copy or source file. **SNIPPET-ONLY** means the claim comes from a search-engine summary, not the page itself, so do not quote it as verbatim. **UNVERIFIED** means it could not be checked at all, or rests on background knowledge. **RE-CHECK** marks a limit, quota or price the vendor can change at any time. **CONFLICT** marks sources that disagree. "Inference (ours)" is the researchers' own synthesis, not a rule or standard.

## Datathon rules: Kaggle's template allows external data by default but bans private sharing outright

### What is sourced

The Kaggle text below comes from two 2026 verbatim copies of official rules pages that participants committed to GitHub: the NVIDIA Nemotron 3 Reasoning Challenge and Playground Series S6. They match almost word for word, which is good evidence that this is the current template. They are still third-party copies, so treat them as reliable but not primary (**FULL-TEXT** of the copies) ([Nemotron rules copy](https://github.com/ahb-sjsu/agi-hpc/blob/HEAD/benchmarks/nemotron-rules.txt); [Playground S6 copy](https://github.com/leechanwoo-kor/kaggle-playground-series-s6e3/blob/HEAD/docs/rules.md)).

**Accounts and teams.** Submitting "through more than one Kaggle account" means disqualification (§3.5a). Each person may join only one team, and the team cap is set per competition: 5 for Nemotron and 3 for Playground S6. A merged team must have "a total Submission count less than or equal to the maximum allowed as of the Team Merger Deadline", where the maximum is the daily cap multiplied by the number of days the competition has run (§3.5c) ([Nemotron](https://github.com/ahb-sjsu/agi-hpc/blob/HEAD/benchmarks/nemotron-rules.txt)).

**Submissions and leaderboards.** Both 2026 copies allow **5 submissions per day and 2 final selections**. If you do not select, Kaggle picks the final submissions automatically. "The potential winner(s) are determined solely by the leaderboard ranking on the Private Leaderboard". Participants are not told which test rows are public and which are private, and ties go to the submission entered first ([Nemotron §3.7, §3.18](https://github.com/ahb-sjsu/agi-hpc/blob/HEAD/benchmarks/nemotron-rules.txt)).

**CONFLICT: caps and selection rules vary.** One team's notes record a live cap of 1 per day in a 2026 agent competition (SNIPPET-ONLY) ([dalloliogm notes](https://github.com/dalloliogm/kaggle_competitions/blob/HEAD/competitions/autonomous-agent-prediction-beta/AGENTS.md)), and WiDS 2020 allowed 10 per day (SNIPPET-ONLY). Kaggle's 2026 "kaggriculture" simulation competition overrode "select up to two" with "only the latest 2 submissions are tracked", and ranked teams with a post-deadline Bradley-Terry tournament ([participant notes quoting official pages](https://github.com/romansvet/kaggriculture/blob/HEAD/docs/strategy/2026-09-26-rules1.md)).

**Code, data and labelling.**

- Hand-labelling or human prediction of validation or test records is banned outside Hackathon-format competitions (§3.4b).
- "Privately sharing code or data outside of Teams is not permitted" (§3.5d). Public sharing is allowed only on that competition's Kaggle forum or notebooks, and doing so licenses the code under an OSI-approved licence "that in no event limits commercial use" (§3.6b).
- Any open-source code inside your model must also carry a licence that does not limit commercial use (§3.6c).
- AutoML is allowed if you hold an appropriate licence for it (§2.6c).

([Nemotron](https://github.com/ahb-sjsu/agi-hpc/blob/HEAD/benchmarks/nemotron-rules.txt))

**External data and models.** These are "acceptable unless specifically prohibited by the Host", provided they are "reasonably accessible to all" and of "minimal cost". The template's own examples: a small LLM subscription is reasonable, but a proprietary dataset licence costing more than a prize is not (§2.6a/b) ([Nemotron](https://github.com/ahb-sjsu/agi-hpc/blob/HEAD/benchmarks/nemotron-rules.txt); [Playground S6](https://github.com/leechanwoo-kor/kaggle-playground-series-s6e3/blob/HEAD/docs/rules.md)).

**CONFLICT: older default.** WiDS 2020 used a stricter default: "you may not use data other than the Competition Data" unless the competition website says otherwise. The current WiDS rules could not be read, so the WiDS position for 2025/26 is UNVERIFIED (SNIPPET-ONLY, kaggle.com/c/widsdatathon2020/rules). A data-security clause in both copies forbids giving Competition Data to non-participants, even when the data is CC BY 4.0 ([Playground S6](https://github.com/leechanwoo-kor/kaggle-playground-series-s6e3/blob/HEAD/docs/rules.md)).

**Winner obligations.** Winners must deliver training code, inference code and an environment description, plus a reproducible write-up, and must license the code under the competition's winner licence. Observed winner licences:

- CC BY 4.0 for Nemotron and "None" for Playground S6 (both from the FULL-TEXT copies).
- Apache 2.0 and MIT in other competitions (SNIPPET-ONLY).

Data or models with incompatible licences are exempt from the open-source grant. Winners must answer the winner notification within 1 week and return prize documents within 2 weeks or forfeit. Team prize money is split evenly unless the team unanimously agrees otherwise (§2.5, §2.8, §3.8–3.9) ([Nemotron](https://github.com/ahb-sjsu/agi-hpc/blob/HEAD/benchmarks/nemotron-rules.txt)).

**Dataset licences** (licence texts are FULL-TEXT via SPDX; the CC deed pages were blocked):

- **CC BY 4.0** allows any use with attribution ([SPDX CC-BY-4.0](https://github.com/spdx/license-list-data/blob/main/text/CC-BY-4.0.txt)).
- **CC BY-NC** bars use "primarily intended for or directed towards commercial advantage or monetary compensation" ([SPDX CC-BY-NC-4.0](https://github.com/spdx/license-list-data/blob/main/text/CC-BY-NC-4.0.txt)).
- **CC BY-SA** adds share-alike to adapted material ([SPDX CC-BY-SA-4.0](https://github.com/spdx/license-list-data/blob/main/text/CC-BY-SA-4.0.txt)).
- **ODbL** distinguishes two cases. A "Produced Work" (a chart, model output or dashboard) does not create a Derivative Database, but publishing it requires a notice. A publicly used Derivative Database must stay under ODbL or a compatible licence, and you must offer it, or a file of the alterations, in machine-readable form (§4.3–4.6) ([SPDX ODbL-1.0](https://github.com/spdx/license-list-data/blob/main/text/ODbL-1.0.txt)).

**In-person datathons** (all SNIPPET-ONLY; pages blocked):

- **ASA DataFest:** sign an NDA, store the data securely, "delete all data from thumb drives, hard drives, etc." at the end, and do not name the data source in publicity until all DataFest events are complete ([Penn State DataFest FAQ](https://datafest.psu.edu/faqs/); [ASA DataFest in a box](https://ww2.amstat.org/education/datafest/datafestinabox.cfm)).
- **Citadel datathons:** only "publicly-available data sources" plus data Citadel provides, with no material non-public information or former-employer data ([Citadel terms PDF](https://www.citadel.com/wp-content/uploads/2026/03/Citadel-High-School-Terminals-Programs-Certification-and-Terms.pdf)). The Women's and Summer Invitationals are open to undergraduates aged 18+ at US or Canadian universities, graduating between December 2026 and June 2028 ([Citadel datathons](https://www.citadel.com/careers/programs-and-events/datathons/)). The deliverable is reported as about a 15-page report plus code (secondary source).
- **WiDS:** teams of up to 4, with at least half identifying as women ([WiDS](https://www.widsworldwide.org/learn/datathon/)).

**Disqualification precedent.** The best-documented case is PetFinder.my (announced January 2020). The first-place team hid about 3,500 scraped samples with leaked test labels inside an "external" Pixabay dataset, and swapped in the leaked labels for only about 1 in 10 pets to avoid suspicion. It was caught when a later participant got access to the winning code to put it into production. The team was removed from the leaderboard, and the key member, a Grandmaster, was banned permanently (SNIPPET-ONLY) ([TDS/Medium](https://medium.com/data-science/kaggle-1st-place-winner-cheated-10-000-prize-declared-irrecoverable-bb7e1b639365); [Vice](https://www.vice.com/en/article/kaggle-data-science-community-rocked-by-pet-adoption-contest-cheating-scandal/); [The Register](https://www.theregister.com/2020/01/21/ai_kaggle_contest_cheat/)). The rules allow disqualification for "cheating, deception, or other unfair playing practices", along with removal from the leaderboard and loss of points or medals (§3.8d–e) ([Nemotron](https://github.com/ahb-sjsu/agi-hpc/blob/HEAD/benchmarks/nemotron-rules.txt)). **We found no accessible, named case of disqualification specifically for multiple accounts or private sharing.** Any claim that these are "common" is UNVERIFIED.

### Inference (ours)

The competition-specific rules override the general template, so the daily cap, the final-selection mechanism and the external-data default must be read for each competition, not assumed. The merger cap means a late merge between two teams that have both used most of their submissions can be blocked, so merge early. Pick your final submissions yourself; we could not confirm what Kaggle's automatic pick chooses. PetFinder shows that hosts inspect winning code after the competition, so declare every external dataset publicly in the forum. A model under a non-commercial licence may clash with §3.6c unless the host allows it; check the forum for host clarifications. Whether model weights trained on CC BY-NC data count as "Adapted Material" is legally unsettled and was not researched. For DataFest-style NDA events, a portfolio repo should hold code only, never data.

### Ready-to-use checklist

- [ ] Read the competition-specific rules section first; it overrides the template.
- [ ] Record these terms:
  - team cap and merger deadline
  - daily submission cap
  - number of final selections and how they are chosen (selected by you, or "latest N")
  - data-use tier (Competition Use / Non-Commercial / Commercial)
  - winner licence
  - external-data and pretrained-model policy
- [ ] Use one Kaggle account per person, ever, and only one team per person.
- [ ] Share code only publicly on the competition's forum or notebooks, never privately with another team.
- [ ] Do not hand-label or manually predict test rows.
- [ ] Post every external dataset or model in the forum. Confirm it is free or cheap and accessible to all.
- [ ] Keep dependencies under licences that do not restrict commercial use (no GPL or non-commercial code unless the host allows it).
- [ ] Select your final submissions manually before the deadline: one CV-best and one diverse.
- [ ] From day 1, keep a reproducible repo with training code, inference code, an environment file and a write-up skeleton.
- [ ] For winners: reply to the winner notification within 1 week and return prize documents within 2 weeks.
- [ ] Note the data licence's consequences: CC BY means attribute; CC BY-NC means no product built on it; CC BY-SA or ODbL means published cleaned data keeps the same licence; an ODbL chart or model output needs a notice; a DUA or NDA means code only and delete the data afterwards.
- [ ] Never repost raw Competition Data anywhere during the competition.

Related: [`judging-and-winning-evidence.md`](judging-and-winning-evidence.md) (rubrics and intake checklist) · [`privacy-and-pii.md`](../../01-hackathon-playbook/docs/privacy-and-pii.md) (personal data).
