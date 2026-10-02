# PII Handling, De-identification and Synthetic Data for Hackathon/Datathon Teams

Research date: 2026-10-02. Not legal advice. GDPR = EU/EEA (UK GDPR is near-identical); HIPAA = US, applies only to covered entities/business associates, but its Safe Harbor list is a widely used practical checklist. Sandbox blocked gdpr-info.eu, eur-lex, hhs.gov, physionet.org, openai.com, ai.google.dev, so most legal/policy text below is from search-engine snippets (marked SNIPPET-ONLY). Tool facts (PyPI JSON, GitHub READMEs) were fetched directly and are confirmed.

## 1. What counts as personal data / PHI, and simple de-identification techniques

### Takeaway
Under GDPR, anything relating to a person identifiable "directly or indirectly" is personal data, including online identifiers, and pseudonymised data is STILL personal data; only truly anonymous data falls outside GDPR. HIPAA Safe Harbor gives a concrete 18-identifier removal list (plus "no actual knowledge" of re-identifiability), with special rules for ZIP codes (3 digits, >20,000 pop.) and ages >89.

### Cited Findings
- GDPR Art. 4(1): "'personal data' means any information relating to an identified or identifiable natural person ('data subject'); an identifiable natural person is one who can be identified, directly or indirectly, in particular by reference to an identifier such as a name, an identification number, location data, an online identifier or to one or more factors specific to the physical, physiological, genetic, mental, economic, cultural or social identity of that natural person." — [gdpr-info.eu Art. 4](https://gdpr-info.eu/art-4-gdpr/); [gdpr-text.com Art. 4](https://gdpr-text.com/read/article-4/) (SNIPPET-ONLY, text matches well-known wording)
- Online identifiers include IP addresses and cookie identifiers; devices with unique IDs let people be "singled out" even without names — [ICO: personal data](https://ico.org.uk/for-organisations/direct-marketing-and-privacy-and-electronic-communications/guidance-for-the-use-of-personal-data-in-political-campaigning-1/personal-data/) (SNIPPET-ONLY)
- GDPR Art. 4(5) pseudonymisation: processing so data "can no longer be attributed to a specific data subject without the use of additional information, provided that such additional information is kept separately and is subject to technical and organisational measures..." — [gdpr-info.eu Art. 4](https://gdpr-info.eu/art-4-gdpr/) (UNVERIFIED: wording from memory, page blocked)
- Recital 26: "Personal data which have undergone pseudonymisation, which could be attributed to a natural person by the use of additional information should be considered to be information on an identifiable natural person." Data protection principles "should therefore not apply to anonymous information... or to personal data rendered anonymous in such a manner that the data subject is not or no longer identifiable." — [gdpr-info.eu Recital 26](https://gdpr-info.eu/recitals/no-26/) (SNIPPET-ONLY)
- Practical framing: "Pseudonymisation is effectively only a security measure. It does not change the status of the data as personal data." — [ICO: what is personal data](https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/personal-information-what-is-it/what-is-personal-data/what-is-personal-data/) (SNIPPET-ONLY; attribution of exact sentence to ICO vs. search summarizer uncertain)
- GDPR Art. 9 special categories (health, genetic, biometric, racial/ethnic origin, political opinions, religion, trade-union membership, sex life/orientation) need extra legal basis — [gdpr-info.eu Art. 9](https://gdpr-info.eu/art-9-gdpr/) (UNVERIFIED, page blocked)
- HIPAA de-identification has two routes: Safe Harbor (§164.514(b)(2)) and Expert Determination (§164.514(b)(1)) — [HHS de-identification guidance](https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html) (UNVERIFIED: page blocked; well established)
- Safe Harbor 18 identifiers: (1) names; (2) geographic subdivisions smaller than a state (street, city, county, precinct, ZIP, geocodes), except first 3 ZIP digits if that 3-digit area has >20,000 people per current Census, else "000"; (3) all date elements (except year) directly related to an individual, and ages over 89 aggregated into "90 or older"; (4) phone numbers; (5) fax numbers; (6) email addresses; (7) SSNs; (8) medical record numbers; (9) health plan beneficiary numbers; (10) account numbers; (11) certificate/license numbers; (12) vehicle identifiers/serial numbers incl. license plates; (13) device identifiers/serial numbers; (14) URLs; (15) IP addresses; (16) biometric identifiers incl. finger/voice prints; (17) full-face photos and comparable images; (18) any other unique identifying number, characteristic or code — [CASRAI: 18 HIPAA identifiers](https://casrai.org/guides/18-hipaa-identifiers); [AccountableHQ](https://www.accountablehq.com/post/hipaa-s-18-identifiers-the-phi-safe-harbor-list-explained) (SNIPPET-ONLY; items 9-18 completed from the standard list, secondary sources)
- Safe Harbor also requires the covered entity to have no actual knowledge that remaining info could identify the individual — [HHS guidance](https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html) (UNVERIFIED, page blocked)
- Quasi-identifiers: 87% of the US population (1990 Census) was likely unique on {5-digit ZIP, sex, full date of birth} (Sweeney) — [Sweeney, Simple Demographics Often Identify People Uniquely](https://www.researchgate.net/publication/267716853_Simple_Demographics_Often_Identify_People_Uniquely); [johndcook simulation](https://www.johndcook.com/blog/2018/12/07/simulating-zipcode-sex-birthdate/) (SNIPPET-ONLY)
- k-anonymity: a release is k-anonymous if each record is indistinguishable from at least k-1 others on the quasi-identifiers; achieved via generalization (e.g. DOB -> year, ZIP -> 3 digits, age -> 10-yr band) and suppression (drop rare rows/values) (Sweeney 2002, Int. J. Uncertainty, Fuzziness & Knowledge-Based Systems) — UNVERIFIED (no source fetched; standard definition)

### Inferences
- Practical technique ladder for teams: (1) drop columns you don't need (names, emails, phones, free-text notes); (2) generalize quasi-identifiers (DOB->year/age band, ZIP->3 digits/region, dates->month or relative day offsets); (3) suppress rare combinations (check group sizes with `df.groupby(QIs).size().min() >= k`, k=5 is a common rule of thumb, UNVERIFIED as a standard); (4) replace direct IDs with salted hashes/random IDs = pseudonymisation (still personal data under GDPR, keep the key off the repo); (5) for demos, use synthetic/fake data.
- Free text (clinical notes, chat logs, support tickets) is the hardest place for PII to hide; run a detector (Presidio) AND manually spot-check.
- "Hashing emails" is not anonymisation: the same input gives the same hash, so it is linkable and often brute-forceable.

### Gaps
- Could not fetch primary GDPR/HHS text (blocked); verbatim quotes come via search snippets. Report writer should link to primary URLs listed but treat quotes as snippet-sourced.
- No authoritative source fetched for a recommended k value.

## 2. Tools: Presidio, scrubadub, Faker, SDV (versions + minimal usage)

### Takeaway
Presidio (2.2.364, Jul 2026, MIT, now community-governed under "Data Privacy Stack") is the best-maintained PII detect+anonymize option; scrubadub (2.0.1, Sep 2023) is simple but stale; Faker (40.40.0, Sep 2026, MIT) generates fake values; SDV (1.38.5, Sep 2026) learns from real tables to produce synthetic tables but is under the Business Source License (not OSI open source).

### Cited Findings
Versions (fetched from PyPI JSON on 2026-10-02):
- presidio-analyzer 2.2.364 and presidio-anonymizer 2.2.364, uploaded 2026-07-22, Python >=3.10,<3.15 — [PyPI presidio-analyzer](https://pypi.org/pypi/presidio-analyzer/json); [PyPI presidio-anonymizer](https://pypi.org/pypi/presidio-anonymizer/json)
- scrubadub 2.0.1, uploaded 2023-09-01, MIT — [PyPI scrubadub](https://pypi.org/pypi/scrubadub/json)
- Faker 40.40.0, uploaded 2026-09-29, Python >=3.10, MIT — [PyPI Faker](https://pypi.org/pypi/Faker/json)
- sdv 1.38.5, uploaded 2026-09-28, Python >=3.9,<3.15 — [PyPI sdv](https://pypi.org/pypi/sdv/json)

Presidio:
- Project is "transitioning from a Microsoft-owned project to an independent, community-governed open source project under the new GitHub organization Data Privacy Stack... https://github.com/data-privacy-stack/presidio"; stays MIT; Docker images move from `mcr.microsoft.com/presidio-*` to `ghcr.io/data-privacy-stack/presidio-*` — [Presidio project_transition.md](https://github.com/microsoft/presidio/blob/main/docs/project_transition.md)
- Detects "credit card numbers, names, locations, social security numbers, bitcoin wallets, US phone numbers, financial data and more"; recognizers use NER, regex, rule-based logic and checksums; has image redaction incl. DICOM; components: Analyzer, Anonymizer, Image-Redactor, Structured — [Presidio README](https://github.com/microsoft/presidio/blob/main/README.MD)
- Warning in README: "because it is using automated detection mechanisms, there is no guarantee that Presidio will find all sensitive information. Consequently, additional systems and protections should be employed." — [Presidio README](https://github.com/microsoft/presidio/blob/main/README.MD)
- Minimal usage (quoted from getting-started doc):
  ```sh
  pip install presidio-analyzer
  pip install presidio-anonymizer
  python -m spacy download en_core_web_lg
  ```
  ```py
  from presidio_analyzer import AnalyzerEngine
  from presidio_anonymizer import AnonymizerEngine
  text="My phone number is 212-555-5555"
  analyzer = AnalyzerEngine()
  results = analyzer.analyze(text=text, entities=["PHONE_NUMBER"], language='en')
  anonymizer = AnonymizerEngine()
  anonymized_text = anonymizer.anonymize(text=text,analyzer_results=results)
  print(anonymized_text)
  ```
  — [Presidio getting_started_text.md](https://github.com/microsoft/presidio/blob/main/docs/getting_started/getting_started_text.md). A transformers variant uses `pip install "presidio-analyzer[transformers]"` and `TransformersNlpEngine` with `dslim/bert-base-NER` (same file).
- Hosted demo: https://huggingface.co/spaces/presidio/presidio_demo — [Presidio README](https://github.com/microsoft/presidio/blob/main/README.MD)

scrubadub:
- Removes names, emails, addresses/postcodes (US, GB, CA), credit cards, DOBs, URLs, phone numbers, username/password combos, Skype/Twitter handles, SSNs/UK NI numbers, GB tax and driving licence numbers; name/address detectors need extra packages (scrubadub_spacy, scrubadub_address) — [scrubadub README](https://github.com/LeapBeyond/scrubadub/blob/master/README.rst)
- Minimal usage: `import scrubadub; scrubadub.clean("My cat can be contacted on example@example.com, or 1800 555-5555")` -> `'My cat can be contacted on {{EMAIL}}, or {{PHONE}}'` — [scrubadub README](https://github.com/LeapBeyond/scrubadub/blob/master/README.rst)

Faker:
- `pip install Faker`; `from faker import Faker; fake = Faker(); fake.name(); fake.address()`; reproducible with `Faker.seed(4321)` ("A Seed produces the same result when the same methods with the same version of faker are called") — [Faker README](https://github.com/joke2k/faker/blob/master/README.rst)

SDV:
- "The SDV is publicly available under the Business Source License"; models from GaussianCopula to CTGAN; single, multi-table and sequential data; built-in quality evaluation — [SDV README](https://github.com/sdv-dev/SDV/blob/main/README.md)
- License: Additional Use Grant allows use "provided that you do not use the Licensed Work... for a Synthetic Data Service" (a commercial offering letting third parties access SDV's functionality); converts to MIT four years after each release — [SDV LICENSE](https://github.com/sdv-dev/SDV/blob/main/LICENSE)
- Minimal usage (README):
  ```python
  from sdv.datasets.demo import download_demo
  real_data, metadata = download_demo(modality='single_table', dataset_name='fake_hotel_guests')
  from sdv.single_table import GaussianCopulaSynthesizer
  synthesizer = GaussianCopulaSynthesizer(metadata)
  synthesizer.fit(data=real_data)
  synthetic_data = synthesizer.sample(num_rows=500)
  from sdv.evaluation import evaluate_quality
  quality_report = evaluate_quality(real_data, synthetic_data, metadata)
  ```
  README claims: sensitive columns (email, billing address, credit card) "are fully anonymized", other columns follow statistical patterns, keys/relationships intact — [SDV README](https://github.com/sdv-dev/SDV/blob/main/README.md)

### Inferences
- Hackathon recipe: Presidio for free text; pandas drop/generalize for tables; Faker for demo user profiles/seed DBs; SDV when you need realistic-looking tabular data with correlations (fine for hackathon use; avoid building a commercial "synthetic data as a service" product on it given BSL).
- Synthetic data trained on real records is not automatically anonymous (models can memorize outliers); treat SDV output from sensitive data with care and check the dataset's DUA before training on it outside the approved environment. (Inference; no source fetched on SDV privacy metrics.)
- scrubadub has had no release since Sep 2023; prefer Presidio for anything beyond regex-level scrubbing.

### Gaps
- Did not verify SDV's privacy metrics (e.g. DCR) or the README import path `sdv.evaluation` vs older `sdv.evaluation.single_table` against the 1.38.5 release; README on `main` was used.
- Presidio PyPI JSON lists no license string (license field None); MIT per README/transition doc.

## 3. LLM APIs with personal data; DUAs; consent

### Takeaway
Major paid APIs don't train on inputs by default but may retain them (e.g. OpenAI up to 30 days for abuse monitoring); consumer chat apps and free tiers are different (Anthropic consumer opt-out since Sep 2025; Gemini free tier may be used for training and human review). Credentialed datasets like MIMIC (PhysioNet) forbid sending data to third-party online services except specific configured providers; default to local models.

### Cited Findings
- OpenAI: since March 1, 2023, API data is not used to train models unless you opt in; API inputs/outputs may be retained up to 30 days for abuse monitoring, then deleted unless legally required; Zero Data Retention available for eligible endpoints/qualifying use cases — [OpenAI: Data controls](https://developers.openai.com/api/docs/guides/your-data); [TechCrunch 2023-03-01](https://techcrunch.com/2023/03/01/addressing-criticism-openai-will-no-longer-use-customer-data-to-train-its-models-by-default/) (SNIPPET-ONLY)
- Anthropic: Aug 28, 2025 consumer terms update — Free/Pro/Max chats and coding sessions used for training unless users opt out (deadline Sep 28, 2025); retention 5 years if opted in, 30 days if not; does not apply to Commercial Terms services (Claude for Work/Gov/Education, API incl. via third-party platforms) — [Anthropic: Updates to consumer terms](https://www.anthropic.com/news/updates-to-our-consumer-terms) (SNIPPET-ONLY)
- Google Gemini API: on unpaid/free tier, prompts, files and responses may be used to improve products and read by human reviewers; paid tier (and Vertex AI) excluded from training under the data processing addendum — [Gemini API terms](https://ai.google.dev/gemini-api/terms) (primary blocked); [ampm-aiops summary](https://ampm-aiops.com/en/guides/gemini-free-tier-data-tradeoff-2026/) (SNIPPET-ONLY, secondary)
- PhysioNet (MIMIC): the Credentialed DUA "explicitly prohibits sharing access to the data with third parties, including sending it through APIs provided by companies like OpenAI, or using it in online platforms like ChatGPT"; researchers "strongly recommended to use locally deployed LLMs"; permitted online options with conditions: Azure OpenAI (must opt out of human review), Google Gemini via Vertex AI, Anthropic Claude (no training by default, no routine human review) — [PhysioNet: Use of MIMIC data with LLMs and online services](https://physionet.org/news/post/llm-responsible-use/); earlier post [PhysioNet: Responsible use of MIMIC with GPT](https://physionet.org/news/post/gpt-responsible-use/) (SNIPPET-ONLY; exact provider list/conditions should be checked on the live page)
- Credentialed PhysioNet access involves training plus signing a DUA per project — [CASRAI: PhysioNet credentialed access](https://casrai.org/guides/physionet-credentialed-access-restricted-data) (SNIPPET-ONLY)

### Inferences
- Rule of thumb for teams: (a) read the dataset's DUA/licence before anything else; if it restricts sharing, no hosted LLM, no public GitHub/Colab/Kaggle uploads, no pasting into chat apps; (b) never use consumer chat UIs or free API tiers for real personal data; (c) if a hosted API is allowed, use a paid/commercial tier, minimise and de-identify first (Presidio before the prompt), and log what was sent; (d) a provider "not training" does not mean "not retaining".
- Consent for user testing/demos: get explicit, informed opt-in (what is recorded, why, who sees it, how long kept, how to withdraw); prefer not to record faces/voices; use teammates or Faker personas in demo videos and screenshots; never show real participant data on stage or in a public repo. (No authoritative source fetched; GDPR Art. 6/7 consent conditions apply in EU — UNVERIFIED.)
- Secrets hygiene overlaps: add raw data paths to .gitignore; public repo commits of personal data are effectively permanent.

### Gaps
- Could not fetch any provider's primary policy page (blocked); policies change often, so the guide should say "check current terms" and date-stamp.
- No authoritative source collected on consent forms for hackathon user testing (e.g. university IRB templates, ICO consent guidance).
- HIPAA BAA availability per provider not researched.

## 4. Re-identification cases that justify caution

### Takeaway
Three classic cases show that removing names is not enough: linkage on quasi-identifiers (Massachusetts GIC/Weld), on sparse behavioural data (Netflix Prize), and on the content itself (AOL search logs). Recent work suggests LLMs make linkage attacks cheaper.

### Cited Findings
- Massachusetts GIC: in the 1990s the state Group Insurance Commission released "anonymized" hospital data for research; Gov. William Weld assured the public identifiers were removed; Latanya Sweeney (1997) re-identified Weld's records by linking ZIP, birth date and sex with the Cambridge voter roll — [centerconsulting summary](https://www.centerconsulting.com/ai-library/facts/sweeney-87-percent-reidentifiable); [EFF: What information is personally identifiable?](https://www.eff.org/deeplinks/2009/09/what-information-personally-identifiable) (SNIPPET-ONLY; voter-roll detail from general knowledge, UNVERIFIED in fetched text)
- Sweeney: 87% (87.1%) of the US population per 1990 Census likely uniquely identified by 5-digit ZIP, sex, date of birth — [Sweeney paper (ResearchGate)](https://www.researchgate.net/publication/267716853_Simple_Demographics_Often_Identify_People_Uniquely) (SNIPPET-ONLY); later revisited by Golle with a lower estimate (~63% on 2000 Census) — [Golle, Revisiting the Uniqueness of Simple Demographics](https://www.researchgate.net/publication/221342213_Revisiting_the_Uniqueness_of_Simple_Demographics_in_the_US_Population) (63% figure UNVERIFIED)
- Netflix Prize: Narayanan & Shmatikov (IEEE S&P 2008) de-anonymised the dataset of ~500,000 subscribers' ratings; "with 8 movie ratings (of which 2 may be completely wrong) and dates that may have a 14-day error, 99% of records can be uniquely identified"; linked to public IMDb ratings — [Narayanan & Shmatikov paper (Cornell PDF)](https://www.cs.cornell.edu/~shmat/shmat_oak08netflix.pdf); [arXiv cs/0610105](https://arxiv.org/pdf/cs/0610105) (SNIPPET-ONLY)
- AOL (Aug 2006): AOL released "anonymized" search logs keyed by user number; the New York Times identified user No. 4417749 as Thelma Arnold, a 62-year-old widow in Lilburn, Georgia, purely from her search terms — [NYT "A Face Is Exposed for AOL Searcher No. 4417749" (PDF copy)](https://www2.hawaii.edu/~strev/ICS614/materials/NYT%20-%20confidentiality%20-%20A%20Face%20is%20Exposed%20for%20AOL%20Searcher%20%202006-08-24.pdf); [TechCrunch 2006-08-09](https://techcrunch.com/2006/08/09/first-person-identified-from-aol-data-thelma-arnold/) (SNIPPET-ONLY)
- 2026 paper "Large-scale online deanonymization with LLMs" — [arXiv 2602.16800](https://arxiv.org/html/2602.16800) (SNIPPET-ONLY: title only seen; findings not read, UNVERIFIED)

### Inferences
- Lesson for the guide: "anonymised" claims fail via (1) quasi-identifier linkage, (2) sparse high-dimensional behaviour data (ratings, locations, purchases, search queries are fingerprints), (3) free-text content. Teams publishing derived datasets or demos should aggregate, generalize, or use synthetic data rather than releasing row-level records.

### Gaps
- Did not read the 2026 LLM deanonymization paper; cite cautiously or omit.
- Primary Sweeney 1997/2000 documents not fetched directly.
