# 🔒 Privacy, PII & De-identification

<!-- markdownlint-disable MD013 -->

> 📚 Full research report and raw notes: [`technical-datathon-gap-fill-round-two-2026-10-02`](../../_research/technical-datathon-gap-fill-round-two-2026-10-02/research.md).

> How to handle personal data in a hackathon or datathon: what counts as personal data, how to de-identify it, which tools to use, and when you may not send data to an LLM. **Not legal advice**; rules depend on your jurisdiction and the dataset's agreement. Research round two, 2026-10-02. The sandbox blocked many primary sites, so read the labels: **FULL-TEXT** means the page or file was read in full, usually a GitHub copy or source file. **SNIPPET-ONLY** means the claim comes from a search-engine summary, not the page itself, so do not quote it as verbatim. **UNVERIFIED** means it could not be checked at all, or rests on background knowledge. **RE-CHECK** marks a limit, quota or price the vendor can change at any time. **CONFLICT** marks sources that disagree. "Inference (ours)" is the researchers' own synthesis, not a rule or standard.

## PII handling: pseudonymised is still personal, and a dataset agreement can rule out hosted LLMs

**Not legal advice.** GDPR applies in the EU/EEA, and UK GDPR is near-identical. HIPAA applies only to US covered entities and their business associates, but its Safe Harbor list is a useful practical checklist. Most legal and policy text below is SNIPPET-ONLY because the primary sites were blocked. Policies change, so date-stamp this section and check current terms.

### What is sourced

GDPR Art. 4(1) defines personal data as information relating to a person identifiable "directly or indirectly", including by "location data" or "an online identifier" (SNIPPET-ONLY) ([gdpr-text.com Art. 4](https://gdpr-text.com/read/article-4/)). Recital 26 says pseudonymised data that can be re-attributed "should be considered to be information on an identifiable natural person". Only data rendered anonymous so that the person "is not or no longer identifiable" falls outside GDPR (SNIPPET-ONLY) ([Recital 26](https://gdpr-info.eu/recitals/no-26/)). The ICO calls pseudonymisation "effectively only a security measure" (SNIPPET-ONLY; the exact attribution is uncertain) ([ICO](https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/personal-information-what-is-it/what-is-personal-data/what-is-personal-data/)). The Art. 9 special categories, such as health, biometrics and ethnicity, need an extra legal basis (UNVERIFIED; page blocked).

HIPAA allows de-identification by two routes, Safe Harbor or Expert Determination (UNVERIFIED; [HHS guidance](https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html) was blocked). Safe Harbor removes **18 identifier types**. The less obvious rules are:

- ZIP codes may keep only the first 3 digits, and only if that 3-digit area has more than 20,000 people; otherwise use "000".
- All date elements except the year are removed.
- Ages over 89 are grouped as "90 or older".
- IP addresses, URLs, device serial numbers, full-face photos, and "any other unique identifying number, characteristic or code" are all on the list.

Safe Harbor also requires that the covered entity have no actual knowledge that the remaining data could identify someone (SNIPPET-ONLY; items 9–18 from secondary lists) ([CASRAI](https://casrai.org/guides/18-hipaa-identifiers); [AccountableHQ](https://www.accountablehq.com/post/hipaa-s-18-identifiers-the-phi-safe-harbor-list-explained)).

**Why removing names is not enough.**

- Sweeney estimated that **87%** of the US population (1990 Census) was likely unique on {5-digit ZIP, sex, full date of birth}, and re-identified Governor Weld's hospital records ([Sweeney](https://www.researchgate.net/publication/267716853_Simple_Demographics_Often_Identify_People_Uniquely); [EFF](https://www.eff.org/deeplinks/2009/09/what-information-personally-identifiable)). **CONFLICT:** Golle's re-analysis of the 2000 Census gives a lower figure, about 63% (UNVERIFIED) ([Golle](https://www.researchgate.net/publication/221342213_Revisiting_the_Uniqueness_of_Simple_Demographics_in_the_US_Population)).
- Netflix Prize: with "8 movie ratings (of which 2 may be completely wrong) and dates that may have a 14-day error, 99% of records can be uniquely identified" ([Narayanan and Shmatikov](https://www.cs.cornell.edu/~shmat/shmat_oak08netflix.pdf)).
- AOL: the New York Times identified searcher No. 4417749 from her search terms alone ([NYT PDF copy](https://www2.hawaii.edu/~strev/ICS614/materials/NYT%20-%20confidentiality%20-%20A%20Face%20is%20Exposed%20for%20AOL%20Searcher%20%202006-08-24.pdf)).

All three are SNIPPET-ONLY. A 2026 paper on LLM-driven deanonymisation exists, but only its title was seen ([arXiv 2602.16800](https://arxiv.org/html/2602.16800), UNVERIFIED findings).

**Tools** (versions from PyPI, FULL-TEXT; READMEs read directly):

- **Presidio 2.2.364** (July 2026, MIT) is moving from Microsoft to the community-run "Data Privacy Stack" organisation, and its Docker images move to `ghcr.io/data-privacy-stack/presidio-*` ([transition doc](https://github.com/microsoft/presidio/blob/main/docs/project_transition.md)). Its README warns "there is no guarantee that Presidio will find all sensitive information" ([Presidio README](https://github.com/microsoft/presidio/blob/main/README.MD)).
- **scrubadub 2.0.1** dates from September 2023 ([PyPI](https://pypi.org/pypi/scrubadub/json)).
- **Faker 40.40.0** (September 2026, MIT) can be seeded so it returns the same values each run ([Faker README](https://github.com/joke2k/faker/blob/master/README.rst)).
- **SDV 1.38.5** is under the **Business Source License**, not OSI open source. You may use it except "for a Synthetic Data Service", and each release converts to MIT after four years ([SDV LICENSE](https://github.com/sdv-dev/SDV/blob/main/LICENSE)).

**LLM data-sharing** (all SNIPPET-ONLY, provider pages blocked; RE-CHECK):

- **OpenAI:** API data has not been used for training by default since 1 March 2023, but may be retained for up to 30 days for abuse monitoring. Zero Data Retention exists for eligible use cases ([OpenAI data controls](https://developers.openai.com/api/docs/guides/your-data)).
- **Anthropic:** its 2025 consumer-terms update made Free, Pro and Max chats eligible for training unless users opt out, with 5-year retention if opted in and 30 days if not. This does not apply to the API or other commercial-terms services ([Anthropic](https://www.anthropic.com/news/updates-to-our-consumer-terms)).
- **Gemini:** on the free API tier, prompts may be used for product improvement and read by human reviewers. The paid tier and Vertex AI are excluded ([Gemini API terms](https://ai.google.dev/gemini-api/terms)).
- **PhysioNet (MIMIC):** the credentialed DUA "explicitly prohibits... sending it through APIs provided by companies like OpenAI, or using it in online platforms like ChatGPT", and strongly recommends local LLMs. It names conditional exceptions: Azure OpenAI with human review opted out, Gemini via Vertex AI, and Claude. Check the live page for the exact list ([PhysioNet LLM guidance](https://physionet.org/news/post/llm-responsible-use/)).

### Inference (ours)

The workable ladder for a team, in order:

1. Drop columns you do not need.
2. Generalise quasi-identifiers: date of birth to year or age band, ZIP to 3 digits or region, dates to month or relative offsets.
3. Suppress rare combinations. k = 5 is a common rule of thumb, not a sourced standard.
4. Replace direct IDs with random or salted IDs. This is pseudonymisation, so the data is still personal data, and the key stays out of the repo.
5. Use Faker or SDV output for demos.

Hashing emails is not anonymisation: the same input always gives the same hash, so the result is linkable and often brute-forceable. Free text such as clinical notes, chats and support tickets is where PII hides, so run Presidio and then spot-check by hand. Ratings, locations, purchases and search queries act as fingerprints even with no names attached. SDV output trained on real records is not automatically anonymous, because outliers can be memorised, and the BSL rules out building a commercial synthetic-data service on it. For LLMs: "not training" is different from "not retaining", consumer chat apps and free API tiers are off-limits for real personal data, and a hosted API is acceptable only when the DUA allows it, on a paid or commercial tier, after de-identification, with a log of what was sent. Consent for user testing should be explicit and opt-in, covering what is recorded, why, who sees it, how long it is kept and how to withdraw. GDPR consent conditions are UNVERIFIED here, and no consent-form template was sourced.

### Ready-to-use checklist and code

- [ ] Read the dataset's licence or DUA **before** loading it anywhere. If it restricts sharing: no hosted LLM, no Colab, Kaggle or HF upload, no public repo, no chat apps.
- [ ] List direct identifiers (against the 18 Safe Harbor types) and quasi-identifiers (ZIP, date of birth, sex, rare categories, location traces, free text).
- [ ] Drop, generalise, suppress, then pseudonymise, in that order. Keep any re-identification key off the repo and off shared drives.
- [ ] Check the minimum group size on the quasi-identifiers before sharing any derived table.
- [ ] Run Presidio over free-text columns, then spot-check a sample by hand.
- [ ] For demos, screenshots and videos, use Faker personas or synthetic data, never real rows.
- [ ] LLM use: paid or commercial tier only, de-identified input only, provider named in the README, prompts logged. For credentialed health data, use a local model unless the DUA lists the provider.
- [ ] Delete data at the end if the agreement requires it, and record that you did.

```python
# Presidio quickstart, quoted from the official getting-started doc (FULL-TEXT source; not run by us)
# pip install presidio-analyzer presidio-anonymizer && python -m spacy download en_core_web_lg
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
text = "My phone number is 212-555-5555"
results = AnalyzerEngine().analyze(text=text, entities=["PHONE_NUMBER"], language='en')
print(AnonymizerEngine().anonymize(text=text, analyzer_results=results))

# k-anonymity spot check (assembled by us; not run; k=5 is a rule of thumb, UNVERIFIED as a standard)
QIS = ['zip3', 'age_band', 'sex']
assert df.groupby(QIS).size().min() >= 5, "rare quasi-identifier combinations: generalise or suppress"

# Faker demo personas (README pattern; not run by us)
from faker import Faker
Faker.seed(4321); fake = Faker()
demo_users = [{'name': fake.name(), 'address': fake.address()} for _ in range(50)]
```
Source for the Presidio code: [Presidio getting started](https://github.com/microsoft/presidio/blob/main/docs/getting_started/getting_started_text.md). Source for the Faker pattern: [Faker README](https://github.com/joke2k/faker/blob/master/README.rst).

Related: the short privacy minimum in [`rules-hardware-a11y-remote.md`](rules-hardware-a11y-remote.md) section 3 · dataset licences in [`datathon-rules-and-licences.md`](../../08-datathon-handbook/01-playbook/datathon-rules-and-licences.md).
