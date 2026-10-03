# Getting training/evaluation data at a hackathon when no dataset is provided

Research date: 2026-10-03. Method notes: PyPI JSON fetched live via curl (`https://pypi.org/pypi/<pkg>/json`) on 2026-10-03; READMEs fetched from raw.githubusercontent.com. huggingface.co, arxiv.org and rfc-editor.org were blocked by the egress proxy, so claims about those pages come from search-result snippets and are marked SNIPPET-ONLY. GitHub API was not available, so "archived/maintained" status comes from release dates and README text only. Items stated from background knowledge without a fetched source are marked UNVERIFIED. This is not legal advice.

## Q1: Tools — licence, latest version (PyPI JSON), minimal usage, realistic hackathon setup time

### Takeaway
Every listed labelling, weak-supervision, active-learning and augmentation tool is under a permissive licence (Apache-2.0, MIT or BSD) except SDV, which uses the Business Source License 1.1. That licence is not open source and bars use in a commercial "Synthetic Data Service". Label Studio, Faker and SDV had releases in the last week. Snorkel, modAL and doccano have had no release since 2023–2024, so expect dependency friction.

### Cited Findings

**PyPI snapshot (fetched 2026-10-03 from `https://pypi.org/pypi/<pkg>/json`)**

| Package | Latest version | Upload date of latest | Licence (PyPI metadata) | requires_python |
|---|---|---|---|---|
| label-studio | 1.23.2 | 2026-09-29 | Apache-2.0 | >=3.10,<4 |
| doccano | 1.8.4 | 2023-07-20 | MIT | >=3.8,<4.0 |
| argilla (client) | 2.8.0 | 2025-03-10 | Apache 2.0 | >=3.9 |
| snorkel | 0.10.0 | 2024-02-27 | Apache 2.0 | >=3.11 |
| modAL-python | 0.4.2.1 | 2023-06-01 | MIT | (none declared) |
| small-text | 1.4.1 (stable); 2.0.0.dev4 pre-release exists | 2024-08-18 | MIT | >=3.7 (README says 3.10+) |
| Faker | 40.40.0 | 2026-09-29 | MIT | >=3.10 |
| sdv | 1.38.5 | 2026-09-28 | BUSL-1.1 | >=3.9,<3.15 |
| cleanlab | 2.9.0 | — | Apache-2.0 | >=3.10 |
| nlpaug | 1.1.11 | — | MIT | >=3.7 |
| albumentations | 2.0.8 | — | MIT | >=3.9 |
| augly | 1.0.0 | — | MIT (classifier) | >=3.6 |
| distilabel | 1.5.3 | 2025-01-28 | Apache-2.0 | >=3.9 |
| cvat-cli | 2.77.0 | — | MIT | >=3.10 |
| kaggle (API client) | 2.2.4 | — | Apache-2.0 | >=3.11 |
| datasets (Hugging Face) | 5.0.1 | — | Apache 2.0 | >=3.10 |
| openml | 0.15.1 | — | BSD-3-Clause | >=3.8 |
| ucimlrepo | 0.0.7 | — | MIT (classifier) | >=3.7 |

Source for the table: PyPI JSON API, e.g. [label-studio](https://pypi.org/pypi/label-studio/json), [sdv](https://pypi.org/pypi/sdv/json), [snorkel](https://pypi.org/pypi/snorkel/json), [modAL-python](https://pypi.org/pypi/modAL-python/json), [small-text](https://pypi.org/pypi/small-text/json), [doccano](https://pypi.org/pypi/doccano/json), [argilla](https://pypi.org/pypi/argilla/json), [Faker](https://pypi.org/pypi/Faker/json), [datasets](https://pypi.org/pypi/datasets/json), [openml](https://pypi.org/pypi/openml/json), [ucimlrepo](https://pypi.org/pypi/ucimlrepo/json), [kaggle](https://pypi.org/pypi/kaggle/json), [cleanlab](https://pypi.org/pypi/cleanlab/json), [distilabel](https://pypi.org/pypi/distilabel/json), [cvat-cli](https://pypi.org/pypi/cvat-cli/json), [nlpaug](https://pypi.org/pypi/nlpaug/json), [albumentations](https://pypi.org/pypi/albumentations/json), [augly](https://pypi.org/pypi/augly/json).

**Labelling tools**
- Label Studio: `pip install label-studio`, then `label-studio` starts the server at http://localhost:8080 and needs Python >=3.10. The Docker equivalent is `docker run -it -p 8080:8080 -v $(pwd)/mydata:/label-studio/data heartexlabs/label-studio:latest`, which stores data in SQLite under `./mydata` by default — [Label Studio README](https://raw.githubusercontent.com/HumanSignal/label-studio/develop/README.md)
- doccano (text-focused: classification, sequence labelling, seq2seq): `pip install doccano`, which uses SQLite by default. A Docker image `doccano/doccano` with admin credentials passed through env vars is on port 8000 — [doccano README](https://raw.githubusercontent.com/doccano/doccano/master/README.md). Its latest PyPI release is from July 2023 ([PyPI](https://pypi.org/pypi/doccano/json)).
- Argilla: `pip install argilla` installs only the client. You also need a server, and the README calls the "free Hugging Face Spaces deployment integration" the easiest route. Client: `import argilla as rg; client = rg.Argilla(api_url="https://[owner]-[space].hf.space", api_key=...)`. It loads HF datasets through `dataset.records.log(records=data, mapping={"text": "review"})` — [Argilla README](https://raw.githubusercontent.com/argilla-io/argilla/develop/README.md)
- CVAT (image, video and 3D annotation): its core is "MIT-licensed… Some serverless assets and dependencies may have separate licenses". You can self-host with Docker Engine, Docker Compose and Git (`docker compose up -d`), or use "CVAT Online (Free plan)" at app.cvat.ai, where "Feature availability and usage limits vary by plan" — [CVAT README](https://raw.githubusercontent.com/cvat-ai/cvat/develop/README.md); [CVAT LICENSE](https://raw.githubusercontent.com/cvat-ai/cvat/develop/LICENSE) (MIT, © Intel 2018–2022, CVAT.ai 2022–2025)

**Weak supervision**
- Snorkel: `pip install snorkel` and Python 3.11+. The README announces that "The Snorkel team is now focusing their efforts on Snorkel Flow", the commercial platform. Learning material is at snorkel.org/get-started and in the snorkel-tutorials repo — [Snorkel README](https://raw.githubusercontent.com/snorkel-team/snorkel/main/README.md). Licence: Apache 2.0 ([LICENSE](https://raw.githubusercontent.com/snorkel-team/snorkel/main/LICENSE)). Latest release 0.10.0, Feb 2024 ([PyPI](https://pypi.org/pypi/snorkel/json)).
- Typical Snorkel pattern (UNVERIFIED, from background knowledge of the snorkel.org tutorial, not fetched): decorate Python functions with `@labeling_function()` that return a label or `ABSTAIN`, apply them with `PandasLFApplier`, then fit `LabelModel` to combine the noisy votes into probabilistic labels.

**Active learning**
- modAL. Minimal usage from the README: `learner = ActiveLearner(estimator=RandomForestClassifier(), X_training=X_training, y_training=y_training)`, then `query_idx, query_inst = learner.query(X_pool)`, then `learner.teach(X_pool[query_idx], y_new)`. Query strategies can be swapped (e.g. `entropy_sampling`) or replaced with a custom function — [modAL README](https://raw.githubusercontent.com/modAL-python/modAL/master/README.md). Latest release 0.4.2.1, June 2023 ([PyPI](https://pypi.org/pypi/modAL-python/json)).
- small-text: `pip install small-text`, or `pip install small-text[transformers]` for the transformer-based classifiers. The README says it "requires Python 3.10 or newer" and needs CUDA 10.1+ for GPU. Quick-start examples are binary classification, PyTorch multi-class and transformers multi-class. The README carries a "maintained: yes" badge and an MIT licence — [small-text README](https://raw.githubusercontent.com/webis-de/small-text/main/README.md). The README's doc links point to v2.0.0.dev4, while PyPI stable is 1.4.1 ([PyPI](https://pypi.org/pypi/small-text/json)).

**Synthetic data**
- Faker: `pip install Faker`, then use "`faker.Faker()` to create and initialize a faker generator, which can generate data by accessing properties named after the type of data you want" (e.g. `fake.name()`, `fake.address()`) — [Faker README](https://raw.githubusercontent.com/joke2k/faker/master/README.rst). MIT licence.
- SDV: `pip install sdv`. Minimal flow: `download_demo(modality='single_table', dataset_name='fake_hotel_guests')`, then `GaussianCopulaSynthesizer(metadata)`, `.fit(data=real_data)`, `.sample(num_rows=500)`, then `evaluate_quality(real_data, synthetic_data, metadata)`, which reports "Column Shapes" and "Column Pair Trends" scores — [SDV README](https://raw.githubusercontent.com/sdv-dev/SDV/main/README.md)
- SDV licence: "Business Source License 1.1… is not an Open Source license." The Additional Use Grant allows use "provided that you do not use the Licensed Work… for a Synthetic Data Service", defined as "a commercial offering that allows third parties… to access the functionality… [of] synthetic data creation". Change Date: "four years from release date", after which it becomes MIT — [SDV LICENSE](https://raw.githubusercontent.com/sdv-dev/SDV/main/LICENSE)

**Dataset loaders**
- UCI via `ucimlrepo`: `from ucimlrepo import fetch_ucirepo; heart_disease = fetch_ucirepo(id=45)`, then `X = heart_disease.data.features; y = heart_disease.data.targets`. Metadata is available through `.metadata`, and `list_available_datasets()` lists what can be imported — [ucimlrepo README](https://raw.githubusercontent.com/uci-ml-repo/ucimlrepo/main/README.md)
- Hugging Face `datasets`: `from datasets import load_dataset; load_dataset("imdb", split="train[:100]")` (example quoted in the [Argilla README](https://raw.githubusercontent.com/argilla-io/argilla/develop/README.md))
- Not fetched (UNVERIFIED): OpenML `openml.datasets.get_dataset(<id>)` followed by `.get_data(target=...)`; Kaggle CLI `kaggle datasets download -d <owner>/<slug>`, which needs an API token from the Kaggle account settings.

### Inferences
- Realistic setup times, based on install paths in the READMEs. These are my judgement, not measured.
  - Under 5 minutes: Faker, ucimlrepo, `datasets`, openml, modAL (all pip-only, no server).
  - About 10–20 minutes: Label Studio (pip or a single docker run, SQLite, no external DB) and doccano (pip, or Docker with env-var admin).
  - Argilla needs a server. A HF Space is easiest but adds account and Space spin-up time, and huggingface.co may be blocked on some networks, as it was in this research sandbox.
  - CVAT self-hosting needs Docker Compose with several services, so it is the heaviest. The hosted free plan is faster for image or video work.
  - Snorkel's dependency pins may clash with current stacks given no release since Feb 2024. Plan an isolated venv.
- Default picks: Label Studio for mixed modalities, doccano for text-only, and CVAT Online for bounding boxes and video.
- SDV is fine for a hackathon prototype under the Additional Use Grant. A team pitching a commercial synthetic-data product should flag the BSL. This is not legal advice.
- modAL and doccano still work but are stale, so pin versions. small-text is the more actively signposted active-learning option for text.

### Gaps
- Could not check GitHub "archived" status via the API (it returned 403 in this session). Maintenance status is inferred only from PyPI release dates and README text.
- No published measurements of setup time for these tools were found. The times above are estimates.
- Did not fetch the Snorkel get-started page or the OpenML and Kaggle docs, so their usage snippets are UNVERIFIED.
- Argilla's latest PyPI release is March 2025, and I did not confirm whether the project is still actively developed.

## Q2: Public dataset sources and licence caveats

### Takeaway
Use Kaggle, Hugging Face Hub, UCI, OpenML, government portals and Google Dataset Search, and check the per-dataset licence every time. Kaggle shows a licence per dataset, competition data carries stricter rules, and HF licence fields are declared by uploaders. Papers with Code shut down in July 2025. Its successor is Hugging Face Trending Papers, and the historical data survives on GitHub.

### Cited Findings
- Kaggle lists a licence on every dataset page. CC0 is public domain, CC-BY needs attribution, and other CC variants may block commercial use or require share-alike — [labelyourdata.com Kaggle guide](https://labelyourdata.com/articles/machine-learning/kaggle-datasets) (SNIPPET-ONLY, secondary source)
- In Kaggle competition rules, "Unless otherwise expressly stated on the Competition Website, Participants must not use data other than the Data to develop and test their models". Competition data must be accepted under its own rules, and use outside the competition may be restricted — [Kaggle competition rules example](https://www.kaggle.com/competitions/data-science-and-ai/rules) (SNIPPET-ONLY)
- Papers with Code was sunsetted on 24 July 2025 and now redirects to Hugging Face Trending Papers, built with Meta. The historical dataset is preserved in the paperswithcode-data GitHub repo — [Coursera article](https://www.coursera.org/articles/papers-with-code); [AK on X](https://x.com/_akhaliq/status/1948729112626946080) (both SNIPPET-ONLY)
- Google Dataset Search indexes pages marked up with schema.org `Dataset` or W3C DCAT metadata — [Google developers: Dataset structured data](https://developers.google.com/search/docs/appearance/structured-data/dataset); [Google Dataset Search paper, WWW 2019](https://dl.acm.org/doi/fullHtml/10.1145/3308558.3313685) (SNIPPET-ONLY). Because it is an index, the licence is whatever the hosting site states.
- The Data.gov catalog "automatically harvests over 1000 different sources from federal, state and local open data sources". One snippet gave a count of about 604,843 datasets and said the number fluctuates — [catalog.data.gov](https://catalog.data.gov/); [Data.gov user guide](https://data.gov/user-guide/) (SNIPPET-ONLY, count not confirmed)
- UCI datasets can be fetched programmatically with `ucimlrepo`, which exposes metadata including `num_instances` and a summary — [ucimlrepo README](https://raw.githubusercontent.com/uci-ml-repo/ucimlrepo/main/README.md)

### Inferences
- Practical licence check: look at the dataset card or page licence first, then the original upstream source, since re-uploads on Kaggle or HF may not have rights to relicense. If the event allows outside data, record the URL and licence in the README for judges.
- For benchmarks with leaderboards now that Papers with Code is gone, use HF Papers/Trending and the archived paperswithcode-data repo.

### Gaps
- Could not fetch huggingface.co (blocked), so HF dataset-card licence and gating behaviour is unconfirmed (UNVERIFIED: licences are declared in dataset-card YAML by uploaders, and some datasets are "gated" behind accepting terms).
- UCI's default dataset licence (UNVERIFIED: many UCI datasets list CC BY 4.0) was not confirmed.
- Did not verify the national open-data portals of other countries (data.gov.uk, data.europa.eu).

## Q3: How much to label — published guidance

### Takeaway
I found no authoritative universal number. The sources available here point to diminishing returns after a few hundred labelled examples for simple text classification. Plan an initial batch of about 100–300 items, then plot a learning curve on a held-out set and stop when it flattens.

### Cited Findings
- "High classification accuracy can be achieved using a manually annotated dataset of only 300 examples"; adding more "rarely substantially increases classification performance" — [ScienceDirect, "Text classification using a few labeled examples" (Computers in Human Behavior)](https://www.sciencedirect.com/science/article/abs/pii/S0747563213002823) (SNIPPET-ONLY; the attribution of this exact quote to this paper is unconfirmed)
- One practitioner experiment found accuracy rose steeply then flattened. Going from 2 to 5 examples per category gained 15 points, and from 5 to 10 only 5 more — [Towards Data Science, "How Many Labeled Examples Does a Text Classifier Actually Need?"](https://towardsdatascience.com/how-many-labeled-examples-does-a-text-classifier-actually-need-i-measured-it/) (SNIPPET-ONLY, blog, single task)
- "Simple baseline classifiers can get surprisingly close to state-of-the-art" in few-shot settings — [A Neural Few-Shot Text Classification Reality Check (arXiv 2101.12073)](https://arxiv.org/pdf/2101.12073) (SNIPPET-ONLY)
- Active learning (modAL, small-text) is designed to choose which pool items to label next, e.g. by uncertainty or entropy sampling — [modAL README](https://raw.githubusercontent.com/modAL-python/modAL/master/README.md)

### Inferences
- Hackathon heuristic, my synthesis and not a published rule: two or three people labelling for one hour in Label Studio or doccano can plausibly produce a few hundred short-text labels. Hold out a fixed test split first, before any active learning or LLM labelling, so the evaluation set stays untouched.
- Report a learning curve (e.g. 50, 100, 200, 400 labels) instead of claiming a magic number.

### Gaps
- Could not reach arXiv or full texts, so no rigorous published per-class guidance was confirmed. No image or vision guidance was found.

## Q4: LLM-assisted labelling and how to validate it

### Takeaway
LLMs can match or beat crowd workers on some text annotation tasks, but performance varies by task, and outputs are non-deterministic. The literature's consistent recommendation is to validate against a human-labelled sample using agreement metrics before trusting the labels.

### Cited Findings
- Gilardi, Alizadeh and Kubli (PNAS 2023) used 6,183 tweets and news articles across relevance, stance, topic and frame tasks. They found ChatGPT zero-shot accuracy exceeded crowd workers by about 25 percentage points on average. Intercoder agreement was about 56% for MTurk, 79% for trained annotators, 91% for ChatGPT at temperature 1 and 97% at temperature 0.2. Cost was under $0.003 per annotation, about 30x cheaper than MTurk — [PNAS](https://www.pnas.org/doi/pdf/10.1073/pnas.2305016120); [PMC copy](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10372638/) (SNIPPET-ONLY; full text not fetched)
- Pangakis, Wolken and Fasching, "Automated Annotation with Generative AI Requires Validation". Any LLM annotation process must validate against human-generated labels, and the paper outlines a workflow for doing so — [arXiv 2306.00176](https://arxiv.org/pdf/2306.00176) (SNIPPET-ONLY; author names from background knowledge, UNVERIFIED)
- Reiss, "Testing the Reliability of ChatGPT for Text Annotation and Classification: A Cautionary Remark". ChatGPT is non-deterministic, consistency can fall below scientific reliability thresholds, and pooling repeated runs (majority vote) improves reliability — [arXiv 2304.11085](https://arxiv.org/pdf/2304.11085) (SNIPPET-ONLY)
- Törnberg warns that LLMs' apparent simplicity can mislead and that they are prone to bias and unreliable results — referenced in [Frontiers in Social Psychology 2025](https://www.frontiersin.org/journals/social-psychology/articles/10.3389/frsps.2025.1460277/full) (SNIPPET-ONLY)
- A framework for consistency of LLM binary classification covers sample-size planning, prompt design and reliability assessment — [arXiv 2505.14918](https://arxiv.org/html/2505.14918) / [Taylor & Francis 2026](https://www.tandfonline.com/doi/full/10.1080/2573234X.2026.2652281) (SNIPPET-ONLY)
- Tooling: Argilla is built for collecting human feedback, including reviewing model-suggested labels ([Argilla README](https://raw.githubusercontent.com/argilla-io/argilla/develop/README.md)). distilabel (Apache-2.0, 1.5.3) is Argilla's synthetic-data and LLM-labelling pipeline library ([PyPI](https://pypi.org/pypi/distilabel/json)). cleanlab (Apache-2.0, 2.9.0) is for finding label errors ([PyPI](https://pypi.org/pypi/cleanlab/json)).

### Inferences
Suggested validation recipe, synthesised from the sources above:
1. Have humans label a random gold sample, independently of the LLM, with two annotators on an overlap.
2. Compute LLM-vs-human accuracy and Cohen's kappa (or Krippendorff's alpha), plus human-vs-human kappa as the ceiling.
3. Run the LLM at low temperature, or several times with a majority vote (per Reiss), and report run-to-run agreement.
4. Spot-check disagreements and low-confidence items.
5. Never let the LLM label the test set alone.

Judges will likely ask "how do you know the labels are right?" Showing a kappa against a human sample answers that question.

### Gaps
- Could not access full texts (arXiv blocked), so I did not confirm specific recommended validation sample sizes or kappa thresholds. No numbers are invented here.

## Q5: Synthetic data, augmentation and pitfalls

### Takeaway
Synthetic data (Faker for schema-valid fake records, SDV for statistically fitted tabular data, LLM-generated text, and augmentation libraries) fills gaps quickly. The main pitfalls are mismatch with real distributions, test-into-train leakage (e.g. synthesising from data that includes test rows, or LLM paraphrases of test items), collapse of distribution tails under recursive generation, and judges' scepticism.

### Cited Findings
- Model collapse: "indiscriminate use of model-generated content in training causes irreversible defects in the resulting models, in which tails of the original content distribution disappear" (Shumailov et al., Nature 631:755–759, 2024) — [Nature](https://www.nature.com/articles/s41586-024-07566-y) (SNIPPET-ONLY). A follow-up note debates how universal the effect is — [arXiv 2410.12954](https://arxiv.org/html/2410.12954v2) (SNIPPET-ONLY)
- SDV ships a quality report scoring column shapes and pairwise trends of synthetic data against real data. These are fidelity checks, not proof of downstream utility — [SDV README](https://raw.githubusercontent.com/sdv-dev/SDV/main/README.md)
- Faker produces format-realistic values (names, addresses) by provider, not data drawn from a real-world distribution — [Faker README](https://raw.githubusercontent.com/joke2k/faker/master/README.rst)
- Augmentation libraries are all permissively licensed: nlpaug for text (MIT, 1.1.11), albumentations for images (MIT, 2.0.8) and AugLy for multimodal (MIT, 1.0.0) — PyPI JSON ([nlpaug](https://pypi.org/pypi/nlpaug/json), [albumentations](https://pypi.org/pypi/albumentations/json), [augly](https://pypi.org/pypi/augly/json))

### Inferences
- Split real data into train and test before fitting any synthesizer or generating any augmentations. Fit SDV or prompt LLMs only on the train split, and deduplicate synthetic items against the test set (near-duplicates too).
- Always evaluate on real data, even if it is small. A model trained and tested on synthetic data mostly measures how well it learned the generator.
- Be upfront in the pitch: state what fraction of the data is synthetic, how it was made, and show real-data metrics. This pre-empts judge scepticism, which is my inference since no source on judges was found.
- Faker suits demo and UI data and pipeline testing, not training a model expected to generalise.

### Gaps
- Found no published source on hackathon judges' attitudes to synthetic data.
- Did not find quantified guidance on what ratio of synthetic to real data is safe.

## Q6: Scraping legality basics (not legal advice)

### Takeaway
robots.txt is a voluntary crawler convention, not an access-control mechanism. Ignoring it is still bad practice, and Terms of Service can be enforced as a contract: hiQ won its Computer Fraud and Abuse Act (CFAA) arguments but lost on breach of contract.

### Cited Findings
- RFC 9309 (Robots Exclusion Protocol) states that "These rules are not a form of access authorization." It is honour-based with no enforcement mechanism — [RFC 9309 (IETF datatracker)](https://datatracker.ietf.org/doc/html/rfc9309) (SNIPPET-ONLY; direct fetch blocked)
- hiQ v. LinkedIn: hiQ prevailed on the CFAA at the Ninth Circuit (2019 and 2022). In November 2022 the N.D. Cal. district court held that LinkedIn's user-agreement anti-scraping provisions were enforceable in a breach-of-contract claim. The December 2022 settlement included a permanent injunction, $500,000 in damages and deletion of scraped data and code — [ZwillGen](https://www.zwillgen.com/alternative-data/hiq-v-linkedin-wrapped-up-web-scraping-lessons-learned/); [Privacy World](https://www.privacyworld.blog/2022/12/linkedins-data-scraping-battle-with-hiq-labs-ends-with-proposed-judgment/); [9th Cir. opinion on Justia](https://law.justia.com/cases/federal/appellate-courts/ca9/17-16783/17-16783-2022-04-18.html) (SNIPPET-ONLY)
- A paper on the legal liabilities of robots.txt exists — [arXiv 2503.06035](https://arxiv.org/pdf/2503.06035) (SNIPPET-ONLY, not read)

### Inferences
Practical hackathon checklist (not legal advice):
- Prefer official APIs or published datasets.
- Respect robots.txt and rate limits.
- Read the ToS, especially if you log in or click "agree".
- Avoid personal data, since privacy law such as GDPR may apply. This is UNVERIFIED and was not researched here.
- Don't redistribute scraped content in your public repo.
- Note the source in the README.

### Gaps
- US-centric sources only. EU/UK database rights, GDPR and text-and-data-mining exceptions were not researched.
- No 2025–2026 case law was checked.
