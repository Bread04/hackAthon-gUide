# 🗂️ Getting Data When None Is Provided

<!-- markdownlint-disable MD013 -->

> Public datasets and their licences, fast labelling, weak supervision, LLM-assisted labels (and how to validate them), synthetic data and scraping basics. **Not legal advice.** Personal data: [`privacy-and-pii.md`](../../01-hackathon-playbook/docs/privacy-and-pii.md).
>
> 📚 Source research: [`technical-ml-gaps-2026-10-03`](../../_research/technical-ml-gaps-2026-10-03/research.md). **Label key used throughout.** **SNIPPET-ONLY** means the claim comes from a search-result snippet because the primary page could not be fetched; check the source before relying on it. **UNVERIFIED** means it comes from general knowledge or an unconfirmed attribution. **RE-CHECK** marks prices, quotas and limits that change often; confirm them on the day. **Vendor-reported** or **author-reported** marks numbers from the people selling or publishing the tool. Every code block is labelled **not run by us**: each is either quoted from the cited source or assembled from cited calls, and none was executed. All version and licence tables are dated **2026-10-03** and were taken from PyPI or npm JSON on that date. Nothing here is legal advice.

## 5. Getting data with no dataset provided: hold out a human-labelled test set first

### Plain-English explanation

Many software hackathons hand you a problem but no data. You have six ways to get some:

1. **Find a public dataset.** Sources include Kaggle, the Hugging Face Hub, UCI, OpenML, government portals and Google Dataset Search.
2. **Label your own** in a labelling tool.
3. **Write rules that label for you** (weak supervision), then combine their noisy votes.
4. **Ask an LLM to label.**
5. **Generate synthetic data.**
6. **Scrape the web.**

One rule ties all six together. **Before you use any shortcut, hand-label a small random sample yourself and set it aside as the test set.** That sample is how you prove the shortcut works. It answers the judge's inevitable question, "how do you know the labels are right?" Pangakis et al.'s paper title states the principle: "Automated Annotation with Generative AI **Requires Validation**" (**SNIPPET-ONLY**; author names **UNVERIFIED**, [arXiv 2306.00176](https://arxiv.org/pdf/2306.00176)).

### Decision table: where data can come from

| Route | When | Tooling | Main caveat |
|---|---|---|---|
| Public dataset | A close-enough dataset exists | Kaggle CLI, HF `datasets`, `ucimlrepo`, `openml`, Data.gov, Google Dataset Search | Licence per dataset; re-uploads may lack rights to relicense |
| Hand labelling | Fewer than a few hundred items needed; any modality | Label Studio (mixed), doccano (text), CVAT (boxes, video), Argilla (needs a server) | Time: plan 100–300 items first |
| Weak supervision | Domain rules are expressible as code (keywords, regexes, lookups) | Snorkel (stale since Feb 2024) | Label quality unknown until checked against gold |
| Active learning | Big unlabelled pool, small labelling budget | modAL (stale), small-text | Setup time; needs a decent initial model |
| LLM labelling | Text tasks with clear categories | Any LLM API; distilabel; review in Argilla | Non-deterministic; must validate with kappa |
| Synthetic | Demo or UI data; filling rare classes; schema testing | Faker (format-valid), SDV (fitted tabular, **BUSL**), LLM text, augmentation (nlpaug, albumentations, AugLy) | Distribution mismatch; leakage; judge scepticism |
| Scraping | No API or dataset exists, and ToS allow it | requests/BeautifulSoup (not researched) | ToS enforceable as contract; robots.txt is etiquette, not permission |

The sources are cited in the subsections below.

**Public sources and their licences.**

- **Kaggle** shows a licence on every dataset page (**SNIPPET-ONLY**, [secondary](https://labelyourdata.com/articles/machine-learning/kaggle-datasets)). Competition data is stricter: "Participants must not use data other than the Data" unless the rules say otherwise (**SNIPPET-ONLY**, [example rules](https://www.kaggle.com/competitions/data-science-and-ai/rules)).
- **Papers with Code** was sunsetted on 24 July 2025 and now redirects to Hugging Face Trending Papers. Its historical data survives in the paperswithcode-data GitHub repo (**SNIPPET-ONLY**, [Coursera](https://www.coursera.org/articles/papers-with-code)).
- **Google Dataset Search** indexes pages marked up with schema.org `Dataset` or DCAT metadata, so the licence is whatever the hosting site states (**SNIPPET-ONLY**, [Google](https://developers.google.com/search/docs/appearance/structured-data/dataset)).
- **Data.gov** harvests "over 1000 different sources". A dataset count seen in a snippet was not confirmed (**SNIPPET-ONLY**, [catalog](https://catalog.data.gov/)).
- **Hugging Face** dataset-card licences are declared by uploaders, and some datasets are gated. Both points are **UNVERIFIED** because huggingface.co was blocked.
- **UCI**'s default licence is **UNVERIFIED**.

A practical licence check (our inference):

1. Read the dataset page's licence.
2. Trace it back to the upstream source, because re-uploads may not have the right to relicense.
3. Record the URL and licence in the README for the judges.

**How much to label.** No authoritative universal number exists. The available evidence points to diminishing returns after a few hundred examples for simple text classification:

- One study says 300 examples can give high accuracy, and that more "rarely substantially increases" performance. The quote's attribution to this exact paper is unconfirmed (**SNIPPET-ONLY**, [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0747563213002823)).
- A single-task blog experiment found going from 2 to 5 examples per class gained 15 points, and from 5 to 10 gained only 5 more (**SNIPPET-ONLY**, [TDS](https://towardsdatascience.com/how-many-labeled-examples-does-a-text-classifier-actually-need-i-measured-it/)).
- In few-shot settings, "simple baseline classifiers can get surprisingly close to state-of-the-art" (**SNIPPET-ONLY**, [arXiv 2101.12073](https://arxiv.org/pdf/2101.12073)).

Teach the method rather than a magic number:

1. Label 100–300 items.
2. Plot a learning curve at 50, 100, 200 and 400 labels, using `LearningCurveDisplay` from section 1.
3. Stop when the curve flattens.

Combine this with SetFit or frozen embeddings from section 3. No guidance for images was found.

**LLM labelling: promising, but it needs validating.**

- **Promising (SNIPPET-ONLY).** Gilardi, Alizadeh and Kubli (PNAS 2023) tested 6,183 tweets and news articles. ChatGPT zero-shot beat crowd workers by about 25 points on average. Intercoder agreement was about 56% for MTurk, 79% for trained annotators, 91% for ChatGPT at temperature 1 and 97% at temperature 0.2. Cost was under $0.003 per annotation ([PNAS](https://www.pnas.org/doi/pdf/10.1073/pnas.2305016120)).
- **But inconsistent.** Reiss finds ChatGPT non-deterministic enough that consistency can fall below scientific reliability thresholds, and that pooling repeated runs by majority vote helps (**SNIPPET-ONLY**, [arXiv 2304.11085](https://arxiv.org/pdf/2304.11085)).
- **And biased.** Törnberg warns of bias and unreliable results (**SNIPPET-ONLY**, via [Frontiers 2025](https://www.frontiersin.org/journals/social-psychology/articles/10.3389/frsps.2025.1460277/full)).
- **The literature's consistent recommendation** is to validate against human labels with agreement statistics. No specific sample sizes or kappa thresholds were confirmed, and none are invented here.

A validation recipe synthesised from these sources:

1. Have humans label a random gold sample independently, with two annotators overlapping on part of it.
2. Compute LLM-vs-human accuracy and Cohen's kappa (`cohen_kappa_score`), plus human-vs-human kappa as the ceiling.
3. Run the LLM at low temperature, or several times with a majority vote, and report run-to-run agreement.
4. Review the disagreements by hand.
5. **Never let the LLM alone label the test set.**

**Synthetic data.**

- **Faker** produces format-realistic values such as names and addresses. They are not drawn from any real distribution, so Faker is for demos and pipeline tests, not for training a model meant to generalise ([Faker README](https://raw.githubusercontent.com/joke2k/faker/master/README.rst)).
- **SDV** fits tabular data and reports "Column Shapes" and "Column Pair Trends" quality scores. These measure fidelity, not downstream usefulness ([SDV README](https://raw.githubusercontent.com/sdv-dev/SDV/main/README.md)).
- **Model collapse (contested scope).** Shumailov et al. (Nature 2024) warn that "indiscriminate use of model-generated content in training causes irreversible defects", in which the tails of the distribution disappear (**SNIPPET-ONLY**, [Nature](https://www.nature.com/articles/s41586-024-07566-y)). A follow-up questions how universal that is (**SNIPPET-ONLY**, [arXiv 2410.12954](https://arxiv.org/html/2410.12954v2)).
- **Three rules (our inference):**
  - Split before you synthesise: fit generators on the train split only, and deduplicate synthetic items against test.
  - Always evaluate on real data.
  - State the synthetic fraction in the pitch.
- No source on what ratio of synthetic to real data is safe was found.

**Scraping basics (not legal advice; US-centric).**

- RFC 9309 says robots.txt rules "are not a form of access authorization". They are an honour-based convention (**SNIPPET-ONLY**, [RFC 9309](https://datatracker.ietf.org/doc/html/rfc9309)).
- Terms of service bite. In *hiQ v. LinkedIn*, hiQ won its Computer Fraud and Abuse Act (CFAA) arguments at the Ninth Circuit, but the district court held LinkedIn's anti-scraping user agreement enforceable as a contract. The December 2022 settlement included a permanent injunction, $500,000 and deletion of the scraped data (**SNIPPET-ONLY**, [ZwillGen](https://www.zwillgen.com/alternative-data/hiq-v-linkedin-wrapped-up-web-scraping-lessons-learned/); [Ninth Circuit opinion](https://law.justia.com/cases/federal/appellate-courts/ca9/17-16783/17-16783-2022-04-18.html)).
- Checklist (our inference):
  - Prefer official APIs and published datasets.
  - Respect robots.txt and rate limits.
  - Read the ToS, especially behind a login.
  - Avoid personal data. GDPR exposure is **UNVERIFIED** here.
  - Do not commit scraped content to a public repo.
  - Cite the source in the README.
- EU/UK database rights and text-and-data-mining exceptions were not researched.

### Minimal code (not run by us)

Public data loaders, quoted from the [ucimlrepo README](https://raw.githubusercontent.com/uci-ml-repo/ucimlrepo/main/README.md) and the [Argilla README](https://raw.githubusercontent.com/argilla-io/argilla/develop/README.md). The OpenML and Kaggle lines are **UNVERIFIED**. Not run by us.

```python
# not run by us
from ucimlrepo import fetch_ucirepo
heart = fetch_ucirepo(id=45); X, y = heart.data.features, heart.data.targets
from datasets import load_dataset
ds = load_dataset("imdb", split="train[:100]")
# UNVERIFIED: openml.datasets.get_dataset(<id>).get_data(target=...)
# UNVERIFIED: kaggle datasets download -d <owner>/<slug>   (needs API token)
```

Labelling tool start-up commands, quoted from the [Label Studio README](https://raw.githubusercontent.com/HumanSignal/label-studio/develop/README.md). The tool uses SQLite by default. Not run by us.

```bash
# not run by us — quoted
pip install label-studio && label-studio            # http://localhost:8080
docker run -it -p 8080:8080 -v $(pwd)/mydata:/label-studio/data heartexlabs/label-studio:latest
```

Active learning, quoted from the [modAL README](https://raw.githubusercontent.com/modAL-python/modAL/master/README.md). Not run by us.

```python
# not run by us — quoted
from modAL.models import ActiveLearner
from sklearn.ensemble import RandomForestClassifier
learner = ActiveLearner(estimator=RandomForestClassifier(), X_training=X_training, y_training=y_training)
query_idx, query_inst = learner.query(X_pool)
learner.teach(X_pool[query_idx], y_new)
```

Validating LLM labels against a human gold set. Assembled from scikit-learn functions; **tested** on toy labels (runs).

```python
# tested 2026-10-03 on toy labels — assembled
from sklearn.metrics import cohen_kappa_score, accuracy_score
from collections import Counter
# gold: human labels on a random sample; llm_runs: list of 3 label lists from repeated LLM runs
llm_vote = [Counter(v).most_common(1)[0][0] for v in zip(*llm_runs)]   # majority vote (Reiss)
print("LLM vs human  acc", accuracy_score(gold, llm_vote), "kappa", cohen_kappa_score(gold, llm_vote))
print("human vs human kappa (ceiling)", cohen_kappa_score(gold_annot_a, gold_annot_b))
print("run-to-run kappa", cohen_kappa_score(llm_runs[0], llm_runs[1]))
```

Synthetic data, quoted from the [SDV README](https://raw.githubusercontent.com/sdv-dev/SDV/main/README.md) and the [Faker README](https://raw.githubusercontent.com/joke2k/faker/master/README.rst). The licence is **BUSL-1.1** for SDV. Not run by us.

```python
# not run by us — quoted (fit on the TRAIN split only)
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.evaluation.single_table import evaluate_quality
synth = GaussianCopulaSynthesizer(metadata)
synth.fit(data=train_df)
fake = synth.sample(num_rows=500)
evaluate_quality(train_df, fake, metadata)
from faker import Faker
fake_person = Faker(); fake_person.name(); fake_person.address()
```

The SDV import paths are from background knowledge, because the README snippet in the notes showed calls but not imports (**UNVERIFIED**).

### Pitfalls

**Labelling the test set with the same shortcut you are testing.** This makes the evaluation circular. Hold out the human gold set first.

**Leakage from synthesis.** Synthesising or paraphrasing from data that includes test rows leaks the test set into training.

**Training and testing on synthetic data.** That mostly measures how well the model learned the generator.

**Stale tools.**

- Snorkel's last release was February 2024, and the team "is now focusing their efforts on Snorkel Flow", its commercial product ([Snorkel README](https://raw.githubusercontent.com/snorkel-team/snorkel/main/README.md)).
- modAL's last release was June 2023, and doccano's July 2023.
- Use an isolated virtual environment and pin versions.
- The Snorkel `@labeling_function` / `LabelModel` pattern is **UNVERIFIED** here.

**Argilla needs a server.** The README calls the HF Spaces deployment "easiest" ([Argilla README](https://raw.githubusercontent.com/argilla-io/argilla/develop/README.md)), but that is now affected by the Spaces paid-plan change in section 4 (our inference; **RE-CHECK**). huggingface.co may also be blocked on some venue networks.

**CVAT self-hosting is heavy.** It needs Docker Compose with several services. CVAT Online's free plan has limits that "vary by plan" (**RE-CHECK**, [CVAT README](https://raw.githubusercontent.com/cvat-ai/cvat/develop/README.md)).

**Setup times are estimates, not measurements** (our judgement from the install paths):

- Under 5 minutes: Faker, ucimlrepo, `datasets`, modAL.
- About 10–20 minutes: Label Studio and doccano.

### Versions and licences (as of 2026-10-03)

| Package | Version | Released | Licence | Python |
|---|---|---|---|---|
| label-studio | 1.23.2 | 2026-09-29 | Apache-2.0 | ≥3.10,<4 |
| doccano | 1.8.4 | 2023-07-20 | MIT | ≥3.8,<4 |
| argilla (client) | 2.8.0 | 2025-03-10 | Apache-2.0 | ≥3.9 |
| cvat-cli | 2.77.0 | — | MIT (core MIT; some assets separately licensed) | ≥3.10 |
| snorkel | 0.10.0 | 2024-02-27 | Apache-2.0 | ≥3.11 |
| modAL-python | 0.4.2.1 | 2023-06-01 | MIT | none declared |
| small-text | 1.4.1 (2.0.0.dev4 pre-release) | 2024-08-18 | MIT | ≥3.7 (README says 3.10+) |
| Faker | 40.40.0 | 2026-09-29 | MIT | ≥3.10 |
| **sdv** | 1.38.5 | 2026-09-28 | **BUSL-1.1** | ≥3.9,<3.15 |
| distilabel | 1.5.3 | 2025-01-28 | Apache-2.0 | ≥3.9 |
| cleanlab | 2.9.0 | — | Apache-2.0 | ≥3.10 |
| nlpaug | 1.1.11 | — | MIT | ≥3.7 |
| albumentations | 2.0.8 | — | MIT | ≥3.9 |
| augly | 1.0.0 | — | MIT | ≥3.6 |
| datasets | 5.0.1 | — | Apache-2.0 | ≥3.10 |
| openml | 0.15.1 | — | BSD-3-Clause | ≥3.8 |
| ucimlrepo | 0.0.7 | — | MIT | ≥3.7 |
| kaggle | 2.2.4 | — | Apache-2.0 | ≥3.11 |

All versions are from PyPI JSON, e.g. [sdv](https://pypi.org/pypi/sdv/json) and [label-studio](https://pypi.org/pypi/label-studio/json). Release dates are given only where they were fetched.

**SDV licence warning.** SDV's Business Source License 1.1 "is not an Open Source license". Its Additional Use Grant permits use provided you do not offer "a Synthetic Data Service", meaning a commercial offering that gives third parties synthetic-data-creation functionality. Each release converts to MIT "four years from release date" ([SDV LICENSE](https://raw.githubusercontent.com/sdv-dev/SDV/main/LICENSE)). A hackathon prototype is fine under the grant. A team pitching a commercial synthetic-data product must flag it (our reading; not legal advice).

### Learning resources

**Loaders and labelling tools.** The [ucimlrepo README](https://raw.githubusercontent.com/uci-ml-repo/ucimlrepo/main/README.md), [Label Studio README](https://raw.githubusercontent.com/HumanSignal/label-studio/develop/README.md) and [small-text README](https://raw.githubusercontent.com/webis-de/small-text/main/README.md).

**Weak supervision.** The Snorkel tutorials linked from the [Snorkel README](https://raw.githubusercontent.com/snorkel-team/snorkel/main/README.md).

**LLM-labelling evidence.** [Gilardi et al., PNAS 2023](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10372638/), [Pangakis et al.](https://arxiv.org/pdf/2306.00176) and [Reiss](https://arxiv.org/pdf/2304.11085) (all **SNIPPET-ONLY**).

**Scraping law.** [RFC 9309](https://datatracker.ietf.org/doc/html/rfc9309) and the [ZwillGen hiQ wrap-up](https://www.zwillgen.com/alternative-data/hiq-v-linkedin-wrapped-up-web-scraping-lessons-learned/).
