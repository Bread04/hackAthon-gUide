# 🧩 ML Inside Your Hackathon App

<!-- markdownlint-disable MD013 -->

> For software hackathons: call an API first, ship a pretrained or ONNX model second, train your own last. Covers Hugging Face in 2026, serving with FastAPI/ONNX, in-browser models, and fine-tuning vs prompting vs RAG. Related: [`model-selection.md`](model-selection.md), [`rag-architecture.md`](rag-architecture.md), [`how-llms-work.md`](how-llms-work.md).
>
> 📚 Source research: [`technical-ml-gaps-2026-10-03`](../../_research/technical-ml-gaps-2026-10-03/research.md). **Label key used throughout.** **SNIPPET-ONLY** means the claim comes from a search-result snippet because the primary page could not be fetched; check the source before relying on it. **UNVERIFIED** means it comes from general knowledge or an unconfirmed attribution. **RE-CHECK** marks prices, quotas and limits that change often; confirm them on the day. **Vendor-reported** or **author-reported** marks numbers from the people selling or publishing the tool. Every code block is labelled **not run by us**: each is either quoted from the cited source or assembled from cited calls, and none was executed. All version and licence tables are dated **2026-10-03** and were taken from PyPI or npm JSON on that date. Nothing here is legal advice.

## 4. ML inside a hackathon app: call an API first, ship ONNX second, train last

### Plain-English explanation

A software hackathon has a different goal from a datathon. You need a feature that works live in front of judges, not a leaderboard score. There are three ways to get ML into an app:

1. **Call a hosted API.** This means an LLM provider, or Hugging Face Inference Providers for open-weights models. It is fastest to build, but it costs per call, adds seconds of latency, and fails if the Wi-Fi does.
2. **Run a pretrained open model yourself.** You can run it on a server, or directly in the user's browser. It is free per call and private, but you handle the hosting.
3. **Train your own.** This is realistic mainly for small tabular or classical-ML problems where you already have data. scikit-learn trains in minutes, and the model exports to a tiny ONNX file.

For LLM features specifically, the order of attempts is:

1. Prompting.
2. Putting the documents in the prompt.
3. Retrieval-augmented generation (RAG).
4. Fine-tuning, last.

Anthropic's guidance is that if a knowledge base is "smaller than 200,000 tokens (about 500 pages of material), you can just include the entire knowledge base in the prompt ... with no need for RAG." Prompt caching cuts latency by more than 2× and cost by up to 90% (**vendor-reported**, [Anthropic contextual retrieval](https://www.anthropic.com/news/contextual-retrieval)). Fine-tuning helps format, tone, consistency, cost and latency. Secondary sources summarise OpenAI's position as "Fine-tuning doesn't make the model smarter" (**SNIPPET-ONLY**, not checked against OpenAI docs, [secondary source](https://theneuralbase.com/openai-fine-tuning/qna/when-to-fine-tune-vs-prompt-engineer/)).

### Decision table: API vs pretrained vs train your own

| Criterion | Hosted API (LLM API / HF Inference Providers) | Pretrained open model (server or browser) | Train your own (sklearn / small net) |
|---|---|---|---|
| Labelled data needed | None | None (zero-shot) or a few hundred for embeddings + kNN/logistic | Yes, clean and relevant |
| Build time | Hours | Hours to a day | A day+ including evaluation |
| Latency | Seconds per call | Fast once warm; browser has a first-load download | Near-instant on CPU (ONNX) |
| Cost at demo | Per call; HF free credits are **$0.10/month (RE-CHECK)** | Free per call; hosting may cost | Free |
| Privacy | Data leaves the device | Browser or on-device keeps data local | Local |
| Demo risk | Rate limits, Wi-Fi, keys | Cold starts, model download size | Lowest |
| Best for | Generation, open-ended text and vision tasks | Classification, embeddings, speech, detection; offline or privacy pitches | Tabular prediction on your own data |

The table synthesises [HF Inference Providers pricing](https://raw.githubusercontent.com/huggingface/hub-docs/main/docs/inference-providers/pricing.md), the [ONNX Runtime Web README](https://raw.githubusercontent.com/microsoft/onnxruntime/main/js/web/README.md) and [scikit-learn model persistence](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/model_persistence.rst). No primary source gives a hackathon-specific table, so the arrangement is ours.

### Decision table: fine-tuning vs prompting vs RAG

| Need | Approach | 24–48 h realism | Source |
|---|---|---|---|
| Behaviour, format, few examples | Prompt + few-shot + structured output | Hours | [Anthropic prompt engineering overview](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview) |
| Answers grounded in <200k tokens of docs | Whole corpus in context + prompt caching | Hours | [Anthropic](https://www.anthropic.com/news/contextual-retrieval) (vendor-reported) |
| Larger corpus | RAG (see the guide's `04-ai-and-rag/docs/rag-architecture.md`) | Half a day for basic | Contextual Retrieval cuts failed retrievals 49%, 67% with reranking (vendor-reported, same source) |
| Consistent style or format, lower cost or latency at scale | LoRA/QLoRA of a 1–8B model (Unsloth free Colab, Axolotl) | Feasible: a few hours of training **after** building hundreds of clean examples | [PEFT](https://raw.githubusercontent.com/huggingface/peft/main/README.md); [Unsloth](https://raw.githubusercontent.com/unslothai/unsloth/main/README.md); [Axolotl](https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/README.md) |
| New knowledge | Not fine-tuning; use context or RAG | — | (secondary sources, SNIPPET-ONLY) |
| Hosted fine-tuning on a vendor API | Check tier gating and price | **RE-CHECK** | OpenAI Cookbook notes GPT-4o mini fine-tuning was gated to tiers 4–5 at time of writing, possibly outdated ([cookbook](https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/How_to_finetune_chat_models.ipynb)) |

Before choosing any of these, follow Anthropic's instruction: have "a clear definition of the success criteria" and "ways to empirically test" ([prompt engineering overview](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview)). In practice that means a 20–50-case evaluation set. Unsloth's "2× faster with 70% less VRAM with no accuracy loss" is a **vendor claim**. Its free notebooks cover Gemma 4 E2B, Qwen3.5 4B, gpt-oss 20B, Llama 3.1 8B and Llama 3.2 1B/3B ([Unsloth README](https://raw.githubusercontent.com/unslothai/unsloth/main/README.md)). Axolotl requires Python ≥3.12 and PyTorch ≥2.13. Its quickstart is `axolotl fetch examples` followed by `axolotl train examples/llama-3/lora-1b.yml` ([Axolotl README](https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/README.md)). The bottleneck is building the dataset, not GPU time (our inference). Serving a fine-tuned open model also brings back GPU hosting and cold-start risk.

### Decision table: serving pattern

| Model | Serve with | Why |
|---|---|---|
| scikit-learn / GBDT, your own | **skl2onnx → onnxruntime inside FastAPI** | No pickle code-execution risk; no sklearn version lock; "much less RAM than Python" ([sklearn persistence](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/model_persistence.rst)) |
| scikit-learn, same environment guaranteed | joblib in FastAPI or Streamlit | Simple, but needs identical versions; "Loading can execute arbitrary code" (same source) |
| PyTorch model | `torch.onnx.export(..., dynamo=True)` → onnxruntime | Dynamo is "the recommended exporter"; TorchScript path "no longer recommended" ([PyTorch ONNX tutorial](https://raw.githubusercontent.com/pytorch/tutorials/main/beginner_source/onnx/export_simple_model_to_onnx_tutorial.py)) |
| HF transformers model, need batching | BentoML (`@bentoml.api(batchable=True)`) or LitServe | Packaging plus batching ([BentoML](https://raw.githubusercontent.com/bentoml/BentoML/main/README.md); [LitServe](https://raw.githubusercontent.com/Lightning-AI/LitServe/main/README.md)) |
| Need a UI and an API in 10 lines | Gradio (`gr.Interface(..., api_name="predict")`) | ([Gradio README](https://raw.githubusercontent.com/gradio-app/gradio/main/README.md)) |
| Need it to run with no server | transformers.js or onnxruntime-web in the browser | Privacy, no hosting; WASM runs all `ai.onnx` and `ai.onnx.ml` ops, so sklearn ONNX runs in-browser ([ORT Web](https://raw.githubusercontent.com/microsoft/onnxruntime/main/js/web/README.md)) |

### Hugging Face in 2026: what changed

There are three ways to use a Hugging Face model:

- **Locally:** `pipeline(task, model)` is the fastest path ([pipeline tutorial](https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/en/pipeline_tutorial.md)).
- **Hosted:** **Inference Providers** exposes "200+ models" through an OpenAI-compatible router at `https://router.huggingface.co/v1`, with "no markup from Hugging Face" ([Inference Providers](https://raw.githubusercontent.com/huggingface/hub-docs/main/docs/inference-providers/index.md)). You need a fine-grained token with the "Make calls to Inference Providers" permission. Monthly credits are **$0.10 free and $2.00 PRO, "subject to change" (RE-CHECK)**, so the free credits will not cover a demo day. A "Custom Provider Key" is billed by the provider ([pricing](https://raw.githubusercontent.com/huggingface/hub-docs/main/docs/inference-providers/pricing.md)).
- **On Spaces:** these terms changed in 2026 and are covered next.

**The Spaces paid-plan change (RE-CHECK).** The current docs say:

> "Static Spaces are free for everyone. Gradio and Docker Spaces run on compute and require a paid plan to create ... Free personal accounts in good standing can still host up to 2 Gradio Spaces running on ZeroGPU."

Free hardware still "go[es] to sleep" when unused, and the 50 GB disk is not persistent ([Spaces overview](https://raw.githubusercontent.com/huggingface/hub-docs/main/docs/hub/spaces-overview.md)). ZeroGPU details ([ZeroGPU docs](https://raw.githubusercontent.com/huggingface/hub-docs/main/docs/hub/spaces-zerogpu.md)):

- It supports the Gradio SDK only.
- Functions get 60 s of GPU by default, adjustable with `@spaces.GPU(duration=120)`.
- Free accounts need a verified email and must be older than 30 days.
- PRO gets 8× the daily quota.

The date the change took effect was not confirmed here. The guide's `tooling-2026-update.md` hosting table cites forum posts placing the Docker restriction around July 2026. **Older tutorials that say "deploy free to a Gradio Space" are now wrong for new free accounts.**

**Licences and safety on the Hub.**

- Hub licence tags include `apache-2.0` and `mit`, the OpenRAIL family, the Llama community licences, `gemma` and `other` ([Hub licences](https://raw.githubusercontent.com/huggingface/hub-docs/main/docs/hub/repositories-licenses.md)). Prefer `apache-2.0` or `mit`.
- Gated models require you to "share contact information", so request access before the event ([gated models](https://raw.githubusercontent.com/huggingface/hub-docs/main/docs/hub/models-gated.md)).
- Prefer `.safetensors` weights. Pickle files allow "arbitrary code execution attacks" ([HF pickle security](https://raw.githubusercontent.com/huggingface/hub-docs/main/docs/hub/security-pickle.md)).

### In-browser models

transformers.js v4 mirrors the Python `pipeline` API ([transformers.js README](https://raw.githubusercontent.com/huggingface/transformers.js/main/README.md)):

- It runs on WASM (CPU) by default, or on WebGPU with `device: 'webgpu'`.
- The default dtype is `q8` on WASM and `fp32` on WebGPU, and `q4` is available.

The transformers.js WebGPU guide states: "As of March 2026, global WebGPU support is around 85% (according to caniuse.com)". Firefox needs a flag, and Safari support is version-dependent ([WebGPU guide](https://raw.githubusercontent.com/huggingface/transformers.js/main/packages/transformers/docs/source/guides/webgpu.md)). ONNX Runtime Web's compatibility table lists WebGPU on Chrome and Edge but not on Safari, iOS or Firefox. It also calls WebGL "in maintenance mode" ([ORT Web](https://raw.githubusercontent.com/microsoft/onnxruntime/main/js/web/README.md)). The two sources may differ in how recent they are. Either way, **make WASM the default and WebGPU an enhancement.**

TensorFlow.js has had no npm release since October 2024 ([npm](https://registry.npmjs.org/@tensorflow/tfjs)). For new projects, prefer transformers.js, onnxruntime-web or MediaPipe (`@mediapipe/tasks-vision` 1.0.1). The MediaPipe web code pattern could not be fetched and is **UNVERIFIED**.

### Minimal code (not run by us)

Train, export to ONNX and serve with FastAPI, assembled from the [skl2onnx README](https://raw.githubusercontent.com/onnx/sklearn-onnx/main/README.md) and the [FastAPI README](https://raw.githubusercontent.com/fastapi/fastapi/master/README.md). **Tested 2026-10-03** (skl2onnx 1.20.0, onnxruntime 1.30.0, fastapi 0.142.2, scikit-learn 1.9.1) with FastAPI's `TestClient`: ONNX probabilities matched scikit-learn's to 4 decimals. Two things we found: `to_onnx` accepts a DataFrame row as the example input, and with `zipmap=False` the outputs are named `label` and `probabilities` (not `output_label` / `output_probability`), so read the names from the session instead of hard-coding them.

```python
# tested 2026-10-03. export.py
import numpy as np
from skl2onnx import to_onnx
onx = to_onnx(model, X_train[:1].astype(np.float32),   # float32 example row
              options={"zipmap": False})                 # probabilities as a plain array
with open("model.onnx", "wb") as f:
    f.write(onx.SerializeToString())
```

```python
# tested 2026-10-03. app.py ; run locally with: uvicorn app:app --reload
import numpy as np, onnxruntime as rt
from fastapi import FastAPI
from pydantic import BaseModel

sess = rt.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])  # load once, at startup
inp = sess.get_inputs()[0].name
out_label, out_proba = [o.name for o in sess.get_outputs()]  # "label", "probabilities"
app = FastAPI()

class Rows(BaseModel):
    rows: list[list[float]]

@app.post("/predict")
def predict(body: Rows):
    X = np.asarray(body.rows, dtype=np.float32)
    label, proba = sess.run([out_label, out_proba], {inp: X})
    return {"label": label.tolist(), "probability": proba[:, 1].round(4).tolist()}
```

PyTorch to ONNX, quoted from the [PyTorch ONNX tutorial](https://raw.githubusercontent.com/pytorch/tutorials/main/beginner_source/onnx/export_simple_model_to_onnx_tutorial.py). It needs `pip install --upgrade onnx onnxscript`. Not run by us.

```python
# not run by us — quoted
example_inputs = (torch.randn(1, 1, 32, 32),)
onnx_program = torch.onnx.export(torch_model, example_inputs, dynamo=True)
```

Hosted open-weights model through HF Inference Providers, quoted from the [HF docs](https://raw.githubusercontent.com/huggingface/hub-docs/main/docs/inference-providers/index.md). Not run by us.

```python
# not run by us — quoted
import os
from openai import OpenAI
client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=os.environ["HF_TOKEN"])
completion = client.chat.completions.create(model="openai/gpt-oss-120b:fastest",
    messages=[{"role": "user", "content": "How many 'G's in 'huggingface'?"}])
```

In-browser model, quoted from the [transformers.js README](https://raw.githubusercontent.com/huggingface/transformers.js/main/README.md) and [dtypes guide](https://raw.githubusercontent.com/huggingface/transformers.js/main/packages/transformers/docs/source/guides/dtypes.md). Not run by us.

```js
// not run by us — quoted
import { pipeline } from '@huggingface/transformers';
const pipe = await pipeline('sentiment-analysis');            // WASM, q8 by default
const out = await pipe('I love transformers!');
const gen = await pipeline("text-generation", "onnx-community/Qwen2.5-0.5B-Instruct",
                           { dtype: "q4", device: "webgpu" });
```

Gradio UI plus API, quoted from the [Gradio README](https://raw.githubusercontent.com/gradio-app/gradio/main/README.md). Not run by us.

```python
# not run by us — quoted
import gradio as gr
def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)
demo = gr.Interface(fn=greet, inputs=["text", "slider"], outputs=["text"], api_name="predict")
demo.launch()
```

### Pitfalls

**Pickle is not safe to load from untrusted sources.** Python's docs say "The `pickle` module **is not secure** ... malicious pickle data ... execute arbitrary code during unpickling" ([Python pickle docs](https://raw.githubusercontent.com/python/cpython/main/Doc/library/pickle.rst)). Never accept user-uploaded model files. `skops.io` is a safer format for scikit-learn models ([sklearn persistence](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/model_persistence.rst)).

**Version mismatches.** A model saved with one scikit-learn version and loaded with another raises `InconsistentVersionWarning`. None of the pickle-based methods support loading across versions (same source). Pin the training version in the serving requirements, or ship ONNX.

**Not every model converts to ONNX.** Some models and pipelines are unsupported (same source), and skl2onnx needs float32 inputs. The last supported opset is 21 ([skl2onnx](https://raw.githubusercontent.com/onnx/sklearn-onnx/main/README.md)).

**Cold starts.**

- Free hosts sleep. Render's free tier reportedly spins down after 15 minutes idle (**SNIPPET-ONLY, RE-CHECK**).
- Reported wake times of 30–90 s are anecdotal and **UNVERIFIED** ([GitHub issue](https://github.com/Sponti-App/Sponti/issues/125)).
- Mitigations (our inference):
  - Load the model once at startup, never per request.
  - Bake weights into the image at build time.
  - Hit the endpoint about 5 minutes before judging.
  - Keep a recorded fallback.
- The guide's `tooling-2026-update.md` has the free-host table.

**Browser model size.** The model download is the main cost. Our inference:

- Choose models in the tens to low hundreds of MB.
- Use `q4` or `q8`.
- Load in a Web Worker so the UI does not freeze.
- Show a progress bar.
- Preload before the demo.

No primary source on per-tab memory limits was found.

**Batching does not always help.** In `pipeline`, batching "may improve speed ... but it isn't guaranteed" ([pipeline tutorial](https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/en/pipeline_tutorial.md)).

### Versions and licences (as of 2026-10-03; RE-CHECK on the day)

| Package | Version | Released | Licence |
|---|---|---|---|
| fastapi | 0.142.2 | 2026-09-30 | not in PyPI `license` field (not checked) |
| onnxruntime | 1.30.0 | 2026-09-10 | MIT |
| onnx | 1.23.1 | — | not checked |
| skl2onnx | 1.20.0 | 2026-01-30 | Apache-2.0 |
| bentoml | 1.4.39 | 2026-05-07 | Apache-2.0 |
| litserve | 0.2.19 | 2026-09-09 | Apache-2.0 |
| gradio | 6.29.1 | 2026-10-02 | not in PyPI `license` field (not checked) |
| streamlit | 1.65.0 | 2026-10-02 | not in PyPI `license` field (not checked) |
| huggingface_hub | 2.1.1 | — | not checked |
| unsloth | 2026.9.14 | 2026-10-01 | not in PyPI `license` field (not checked) |
| axolotl | 0.20.0 | 2026-09-30 | not in PyPI `license` field (not checked) |
| peft | 0.21.2 | 2026-10-01 | Apache |
| npm @huggingface/transformers | 4.3.0 | 2026-09-16 | not checked |
| npm onnxruntime-web | 1.30.0 | 2026-09-14 | not checked (ORT is MIT) |
| npm @mediapipe/tasks-vision | 1.0.1 | 2026-07-31 | not checked |
| npm @tensorflow/tfjs | 4.22.0 | **2024-10-21 (stale)** | not checked |

Versions are from the PyPI JSON API (e.g. [fastapi](https://pypi.org/pypi/fastapi/json), [onnxruntime](https://pypi.org/pypi/onnxruntime/json)) and the npm registry (e.g. [@huggingface/transformers](https://registry.npmjs.org/@huggingface/transformers)).

### Learning resources

**Python serving.**

- transformers [pipeline tutorial](https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/en/pipeline_tutorial.md).
- scikit-learn [model persistence](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/model_persistence.rst).
- PyTorch [ONNX export tutorial](https://raw.githubusercontent.com/pytorch/tutorials/main/beginner_source/onnx/export_simple_model_to_onnx_tutorial.py).

**Hugging Face hosting.** HF [Inference Providers](https://raw.githubusercontent.com/huggingface/hub-docs/main/docs/inference-providers/index.md) and [Spaces overview](https://raw.githubusercontent.com/huggingface/hub-docs/main/docs/hub/spaces-overview.md).

**In-browser.** [transformers.js README](https://raw.githubusercontent.com/huggingface/transformers.js/main/README.md) and [WebGPU guide](https://raw.githubusercontent.com/huggingface/transformers.js/main/packages/transformers/docs/source/guides/webgpu.md).

**LLM strategy.**

- Anthropic [contextual retrieval](https://www.anthropic.com/news/contextual-retrieval) and [prompt engineering overview](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview).
- OpenAI Cookbook [fine-tuning chat models](https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/How_to_finetune_chat_models.ipynb).
- [Unsloth notebooks](https://raw.githubusercontent.com/unslothai/unsloth/main/README.md).
