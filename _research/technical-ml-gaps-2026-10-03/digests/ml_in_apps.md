# Putting ML into a hackathon app (API vs open model vs train; HF; serving; in-browser; fine-tune vs prompt vs RAG)

Research date: 2026-10-03. Versions pulled live from PyPI JSON (`https://pypi.org/pypi/<pkg>/json`) and npm registry (`https://registry.npmjs.org/<pkg>`) on that date. Several vendor doc domains (platform.openai.com, docs.python.org, docs.pytorch.org, render.com, ai.google.dev) were blocked by the egress proxy, so GitHub raw sources were used instead where they exist.

## Latest versions (as of 2026-10-03; RE-CHECK on the day)

### Takeaway
Everything in the stack shipped a release in the last month except skl2onnx (Jan 2026), bentoml (May 2026) and @tensorflow/tfjs (Oct 2024, which looks stale). Pin versions in requirements.txt.

### Cited Findings
- Python (PyPI JSON, version + upload date of the first file):
  - fastapi 0.142.2 (2026-09-30) — https://pypi.org/pypi/fastapi/json
  - onnxruntime 1.30.0 (2026-09-10, MIT) — https://pypi.org/pypi/onnxruntime/json
  - skl2onnx 1.20.0 (2026-01-30, Apache-2.0) — https://pypi.org/pypi/skl2onnx/json
  - bentoml 1.4.39 (2026-05-07, Apache-2.0) — https://pypi.org/pypi/bentoml/json
  - litserve 0.2.19 (2026-09-09, Apache-2.0) — https://pypi.org/pypi/litserve/json
  - unsloth 2026.9.14 (2026-10-01) — https://pypi.org/pypi/unsloth/json
  - axolotl 0.20.0 (2026-09-30) — https://pypi.org/pypi/axolotl/json
  - peft 0.21.2 (2026-10-01) — https://pypi.org/pypi/peft/json
  - Context: transformers 5.18.0 (2026-09-30), gradio 6.29.1 (2026-10-02), streamlit 1.65.0 (2026-10-02), scikit-learn 1.9.1 (2026-09-10), onnx 1.23.1, torch 2.14.1 (2026-09-30), huggingface_hub 2.1.1, joblib 1.6.0 — same PyPI JSON endpoint per package
- npm (dist-tags.latest + publish time):
  - @huggingface/transformers 4.3.0 (2026-09-16) — https://registry.npmjs.org/@huggingface/transformers
  - onnxruntime-web 1.30.0 (2026-09-14) — https://registry.npmjs.org/onnxruntime-web
  - @mediapipe/tasks-vision 1.0.1 (2026-07-31) — https://registry.npmjs.org/@mediapipe/tasks-vision
  - @tensorflow/tfjs 4.22.0 (2024-10-21; no release in about 2 years) — https://registry.npmjs.org/@tensorflow/tfjs

### Inferences
- transformers is on major version 5 and transformers.js on v4, so older tutorials (v4/v2 era) may show deprecated APIs.
- TensorFlow.js has had no npm release since Oct 2024. For new browser projects, transformers.js, onnxruntime-web or MediaPipe are the more actively maintained options.

### Gaps
- Licences of fastapi, unsloth, axolotl, gradio and streamlit were empty in the PyPI `license` field and were not checked elsewhere.

## (1) Hosted API vs pretrained open model vs train your own: criteria

### Takeaway
For a 24–48h build, the default is a hosted API (an LLM API, or HF Inference Providers for open-weights models) for anything generative. Use a small pretrained open model, ideally in the browser or on CPU, when privacy, offline use or zero per-call cost matters. Train your own only for small tabular or classical-ML problems where the team has the data (scikit-learn trains in minutes and exports to ONNX). No primary source gives a hackathon-specific decision table. The criteria below combine documented facts with the inferences that follow from them.

### Cited Findings
- **Cost/latency, hosted open-weights:** HF Inference Providers gives "200+ models from leading AI inference providers", pay-as-you-go, "no markup from Hugging Face". Monthly credits are **$0.10 for free users ("subject to change")** and $2.00 for PRO (RE-CHECK) — https://raw.githubusercontent.com/huggingface/hub-docs/main/docs/inference-providers/pricing.md
- It exposes an OpenAI-compatible endpoint: `base_url="https://router.huggingface.co/v1"` with `api_key=os.environ["HF_TOKEN"]`, model e.g. `"openai/gpt-oss-120b:fastest"`. The token must be fine-grained with "Make calls to Inference Providers" permission — https://raw.githubusercontent.com/huggingface/hub-docs/main/docs/inference-providers/index.md
- **Privacy/latency, local:** the ONNX Runtime Web README says in-browser scoring offers "reducing server-client communication and protecting user privacy, as well as offering install-free and cross-platform in-browser ML experience" — https://raw.githubusercontent.com/microsoft/onnxruntime/main/js/web/README.md
- **Data:** if the knowledge base is "smaller than 200,000 tokens (about 500 pages of material), you can just include the entire knowledge base in the prompt ... with no need for RAG". Prompt caching cuts "latency by > 2x and costs by up to 90%" — https://www.anthropic.com/news/contextual-retrieval
- **Fine-tuning does not add capability (SNIPPET-ONLY, secondary sources):** "Fine-tuning doesn't make the model smarter; it makes it cheaper and faster for a specific task" — search snippet citing https://theneuralbase.com/openai-fine-tuning/qna/when-to-fine-tune-vs-prompt-engineer/ (not primary; UNVERIFIED against OpenAI docs)

### Inferences
- Decision criteria for the guide:
  - **Data:** no labelled data means API or pretrained model. A small clean tabular dataset suits training a sklearn model. Text or images with fewer than a few hundred labels suit a pretrained model with zero-shot classification or embeddings plus kNN.
  - **Latency:** hosted LLM calls take seconds per call. A small ONNX/sklearn model on CPU responds almost instantly once warm. Free hosts add cold starts (see section 3).
  - **Cost:** free HF credits ($0.10/month) will not cover a demo day. Teams should bring their own API key or credits from sponsors.
  - **Privacy:** browser or on-device inference keeps user data off the network.
  - **Demo risk:** every network dependency (API rate limits, cold start, Wi-Fi) is a failure point. Keep a cached or recorded fallback, and pre-warm the server before judging.

### Gaps
- No official OpenAI page could be fetched (platform.openai.com blocked). The OpenAI "when to fine-tune" guidance above is SNIPPET-ONLY from secondary sites.

## (2) Using Hugging Face models: pipelines, Inference Providers/Endpoints, Spaces, licences

### Takeaway
`pipeline(task, model)` is the fastest local path. Inference Providers is the fastest hosted path through an OpenAI-compatible router. Spaces is now restricted: Gradio/Docker Spaces on CPU need a paid plan, except that free accounts can host up to 2 ZeroGPU Gradio Spaces. Check each model's licence and gating before building on it.

### Cited Findings
- Minimal pipeline (from the transformers docs):
  ```py
  from transformers import pipeline
  pipeline = pipeline(task="text-generation", model="google/gemma-2-2b")
  pipeline("the secret to baking a really good cake is ")
  ```
  The same docs show `device_map="auto"` via Accelerate and a `batch_size=` argument; batching "may improve speed, especially on a GPU, but it isn't guaranteed" — https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/en/pipeline_tutorial.md
- Inference Providers OpenAI-compatible snippet:
  ```py
  from openai import OpenAI
  client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=os.environ["HF_TOKEN"])
  completion = client.chat.completions.create(model="openai/gpt-oss-120b:fastest",
      messages=[{"role": "user", "content": "How many 'G's in 'huggingface'?"}])
  ```
  `hf models ls --warm` lists every model served by at least one provider — https://raw.githubusercontent.com/huggingface/hub-docs/main/docs/inference-providers/index.md
- Billing: "Routed by Hugging Face" uses HF credits. "Custom Provider Key" is billed by the provider, and credits do not apply. PRO/Team credits also cover Inference Endpoints, Spaces hardware and Jobs — https://raw.githubusercontent.com/huggingface/hub-docs/main/docs/inference-providers/pricing.md
- **Spaces (RE-CHECK):** "Static Spaces are free for everyone. Gradio and Docker Spaces run on compute and require a paid plan to create ... Free personal accounts in good standing can still host up to 2 Gradio Spaces running on ZeroGPU." The default is "16GB RAM, 2 CPU cores and 50GB of (not persistent) disk". CPU Basic is listed as FREE, T4 small at $0.40/h. "On free hardware, your Space will 'go to sleep' ... if unused" — https://raw.githubusercontent.com/huggingface/hub-docs/main/docs/hub/spaces-overview.md
- **ZeroGPU (RE-CHECK):**
  - Gradio SDK only (Gradio 4+). `large` is "Half NVIDIA RTX Pro 6000 Blackwell" (48GB). `xlarge` is the full card (96GB) at 2× quota.
  - The default function GPU runtime is 60 s, set with `@spaces.GPU(duration=120)`.
  - Free accounts must have a verified email and be older than 30 days. PRO gets 8× daily quota.
  - Source: https://raw.githubusercontent.com/huggingface/hub-docs/main/docs/hub/spaces-zerogpu.md
- **Licences:** Hub licence identifiers include `apache-2.0`, `mit`, the OpenRAIL family (`openrail`, `creativeml-openrail-m`, `openrail++`), `llama2`/`llama3`/`llama3.1`/`llama3.2`/`llama3.3`/`llama4` community licences, `gemma` (Gemma Terms of Use) and `other` (custom LICENSE file) — https://raw.githubusercontent.com/huggingface/hub-docs/main/docs/hub/repositories-licenses.md
- **Gated models:** users "must agree to share their contact information ... to access the model files". Teams need an HF token and approval before the demo — https://raw.githubusercontent.com/huggingface/hub-docs/main/docs/hub/models-gated.md
- **Pickle in model repos:** "There are dangerous arbitrary code execution attacks that can be perpetrated when you load a pickle file." HF scans pickles and lists their imports, and recommends loading from trusted orgs. The doc points to safetensors — https://raw.githubusercontent.com/huggingface/hub-docs/main/docs/hub/security-pickle.md

### Inferences
- Pick models tagged `apache-2.0` or `mit` to avoid licence questions. Llama, Gemma and OpenRAIL models carry use restrictions or attribution terms that teams should read before shipping a product.
- Request gated-model access before the event, since approval can be manual.
- Prefer `.safetensors` weights over `.bin`/`.pt` pickles.

### Gaps
- Inference Endpoints (dedicated) pricing and scale-to-zero cold-start times were not fetched.
- I did not confirm the date the Spaces paid-plan change took effect. It is in the current hub-docs main branch.

## (3) Serving a model: FastAPI + joblib, ONNX, BentoML, LitServe, Gradio/Streamlit; cold starts and size limits

### Takeaway
For sklearn, the simplest robust pattern is to train, export with `skl2onnx.to_onnx`, and serve with `onnxruntime.InferenceSession` inside FastAPI. This avoids pickle's security and version-pinning problems and runs with less RAM. joblib/pickle is fine for your own model only if training and serving use identical library versions. BentoML and LitServe add batching and packaging. Gradio gives a UI plus an API in about 10 lines. Free hosts sleep, and cold starts that also download models make it worse, so bake model files into the image and pre-warm.

### Cited Findings
- **FastAPI minimal** (README):
  ```py
  from fastapi import FastAPI
  app = FastAPI()
  @app.get("/")
  def read_root():
      return {"Hello": "World"}
  @app.get("/items/{item_id}")
  def read_item(item_id: int, q: str | None = None):
      return {"item_id": item_id, "q": q}
  ```
  Source: https://raw.githubusercontent.com/fastapi/fastapi/master/README.md
- **skl2onnx + onnxruntime** (README "Getting started"):
  ```py
  from skl2onnx import to_onnx
  onx = to_onnx(clr, X[:1])            # X must be float32
  with open("rf_iris.onnx", "wb") as f:
      f.write(onx.SerializeToString())
  import onnxruntime as rt
  sess = rt.InferenceSession("rf_iris.onnx", providers=["CPUExecutionProvider"])
  input_name = sess.get_inputs()[0].name
  label_name = sess.get_outputs()[0].name
  pred_onx = sess.run([label_name], {input_name: X_test.astype(np.float32)})[0]
  ```
  "Last supported opset is 21" — https://raw.githubusercontent.com/onnx/sklearn-onnx/main/README.md
- **PyTorch → ONNX:** `torch.onnx.export(..., dynamo=True)` "is the recommended exporter". The TorchScript-based path "is no longer recommended". It requires `pip install --upgrade onnx onnxscript`:
  ```py
  example_inputs = (torch.randn(1, 1, 32, 32),)
  onnx_program = torch.onnx.export(torch_model, example_inputs, dynamo=True)
  # save, then:
  ort_session = onnxruntime.InferenceSession("./image_classifier_model.onnx", providers=["CPUExecutionProvider"])
  onnxruntime_outputs = ort_session.run(None, onnxruntime_input)[0]
  ```
  Source: https://raw.githubusercontent.com/pytorch/tutorials/main/beginner_source/onnx/export_simple_model_to_onnx_tutorial.py
- **sklearn persistence comparison** (official docs):
  - pickle, joblib and cloudpickle all have the con "Loading can execute arbitrary code" and "Requires the same environment as the training environment".
  - "none of these methods support loading a model trained with a different version of scikit-learn, and possibly different versions of other dependencies such as numpy and scipy".
  - ONNX environments can be minimal and need no Python. "onnxruntime typically requires much less RAM than Python to compute predictions from small models". Not all models are supported by ONNX.
  - `skops.io` is the safer alternative: `sio.get_untrusted_types(file=...)` then `sio.load(..., trusted=unknown_types)`.
  - Source: https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/model_persistence.rst
- **Version-mismatch pitfall:** loading with a different sklearn version raises `InconsistentVersionWarning`. You can turn it into an error and read `w.original_sklearn_version` — same source as above; class defined in https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/exceptions.py
- **Pickle security** (Python docs): "The `pickle` module **is not secure**. Only unpickle data you trust. It is possible to construct malicious pickle data which will **execute arbitrary code during unpickling**." — https://raw.githubusercontent.com/python/cpython/main/Doc/library/pickle.rst
- **BentoML minimal** (README):
  ```py
  import bentoml
  @bentoml.service(image=bentoml.images.Image(python_version="3.11").python_packages("torch", "transformers"))
  class Summarization:
      def __init__(self) -> None:
          import torch
          from transformers import pipeline
          device = "cuda" if torch.cuda.is_available() else "cpu"
          self.pipeline = pipeline('summarization', device=device)
      @bentoml.api(batchable=True)
      def summarize(self, texts: list[str]) -> list[str]:
          results = self.pipeline(texts)
          return [item['summary_text'] for item in results]
  ```
  Run with `bentoml serve`, which serves on http://localhost:3000 — https://raw.githubusercontent.com/bentoml/BentoML/main/README.md
- **LitServe minimal** (README):
  ```py
  import litserve as ls
  class InferenceEngine(ls.LitAPI):
      def setup(self, device):
          self.text_model = lambda x: x**2
          self.vision_model = lambda x: x**3
      def predict(self, request):
          x = request["input"]
          return {"output": self.text_model(x) + self.vision_model(x)}
  if __name__ == "__main__":
      server = ls.LitServer(InferenceEngine(max_batch_size=1), accelerator="auto")
      server.run(port=8000)
  ```
  Source: https://raw.githubusercontent.com/Lightning-AI/LitServe/main/README.md
- **Gradio as server** (README):
  ```py
  import gradio as gr
  def greet(name, intensity):
      return "Hello, " + name + "!" * int(intensity)
  demo = gr.Interface(fn=greet, inputs=["text", "slider"], outputs=["text"], api_name="predict")
  demo.launch()
  ```
  Source: https://raw.githubusercontent.com/gradio-app/gradio/main/README.md
- **Cold starts on free hosts:**
  - HF Spaces on free hardware "go to sleep" when unused, and disk is not persistent (50GB) — https://raw.githubusercontent.com/huggingface/hub-docs/main/docs/hub/spaces-overview.md
  - Render free web services spin down after 15 minutes of inactivity (SNIPPET-ONLY, RE-CHECK; render.com/docs/free was blocked) — search snippets from https://dashdashhard.com/posts/ultimate-guide-to-renders-free-tier/ and https://www.luckymedia.dev/insights/render
  - Reported wake times range from 30–60 s (https://github.com/Sponti-App/Sponti/issues/125) to 50–90 s (snippet from https://kuberns.com/blogs/cold-start-problem-deployment/). These sources are anecdotal; UNVERIFIED.

### Inferences
- Load the model once at startup (module scope, `setup()`, or `__init__`), never per request.
- Download weights at build time (Dockerfile `RUN` or commit them) rather than on first request. Otherwise a cold start pays both container boot and the model download.
- Pin `scikit-learn==<training version>` in the serving requirements, or ship ONNX.
- Never accept user-uploaded pickles.
- Hit the endpoint about 5 minutes before judging to wake it.

### Gaps
- Streamlit minimal snippet was not fetched (Streamlit README not retrieved).
- Official free-tier numbers for Render, Railway, Fly.io, Vercel and Modal could not be verified (domains blocked). Image-size and memory limits on those hosts are unknown here, so all are RE-CHECK.

## (4) In-browser / on-device models: transformers.js, ONNX Runtime Web, WebGPU, MediaPipe, TF.js

### Takeaway
transformers.js v4 mirrors the Python `pipeline` API and runs ONNX models on WASM (CPU) by default or WebGPU with `device: 'webgpu'`. Use quantized dtypes (`q4`/`q8`) to cut download size. WebGPU covers about 85% of users globally and is patchy on Safari and Firefox, so keep a WASM fallback.

### Cited Findings
- transformers.js quick tour:
  ```js
  import { pipeline } from '@huggingface/transformers';
  const pipe = await pipeline('sentiment-analysis');
  const out = await pipe('I love transformers!');
  ```
  Use `{ device: 'webgpu' }` for GPU. The default dtype is `"fp32"` on WebGPU and `"q8"` on WASM, and `"q4"` is available. Local or offline models are set with `env.localModelPath = '/path/to/models/'` and `env.allowRemoteModels = false` — https://raw.githubusercontent.com/huggingface/transformers.js/main/README.md
- In-browser LLM example: `pipeline("text-generation", "onnx-community/Qwen2.5-0.5B-Instruct", { dtype: "q4", device: "webgpu" })`. `ModelRegistry.get_available_dtypes(id)` lists the available quantizations — https://raw.githubusercontent.com/huggingface/transformers.js/main/packages/transformers/docs/source/guides/dtypes.md
- WebGPU guide: "As of March 2026, global WebGPU support is around 85% (according to caniuse.com)". Firefox needs `dom.webgpu.enabled`; Safari support is version-dependent. The guide's embeddings example is `pipeline("feature-extraction", "mixedbread-ai/mxbai-embed-xsmall-v1", { device: "webgpu" })` — https://raw.githubusercontent.com/huggingface/transformers.js/main/packages/transformers/docs/source/guides/webgpu.md
- ONNX Runtime Web: WASM runs on all listed browsers. The compatibility table lists WebGPU on Chrome/Edge (Windows, Android, macOS) but not on Safari, iOS or Firefox. WebGL is "in maintenance mode" and WebGPU is recommended. WASM supports all `ai.onnx` and `ai.onnx.ml` operators, so sklearn ONNX models (`ai.onnx.ml`) can run in-browser — https://raw.githubusercontent.com/microsoft/onnxruntime/main/js/web/README.md
- TF.js: install via `<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js">` or `import * as tf from '@tensorflow/tfjs'` — https://raw.githubusercontent.com/tensorflow/tfjs/master/README.md. The latest npm release is 4.22.0 (2024-10-21) — https://registry.npmjs.org/@tensorflow/tfjs
- MediaPipe targets "Android, iOS, web, desktop, edge devices, and IoT". Web setup guide: https://developers.google.com/mediapipe/solutions/setup_web — https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/README.md. The npm package `@mediapipe/tasks-vision` is at 1.0.1.

### Inferences
- The model download is the main browser cost:
  - Pick models in the tens to low hundreds of MB.
  - Use `q4`/`q8`.
  - Load in a Web Worker so the UI does not freeze.
  - Show a progress bar.
  - Pre-load before the demo, since the browser caches it.
- Make WASM the default and WebGPU an enhancement, given that iOS Safari is not in ORT Web's WebGPU column and transformers.js says Safari support is version-dependent. The two sources may differ in recency.

### Gaps
- Could not fetch the MediaPipe Tasks Vision web code sample (ai.google.dev blocked; sample repo paths 404). The usual `FilesetResolver.forVisionTasks(...)` / `ImageClassifier.createFromOptions(...)` pattern is UNVERIFIED here.
- No primary source found for browser per-tab memory limits or a maximum practical model size.

## (5) LLM fine-tuning vs prompting vs RAG: criteria, costs, tooling, 24–48h realism

### Takeaway
Order of attempts: prompt (with few-shot examples), then put the docs in context with caching if they are under about 200k tokens, then RAG for larger corpora, and fine-tune last. Fine-tuning helps format, tone, consistency, cost and latency, not new knowledge or reasoning. In 24–48h a LoRA/QLoRA fine-tune of a 1–8B model on a free Colab notebook (Unsloth) or a 1B LoRA example (Axolotl) is feasible. The bottleneck is building a clean dataset of hundreds of examples plus evaluation, not GPU time.

### Cited Findings
- Anthropic: for a knowledge base under 200,000 tokens (about 500 pages), put the whole thing in the prompt. Prompt caching gives ">2x" lower latency and "up to 90%" lower cost. Beyond that, Contextual Retrieval (contextual embeddings plus contextual BM25) "can reduce the number of failed retrievals by 49% and, when combined with reranking, by 67%" — https://www.anthropic.com/news/contextual-retrieval
- Anthropic prompt-engineering overview: before prompting, have "a clear definition of the success criteria", "ways to empirically test" and "a first draft prompt". "Not every success criteria or failing eval is best solved by prompt engineering ... you can sometimes improve latency and cost more easily by selecting a different model." — https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview
- OpenAI Cookbook (fine-tune chat models): "Fine-tuning improves the model by training on many more examples than can fit in a prompt". "Fine-tuning works best when focused on a particular domain". At the time of writing, GPT-4o mini fine-tuning was gated to Tier 4–5 usage tiers (may be outdated, RE-CHECK) — https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/How_to_finetune_chat_models.ipynb
- OpenAI fine-tuning-guide paraphrases (SNIPPET-ONLY; secondary): try prompt engineering, prompt chaining and function calling first. Fine-tuning is good for "following a given format or tone", complex instructions, "improving latency, and reducing token usage" — snippets from https://www.datacamp.com/tutorial/fine-tuning-openais-gpt-4-step-by-step-guide and https://theneuralbase.com/openai-fine-tuning/learn/beginner/the-prompting-first-rule/
- **PEFT/LoRA:**
  - The README example wraps Qwen2.5-3B-Instruct with `LoraConfig(r=16, lora_alpha=32, task_type=TaskType.CAUSAL_LM)`, which prints "trainable params: 3,686,400 || all params: 3,089,625,088 || trainable%: 0.1193". The adapter is saved with `model.save_pretrained(...)` and loaded with `PeftModel.from_pretrained(model, "qwen2.5-3b-lora")`.
  - Memory table on an A100 80GB: T0_3B needs 47.14GB GPU for a full fine-tune vs 14.4GB with LoRA.
  - The T0_3B LoRA checkpoint is "just 19MB compared to 11GB".
  - QLoRA of Llama-2-7b on a 16GB GPU is referenced.
  - Source: https://raw.githubusercontent.com/huggingface/peft/main/README.md
- **Unsloth:**
  - Claims "2× faster with 70% less VRAM with no accuracy loss" (vendor claim).
  - Free Colab notebooks are listed for Gemma 4 (E2B), Qwen3.5 (4B), gpt-oss (20B), Llama 3.1 (8B) Alpaca, Llama 3.2 (1B/3B) conversational, Orpheus-TTS and embeddinggemma (300M), plus GRPO notebooks.
  - Source: https://raw.githubusercontent.com/unslothai/unsloth/main/README.md
- **Axolotl:**
  - YAML-config based. Requires Python ≥3.12 and PyTorch ≥2.13.0; it is "uv-first".
  - Quickstart is `axolotl fetch examples` then `axolotl train examples/llama-3/lora-1b.yml`.
  - Supports full fine-tuning, LoRA, QLoRA, DPO/ORPO/KTO and GRPO.
  - Source: https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/README.md

### Inferences
- Realistic in 24–48h:
  - prompting plus structured outputs (hours);
  - long-context or cached-context "RAG-lite" (hours);
  - basic embeddings RAG (half a day);
  - a LoRA fine-tune of a ≤8B model on a free or cheap GPU via an Unsloth notebook (a few hours of training after dataset prep).
- Not realistic: full fine-tunes, training from scratch, or multi-GPU Axolotl setups unless the team has done it before.
- Serving a fine-tuned open model adds hosting risk (GPU needed, cold starts). Teams should weigh that against a hosted API with a good prompt.
- Make a small eval set (20–50 cases) before choosing. Both Anthropic and OpenAI guidance put success criteria and testing first.

### Gaps
- No primary OpenAI or Anthropic doc on current hosted fine-tuning availability or prices could be fetched (OpenAI docs blocked; no Anthropic fine-tuning page retrieved). All hosted fine-tuning prices are a gap, RE-CHECK.
- No sourced wall-clock training times for Unsloth notebooks on a Colab T4.
