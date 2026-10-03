# Deep learning and transfer learning for image, text and audio at 24-48h hackathons/datathons (as of 2026-10-03)

Research method note: huggingface.co, docs.ultralytics.com, arxiv.org, pytorch.org, kaggle.com and nvidia.com were blocked by the sandbox proxy. Most facts below come from raw GitHub READMEs/docs sources (raw.githubusercontent.com) and PyPI JSON, fetched 2026-10-03. Hugging Face docs are cited by their GitHub source files (docs/source/en/...). Web-search-only claims are marked SNIPPET-ONLY.

## Q1. Which approach when: frozen embeddings + linear head vs fine-tuning the last layers vs full fine-tuning vs PEFT/LoRA (data size, T4 compute, minimal code)

### Takeaway
Treat this as a ladder. (1) Start with frozen embeddings plus a linear or logistic head, which takes minutes and is hard to overfit. (2) Move to SetFit or fine-tuning only the head and top layers when you have very few labels or a small dataset. (3) Use full fine-tuning of a small or base backbone when you have thousands of labels or the domain differs from pretraining. (4) Use LoRA/PEFT mainly when the model is too large for a 16 GB T4 to fully fine-tune (LLMs, Whisper-large, diffusion). The sources give rules of thumb based on dataset size and similarity to the pretraining data, not hard row-count cutoffs.

### Cited Findings
**Decision rules (data size x domain similarity)**
- The CS231n rules of thumb cover four cases. (1) Small dataset, similar to pretraining: "not a good idea to fine-tune the ConvNet due to overfitting concerns... best idea might be to train a linear classifier on the CNN codes." (2) Large and similar: fine-tune the whole network. (3) Small but very different: train only a linear classifier, possibly on activations from earlier layers. (4) Large and very different: "very often still beneficial to initialize with weights from a pretrained model" and fine-tune the whole network. — [CS231n transfer learning notes](https://github.com/cs231n/cs231n.github.io/blob/master/transfer-learning.md)
- The PyTorch official tutorial defines the two main scenarios: "finetuning the ConvNet" and "ConvNet as fixed feature extractor" (freeze all weights except the final FC layer with `param.requires_grad = False`). The tutorial says training "should take around 15-25 min on CPU" for the feature-extractor case and less than a minute on GPU. — [pytorch/tutorials transfer_learning_tutorial.py](https://github.com/pytorch/tutorials/blob/main/beginner_source/transfer_learning_tutorial.py)
- Kumar et al. (ICLR 2022), "Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution": full fine-tuning can improve in-distribution accuracy but can underperform linear probing out-of-distribution, especially when pretrained features are good and the distribution shift is large. The two-stage fix is LP-FT (linear probe first, then fine-tune), which changes features 10-100x less than plain fine-tuning — SNIPPET-ONLY (search summary) — [ICLR 2022 paper (NSF PAR copy)](https://par.nsf.gov/servlets/purl/10337813); [NeurIPS 2024 follow-up on LP-FT for LMs](https://proceedings.neurips.cc/paper_files/paper/2024/file/fcc22e5b7d5d2155d994da22d045f0a6-Paper-Conference.pdf)
- fastai's `fine_tune` implements LP-FT by default: "Fine tune with `Learner.freeze` for `freeze_epochs`, then with `Learner.unfreeze` for `epochs`, using discriminative LR". The signature is `fine_tune(self, epochs, base_lr=2e-3, freeze_epochs=1, lr_mult=100, ...)`. — [fastai/callback/schedule.py](https://github.com/fastai/fastai/blob/main/fastai/callback/schedule.py)
- The DINOv3 README describes its models as producing high-quality dense features that outperform "the specialized state of the art across a broad range of settings, without fine-tuning". It ships linear-probing code for classification, ADE20K segmentation and NYUv2 depth, plus a Colab notebook that trains a linear foreground-segmentation model on DINOv3 features. — [facebookresearch/dinov3 README](https://github.com/facebookresearch/dinov3)

**Few-shot text (SetFit)**
- SetFit claims that "with only 8 labeled examples per class on the Customer Reviews sentiment dataset, SetFit is competitive with fine-tuning RoBERTa Large on the full training set of 3k examples". It needs no prompts and is "typically an order of magnitude (or more) faster to train". — [huggingface/setfit README](https://github.com/huggingface/setfit)
- Compute: "training SetFit on an NVIDIA V100 with 8 labeled examples takes just 30 seconds, at a cost of $0.025", compared with T-Few 3B, which needs an A100 for 11 minutes (about $0.7). V100, not T4; I found no T4 number. — [HF blog: setfit.md](https://github.com/huggingface/blog/blob/main/setfit.md)
- Method: contrastive fine-tuning of a Sentence Transformer on in-class and out-of-class pairs, then a classification head (scikit-learn LogisticRegression by default, or a differentiable `SetFitHead`). — [setfit README](https://github.com/huggingface/setfit); [HF blog setfit.md](https://github.com/huggingface/blog/blob/main/setfit.md)

**PEFT/LoRA**
- On `bigscience/mt0-large`, LoRA trains "only 0.19% of the parameters". The README example on Qwen2.5-3B prints `trainable params: 3,686,400 || all params: 3,089,625,088 || trainable%: 0.1193`. — [huggingface/peft README](https://github.com/huggingface/peft)
- Memory measured on an A100 80GB (not a T4): T0_3B full fine-tuning uses 47.14 GB GPU, against 14.4 GB with LoRA. bloomz-7b1 full fine-tuning runs out of memory, against 32 GB with LoRA. The T0_3B LoRA checkpoint is 19 MB, against 11 GB for the full model. Stable Diffusion v1-4 with LoRA and gradient checkpointing needs 8.12 GB. — [peft README](https://github.com/huggingface/peft)
- PEFT says adapters "demonstrate performance comparable to a fully finetuned model" and avoid "catastrophic forgetting or overfitting the backbone". — [peft README](https://github.com/huggingface/peft)
- 16 GB-class GPU references from the PEFT README: QLoRA fine-tuning of Llama-2-7B "on a 16GB GPU" (PyTorch blog); a Whisper-large-v2 LoRA plus 8-bit Colab notebook; a Stable Diffusion LoRA DreamBooth Space "running on a T4 instance". — [peft README](https://github.com/huggingface/peft)

**Full fine-tuning compute data points (T4-class / Colab)**
- Whisper-small (244M) fine-tuning on about 8 hours of Hindi audio in Colab uses `per_device_train_batch_size=16, max_steps=5000, gradient_checkpointing=True, fp16=True, learning_rate=1e-5`. "Training will take approximately 5-10 hours depending on your GPU or the one allocated to the Google Colab." If you hit OOM, halve the batch size and raise `gradient_accumulation_steps`. Logged WER: 34.63 at step 1000, 32.01 at step 4000. — [HF blog fine-tune-whisper.md](https://github.com/huggingface/blog/blob/main/fine-tune-whisper.md)
- fastai quick start: a cat/dog classifier with `vision_learner(dls, resnet34, metrics=error_rate); learn.fine_tune(1)`. — [fastai nbs/quick_start.ipynb](https://github.com/fastai/fastai/blob/main/nbs/quick_start.ipynb)

**Minimal code quoted from official sources**
- timm, frozen feature extractor (pooled embeddings): `m = timm.create_model('resnet50', pretrained=True, num_classes=0)` returns `torch.Size([2, 2048])`. To replace the head, use `m.reset_classifier(0)`, or set `num_classes=N` at creation. — [timm hfdocs feature_extraction.mdx](https://github.com/huggingface/pytorch-image-models/blob/main/hfdocs/source/feature_extraction.mdx)
- transformers text classification (task guide): `TrainingArguments(output_dir="my_awesome_model", learning_rate=2e-5, per_device_train_batch_size=16, per_device_eval_batch_size=16, num_train_epochs=2, weight_decay=0.01, eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True, push_to_hub=True)`, then `Trainer(model=..., args=..., train_dataset=..., eval_dataset=..., processing_class=tokenizer, data_collator=..., compute_metrics=...)` and `trainer.train()`. — [transformers docs/source/en/tasks/sequence_classification.md](https://github.com/huggingface/transformers/blob/main/docs/source/en/tasks/sequence_classification.md)
- transformers image classification fine-tunes `google/vit-base-patch16-224-in21k` on 5,000 Food-101 images (`split="train[:5000]"`). Settings: `learning_rate=5e-5, per_device_train_batch_size=16, gradient_accumulation_steps=4, num_train_epochs=3, remove_unused_columns=False, metric_for_best_model="accuracy"`. Augmentation is `Compose([RandomResizedCrop(size), ToTensor(), normalize])`. — [transformers tasks/image_classification.md](https://github.com/huggingface/transformers/blob/main/docs/source/en/tasks/image_classification.md)
- transformers audio classification fine-tunes `facebook/wav2vec2-base` on MInDS-14 (intent). It resamples to 16 kHz with `dataset.cast_column("audio", Audio(sampling_rate=16000))` and truncates with `max_length=16000`. — [transformers tasks/audio_classification.md](https://github.com/huggingface/transformers/blob/main/docs/source/en/tasks/audio_classification.md)
- SetFit, end to end (README, condensed): `model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2", labels=["negative","positive"])`, `args = TrainingArguments(batch_size=16, num_epochs=4, eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True)`, `trainer = Trainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, metric="accuracy", column_mapping={"sentence": "text", "label": "label"})`, `trainer.train()`. Uses `sample_dataset(..., num_samples=8)` for 8 examples per class; the README prints `{'accuracy': 0.869...}` on SST-2. — [setfit README](https://github.com/huggingface/setfit)
- PEFT: `from peft import LoraConfig, TaskType, get_peft_model`, then `peft_config = LoraConfig(r=16, lora_alpha=32, task_type=TaskType.CAUSAL_LM)`, `model = get_peft_model(model, peft_config)`, `model.print_trainable_parameters()`. Reload with `PeftModel.from_pretrained(model, "qwen2.5-3b-lora")`. Inside transformers: `model.add_adapter(peft_config, adapter_name="lora_1")`. — [peft README](https://github.com/huggingface/peft)
- Ultralytics: `model = YOLO("yolo26n.pt")`, `model.train(data="coco8.yaml", epochs=100, imgsz=640, device="cpu")`, `metrics = model.val()`, `model.export(format="onnx")`. — [ultralytics README](https://github.com/ultralytics/ultralytics)
- sentence-transformers embeddings for a frozen linear head: `model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")`, `embeddings = model.encode(sentences)` gives shape `(3, 384)`, and `model.similarity(embeddings, embeddings)`. — [sentence-transformers README](https://github.com/UKPLab/sentence-transformers)

### Inferences
- Hackathon default ladder: (a) frozen DINOv2/v3 / SigLIP / sentence-transformer / CLAP / BEATs embeddings plus scikit-learn LogisticRegression as a baseline within the first hour; (b) SetFit for text under about 100 labels per class; (c) full fine-tuning of a base-sized model (ViT-B/ConvNeXt-T, DeBERTa-v3-base/ModernBERT-base, wav2vec2-base/AST) when you have thousands of labels; (d) LoRA only for models above roughly 1B parameters, or Whisper-medium/large, on a T4.
- Applying LP-FT is cheap. fastai `fine_tune` does it automatically; in HF you train the head with the backbone frozen for one epoch, then unfreeze.
- Whisper-small fine-tuning at 5-10 h for 5,000 steps uses a large share of a 24-48 h hackathon. Cut `max_steps`, or use whisper-tiny/base.

### Gaps
- I found no official, documented T4 wall-clock numbers for timm, ViT or DeBERTa fine-tuning. The Whisper "5-10 hours ... Colab" figure is the only Colab-class training-time figure I found in official sources.
- I found no authoritative numeric thresholds (for example "use LoRA above N examples") for encoder-sized models. The guidance is qualitative.
- The arXiv originals (LoRA 2106.09685, LP-FT 2202.10054, SetFit 2209.11055) could not be fetched; the LP-FT claims are SNIPPET-ONLY.

## Q2. Recommended pretrained backbones per modality in 2026, with licences

### Takeaway
Images: timm for supervised backbones; DINOv2 (Apache-2.0) or DINOv3 (custom DINOv3 License, gated) for frozen features; SigLIP 2 or OpenCLIP for zero-shot and text-image tasks; Ultralytics YOLO26 for detection, segmentation and pose (AGPL-3.0). Text: ModernBERT-base/large (8k context, faster) or DeBERTa-v3 for classification; sentence-transformers for embeddings; SetFit for few-shot. Audio: Whisper (MIT) for ASR; wav2vec2 / AST / BEATs for classification fine-tuning; CLAP (laion-clap) for zero-shot and embeddings.

### Cited Findings
**Images**
- The timm README lists model families including ConvNeXt/V2, EVA/EVA-02/EVA-CLIP, DINOv3, DINOv2 with registers, SigLIP and SigLIP 2 image encoders (including NaFlex). Recent changelog entries add EUPE (DINOv3-style), TIPSv2 and Sapiens2. All models support `features_only=True` feature pyramids and `get_classifier`/`reset_classifier`. — [timm README](https://github.com/huggingface/pytorch-image-models)
- timm licence: code is Apache-2.0. ImageNet-pretrained weights carry ambiguity ("one should assume that the original dataset license applies to the weights", so seek legal advice for commercial use). Some weights are CC-BY-NC 4.0 (Facebook WSL/SSL/SWSL). — [timm README, Licenses](https://github.com/huggingface/pytorch-image-models#licenses)
- DINOv2 "code and model weights are released under the Apache License 2.0". Load with `torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')` (also `_reg` and `_lc` linear-classifier variants). One exception: the XRay-DINO weights are under the FAIR Noncommercial Research License. — [dinov2 README](https://github.com/facebookresearch/dinov2)
- DINOv3: "code and model weights are released under the DINOv3 License" (custom, last updated Aug 19, 2025). Weights are gated: you request access and receive URLs by e-mail. Backbones: ViT-S/16, S+/16, B/16, L/16, H+/16, 7B/16, and ConvNeXt tiny/small/base/large, plus SAT-493M satellite variants. Supported in transformers from 4.56.0; for example `AutoModel.from_pretrained("facebook/dinov3-convnext-tiny-pretrain-lvd1689m")` then `outputs.pooler_output`. — [dinov3 README](https://github.com/facebookresearch/dinov3); [DINOv3 LICENSE.md](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)
- SigLIP 2 zero-shot via `pipeline(task="zero-shot-image-classification", model="google/siglip2-base-patch16-224")`. When calling the model manually you must use `padding="max_length", max_length=64`, and NaFlex variants (`google/siglip2-base-patch16-naflex`) keep native aspect ratio. — [transformers model_doc/siglip2.md](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/siglip2.md)
- OpenCLIP: `open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')`, plus `encode_image` / `encode_text`. The repo includes a `scripts/linear_probe.py` helper and points to CLIP_benchmark for evaluation on 40 datasets. The repo licence is MIT-style (copyright Ilharco et al.). — [open_clip README](https://github.com/mlfoundations/open_clip); [open_clip LICENSE](https://github.com/mlfoundations/open_clip/blob/main/LICENSE)
- Ultralytics YOLO26 is current, and YOLO27 is "Coming Soon ... no launch date". COCO detection rows give params / FLOPs / T4 TensorRT latency: YOLO26n 40.9 mAP, 2.4M, 1.7 ms; YOLO26s 48.6, 9.5M, 2.5 ms; YOLO26m 53.1, 20.4M, 4.7 ms; YOLO26l 55.0, 24.8M, 6.2 ms; YOLO26x 57.5, 55.7M, 11.8 ms. The family also covers seg, semantic seg, depth, cls (YOLO26n-cls 71.4 top-1 ImageNet), pose and OBB. — [ultralytics README](https://github.com/ultralytics/ultralytics)
- **Ultralytics licence: AGPL-3.0** (PyPI also says AGPL-3.0). The README offers "two licensing options": AGPL-3.0 for "students, researchers, and enthusiasts", or an Enterprise License for business products and services, "including internal tools, automated workflows, and production deployments, bypassing the open-source requirements of AGPL-3.0". — [ultralytics README License](https://github.com/ultralytics/ultralytics#license); [PyPI ultralytics JSON](https://pypi.org/pypi/ultralytics/json)

**Text**
- ModernBERT: "trained on 2T tokens", with RoPE supporting "sequences of up to 8192 tokens". Use `answerdotai/ModernBERT-base` / `-large`. Padding-free training needs `attn_implementation="flash_attention_2"`, which is no longer the default. — [transformers model_doc/modernbert.md](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/modernbert.md)
- The ModernBERT blog says DeBERTaV3 "has been the choice of champions for years" on Kaggle NLP, and that ModernBERT is "the first base-size model to beat DeBERTaV3 on GLUE". It also says ModernBERT is about "twice as fast as DeBERTa... up to 4x faster" with mixed-length inputs, though it "slightly lags DeBERTaV3" in natural language understanding (per the blog's comparison). Target inference GPUs include the T4. — [HF blog modernbert.md](https://github.com/huggingface/blog/blob/main/modernbert.md)
- FlashAttention-2 supports Ampere, Ada and Hopper. For Turing GPUs (T4) you need the separate flash-attention-turing repo, and "bf16 requires Ampere, Ada, or Hopper GPUs". — [flash-attention README](https://github.com/Dao-AILab/flash-attention)
- DeBERTa-v3 sizes (backbone params): XSmall 22M (hidden 384), Small 44M, Base 86M, Large 304M; mDeBERTa-v3-base covers 102 languages. Published MNLI-m/mm: Large 91.8/91.9, Base 90.6/90.7, Small 88.3/87.7, XSmall 88.1/88.3. — [microsoft/DeBERTa README](https://github.com/microsoft/DeBERTa)
- sentence-transformers example model: `all-MiniLM-L6-v2` (384-dim). SetFit can use any Sentence Transformer on the Hub, including multilingual ones. — [sentence-transformers README](https://github.com/UKPLab/sentence-transformers); [setfit README](https://github.com/huggingface/setfit)

**Audio**
- Whisper sizes and VRAM: tiny 39M ~1 GB (~10x speed); base 74M ~1 GB; small 244M ~2 GB; medium 769M ~5 GB; large 1550M ~10 GB; turbo 809M ~6 GB (~8x). "the turbo model is not trained for translation tasks". Code and weights are MIT. — [openai/whisper README](https://github.com/openai/whisper)
- Whisper was pretrained on 680,000 hours of labelled audio. Fine-tuning works "with as little as 8 hours of fine-tuning data". — [HF blog fine-tune-whisper.md](https://github.com/huggingface/blog/blob/main/fine-tune-whisper.md)
- AST (Audio Spectrogram Transformer): BSD-3-Clause. It has recipes for ESC-50 (95.75% accuracy in the attached log), Speech Commands V2 and AudioSet, plus a Colab inference notebook. The README recommends `audioset_pretrain=True` for all tasks except AudioSet itself. — [YuanGongND/ast README](https://github.com/YuanGongND/ast); [ast LICENSE](https://github.com/YuanGongND/ast/blob/master/LICENSE)
- BEATs: from microsoft/unilm (MIT repo licence). Weights are OneDrive links, AS20K/AS2M pretrained and fine-tuned. Features via `BEATs_model.extract_features(audio_input_16khz, padding_mask=...)`; input is 16 kHz. — [unilm/beats README](https://github.com/microsoft/unilm/tree/master/beats); [unilm LICENSE](https://github.com/microsoft/unilm/blob/master/LICENSE)
- CLAP: `pip install laion-clap`, `laion_clap.CLAP_Module(enable_fusion=False)`, `get_audio_embedding_from_filelist(...)`. Checkpoints: `630k-audioset-best.pt` (general audio <10 s), fusion variants for variable length, and music or music+speech checkpoints. Repo licence is CC0 1.0. — [LAION-AI/CLAP README](https://github.com/LAION-AI/CLAP); [CLAP LICENSE](https://github.com/LAION-AI/CLAP/blob/main/LICENSE)
- wav2vec2-base is the backbone in the official transformers audio-classification guide. — [transformers tasks/audio_classification.md](https://github.com/huggingface/transformers/blob/main/docs/source/en/tasks/audio_classification.md)

### Inferences
- For a hackathon demo that may become a product, flag the licences. Ultralytics (AGPL-3.0) means network-served modifications must be open-sourced unless you buy an Enterprise licence. DINOv3 has a custom licence and gated download, which adds approval delay at a hackathon, so DINOv2 (Apache-2.0) is the frictionless choice. Some timm weights are non-commercial.
- On a T4, ModernBERT works but without FlashAttention-2 (no Turing support) and without bf16, so the speed advantages measured on an RTX 4090 may shrink. DeBERTa-v3-base remains a safe choice.

### Gaps
- Licences of individual HF model cards (SigLIP 2, ModernBERT, wav2vec2, AST HF port, DeBERTa-v3 weights) could not be verified because huggingface.co was blocked; they are commonly Apache-2.0 or MIT (UNVERIFIED).
- I could not read the AGPL-3.0 implications for hackathon code from Ultralytics docs (docs.ultralytics.com blocked; the FAQ.md grep found no AGPL text).

## Q3. Latest package versions (PyPI JSON, fetched 2026-10-03)

### Takeaway
All core packages had releases in Sept-Oct 2026. Note that transformers is on 5.x and sentence-transformers on 6.x, so old Colab notebooks written for transformers 4.x may need small API fixes (for example `processing_class=` instead of `tokenizer=`, and `eval_strategy` instead of `evaluation_strategy`).

### Cited Findings
| package | version | upload date | licence (PyPI) | requires_python |
|---|---|---|---|---|
| torch | 2.14.1 | 2026-09-30 | Apache-2.0 AND BSD-3/BSD-2/MIT/BSL (composite) | >=3.10 |
| timm | 1.0.30 | 2026-09-22 | Apache-2.0 | >=3.8 |
| transformers | 5.18.0 | 2026-09-30 | Apache 2.0 | >=3.10.0 |
| peft | 0.21.2 | 2026-10-01 | Apache | >=3.10.0 |
| setfit | 1.2.0 | 2026-09-04 | Apache 2.0 | >=3.9 |
| sentence-transformers | 6.1.0 | 2026-09-18 | Apache-2.0 | >=3.10 |
| ultralytics | 8.4.171 | 2026-10-01 | **AGPL-3.0** | >=3.8 |
| fastai | 2.8.12 | 2026-09-09 | Apache-2.0 | >=3.10 |
| accelerate | 1.15.0 | 2026-09-09 | Apache | >=3.10.0 |
— Sources: https://pypi.org/pypi/torch/json, https://pypi.org/pypi/timm/json, https://pypi.org/pypi/transformers/json, https://pypi.org/pypi/peft/json, https://pypi.org/pypi/setfit/json, https://pypi.org/pypi/sentence-transformers/json, https://pypi.org/pypi/ultralytics/json, https://pypi.org/pypi/fastai/json, https://pypi.org/pypi/accelerate/json
- The Whisper blog uses the older `evaluation_strategy="steps"`, while current task guides use `eval_strategy="epoch"` and `processing_class=tokenizer`. — [fine-tune-whisper.md](https://github.com/huggingface/blog/blob/main/fine-tune-whisper.md); [sequence_classification.md](https://github.com/huggingface/transformers/blob/main/docs/source/en/tasks/sequence_classification.md)

### Inferences
- Pin versions in requirements.txt at hackathon start, because Colab and Kaggle preinstalled versions can lag or lead.

### Gaps
- I did not check Colab/Kaggle preinstalled versions.

## Q4. Data augmentation and training on free GPUs (batch size, mixed precision, early stopping)

### Takeaway
On a free T4 (16 GB, Turing): use fp16 AMP rather than bf16, a batch size of 16 with gradient accumulation (or auto batch finding), gradient checkpointing for big models, early stopping with load-best-model, and RAM caching of images. Use light augmentation for classification (RandomResizedCrop, flips). YOLO already applies mosaic, flip and HSV augmentation by default.

### Cited Findings
- Kaggle gives "up to 30 hours per week" of GPU, with a P100 16 GB or a T4 option, 12-hour execution limit per CPU/GPU session (9 h for TPU) and 20 GB of auto-saved disk. — [ultralytics docs/en/integrations/kaggle.md](https://github.com/ultralytics/ultralytics/blob/main/docs/en/integrations/kaggle.md); the 30h quota is shared between P100 and T4x2 (2x16 GB) — SNIPPET-ONLY [Kaggle efficient GPU usage doc](https://www.kaggle.com/docs/efficient-gpu-usage), [aimultiple free cloud GPU](https://aimultiple.com/free-cloud-gpu)
- bf16 needs Ampere or newer ("bf16 requires Ampere, Ada, or Hopper GPUs"), so use `fp16=True` on a T4 or P100. — [flash-attention README](https://github.com/Dao-AILab/flash-attention)
- `TrainingArguments` docstrings: `bf16` is "Generally preferred over FP16 due to better numerical stability and no loss scaling required". `fp16` says "Consider using BF16 instead if your hardware supports it". Effective batch = `per_device_train_batch_size × num_devices × gradient_accumulation_steps`. `auto_find_batch_size` finds "a training batch size that will fit into memory automatically through exponential decay, avoiding CUDA Out-of-Memory errors" and requires accelerate. — [transformers training_args.py](https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py)
- Example LLM recipe: `per_device_train_batch_size=2, gradient_accumulation_steps=8, gradient_checkpointing=True, bf16=True, learning_rate=2e-5, eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True`. — [transformers docs/source/en/training.md](https://github.com/huggingface/transformers/blob/main/docs/source/en/training.md)
- `EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.0)`. It must be used with `metric_for_best_model` and "depends on ... load_best_model_at_end"; if `save_steps` differs from `eval_steps`, stopping waits for the next save. — [transformers trainer_callback.py](https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_callback.py)
- Ultralytics defaults (`default.yaml`):
  - Training and early stopping: `epochs: 100`, `patience: 100` ("early stop after N epochs without val improvement").
  - Batch size: `batch: 16`. Use `-1` for AutoBatch, or a float in (0,1) for a GPU memory fraction.
  - Precision: `amp: True`, where True means fp16 after an AMP check; 'bf16' or False are the alternatives.
  - Data loading and layers: `cache: False` (True/'ram' or 'disk'); `freeze:` takes the first N layers.
  - Augmentation and imbalance: `fliplr: 0.5`, `mosaic: 1.0`, `close_mosaic: 10`, `mixup: 0.0`, `hsv_h: 0.015`, `cls_pw: 0.0` (class-weight power for imbalance).
  - Optimizer: `optimizer: auto`, which picks MuSGD for runs over about 10k iterations and AdamW otherwise, and ignores the user's lr0.
  - Sources: [ultralytics cfg/default.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml); [docs/en/modes/train.md](https://github.com/ultralytics/ultralytics/blob/main/docs/en/modes/train.md)
- Ultralytics `batch=-1` targets "approximately 60% CUDA memory utilization". — [docs/en/modes/train.md](https://github.com/ultralytics/ultralytics/blob/main/docs/en/modes/train.md)
- Ultralytics training tips:
  - "A good starting point is 300 epochs... If the model overfits early, you can reduce the number of epochs".
  - `patience=5` stops training after 5 epochs without improvement.
  - `cache=True` gives the fastest data access.
  - AMP is on by default and falls back to FP32 if the capability check fails.
  - Pretrained `yolo26n.pt` starts from COCO weights.
  - Source: [docs/en/guides/model-training-tips.md](https://github.com/ultralytics/ultralytics/blob/main/docs/en/guides/model-training-tips.md)
- Image augmentation baseline in the HF task guide is `RandomResizedCrop` + `Normalize`. Audio: AST notes a torchaudio SpecAugment behaviour change (a bug that was fixed). — [image_classification.md](https://github.com/huggingface/transformers/blob/main/docs/source/en/tasks/image_classification.md); [ast README](https://github.com/YuanGongND/ast)
- OpenCLIP inference uses `torch.autocast("cuda")` and `torch.no_grad()`. — [open_clip README](https://github.com/mlfoundations/open_clip)

### Inferences
- Precompute frozen embeddings once (inference only under autocast) and cache them to disk as .npy. Head training then runs on CPU in seconds, which saves the GPU quota for one or two fine-tuning runs.
- Ultralytics' default `patience=100` with `epochs=100` effectively disables early stopping. Set `patience=10-20` for hackathon time budgets (inference from the defaults).

### Gaps
- I found no official per-model T4 throughput (images/sec, tokens/sec) for training. Colab free-tier GPU quotas could not be confirmed (research.google.com blocked).

## Q5. Common failure modes (overfitting small data, near-duplicate leakage, class imbalance) and good tutorials

### Takeaway
The biggest silent killer is near-duplicate leakage between train and validation, which inflates validation scores. Split by group (patient, source video, user) and deduplicate using embedding similarity. Fight overfitting with frozen features or LP-FT, early stopping and augmentation. For imbalance, use class weights (`cls_pw` in YOLO) and macro-F1 or mAP metrics.

### Cited Findings
- 3.3% of CIFAR-10 and 10% of CIFAR-100 test images have near-duplicates in training. On the duplicate-free ciFAIR test sets, accuracy dropped by "between 9% and 14% relative" — SNIPPET-ONLY (search summary of Barz & Denzler) — [arXiv 1902.00423 PDF](https://arxiv.org/pdf/1902.00423); [PMC copy](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8321059/)
- cleanlab detects "outliers, duplicates, label errors" in any dataset (Datalab) and finds label issues "in ONE line of code". — [cleanlab README](https://github.com/cleanlab/cleanlab)
- Ultralytics on class imbalance: "some classes have significantly fewer examples... can cause the model to perform poorly on rare classes. Ultralytics YOLO supports class weighting through the `cls_pw` argument" (0 means off, 1.0 means full inverse frequency). — [docs/en/modes/train.md](https://github.com/ultralytics/ultralytics/blob/main/docs/en/modes/train.md); [default.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml)
- Overfitting with small data: fine-tuning a small, similar dataset risks overfitting, so prefer a linear classifier. — [CS231n](https://github.com/cs231n/cs231n.github.io/blob/master/transfer-learning.md)
- Whisper fine-tuning log shows validation loss rising (0.3075 at step 1000 to 0.4519 at step 4000) while WER improved (34.63 to 32.01). Choose the checkpoint on the task metric (`metric_for_best_model="wer", greater_is_better=False`), not on loss. — [fine-tune-whisper.md](https://github.com/huggingface/blog/blob/main/fine-tune-whisper.md)
- The open_clip README warns that models start in train mode ("impacts some models with BatchNorm or stochastic depth"), so call `model.eval()` before extracting embeddings. — [open_clip README](https://github.com/mlfoundations/open_clip)
- The SigLIP 2 doc warns that text must be padded to `max_length=64`, "since the model was trained with this". Wrong padding is a silent accuracy bug. — [siglip2.md](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/siglip2.md)

**Tutorials/notebooks for quick transfer learning**
- fastai Quick Start: about 5-line image, segmentation, text and tabular models, with every docs page available as a Colab notebook — [fastai README](https://github.com/fastai/fastai), [nbs/quick_start.ipynb](https://github.com/fastai/fastai/blob/main/nbs/quick_start.ipynb)
- PyTorch transfer learning tutorial covering fine-tuning vs fixed feature extractor — [pytorch/tutorials](https://github.com/pytorch/tutorials/blob/main/beginner_source/transfer_learning_tutorial.py)
- HF ViT fine-tuning on the beans dataset with a Colab link — [HF blog fine-tune-vit.md](https://github.com/huggingface/blog/blob/main/fine-tune-vit.md)
- HF Whisper fine-tuning Colab — [sanchit-gandhi/notebooks fine_tune_whisper.ipynb](https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/fine_tune_whisper.ipynb) (linked from [blog](https://github.com/huggingface/blog/blob/main/fine-tune-whisper.md))
- SetFit notebooks directory — [setfit/notebooks](https://github.com/huggingface/setfit/tree/main/notebooks)
- DINOv3 foreground segmentation linear-probe Colab — [dinov3 notebooks/foreground_segmentation.ipynb](https://colab.research.google.com/github/facebookresearch/dinov3/blob/main/notebooks/foreground_segmentation.ipynb)
- AST one-click Colab inference — [ast README](https://github.com/YuanGongND/ast)
- Ultralytics Colab tutorial — [examples/tutorial.ipynb](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb) (linked in [modes/train.md](https://github.com/ultralytics/ultralytics/blob/main/docs/en/modes/train.md)) and the Kaggle YOLO26 notebook ([README badge](https://github.com/ultralytics/ultralytics))
- PEFT Whisper-large-v2 LoRA plus 8-bit Colab — [Colab link in peft README](https://github.com/huggingface/peft)
- HF transformers task guides: [image](https://github.com/huggingface/transformers/blob/main/docs/source/en/tasks/image_classification.md), [text](https://github.com/huggingface/transformers/blob/main/docs/source/en/tasks/sequence_classification.md), [audio](https://github.com/huggingface/transformers/blob/main/docs/source/en/tasks/audio_classification.md)

### Inferences
- A practical dedup recipe built from the tools above: embed all images with DINOv2/CLIP, take cosine nearest neighbours, and cluster pairs above a threshold into groups. Then use `GroupKFold` so near-duplicates never straddle train and validation. This is an inference; no single official source prescribes this exact recipe.
- For imbalance outside YOLO: use class-weighted CrossEntropy or a balanced logistic head, and report macro-F1 (general practice; not sourced here).

### Gaps
- No primary source fetched for class-imbalance handling in transformers Trainer (no built-in class-weight argument was confirmed). fastdup / imagededup tooling was not researched.
- The ciFAIR figures are SNIPPET-ONLY; I could not open the arXiv PDF.
