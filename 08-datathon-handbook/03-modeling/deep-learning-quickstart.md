# 🧠 Deep Learning & Transfer Learning Quickstart

<!-- markdownlint-disable MD013 -->

> Images, text and audio in 24-48 hours: start from a pretrained model, climb the ladder only as far as your data allows, and train on a free GPU. Code here could **not** be run in our sandbox (no PyTorch or model downloads); the tabular parts of the guide are tested. Baseline recipes: [`baseline-recipes.md`](baseline-recipes.md); free GPUs: [`tooling-2026-update.md`](tooling-2026-update.md) section 6.
>
> 📚 Source research: [`technical-ml-gaps-2026-10-03`](../../_research/technical-ml-gaps-2026-10-03/research.md). **Label key used throughout.** **SNIPPET-ONLY** means the claim comes from a search-result snippet because the primary page could not be fetched; check the source before relying on it. **UNVERIFIED** means it comes from general knowledge or an unconfirmed attribution. **RE-CHECK** marks prices, quotas and limits that change often; confirm them on the day. **Vendor-reported** or **author-reported** marks numbers from the people selling or publishing the tool. Every code block is labelled **not run by us**: each is either quoted from the cited source or assembled from cited calls, and none was executed. All version and licence tables are dated **2026-10-03** and were taken from PyPI or npm JSON on that date. Nothing here is legal advice.

## 3. Deep learning for images, text and audio: climb a four-rung ladder from frozen embeddings

### Plain-English explanation

At a 24–48 hour event you almost never train a neural network from scratch. You **borrow one**: a "pretrained backbone" that has already learned general features from millions of images, documents or hours of audio. You then adapt it to your task. This is **transfer learning**, and there are four levels of adaptation, from cheapest to most expensive:

1. **Frozen embeddings plus a linear head.** Run your data through the backbone once, save the output vectors, then train a logistic regression on them. It takes minutes, often runs on CPU, and is hard to overfit. The guide's `baseline-recipes.md` already has code for this.
2. **Few-shot or head-and-top-layer training.** Examples include SetFit for text, and fastai's freeze-then-unfreeze. Use these when you have very few labels.
3. **Full fine-tuning of a small or base backbone.** Use this when you have thousands of labels or your domain looks unlike the pretraining data.
4. **LoRA/PEFT.** Train small "adapter" matrices instead of the whole model. Use this mainly when the model is too large to fully fine-tune on a free 16 GB T4: LLMs, Whisper-large, diffusion models.

The CS231n notes give the classic rule of thumb for choosing a level ([CS231n transfer learning](https://github.com/cs231n/cs231n.github.io/blob/master/transfer-learning.md)):

- With a **small dataset similar to the pretraining data**, it is "not a good idea to fine-tune ... due to overfitting concerns". Train a linear classifier on the features instead.
- With a **large dataset**, fine-tune the whole network.
- Even when your data is **very different**, initialising from pretrained weights is "very often still beneficial".

The PyTorch tutorial names the same two basic modes, "finetuning the ConvNet" and "ConvNet as fixed feature extractor" ([PyTorch transfer learning tutorial](https://github.com/pytorch/tutorials/blob/main/beginner_source/transfer_learning_tutorial.py)).

A useful nuance comes from Kumar et al. (ICLR 2022). Full fine-tuning can beat a linear probe in-distribution yet underperform it out-of-distribution. Their fix, **LP-FT**, trains the linear head first and then fine-tunes everything (**SNIPPET-ONLY**, [paper](https://par.nsf.gov/servlets/purl/10337813)). fastai's `fine_tune` already does exactly this by default: it trains frozen for `freeze_epochs`, then unfreezes with discriminative learning rates ([fastai schedule.py](https://github.com/fastai/fastai/blob/main/fastai/callback/schedule.py)).

### Decision table: which rung of the ladder

| Labels you have | Domain vs pretraining | First move | Upgrade if time allows | Compute (free T4) |
|---|---|---|---|---|
| None | Any | Zero-shot: SigLIP 2 / OpenCLIP (images), CLAP (audio), an LLM or zero-shot pipeline (text) | Label 100–300 items (section 5) | Inference only |
| 8–100 per class (text) | Similar | **SetFit** | Frozen embeddings + LogisticRegression as a comparison | Seconds to minutes; V100 timing below |
| Hundreds | Similar | Frozen embeddings (DINOv2, sentence-transformers, CLAP/BEATs) + LogisticRegression | LP-FT (`fastai fine_tune`) | CPU after one embedding pass |
| Hundreds | Very different (X-ray, satellite, machine sounds) | Linear probe, possibly on earlier-layer features (CS231n) | LP-FT with strong augmentation | T4, ≤1 h |
| Thousands | Any | Full fine-tune of a base model (ViT-B / ConvNeXt-T, DeBERTa-v3-base / ModernBERT-base, wav2vec2-base / AST) | Ensemble 2–3 seeds | T4, hours |
| Object detection / segmentation | Any | Ultralytics YOLO26n/s from COCO weights (**AGPL**) | Larger YOLO26 size | T4, hours |
| Speech-to-text | Any | Whisper zero-shot (pick size by VRAM) | Fine-tune whisper-tiny/base/small, or LoRA on large | Whisper-small fine-tune quoted at 5–10 h |
| Any, model >1B params | Any | LoRA / QLoRA via PEFT | — | 16 GB GPU possible for 7B with QLoRA |

The rungs follow [CS231n](https://github.com/cs231n/cs231n.github.io/blob/master/transfer-learning.md), the [SetFit README](https://github.com/huggingface/setfit), the [PEFT README](https://github.com/huggingface/peft) and the [HF Whisper fine-tuning blog](https://github.com/huggingface/blog/blob/main/fine-tune-whisper.md). The sources give **qualitative** rules based on data size and similarity. No authoritative row-count cutoffs were found, so the label counts in the first column are our inference.

**Author- and vendor-reported numbers to label as such.**

- **SetFit (author-reported).** With 8 labelled examples per class, SetFit is "competitive with fine-tuning RoBERTa Large on the full training set of 3k examples". Training takes "30 seconds" on a **V100** (not a T4) and costs about $0.025 ([SetFit README](https://github.com/huggingface/setfit); [HF blog](https://github.com/huggingface/blog/blob/main/setfit.md)).
- **PEFT (author-reported, measured on an A100 80GB, not a T4).**
  - Training T0_3B fully needs 47.14 GB, against 14.4 GB with LoRA.
  - The LoRA checkpoint is 19 MB, against 11 GB for the full model.
  - LoRA on Qwen2.5-3B trains 0.1193% of the parameters ([PEFT README](https://github.com/huggingface/peft)).
- **Whisper-small fine-tuning (author-reported).** About 8 h of Hindi audio, 5,000 steps, "approximately 5-10 hours" on a Colab-class GPU. That is a large share of a hackathon, so cut `max_steps` or use tiny/base ([HF blog](https://github.com/huggingface/blog/blob/main/fine-tune-whisper.md)).

### Decision table: recommended backbones in 2026, with licences

| Modality / job | Backbone | Licence (weights) | Load with | Notes |
|---|---|---|---|---|
| Image features (frozen) | **DINOv2** | Apache-2.0 (XRay-DINO weights excepted: FAIR non-commercial) | `torch.hub.load('facebookresearch/dinov2','dinov2_vits14')` | Frictionless default ([dinov2](https://github.com/facebookresearch/dinov2)) |
| Image features (frozen) | **DINOv3** | **Custom DINOv3 License, gated** (request access; URLs by e-mail) | `AutoModel.from_pretrained("facebook/dinov3-convnext-tiny-pretrain-lvd1689m")` (transformers ≥4.56) | Authors claim it beats specialised SOTA "without fine-tuning" (author-reported); approval delay at a hackathon ([dinov3](https://github.com/facebookresearch/dinov3); [LICENSE](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)) |
| Image classification fine-tune | timm (ConvNeXt, EVA, ViT, etc.) | Code Apache-2.0; **weights may inherit dataset licence; some CC-BY-NC** | `timm.create_model(name, pretrained=True, num_classes=N)` | "assume that the original dataset license applies to the weights" ([timm licences](https://github.com/huggingface/pytorch-image-models#licenses)) |
| Zero-shot image / image–text | SigLIP 2, OpenCLIP | OpenCLIP repo MIT-style; SigLIP 2 card licence **UNVERIFIED** | `pipeline("zero-shot-image-classification", model="google/siglip2-base-patch16-224")` | Text must be padded to `max_length=64` ([siglip2 doc](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/siglip2.md); [open_clip](https://github.com/mlfoundations/open_clip)) |
| Detection / segmentation / pose | **Ultralytics YOLO26** | **AGPL-3.0** or paid Enterprise licence | `YOLO("yolo26n.pt")` | Vendor-reported COCO: YOLO26n 40.9 mAP, 2.4M params, 1.7 ms T4 TensorRT; YOLO26s 48.6; YOLO26m 53.1; YOLO26l 55.0; YOLO26x 57.5 ([ultralytics](https://github.com/ultralytics/ultralytics)) |
| Text classification | DeBERTa-v3 (xsmall 22M → large 304M; mDeBERTa for 102 languages) | Repo MIT; HF card licence **UNVERIFIED** | `AutoModelForSequenceClassification` | Author-reported MNLI-m: base 90.6, large 91.8 ([DeBERTa](https://github.com/microsoft/DeBERTa)) |
| Text classification, long inputs | ModernBERT-base/large | **UNVERIFIED** (card not fetched) | `answerdotai/ModernBERT-base` | 8,192-token context; HF blog claims ~2× faster than DeBERTa and first base model to beat DeBERTaV3 on GLUE, but "slightly lags" it on NLU (author-reported) ([modernbert doc](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/modernbert.md); [blog](https://github.com/huggingface/blog/blob/main/modernbert.md)) |
| Text embeddings / few-shot | sentence-transformers (`all-MiniLM-L6-v2`, 384-d), SetFit | Apache-2.0 (libraries) | `SentenceTransformer(...)`, `SetFitModel.from_pretrained(...)` | ([sentence-transformers](https://github.com/UKPLab/sentence-transformers)) |
| Speech-to-text | **Whisper** | MIT (code and weights) | `openai/whisper` | VRAM: tiny/base ~1 GB, small ~2 GB, medium ~5 GB, large ~10 GB, turbo ~6 GB; turbo "not trained for translation" ([whisper](https://github.com/openai/whisper)) |
| Audio classification | AST; wav2vec2-base; BEATs | AST BSD-3-Clause; BEATs repo MIT (weights via OneDrive) | transformers audio-classification guide | AST ESC-50 95.75% in attached log (author-reported) ([AST](https://github.com/YuanGongND/ast); [BEATs](https://github.com/microsoft/unilm/tree/master/beats)) |
| Zero-shot audio / audio embeddings | CLAP (`laion-clap`) | Repo CC0-1.0 | `laion_clap.CLAP_Module(enable_fusion=False)` | Separate checkpoints for general audio <10 s, music, and speech ([CLAP](https://github.com/LAION-AI/CLAP)) |

**Licence warnings to put in a callout.**

- **Ultralytics: AGPL-3.0.** The README offers AGPL-3.0 for "students, researchers, and enthusiasts" and a paid Enterprise licence for business use, "including internal tools, automated workflows, and production deployments" ([Ultralytics licence](https://github.com/ultralytics/ultralytics#license)). In practice, a team that serves a modified YOLO over a network and later commercialises must open-source it or buy a licence (our reading; not legal advice).
- **DINOv3: custom licence, gated download.** Use DINOv2 if you cannot wait for approval.
- **timm: some weights are non-commercial.**
- **Hugging Face model cards: not checked.** Licences for SigLIP 2, ModernBERT, wav2vec2, the AST HF port and the DeBERTa-v3 weights could not be fetched. Many cards are Apache-2.0 or MIT, but that is **UNVERIFIED**.

### Free-GPU training settings (T4, 16 GB, Turing)

| Setting | Use | Why / source |
|---|---|---|
| Precision | `fp16=True`, **not** bf16 | "bf16 requires Ampere, Ada, or Hopper GPUs" ([flash-attention](https://github.com/Dao-AILab/flash-attention)) |
| Batch | `per_device_train_batch_size=16` + `gradient_accumulation_steps`; or `auto_find_batch_size=True` (needs accelerate) | Effective batch = per-device × devices × accumulation ([training_args.py](https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py)) |
| Memory | `gradient_checkpointing=True` for big models; on OOM halve batch, double accumulation | ([Whisper blog](https://github.com/huggingface/blog/blob/main/fine-tune-whisper.md)) |
| Early stopping | `EarlyStoppingCallback(early_stopping_patience=...)` with `metric_for_best_model` and `load_best_model_at_end=True` | Stops only at save points if `save_steps` ≠ `eval_steps` ([trainer_callback.py](https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_callback.py)) |
| Pick checkpoint by task metric | `metric_for_best_model="wer", greater_is_better=False` | In the Whisper log, validation loss rose 0.3075 → 0.4519 while WER improved 34.63 → 32.01 ([Whisper blog](https://github.com/huggingface/blog/blob/main/fine-tune-whisper.md)) |
| FlashAttention / ModernBERT on T4 | FA2 not supported on Turing (needs separate flash-attention-turing repo) | Measured speed gains may shrink on a T4 (our inference) |
| YOLO | Set `patience=10–20`; `batch=-1` (AutoBatch, ~60% memory); `cache=True`; `cls_pw` for class imbalance | Defaults are `epochs: 100, patience: 100`, which effectively disables early stopping (our inference from [default.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml); [train docs](https://github.com/ultralytics/ultralytics/blob/main/docs/en/modes/train.md)) |
| Augmentation | `RandomResizedCrop` + flip (images); YOLO already applies mosaic, flip and HSV | ([HF image guide](https://github.com/huggingface/transformers/blob/main/docs/source/en/tasks/image_classification.md)) |
| Cache embeddings | Embed once under `torch.no_grad()`/autocast, save `.npy`, train head on CPU | Saves GPU quota (our inference); Kaggle quota is in the guide's tooling page |

### Minimal code (not run by us)

Frozen image embeddings with timm. Quoted from the [timm feature extraction docs](https://github.com/huggingface/pytorch-image-models/blob/main/hfdocs/source/feature_extraction.mdx), with the head training assembled by us. Not run by us.

```python
# not run by us — timm call quoted; rest assembled
import timm, torch, numpy as np
from sklearn.linear_model import LogisticRegression
m = timm.create_model('resnet50', pretrained=True, num_classes=0).eval()  # -> (N, 2048)
with torch.no_grad():
    feats = torch.cat([m(xb) for xb in loader]).numpy()
np.save("feats.npy", feats)
clf = LogisticRegression(max_iter=2000).fit(feats[train_idx], y[train_idx])
```

Few-shot text with SetFit, condensed from the [SetFit README](https://github.com/huggingface/setfit), which prints about 0.869 accuracy on SST-2 with 8 examples per class (author-reported). Not run by us.

```python
# not run by us — condensed from SetFit README
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
train_ds = sample_dataset(dataset["train"], label_column="label", num_samples=8)
model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2",
                                    labels=["negative", "positive"])
args = TrainingArguments(batch_size=16, num_epochs=4, eval_strategy="epoch",
                         save_strategy="epoch", load_best_model_at_end=True)
trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                  metric="accuracy", column_mapping={"sentence": "text", "label": "label"})
trainer.train()
```

Full fine-tune with T4 settings. Assembled from the [transformers text-classification guide](https://github.com/huggingface/transformers/blob/main/docs/source/en/tasks/sequence_classification.md) plus `fp16` and `EarlyStoppingCallback` from the transformers source; `push_to_hub` was dropped. Not run by us.

```python
# not run by us — assembled; transformers 5.x argument names
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
args = TrainingArguments(output_dir="out", learning_rate=2e-5,
    per_device_train_batch_size=16, gradient_accumulation_steps=2,
    num_train_epochs=3, weight_decay=0.01, fp16=True,          # T4: fp16, not bf16
    eval_strategy="epoch", save_strategy="epoch",
    load_best_model_at_end=True, metric_for_best_model="f1")
trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
    processing_class=tokenizer, data_collator=collator, compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)])
trainer.train()
```

LoRA, quoted from the [PEFT README](https://github.com/huggingface/peft). Not run by us.

```python
# not run by us — quoted from PEFT README
from peft import LoraConfig, TaskType, get_peft_model
peft_config = LoraConfig(r=16, lora_alpha=32, task_type=TaskType.CAUSAL_LM)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```

Detection, quoted from the [Ultralytics README](https://github.com/ultralytics/ultralytics), with `patience` added by us. The licence is **AGPL-3.0**. Not run by us.

```python
# not run by us — README calls; patience added
from ultralytics import YOLO
model = YOLO("yolo26n.pt")
model.train(data="data.yaml", epochs=100, imgsz=640, patience=15)
metrics = model.val()
model.export(format="onnx")
```

Audio preprocessing, quoted from the [transformers audio-classification guide](https://github.com/huggingface/transformers/blob/main/docs/source/en/tasks/audio_classification.md). Not run by us.

```python
# not run by us — quoted
from datasets import Audio
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))  # wav2vec2/BEATs expect 16 kHz
```

### Pitfalls

**Near-duplicates across train and validation.** This is the silent killer. Barz and Denzler report that 3.3% of CIFAR-10 and 10% of CIFAR-100 test images have near-duplicates in training, and accuracy fell "between 9% and 14% relative" on duplicate-free test sets (**SNIPPET-ONLY**, [arXiv 1902.00423](https://arxiv.org/pdf/1902.00423)). Our inference for a fix:

- Split by source (patient, video, user).
- Embed every item with DINOv2 or CLIP.
- Group cosine near-neighbours above a threshold.
- Use `GroupKFold` so a group never straddles train and validation.

**Fine-tuning a small dataset.** It overfits. Start frozen (CS231n), and use LP-FT rather than unfreezing everything at once.

**Models left in train mode.** OpenCLIP models start in train mode, so call `model.eval()` before extracting embeddings ([open_clip](https://github.com/mlfoundations/open_clip)).

**Wrong SigLIP 2 text padding.** If you call the model manually and do not pad to `max_length=64`, accuracy drops with no error message ([siglip2 doc](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/siglip2.md)).

**Old notebooks on transformers 5.x.** Use `processing_class=` instead of `tokenizer=`, and `eval_strategy` instead of `evaluation_strategy`. The Whisper blog still uses the old name ([Whisper blog](https://github.com/huggingface/blog/blob/main/fine-tune-whisper.md); [sequence_classification guide](https://github.com/huggingface/transformers/blob/main/docs/source/en/tasks/sequence_classification.md)). Pin versions at the start of the event.

**Choosing checkpoints on loss.** Pick on the task metric instead (see the Whisper example in the table above).

**Class imbalance.**

- In YOLO, use `cls_pw` ([train docs](https://github.com/ultralytics/ultralytics/blob/main/docs/en/modes/train.md)).
- Elsewhere, class-weighted cross-entropy plus macro-F1 is general practice, **UNVERIFIED** here. No built-in class-weight argument in the transformers Trainer was confirmed.

**Ultralytics `optimizer: auto` ignores your learning rate.** It picks MuSGD for runs longer than about 10k iterations and AdamW otherwise, and ignores the user's `lr0` ([default.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml)).

### Versions and licences (as of 2026-10-03)

| Package | Version | Released | Licence (PyPI) | Python |
|---|---|---|---|---|
| torch | 2.14.1 | 2026-09-30 | Apache-2.0 and BSD/MIT (composite) | ≥3.10 |
| transformers | 5.18.0 | 2026-09-30 | Apache-2.0 | ≥3.10 |
| timm | 1.0.30 | 2026-09-22 | Apache-2.0 (code; weights vary) | ≥3.8 |
| peft | 0.21.2 | 2026-10-01 | Apache | ≥3.10 |
| setfit | 1.2.0 | 2026-09-04 | Apache-2.0 | ≥3.9 |
| sentence-transformers | 6.1.0 | 2026-09-18 | Apache-2.0 | ≥3.10 |
| accelerate | 1.15.0 | 2026-09-09 | Apache | ≥3.10 |
| fastai | 2.8.12 | 2026-09-09 | Apache-2.0 | ≥3.10 |
| ultralytics | 8.4.171 | 2026-10-01 | **AGPL-3.0** | ≥3.8 |

All versions are from PyPI JSON, e.g. [transformers](https://pypi.org/pypi/transformers/json) and [ultralytics](https://pypi.org/pypi/ultralytics/json). Ultralytics also lists YOLO27 as "Coming Soon" with no launch date ([README](https://github.com/ultralytics/ultralytics)). Colab and Kaggle preinstalled versions were not checked.

### Learning resources

**Concepts.**

- [CS231n transfer learning notes](https://github.com/cs231n/cs231n.github.io/blob/master/transfer-learning.md).
- [PyTorch transfer learning tutorial](https://github.com/pytorch/tutorials/blob/main/beginner_source/transfer_learning_tutorial.py). It quotes 15–25 minutes on CPU for the feature-extractor case.

**Fastest hands-on.**

- The [fastai Quick Start](https://github.com/fastai/fastai/blob/main/nbs/quick_start.ipynb) (5-line models).
- [fastbook](https://github.com/fastai/fastbook). The code is GPL v3, but the prose "is not licensed for any redistribution or change of format", so link to it rather than copying it.

**Per-modality notebooks.**

- Hugging Face task guides for [image](https://github.com/huggingface/transformers/blob/main/docs/source/en/tasks/image_classification.md), [text](https://github.com/huggingface/transformers/blob/main/docs/source/en/tasks/sequence_classification.md) and [audio](https://github.com/huggingface/transformers/blob/main/docs/source/en/tasks/audio_classification.md).
- [ViT on beans](https://github.com/huggingface/blog/blob/main/fine-tune-vit.md).
- The [Whisper fine-tuning Colab](https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/fine_tune_whisper.ipynb).
- [SetFit notebooks](https://github.com/huggingface/setfit/tree/main/notebooks).
- The [DINOv3 linear-probe segmentation Colab](https://colab.research.google.com/github/facebookresearch/dinov3/blob/main/notebooks/foreground_segmentation.ipynb) (gated weights).
- [AST](https://github.com/YuanGongND/ast) (Colab inference).
- The [Ultralytics Colab tutorial](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb).
- The [PEFT README](https://github.com/huggingface/peft), which links a Whisper-large LoRA Colab.
