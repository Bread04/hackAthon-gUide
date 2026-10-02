# Free and Low-Cost Compute for Datathon Teams (as of Oct 2026)

Method note: every primary vendor page (kaggle.com, research.google.com/colaboratory, huggingface.co, lightning.ai, docs.github.com, docs.digitalocean.com, thundercompute.com) was BLOCKED by the sandbox egress proxy. Almost every figure below comes from WebSearch snippets, so treat it as **SNIPPET-ONLY** unless marked otherwise. The one page I fetched in full was a github.com community discussion (DigitalOcean / Student Pack). **Every limit and price is RE-CHECK.** Vendors change free tiers often, and several changed in 2026 (see Student Pack and Studio Lab).

## Q1. Platform-by-platform limits (Kaggle, Colab, HF ZeroGPU, Lightning AI, Paperspace/others, Codespaces, student credits)

### Takeaway
For a 24-48h datathon, Kaggle is the most predictable free GPU option: about 30 GPU-h/week, 12h sessions, documented RAM/disk, and private notebooks by default. Colab free gives a T4 with no guarantees and a roughly 90-minute idle disconnect. Lightning AI gives a persistent free Studio with monthly credits. Codespaces suits CPU work and dev environments (120 core-hours/month). HF ZeroGPU is only for demos (minutes per day). Paperspace's free tier is unclear after DigitalOcean absorbed it. SageMaker Studio Lab reportedly closed in July 2026. DigitalOcean's student credit ended on Aug 1, 2026.

### Cited Findings

**Kaggle Notebooks**
- About 30 h/week GPU quota and 20 h/week TPU quota. The GPU quota is "floating" and can exceed 30 h in a week depending on demand. It resets weekly. RE-CHECK, SNIPPET-ONLY — [Kaggle product-feedback: floating GPU quota](https://www.kaggle.com/product-feedback/173129); [Kaggle forum: 30h/week reset](https://www.kaggle.com/general/135810); [aicreditmart guide 2026](https://aicreditmart.com/ai-credits-providers/kaggle-free-gpu-tpu-30-hours-week-access-guide-2026/)
- Maximum session execution: 12 h for CPU/GPU, 9 h for TPU. RE-CHECK, SNIPPET-ONLY — [Kaggle Notebooks docs](https://www.kaggle.com/docs/notebooks) (via search summary)
- GPU options: 1x Tesla P100 (16 GB) or 2x T4 (32 GB VRAM total) — [Kaggle forum on GPU options](https://www.kaggle.com/product-feedback/361104); [Kaggle docs](https://www.kaggle.com/docs/notebooks). RE-CHECK, SNIPPET-ONLY
- CPU notebooks: 4 cores and 30 GB RAM. P100 and T4x2 notebooks: 4 CPU cores and 29 GB RAM. RE-CHECK, SNIPPET-ONLY — [Kaggle Notebooks docs](https://www.kaggle.com/docs/notebooks)
- Up to 20 GB of output can be saved in `/kaggle/working`. Extra scratch disk outside that path is not kept after the session ends. RE-CHECK, SNIPPET-ONLY — [Kaggle Notebooks docs](https://www.kaggle.com/docs/notebooks)
- Internet: competition scoring kernels run with no internet (the snippet mentions a 30-minute runtime limit for some competitions; that varies by competition). Offline pip installs are possible through the Dependency Manager, and teams can chain a training notebook (internet on) into an inference notebook (internet off). RE-CHECK, SNIPPET-ONLY — [Kaggle Competitions Setup docs](https://www.kaggle.com/docs/competitions-setup); [TDS: chaining kernels](https://towardsdatascience.com/easy-kaggle-offline-submission-with-chaining-kernels-30bba5ea5c4d/)
- Privacy: new notebooks default to **Private**, visible to you and to competition teammates if the notebook references competition data. **Making a notebook Public is permanent** (the toggle disappears). SNIPPET-ONLY — [Kaggle: Default Privacy launch](https://www.kaggle.com/product-feedback/34719)
- A Secrets add-on stores key/value credentials tied to your account. Forks of a public notebook do not inherit your secrets. SNIPPET-ONLY — [Kaggle: User Secrets discussion](https://www.kaggle.com/general/414523); [Medium: Kaggle User Secrets](https://chioujryu.medium.com/kaggle-feature-launch-user-secrets-42b080d705ac)
- "Save & Run All" runs the notebook top to bottom in a separate session and makes a version snapshot: code, logs, output files and data sources. SNIPPET-ONLY — [Kaggle Notebooks docs](https://www.kaggle.com/docs/notebooks); [Kaggle forum: save vs commit](https://www.kaggle.com/general/224266)
- Private dataset size quota: **UNVERIFIED**. The search did not confirm the commonly cited figures (e.g. 100–200 GB private dataset quota) — see Gaps.

**Google Colab (free)**
- Official FAQ: resources are free partly because of "dynamic usage limits that sometimes fluctuate" with no guaranteed or unlimited resources. Usage limits, idle timeout, maximum VM lifetime and GPU types all "vary over time." SNIPPET-ONLY — [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- Third-party reports: T4 (16 GB), up to about 12 h per session, about 90-minute idle disconnect, and roughly 15–30 GPU-h/week dynamic limit. Requesting a GPU at peak times can still give you CPU. RE-CHECK, SNIPPET-ONLY, not official — [projectech guide](https://projectech.in/guides/google-colab-free-gpu-ml-student-guide/); [aicreditmart Colab 2026](https://aicreditmart.com/ai-credits-providers/google-colab-free-tier-t4-gpu-access-guide-2026/); [Thunder Compute blog, Sept 2026](https://www.thundercompute.com/blog/colab-alternatives-for-cheap-deep-learning-in-2025)
- Official FAQ: code runs in a VM private to your account. VMs are deleted after being idle for a while and have a maximum lifetime. Colab deletes the files you created or downloaded when the session ends. SNIPPET-ONLY — [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- Official FAQ: mounting Drive lets **any code in the notebook access all files in your Drive**. Moves interrupted mid-operation through drive.mount() can lose the data in transit. SNIPPET-ONLY — [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- Colab Pro for Education: free 1-year Pro for students and faculty at **US-based** higher-ed institutions. Sources conflict on whether new signups are still open. One snippet says they are "no longer available for new signups"; a Swarthmore IT post (Jan 2026) advertises it as available. UNVERIFIED, RE-CHECK — [Google blog: Colab for higher ed](https://blog.google/outreach-initiatives/education/colab-higher-education/); [Swarthmore ITS, 2026-01-13](https://blogs.swarthmore.edu/its/2026/01/13/google-colab-free-for-students-and-faculty/)
- Colab Pro reportedly includes about 100 compute units/month, background execution, terminal access and 25 GB+ high-RAM runtimes. RE-CHECK, SNIPPET-ONLY, third-party — [bison KB: Colab Pro vs Pro+ 2026](https://knowledgebase.bison.co.in/view_article.php?id=690)

**Hugging Face ZeroGPU (Spaces)**
- Sources conflict on the free-account daily quota: one says 3.5 min/day (2 min unauthenticated), another says 5 min. PRO ($9/mo) gets about 40 min/day with queue priority; an older figure is 25 min of H200. Quota resets 24 h after first use. PRO/Team/Enterprise can buy more at $1 per 10 GPU-minutes. RE-CHECK, SNIPPET-ONLY — [HF docs: Spaces ZeroGPU](https://huggingface.co/docs/hub/en/spaces-zerogpu); [HF forum: ZeroGPU daily quota](https://discuss.huggingface.co/t/zero-gpu-daily-quota/168376); [istarsoft: $9 HF PRO](https://istarsoft.com/guides/huggingface-pro-9-storage-zerogpu-not-two-dollar-inference/)
- ZeroGPU allocates GPU dynamically per function call inside Spaces. It is meant for hosting demos, not long training jobs. Whether only PRO accounts can *host* ZeroGPU Spaces is UNVERIFIED (the docs page could not be fetched).

**Lightning AI**
- Free tier: $0 with no credit card. 15 Lightning credits/month, which expire at month end. One Studio can run free 24/7 but restarts every 4 h. Snippets also say "80 GPU hours/month on interruptible machines" — that conflicts with another figure of about 22 T4-hours/month, and the actual hours depend on the GPU type. RE-CHECK, SNIPPET-ONLY — [aicreditmart: Lightning free plan](https://aicreditmart.com/ai-credits-providers/lightning-ai-free-plan-22-gpu-hours-month-guide-2026/); [Lightning on-demand GPUs](https://lightning.ai/on-demand-gpus)
- Lightning runs an academic/student program (details not retrieved) — [Lightning docs: Students](https://lightning.ai/docs/team-management/academia/students)

**Paperspace / DigitalOcean Gradient**
- Paperspace is being absorbed into DigitalOcean ("Gradient GPU Droplets"). The Gradient API was deprecated on 15 July 2024. Third-party sources still say free notebooks exist with M4000/P5000-class GPUs, 6 h max sessions and 5 GB storage, while high-end GPUs need a $39/mo Growth plan. Current status is UNVERIFIED and RE-CHECK — the official docs were blocked — [DO docs: Paperspace Notebooks](https://docs.digitalocean.com/products/paperspace/notebooks/); [yangmao.ai Paperspace 2026](https://yangmao.ai/en/compute/paperspace/); [Vagon: Paperspace alternatives](https://vagon.io/blog/paperspace-alternatives)

**AWS SageMaker Studio Lab**
- Historically: free T4, 4 h per session / 4 GPU-h per 24 h, CPU sessions longer, and no AWS account or credit card needed. One search summary says **AWS closed Studio Lab on 30 July 2026**. UNVERIFIED, single snippet, RE-CHECK — [DataCamp Studio Lab guide](https://www.datacamp.com/tutorial/sagemaker-studio-lab); [Thunder Compute blog, Sept 2026](https://www.thundercompute.com/blog/colab-alternatives-for-cheap-deep-learning-in-2025)

**GitHub Codespaces**
- Personal accounts: Free plan includes 120 core-hours and 15 GB storage per month. Pro includes 180 core-hours and 20 GB. 120 core-hours equals 60 h on a 2-core machine or 30 h on a 4-core machine. Going past the included usage needs a payment method. CPU only for free users. RE-CHECK, SNIPPET-ONLY — [GitHub community: Codespaces personal FAQ](https://github.com/orgs/community/discussions/38697); [GitHub plans docs](https://docs.github.com/get-started/learning-about-github/githubs-products)

**Student / education credits**
- GitHub Student Developer Pack: Azure for Students $100 credit; AWS Educate access is also listed. RE-CHECK, SNIPPET-ONLY — [AWS builder: Student Pack](https://builder.aws.com/content/3ImcG3NKgTznSwIE7GPYJ0Wr2GX/github-student-developer-pack-free-tools-and-credits); [creditforstartups: student cloud credits](https://creditforstartups.com/students/cloud-credits-for-students)
- **DigitalOcean left the Student Pack.** Its $200 credit offer closed for redemption on **July 31, 2026**, and all credits expired on **August 1, 2026**. Suggested alternatives: Microsoft Azure, Camber, LocalStack; watch for DO offers via Major League Hacking. FETCHED (confirmed on page) — [GitHub community discussion #201240](https://github.com/orgs/community/discussions/201240)

### Inferences
- Kaggle is the safest default for GPU training at a weekend event. A team of 3–4 people each has their own roughly 30 h weekly quota, so the team pool is about 90–120 GPU-h if work is spread across accounts. Check whether event rules allow that. Its documented 12 h session limit is also more predictable than Colab's.
- Colab free is good for fast prototyping and sharing. Don't plan a critical overnight run on it: the idle timeout and variable GPU access make it risky.
- Codespaces (or Lightning's free CPU Studio) works well as a shared, reproducible dev environment and for CPU-heavy pandas/sklearn work. Most tabular datathon tasks don't need a GPU.
- HF ZeroGPU/Spaces fits the final demo (a Gradio app) more than training.
- Mentors and guides that still point students to DigitalOcean credits or SageMaker Studio Lab are probably out of date as of Oct 2026.

### Gaps
- Could not confirm Kaggle's private dataset storage quota, maximum single-dataset size, idle timeout for interactive sessions, or whether internet is on by default for non-competition notebooks (it needs phone verification, UNVERIFIED).
- No official current numbers for Colab free idle timeout or weekly GPU budget; Google deliberately doesn't publish them.
- No confirmation of the exact current free ZeroGPU quota (3.5 vs 5 min) or of hosting eligibility.
- Lightning's free credit-to-GPU-hour conversion is inconsistent across sources.
- The Paperspace free tier and the Studio Lab closure both need primary-source confirmation.
- Did not research Modal/Saturn Cloud/Deepnote free tiers or GCP/AWS student credits in depth (no time).

## Q2. Practical workflow: persisting work, avoiding session loss, environment pinning, privacy

### Takeaway
Treat every free runtime as disposable. Keep code in Git, keep data and checkpoints in persistent storage (Kaggle Datasets/outputs, Drive, HF Hub), pin dependencies, and never upload restricted or PII data to consumer notebook platforms unless the data provider explicitly allows it.

### Cited Findings
- Colab deletes runtime files when the session ends, so write anything you need to keep to mounted Drive. Interrupted Drive moves can lose data. SNIPPET-ONLY — [Colab FAQ](https://research.google.com/colaboratory/faq.html); [startupik: Colab mistakes 2026](https://startupik.com/6-common-google-colab-mistakes-and-how-to-avoid-them/)
- Drive mount gives notebook code access to your whole Drive, so be careful running notebooks you didn't write. The "Careful who you Colab with" write-up covers abuse of this. SNIPPET-ONLY — [Colab FAQ](https://research.google.com/colaboratory/faq.html); [Medium: Careful Who You Colab With](https://antman1p-30185.medium.com/careful-who-you-colab-with-fa8001f933e7)
- Kaggle: only `/kaggle/working` (up to 20 GB) is kept as output; "Save & Run All" versions capture outputs and can be reused as inputs to other notebooks. SNIPPET-ONLY — [Kaggle Notebooks docs](https://www.kaggle.com/docs/notebooks)
- Kaggle: use the Secrets add-on for API keys instead of hard-coding them; publishing is irreversible. SNIPPET-ONLY — [Kaggle: User Secrets](https://www.kaggle.com/general/414523); [Kaggle: Default Privacy](https://www.kaggle.com/product-feedback/34719)
- Kaggle offline workflow: download wheels and models in an internet-enabled notebook, save them as output/dataset, then attach them to the offline notebook. SNIPPET-ONLY — [TDS: chaining kernels](https://towardsdatascience.com/easy-kaggle-offline-submission-with-chaining-kernels-30bba5ea5c4d/)

### Inferences (best practice; not sourced to a vendor doc)
- Checkpoint models and intermediate features every N minutes or epochs to persistent storage, so a 12 h cap or idle disconnect costs minutes, not hours. Make training resumable from the latest checkpoint.
- Commit code to a shared GitHub repo from the start. Put a `requirements.txt` with exact `==` versions (or `uv`/`pip freeze` output) at the top of every notebook. Platform base images change; Kaggle and Colab preinstall different library versions.
- Record `python --version`, CUDA version and key library versions in the README so judges can reproduce results.
- Run long jobs as Kaggle "Save & Run All" (batch) versions rather than interactive sessions, so a closed browser doesn't kill them.
- Privacy: if the datathon data is under an NDA, a DUA, or contains PII/health data, assume Kaggle, Colab, HF and Lightning are third-party processors. Use only what organizers approve (often an organizer-provided VM or local machine). Keep Kaggle datasets and notebooks Private, and never publish a notebook that has restricted data attached. Don't push data files to public Git repos. Use `.gitignore` and secrets managers for keys.

### Gaps
- Didn't retrieve vendor terms-of-service/data-processing text for Kaggle, Colab, HF or Lightning about training on user content or data residency. Read the ToS before uploading restricted data.
