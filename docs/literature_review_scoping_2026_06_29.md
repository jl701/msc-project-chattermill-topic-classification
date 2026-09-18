# Literature Review Scoping: 2026-06-29

This note records the first pass over the locally downloaded literature in `C:\Msc_DSML\Msc_Project\Project_Preparation`. It is not yet a full literature review; it is a map for deciding the dissertation's broad-to-narrow-to-broad argument.

## Local Literature Inventory

### Background Papers

| Local File | Paper | Likely Role |
| --- | --- | --- |
| `Project_Preparation/Background_Paper/1014052.1014073.pdf` | Hu and Liu, 2004, *Mining and Summarizing Customer Reviews* | Broad starting point: customer review mining, product features/aspects, review summarisation |
| `Project_Preparation/Background_Paper/D16-1021.pdf` | Tang et al., 2016, *Aspect Level Sentiment Classification with Deep Memory Network* | ABSA foundation: aspect-specific sentiment rather than document-level sentiment |
| Chinese translated PDFs and LaTeX folders | Local reading aids for the two papers above | Use for understanding, but cite the original papers |

### Recent Papers

| Local File | Paper | Likely Role |
| --- | --- | --- |
| `Project_Preparation/Recent_Paper/1-s2.0-S0925231223009906-main.pdf` | Kontonatsios et al., 2023, *FABSA: An aspect-based sentiment analysis dataset of user reviews* | Dataset and task anchor; Chattermill-related public benchmark |
| `Project_Preparation/Recent_Paper/2306.17290_Towards_Open_Domain_Topic_Classification.pdf` | Ding et al., 2022, *Towards Open-Domain Topic Classification*; local file is the 2023 arXiv copy | Direct open-topic/candidate-label motivation: user-defined taxonomy in real time |
| `Project_Preparation/Recent_Paper/2308.10092v1.pdf` | Yu et al., 2023, *Open, Closed, or Small Language Models for Text Classification?* | Model-family comparison: hosted LLMs, open LLMs, and smaller supervised models |
| `Project_Preparation/Recent_Paper/2512.19651v1.pdf` | Ventirozos et al., *Exploring Zero-Shot ACSA with Unified Meaning Representation in Chain-of-Thought Prompting* | LLM prompting for ACSA; useful for Qwen/Gemini zero-shot framing |
| `Project_Preparation/Recent_Paper/engproc-128-00015.pdf` | Lim et al., 2026, *Parameter-Efficient Adaptation of Qwen2.5 for Aspect-Based Sentiment Analysis Using Low-Rank Adaptation and Parameter-Efficient Fine-Tuning* | Qwen/LoRA/PEFT motivation for the next modelling stage |
| Chinese translated PDFs and generated LaTeX folders | Local reading aids for FABSA, open/small LLMs, zero-shot ACSA, and Qwen PEFT | Use for understanding, but cite the original papers |

## First-Pass Thematic Map

The local literature naturally forms five layers.

### 1. Customer Feedback Mining

Anchor paper:

- Hu and Liu, 2004.

Purpose in dissertation:

- Start from the broad business/NLP problem: customer reviews contain repeated issues, product/service aspects, and opinions.
- Motivate why topic labels are useful for summarising high-volume feedback.

How it connects to this project:

- The project modernises this original feature/opinion mining problem into multi-label topic classification for industrial feedback.

### 2. Aspect-Based Sentiment Analysis

Anchor papers:

- Tang et al., 2016.
- Kontonatsios et al., 2023.

Purpose in dissertation:

- Explain the aspect+sentiment pair as the task target.
- Justify why global document sentiment is insufficient.
- Ground the FABSA dataset and its multi-domain user-review setting.

How it connects to this project:

- The strongest local non-LLM baseline explicitly replaces global sentiment with aspect-conditioned sentiment.
- This is directly motivated by ABSA's core premise: one text can express different sentiment toward different aspects.

### 3. Multi-Label And Long-Tail Text Classification

Local coverage:

- This theme is implied by FABSA and the implemented metrics, but the local paper set does not yet include a dedicated multi-label evaluation paper.

Purpose in dissertation:

- Justify pair samples F1, pair micro F1, and pair macro F1.
- Explain why macro F1 matters for rare labels and why sample-level F1 is a sensible headline metric for multi-label outputs.

Gap:

- Add at least one or two references on multi-label text classification and evaluation metrics.

### 4. Open-Topic / Candidate-Label Classification

Anchor paper:

- Ding et al., 2022.

Purpose in dissertation:

- Narrow the project from ordinary ABSA to open-topic/new-label generalisation.
- Explain why a fixed classifier head is not enough when topic labels may be unseen during training.
- Motivate the candidate-label formulation: the model receives canonical candidate labels and must select from them.

How it connects to this project:

- The held-out-aspect protocol is the project's practical version of open-topic classification.
- The candidate-aspect DistilBERT selector and Qwen/Gemini indexed prompts are label-aware methods.

### 5. LLMs, Small Models, And Parameter-Efficient Adaptation

Anchor papers:

- Yu et al., 2023.
- Ventirozos et al., zero-shot ACSA/UMR prompting.
- Lim et al., 2026, Qwen2.5 LoRA/PEFT for ABSA.

Purpose in dissertation:

- Motivate why the project compares local encoders, open LLMs, and hosted LLMs.
- Discuss when smaller supervised models may remain competitive.
- Position Qwen/Gemini as next-stage candidate-label methods, not as the only possible solution.

How it connects to this project:

- The local non-LLM baseline now reaches `0.6071` fixed held-out-aspect pair samples F1.
- Qwen indexed zero-shot reaches `0.5374`, giving an initial LLM reference.
- Gemini/Qwen fine-tuning remains the next major phase.

## Hourglass Dissertation Shape

The dissertation can be structured like an hourglass:

### Wide Opening

Start from the broad problem:

```text
Customer feedback analysis requires assigning useful business topics and sentiment to large volumes of noisy, multi-issue text.
```

Relevant broad topics:

- customer review mining;
- sentiment analysis and opinion mining;
- aspect-based sentiment analysis;
- multi-label text classification;
- industrial feedback analytics.

### Narrow Waist

Narrow to the project's exact technical problem:

```text
Can label-aware models generalise FABSA aspect+sentiment prediction to unseen candidate aspects, while also handling cross-organisation shift?
```

This is the narrowest contribution:

- public FABSA-based experimental pipeline;
- three evaluation axes:
  - closed-topic provided split;
  - held-out organisation;
  - held-out aspect/open-topic;
- held-out-aspect candidate-label protocol;
- label-masked vs example-filtered training as incomplete-label-noise ablation;
- candidate-aspect DistilBERT selector;
- DistilBERT aspect-conditioned sentiment;
- Qwen indexed zero-shot reference;
- LOAO robustness diagnostic.

### Wide Ending

Broaden back out to the larger system question:

```text
How should industrial NLP systems handle evolving topic taxonomies, new customers, and practical trade-offs between local encoders, open LLMs, and hosted LLMs?
```

Relevant ending topics:

- open-vocabulary / user-defined taxonomy classification;
- LLM-assisted customer feedback analytics;
- cost, latency, privacy, and operational reliability;
- human-in-the-loop taxonomy design;
- future fine-tuning with Qwen or hosted Gemini baselines.

## Recommended Narrow Topic

The thesis should not be framed as "ABSA with Qwen" or "fine-tuning an LLM" as the core narrow topic. That would make the existing DistilBERT/candidate-label work feel secondary.

Recommended narrow topic:

```text
Candidate-label open-topic aspect+sentiment classification for customer feedback, evaluated through held-out aspects and cross-organisation shift.
```

Why this is stronger:

- It matches Aji's repeated emphasis on open-topic generalisation.
- It explains why closed-topic, held-out organisation, and held-out aspect experiments all belong in one dissertation.
- It allows both non-LLM and LLM methods to be compared as approaches to the same problem.
- It makes the strongest completed result meaningful rather than incidental.
- It leaves room for Gemini/Qwen as future or next-stage systems without making the dissertation depend entirely on them.

## Recommended Broad Opening Topic

Recommended broad opening:

```text
Industrial customer-feedback understanding as multi-label aspect and sentiment classification under changing customers and changing topic taxonomies.
```

This is broad enough to include:

- customer review mining;
- ABSA;
- multi-label classification;
- Chattermill's business setting;
- cross-company deployment concerns.

It is more precise than simply "LLMs for topic classification" and avoids over-promising that the whole dissertation is about LLMs.

## Recommended Broad Ending Topic

Recommended broad ending:

```text
Practical open-vocabulary feedback analytics systems that combine local supervised models, open LLMs, and hosted LLMs under accuracy, cost, latency, and governance constraints.
```

This lets the conclusion discuss:

- when local DistilBERT-style systems are enough;
- when a hosted LLM baseline is worth the cost;
- why Qwen fine-tuning is promising but compute-dependent;
- why candidate-label JSON outputs are more deployable than free-form topic generation;
- why evolving taxonomies require label-aware methods.

## Literature Gaps To Fill Next

The local folder is a good start but not sufficient for a complete literature review. Suggested missing areas:

1. Multi-label text classification and evaluation.
   - Need citations for sample-level F1, micro F1, macro F1, thresholding, and long-tail label imbalance.

2. Zero-shot / entailment-style text classification.
   - Yin et al., 2019 is already mentioned in project notes but should be added to the local paper set.
   - This supports candidate-label classification with label text.

3. Domain generalisation / domain adaptation for sentiment or text classification.
   - Needed to justify held-out organisation as a deployment-oriented split.

4. Instruction-tuned ABSA.
   - InstructABSA and related generative ABSA papers should be reviewed more deeply.

5. Practical LLM evaluation.
   - Need sources or clear methodology for cost, latency, JSON validity, and reasoning-token accounting.

## Immediate Reading Plan

Recommended order:

1. Re-read FABSA paper.
   - Extract dataset motivation, annotation setup, domains, labels, baseline framing, and limitations.

2. Re-read Ding et al., 2022.
   - Extract how they define open-domain topic classification and user-defined taxonomy.
   - Map their setup against this project's held-out-aspect candidate-label formulation.

3. Re-read Hu and Liu, 2004.
   - Use only enough to establish customer review mining roots.

4. Re-read Tang et al., 2016.
   - Use it to justify aspect-conditioned sentiment and attention/target-dependent sentiment.

5. Read Yu et al., 2023.
   - Use it to frame open/closed/small model comparison.

6. Read Qwen/PEFT and zero-shot ACSA papers.
   - Use them for the next-stage LLM discussion, not as the core of the current completed experimental contribution.

7. Fill gaps with targeted searches for multi-label evaluation, NLI-style zero-shot classification, and domain generalisation.
