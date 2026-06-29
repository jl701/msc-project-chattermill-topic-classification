# Tier 1 Core Literature Review Matrix

Last updated: 2026-06-29

This document records the first substantive Tier 1 literature-review pass for the dissertation. It focuses on foundations that remain useful regardless of whether the next modelling phase prioritises Qwen, Gemini, stronger LOAO, or another local baseline.

Working dissertation spine:

```text
Structured candidate-label aspect-sentiment classification under taxonomy shift in customer feedback.
```

This is an evidence matrix, not final prose. Each source is mapped to the argument it can support, the limitation it carries, and the project-specific reason it matters.

## Tier Boundary

Tier 1 themes developed now:

- customer review mining;
- aspect-based sentiment analysis;
- FABSA dataset and task formulation;
- multi-label classification and evaluation;
- domain generalisation;
- open-topic / candidate-label / taxonomy-shift classification;
- structured-output LLM evaluation.

Tier 2 branches intentionally left shallow for now:

- detailed Qwen fine-tuning recipes;
- detailed Gemini token/cost accounting;
- joint aspect-sentiment pair-scoring architectures;
- deep deployment governance.

## Dissertation Argument Map

The related-work chapter should use an hourglass structure:

1. Customer feedback and online reviews create a long-standing need for structured opinion mining.
2. ABSA narrows this into aspect-specific sentiment, where one text can contain multiple aspect-polarity pairs.
3. FABSA gives this project a public, multi-domain, multi-label customer-feedback dataset with organisation/domain metadata.
4. Multi-label evaluation explains why sample-level F1, micro F1, macro F1, thresholding, and per-label diagnostics are all needed.
5. Domain generalisation separates cross-organisation/customer shift from the harder taxonomy/new-aspect shift.
6. Candidate-label and zero-shot methods motivate representing labels as inputs rather than fixed classifier-head outputs.
7. Structured-output LLMs are a natural next model family, but must be evaluated for schema reliability, invalid outputs, latency, token usage, and cost alongside F1.

## Core Evidence Matrix

| Theme | Source | What It Establishes | Limitation | Use In This Dissertation |
| --- | --- | --- | --- | --- |
| Customer review mining | Hu and Liu, 2004, *Mining and Summarizing Customer Reviews*. Local: `Project_Preparation/Background_Paper/1014052.1014073.pdf`. External: ACM DOI `10.1145/1014052.1014073`. | Review mining is not just document summarisation: useful summaries can be structured around product/service features and positive/negative opinions. | Early, product-review-focused, pre-neural, feature-extraction approach. | Use in the wide opening to show that extracting aspects and opinions from customer feedback is a foundational problem, not a recent LLM invention. |
| Customer review mining / sentiment analysis | Pang and Lee, 2008, *Opinion Mining and Sentiment Analysis*. External DOI: `10.1561/9781601981516`. | Provides the broader opinion-mining field context and connects subjectivity, sentiment, and opinion-oriented information access. | Broad survey, not specific to candidate-label ABSA or taxonomy shift. | Use to introduce opinion mining before narrowing to ABSA and FABSA. |
| ABSA shared-task context | Pontiki et al., 2016, *SemEval-2016 Task 5: Aspect Based Sentiment Analysis*. External: ACL Anthology `S16-1002`. | Establishes ABSA as a shared-task family with aspect categories, sentiment polarity, multiple domains/languages, and common evaluation procedures. | Benchmark domains are narrower than FABSA and do not directly address evolving customer taxonomies. | Use to place FABSA in the ABSA tradition and explain why aspect categories and polarity are standard ABSA outputs. |
| Aspect-conditioned sentiment | Tang et al., 2016, *Aspect Level Sentiment Classification with Deep Memory Network*. Local: `Project_Preparation/Background_Paper/D16-1021.pdf`. External: ACL Anthology `D16-1021`. | Aspect-level sentiment depends on the target aspect; models should focus on context words relevant to the given aspect. | Sentence-level aspect-term setting is narrower than FABSA's document-level, multi-label aspect-category task. | Use to justify replacing global document sentiment with aspect-conditioned sentiment in the strongest local non-LLM baseline. |
| ABSA survey | Zhang et al., 2023, *A Survey on Aspect-Based Sentiment Analysis: Tasks, Methods, and Challenges*. External DOI: `10.1109/TKDE.2022.3230975`; arXiv `2203.01054`. | Summarises ABSA task variants, methods, datasets, and open challenges. | Survey coverage is broad; only selected parts should be used. | Use to avoid over-relying on one older ABSA paper and to define the ABSA task landscape cleanly. |
| Generative ABSA | Yan et al., 2021, *A Unified Generative Framework for Aspect-Based Sentiment Analysis*. External: ACL Anthology `2021.acl-long.188`. | Shows ABSA subtasks can be reframed as structured sequence generation rather than only discriminative classification. | Uses standard ABSA benchmarks; not designed for candidate-label taxonomy shift or JSON reliability. | Use as a bridge from discriminative ABSA to structured generative outputs, while keeping it Tier 1 rather than deep LLM tuning. |
| Instruction ABSA | Scaria et al., 2024, *InstructABSA: Instruction Learning for Aspect Based Sentiment Analysis*. External: ACL Anthology `2024.naacl-short.63`. | Shows instruction learning can unify and improve several ABSA subtasks. | Not the same as FABSA held-out-aspect candidate-label evaluation. | Use to motivate Qwen/Gemini as instruction-following ABSA-style candidate-label models without making the dissertation purely about LLMs. |
| FABSA dataset | Kontonatsios et al., 2023, *FABSA: An aspect-based sentiment analysis dataset of user reviews*. Local: `Project_Preparation/Recent_Paper/1-s2.0-S0925231223009906-main.pdf`. External DOI: `10.1016/j.neucom.2023.126867`. | Provides the core dataset: about 10,574 feedback reviews, 14 organisations, 10 domains, 12 child aspects, 3 sentiments, 36 aspect-sentiment labels, and multi-label document-level ABSA. | Original paper focuses on fixed taxonomy and does not directly test held-out aspects or evolving taxonomies. | Use as the dataset anchor; then explain this project extends the evaluation toward organisation shift and taxonomy/new-aspect shift. |
| FABSA candidate-aspect modelling | Kontonatsios et al., 2023, FABSA sentence-pair models. | The FABSA paper includes a sentence-pair setup `(review, target aspect) -> absent/positive/negative/neutral`, and notes that such models can potentially handle zero-shot aspects, though inference grows with candidate aspects. | The original experiments are not this project's held-out-aspect protocol. | Use to show that candidate-aspect scoring is already natural for FABSA, then position this project as making the taxonomy-shift evaluation explicit. |
| Multi-label learning | Tsoumakas and Katakis, 2007/2008, *Multi-Label Classification: An Overview* / *Multi-Label Classification*. External DOI for chapter version: `10.4018/978-1-59904-951-9.ch006`. | Introduces multi-label classification, where an instance can belong to several labels simultaneously. | General ML overview, not customer feedback specific. | Use to define why one review can have multiple aspect-sentiment labels and why binary/multiclass framing is insufficient. |
| Multi-label algorithms and metrics | Zhang and Zhou, 2014, *A Review on Multi-Label Learning Algorithms*. External DOI: `10.1109/TKDE.2013.39`. | Covers formal definitions, multi-label algorithms, and evaluation metrics. | Not NLP-specific and not tied to FABSA. | Use to justify reporting sample-level F1, micro F1, macro F1, and thresholding decisions instead of a single accuracy number. |
| Domain adaptation in sentiment | Blitzer et al., 2007, *Biographies, Bollywood, Boom-boxes and Blenders: Domain Adaptation for Sentiment Classification*. External: ACL Anthology `P07-1056`. | Sentiment classifiers trained in one domain can degrade in another; annotation for every target domain is impractical. | Classic domain adaptation paper; not ABSA or multi-label FABSA. | Use to motivate held-out organisation evaluation and explain why random/provided splits can overstate deployment readiness. |
| Domain adaptation survey | Ramponi and Plank, 2020, *Neural Unsupervised Domain Adaptation in NLP - A Survey*. External: ACL Anthology `2020.coling-main.603`. | Reviews NLP domain-shift settings and methods when labelled target-domain data is unavailable. | Focuses on unsupervised adaptation rather than this project's supervised held-out organisation protocol. | Use to frame domain shift as a recognised NLP problem, while clarifying that this project evaluates rather than solves full domain adaptation. |
| Open-domain topic classification | Ding et al., 2023, *Towards Open-Domain Topic Classification*. Local: `Project_Preparation/Recent_Paper/2306.17290_Towards_Open_Domain_Topic_Classification.pdf`. External: arXiv `2306.17290` / ACL demo. | Introduces user-defined taxonomies at inference time and label-aware zero-shot classification. It explicitly treats labels as semantic inputs. | Topic classification is not ABSA, does not model sentiment, and is not FABSA-specific. | This is the main conceptual bridge from fixed-label FABSA classifiers to candidate-label taxonomy-shift ABSA. |
| Zero-shot text classification | Yin, Hay, and Roth, 2019, *Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach*. External: ACL Anthology `D19-1404`. | Provides a standard zero-shot text-classification benchmark and an entailment-style way to compare texts with label descriptions. | Mostly single-label / benchmark-focused; not aspect-sentiment pair prediction. | Use to justify label-aware candidate scoring and to explain why unseen labels can be handled by representing labels as text. |
| Model-family comparison | Yu et al., 2023, *Open, Closed, or Small Language Models for Text Classification?*. Local: `Project_Preparation/Recent_Paper/2308.10092v1.pdf`. External: arXiv `2308.10092`. | Compares closed LLMs, open LLMs, and smaller supervised models; smaller supervised models can match or beat generative LLMs on many classification tasks, while closed models may help on highly generalisable tasks. | Tasks differ from FABSA and not focused on taxonomy shift. | Use to justify keeping a strong local DistilBERT baseline and evaluating Qwen/Gemini empirically rather than assuming LLM superiority. |
| Zero-shot ACSA with LLMs | Ventirozos et al., 2025/2026, *Exploring Zero-Shot ACSA with Unified Meaning Representation in Chain-of-Thought Prompting*. Local: `Project_Preparation/Recent_Paper/2512.19651v1.pdf`. External: arXiv `2512.19651`. | Directly studies zero-shot ACSA using Qwen3-4B, Qwen3-8B, and Gemini-2.5-Pro, and finds structured reasoning effects are model-dependent. | Preliminary, prompt-dependent, and not the same indexed candidate-label JSON protocol as this project. | Use cautiously as evidence that Qwen/Gemini ABSA prompting is relevant but must be measured carefully. |
| Structured LLM outputs | Willard and Louf, 2023, *Efficient Guided Generation for Large Language Models*. External: arXiv `2307.09702`. | Shows guided generation can enforce output structure using regular expressions / grammars and finite-state machinery. | Method paper, not ABSA and not a task evaluation of Gemini/Qwen. | Use to support the need for structured output control and to explain why JSON/schema validity is a measurable part of LLM evaluation. |
| Structured-output restrictions | Tam et al., 2024, *Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models*. External: ACL Anthology `2024.emnlp-industry.91`. | Studies whether JSON/XML-style format restrictions affect LLM performance; structured generation improves parseability but can interact with reasoning and task behaviour. | Not specific to ABSA or candidate-label classification. | Use to avoid assuming JSON mode is purely beneficial; record both schema reliability and task F1. |
| Qwen PEFT for ABSA | Lim et al., 2026, *Parameter-Efficient Adaptation of Qwen2.5 for Aspect-Based Sentiment Analysis Using Low-Rank Adaptation and Parameter-Efficient Fine-Tuning*. Local: `Project_Preparation/Recent_Paper/engproc-128-00015.pdf`. | Shows Qwen2.5 can be adapted to ABSA using LoRA/PEFT under resource constraints. | SemEval Laptop setting, accuracy metric, and no taxonomy-shift evaluation. | Keep as Tier 1 support for future Qwen feasibility, but do not deep-dive into Qwen fine-tuning until the next modelling phase. |

## Theme Notes

### 1. Customer Review Mining

Core claim:

- The dissertation begins from customer feedback analytics, where the goal is not just to classify overall sentiment but to turn many comments into structured signals about what users discuss and how they feel.

Best sources:

- Hu and Liu, 2004.
- Pang and Lee, 2008.

Project connection:

- FABSA follows this older feature/opinion structure but updates it into a manually labelled, multi-domain ABSA dataset.
- The project's output is also structured: aspect-sentiment pairs rather than free-form summaries.

Writing caution:

- Do not spend many pages on pre-neural feature mining. Use it to set up the problem, then move quickly to ABSA.

### 2. Aspect-Based Sentiment Analysis

Core claim:

- Global document sentiment is not enough because one review can express different sentiment toward different aspects.

Best sources:

- Pontiki et al., 2016.
- Tang et al., 2016.
- Zhang et al., 2023.
- Yan et al., 2021.
- Scaria et al., 2024.

Project connection:

- The local experiment confirms this modelling issue empirically: global sentiment was a limitation of the early held-out-aspect pipeline, and DistilBERT aspect-conditioned sentiment improves the cleaner strong pipeline.

Important project evidence:

- Candidate-aspect DistilBERT + global sentiment, `example_filtered`: pair samples F1 `0.5816`.
- Candidate-aspect DistilBERT + DistilBERT aspect-conditioned sentiment, `example_filtered`: pair samples F1 `0.6071`.
- Sentiment accuracy when gold aspect predicted in the strongest local baseline: `0.9100`.

Writing caution:

- Do not claim every aspect-conditioned implementation is better. The lightweight TF-IDF aspect-conditioned sentiment ablation was weaker; the evidence supports stronger aspect-conditioned modelling, not the shallow variant.

### 3. FABSA Dataset

Core claim:

- FABSA is the empirical anchor because it is public, customer-feedback-focused, multi-domain, multi-label, and annotated with aspect+sentiment labels.

Key dataset facts:

- `10,574` feedback reviews.
- `14` anonymised organisations.
- `10` industries/domains.
- `12` child aspects under `7` parent aspect categories.
- `3` sentiment polarities: positive, negative, neutral.
- `36` aspect-sentiment pair classes.
- Document-level multi-label annotation.

Project connection:

- The original FABSA paper supports fixed-taxonomy ABSA.
- This dissertation extends evaluation by adding:
  - provided closed-topic split;
  - held-out organisation split;
  - fixed held-out aspect split;
  - leave-one-aspect-out diagnostics.

Writing caution:

- The held-out-aspect and LOAO protocols are this project's evaluation design, not claims made by the FABSA paper.

### 4. Multi-Label Evaluation

Core claim:

- FABSA requires multi-label metrics because one review can have multiple valid aspect-sentiment labels, and partial matches matter.

Best sources:

- Tsoumakas and Katakis, 2007/2008.
- Zhang and Zhou, 2014.

Project connection:

- Pair samples F1 reflects per-review prediction quality and is the current headline metric.
- Pair micro F1 captures overall label-decision quality.
- Pair macro F1 exposes long-tail performance and rare labels.
- Aspect-only metrics and sentiment-when-aspect-correct diagnostics separate aspect selection errors from sentiment errors.

Important project evidence:

- Closed-topic DistilBERT test pair samples F1: `0.7803`.
- Held-out organisation DistilBERT test pair samples F1: `0.7575`.
- Held-out aspect best local non-LLM test pair samples F1: `0.6071`.

Writing caution:

- Explain the LOAO all-row metric issue clearly: sample-level F1 does not reward true-negative empty rows and can encourage over-prediction unless micro F1 and false-positive diagnostics are also reported.

### 5. Domain Generalisation

Core claim:

- Cross-organisation/domain shift is a realistic deployment concern, but it is not the same as taxonomy/new-aspect shift.

Best sources:

- Blitzer et al., 2007.
- Ramponi and Plank, 2020.
- FABSA paper's multi-domain motivation.

Project connection:

- Held-out organisation results show that a strong encoder can retain much of closed-topic performance under organisation shift.
- Held-out aspect results are lower, which supports separating domain shift from taxonomy shift.

Important project evidence:

- Closed-topic DistilBERT test pair samples F1: `0.7803`.
- Held-out organisation DistilBERT test pair samples F1: `0.7575`.
- Held-out aspect best fixed non-LLM test pair samples F1: `0.6071`.

Writing caution:

- Do not overclaim solved domain adaptation. The project evaluates held-out organisations; it does not implement a full domain-adaptation algorithm.

### 6. Open-Topic / Candidate-Label / Taxonomy Shift

Core claim:

- Fixed classifier heads are insufficient when the candidate taxonomy can change. Candidate-label methods are appropriate because labels are represented as inputs.

Best sources:

- Ding et al., 2023.
- Yin, Hay, and Roth, 2019.
- FABSA sentence-pair setup.

Project connection:

- Candidate-aspect DistilBERT scores `(review text, candidate aspect) -> relevance`.
- Qwen/Gemini prompts should consume indexed candidate labels and output candidate IDs.
- LOAO is needed because one fixed held-out-aspect choice may be fragile.

Important project evidence:

- Qwen indexed zero-shot test pair samples F1: `0.5374`.
- Strongest local candidate-label non-LLM fixed held-out-aspect result: `0.6071`.
- Lexical all-row LOAO shows large performance spread by held-out aspect and strong threshold sensitivity.

Writing caution:

- Prefer "taxonomy shift", "new-aspect generalisation", or "candidate-label classification" over broad "open vocabulary" unless carefully qualified.
- This project selects from supplied candidate labels; it does not discover arbitrary new topics.

### 7. Structured-Output LLM Evaluation

Core claim:

- LLMs are attractive candidate-label models because they can consume natural-language labels and produce structured outputs, but they must be evaluated on operational reliability as well as F1.

Best sources:

- Yu et al., 2023.
- Ventirozos et al., 2025/2026.
- Willard and Louf, 2023.
- Tam et al., 2024.
- Aji's 2026-06-29 guidance.

Project connection:

- Qwen indexed prompt already showed that candidate IDs reduce brittle string-copying failures.
- Gemini should use JSON mode / response format if supported.
- LLM evaluation should record valid JSON rate, schema-valid rate, hallucinated candidates, invalid sentiments, latency, input tokens, visible output tokens, reasoning/thinking tokens, billable tokens, and estimated cost.

Writing caution:

- Do not frame JSON output as only an implementation detail. It is part of the evaluation protocol because invalid or hallucinated outputs directly affect practical usefulness.
- Do not call Gemini cheap unless reasoning/thinking tokens are included in the cost comparison.

## Literature Gaps To Fill Next

These are still Tier 1 gaps, but they can be filled selectively rather than by deep rabbit-hole reading:

| Gap | Why It Matters | Candidate Sources / Search Direction |
| --- | --- | --- |
| Exact multi-label metric citation for sample-F1 / example-based F1 | Needed for defensible metric definitions. | Zhang and Zhou 2014; Tsoumakas and Katakis 2007/2008; scikit-learn metric docs only as implementation reference, not main literature. |
| Cross-domain ABSA | Would bridge domain generalisation and ABSA more directly than generic sentiment adaptation. | Search: `cross-domain aspect based sentiment analysis survey`, `domain adaptation aspect category sentiment analysis`. |
| Dynamic taxonomies in industrial classification | Would support the "evolving customer-feedback taxonomy" motivation directly. | Search: `taxonomy shift text classification`, `dynamic taxonomy text classification`, `label taxonomy evolution NLP`. |
| Structured-output LLM reliability beyond constrained decoding | Needed if Gemini/Qwen become a main chapter. | Search: `structured generation LLM JSON validity`, `schema-constrained LLM output evaluation`, `format restrictions LLM performance`. |

## Draft Related-Work Spine

The first related-work draft can be organised as:

1. **Customer Feedback And Opinion Mining**
   - Start with Hu and Liu plus Pang and Lee.
   - Explain structured feature/opinion extraction.
2. **Aspect-Based Sentiment Analysis**
   - Move to ABSA shared tasks and aspect-conditioned sentiment.
   - Explain why global sentiment is inadequate.
3. **FABSA As The Project Dataset**
   - Describe FABSA's multi-domain, multi-label, aspect-sentiment setup.
   - Identify the gap: fixed taxonomy in the original paper.
4. **Multi-Label Evaluation**
   - Define multi-label prediction and justify sample, micro, and macro F1.
   - Explain aspect-only and sentiment-conditioned diagnostics.
5. **Generalisation Under Shift**
   - Separate organisation/domain shift from taxonomy/new-aspect shift.
6. **Candidate-Label And Zero-Shot Classification**
   - Use label-aware and entailment-style methods to motivate candidate labels.
   - Position held-out aspects and LOAO as the project's taxonomy-shift evaluation.
7. **Structured-Output LLMs For Candidate-Label ABSA**
   - Treat Qwen/Gemini as structured candidate-label classifiers.
   - Require output reliability, latency, and token-cost reporting.

## Immediate Writing Decisions

- Use FABSA and the strongest local non-LLM baseline as the empirical anchor until LLM experiments are completed.
- Keep Qwen/Gemini important, but conditional: they become central if the next experiments are completed rigorously.
- Keep the narrow topic as candidate-label aspect-sentiment classification under taxonomy shift, with domain shift as an important comparison axis.
- Treat joint pair scoring as a possible modelling extension, not a Tier 1 literature-review deep dive.
- Avoid free-form "open vocabulary" wording unless the text immediately clarifies that the model selects from supplied candidate labels.
