# Literature Review Matrix

Last updated: 2026-06-29

This matrix is the Stage 2 working document for the dissertation literature review. It maps locally downloaded papers and known missing areas to the dissertation spine:

```text
Structured candidate-label aspect-sentiment modelling under taxonomy shift in customer feedback.
```

This is not final prose. It is a planning matrix for deciding what each paper contributes and where more literature is needed.

## Review Structure

The related-work chapter should move from broad to narrow:

1. Customer review mining and feedback analytics.
2. Aspect-based sentiment analysis.
3. Multi-label text classification and evaluation.
4. Domain and organisation generalisation.
5. Open-topic / taxonomy-shift classification.
6. Candidate-label, zero-shot, and label-semantics methods.
7. Structured-output LLMs and practical deployment trade-offs.

## Core Local Papers

| Citation / Local File | Theme | Key Problem | Method / Data | Main Finding Or Use | Limitation | How It Supports This Dissertation |
| --- | --- | --- | --- | --- | --- | --- |
| Hu and Liu, 2004, *Mining and Summarizing Customer Reviews*; `Project_Preparation/Background_Paper/1014052.1014073.pdf` | Customer review mining | Large volumes of product reviews need feature/opinion summarisation. | Early feature extraction and opinion summarisation from customer reviews. | Establishes the historical root of mining product/service aspects from feedback. | Older pre-neural, product-review-focused setting. | Use in the broad opening: customer feedback contains aspects/features and opinions that need structured extraction. |
| Tang et al., 2016, *Aspect Level Sentiment Classification with Deep Memory Network*; `Project_Preparation/Background_Paper/D16-1021.pdf` | ABSA / aspect-conditioned sentiment | Sentiment depends on the target aspect, not only on the whole sentence or document. | Deep memory network with attention over context for aspect-level sentiment classification. | Supports the principle that sentiment must be conditioned on the aspect. | Standard benchmark setting is narrower than this project's multi-label candidate-label setup. | Use to justify replacing global sentiment with DistilBERT aspect-conditioned sentiment. |
| Kontonatsios et al., 2023, *FABSA: An aspect-based sentiment analysis dataset of user reviews*; `Project_Preparation/Recent_Paper/1-s2.0-S0925231223009906-main.pdf` | Dataset / ABSA / customer feedback | Existing ABSA datasets are often small, domain-limited, and costly to annotate. | Multi-domain feedback-review ABSA dataset with aspect+sentiment labels. | Still one public dataset with 12 aspects and anonymised organisations. | Central data anchor; use for dataset motivation, task definition, and baseline comparability. |
| Ding et al., 2023, *Towards Open-Domain Topic Classification*; `Project_Preparation/Recent_Paper/2306.17290_Towards_Open_Domain_Topic_Classification.pdf` | Open-domain topic classification | Users may define arbitrary taxonomies at inference time. | Label-aware zero-shot classifier trained to handle unseen labels and evaluated across domains. | Not specifically ABSA and not tied to aspect+sentiment pairs. | Main conceptual bridge from fixed-head classification to candidate-label taxonomy shift. |
| Yu et al., 2023, *Open, Closed, or Small Language Models for Text Classification?*; `Project_Preparation/Recent_Paper/2308.10092v1.pdf` | Model-family comparison | It is unclear when closed LLMs, open LLMs, or smaller supervised models are preferable for classification. | Empirical comparison across model families and classification tasks. | Different tasks from FABSA and not focused on taxonomy shift. | Use to motivate fair local-vs-LLM comparison and avoid assuming LLMs are always superior. |
| Ventirozos et al., *Exploring Zero-Shot ACSA with Unified Meaning Representation in Chain-of-Thought Prompting*; `Project_Preparation/Recent_Paper/2512.19651v1.pdf` | Zero-shot ACSA / LLM prompting | Can LLMs solve aspect-category sentiment analysis without task-specific training? | Zero-shot prompting with UMR/CoT across Qwen/Gemini-style models and multiple datasets. | Preliminary and prompt/model dependent; may not use the same candidate-label protocol. | Use for LLM ABSA prompting context and to justify evaluating Qwen/Gemini carefully rather than assuming success. |
| Lim et al., 2026, *Parameter-Efficient Adaptation of Qwen2.5 for Aspect-Based Sentiment Analysis Using Low-Rank Adaptation and Parameter-Efficient Fine-Tuning*; `Project_Preparation/Recent_Paper/engproc-128-00015.pdf` | Qwen / PEFT / ABSA | How can Qwen-style models be adapted to ABSA with limited trainable parameters? | LoRA/PEFT adaptation for Qwen2.5 on ABSA-style data. | May be a proceedings paper and may not match FABSA taxonomy-shift setup. | Use as support for possible Qwen fine-tuning, not as the central justification for the dissertation. |

## Existing Project Documents As Evidence Sources

| Document | Use In Dissertation Planning |
| --- | --- |
| `docs/evaluation_protocol.md` | Defines closed-topic, held-out organisation, and fixed held-out-aspect protocols. |
| `docs/generalisation_baselines.md` | Main result tables across closed-topic, held-out organisation, held-out aspect, Qwen zero-shot, and local baselines. |
| `docs/loao_heldout_aspect.md` | LOAO motivation, lexical LOAO results, and threshold-selection caveats. |
| `docs/non_llm_open_topic_baseline.md` | Strongest local non-LLM fixed held-out-aspect baseline and reproduction command. |
| `docs/heldout_aspect_error_analysis.md` | Qwen and DistilBERT row-level error categories and prompt-format findings. |
| `docs/qwen_feasibility.md` | Local Qwen feasibility, indexed candidate-label prompt design, and QLoRA context. |
| `docs/aji_updates_2026_06_21.md` | Aji's LOAO, label-masked/example-filtered, global-sentiment, and Gemini access guidance. |
| `docs/aji_updates_2026_06_29.md` | Aji's latest guidance: LOAO is central, use JSON mode if possible, and count reasoning tokens. |

## Theme Matrix

### 1. Customer Review Mining And Feedback Analytics

Purpose:

- Establish the broad industrial problem.
- Explain why customer feedback needs aspect/topic extraction, sentiment, and summarisation.

Current local anchors:

- Hu and Liu, 2004.
- FABSA paper.

Argument to make:

- The dissertation is a modern, supervised/LLM-era version of a long-standing review-mining problem: turning many comments into structured aspect/opinion signals.

Missing or optional literature:

- Recent surveys on customer feedback analytics.
- Industrial customer-experience analytics papers.
- Review mining surveys after deep learning.

Search strings:

```text
customer feedback analytics topic classification survey
customer review mining aspect sentiment survey
opinion mining customer reviews survey
```

### 2. Aspect-Based Sentiment Analysis

Purpose:

- Ground the aspect+sentiment pair target.
- Justify aspect-conditioned sentiment.

Current local anchors:

- Tang et al., 2016.
- FABSA paper.
- Ventirozos et al., zero-shot ACSA.
- Lim et al., Qwen2.5 PEFT for ABSA.

Argument to make:

- FABSA is not document-level sentiment classification. A review can express different sentiment toward different aspects, so global sentiment is methodologically weak.

How project evidence connects:

- Lightweight TF-IDF aspect-conditioned sentiment was weaker than global sentiment.
- DistilBERT aspect-conditioned sentiment improved the controlled lexical setup and the cleaner strong candidate-aspect pipeline.

Missing literature:

- ABSA survey paper.
- Cross-domain ABSA.
- Aspect-category sentiment classification papers.
- Instruction-tuned ABSA, such as InstructABSA.
- Unified generative ABSA.

Search strings:

```text
aspect based sentiment analysis survey
aspect category sentiment analysis survey
cross domain aspect based sentiment analysis
InstructABSA instruction learning aspect based sentiment analysis
unified generative framework aspect based sentiment analysis
```

### 3. Multi-Label Classification And Evaluation

Purpose:

- Explain why FABSA prediction is multi-label.
- Justify sample-level F1, micro F1, macro F1, thresholding, and per-label diagnostics.

Current local anchors:

- FABSA paper.
- Project metrics implementation and result documents.

Argument to make:

- One review can have multiple aspect+sentiment pairs; partial matches matter. Micro F1 captures overall label decisions, macro F1 exposes long-tail label behaviour, and samples F1 reflects per-review prediction quality.

Project-specific notes:

- Pair-level F1 is stricter than aspect-only F1.
- If aspect is correct but sentiment is wrong, pair prediction is wrong.
- In all-row LOAO, sample-F1 threshold selection can over-predict because true-negative empty rows and false-positive empty-gold rows both get row score zero.

Missing literature:

- Multi-label text classification survey.
- Multi-label evaluation metric references.
- Thresholding and label imbalance references.
- Extreme or long-tail classification references if needed.

Search strings:

```text
multi-label text classification survey micro macro sample F1
multi-label classification evaluation metrics sample F1
threshold selection multi-label classification text
long-tail multi-label text classification
```

### 4. Domain And Organisation Generalisation

Purpose:

- Position the held-out organisation experiment.
- Explain why random/provided splits may overstate deployment readiness.
- Separate organisation/domain shift from taxonomy shift.

Current local anchors:

- FABSA paper.
- Project held-out organisation results.

Argument to make:

- A model can generalise reasonably across organisations while still struggling with unseen aspects. Therefore organisation shift and taxonomy shift should be evaluated separately.

Project-specific evidence:

- Closed-topic DistilBERT test pair samples F1: `0.7803`.
- Held-out organisation DistilBERT test pair samples F1: `0.7575`.
- Held-out aspect best fixed non-LLM test pair samples F1: `0.6071`.

Missing literature:

- Domain adaptation in sentiment analysis.
- Domain generalisation in NLP.
- Cross-domain ABSA.
- Customer/company shift in industrial NLP if available.

Search strings:

```text
domain adaptation sentiment analysis survey
domain generalization natural language processing survey
cross-domain aspect based sentiment analysis
cross-company customer feedback classification
```

### 5. Open-Topic / Taxonomy-Shift Classification

Purpose:

- Narrow the dissertation toward the main topic.
- Explain why fixed classifier heads fail when labels change.

Current local anchor:

- Ding et al., 2023, *Towards Open-Domain Topic Classification*.

Argument to make:

- The dissertation studies a constrained and deployable version of open-topic classification: new candidate labels are supplied at inference, and the model selects from them rather than inventing labels.

Project-specific notes:

- Avoid overclaiming "open vocabulary".
- Prefer "candidate-label taxonomy shift" or "new-aspect generalisation".
- Fixed held-out aspects test one taxonomy shift; LOAO tests robustness across all aspects.

Missing literature:

- Open-set text classification.
- Generalized zero-shot text classification.
- Dynamic taxonomy / taxonomy expansion.
- Label semantics for text classification.

Search strings:

```text
open domain topic classification label aware classifier
zero-shot text classification label descriptions
open set text classification survey NLP
taxonomy shift text classification
dynamic taxonomy classification NLP
```

### 6. Candidate-Label, Zero-Shot, And NLI-Style Classification

Purpose:

- Justify candidate-label scoring as the natural modelling form.
- Connect DistilBERT cross-encoders, Qwen/Gemini prompts, and possible joint pair scoring.

Current local anchors:

- Ding et al., 2023.
- Yu et al., 2023.
- Ventirozos et al., zero-shot ACSA.

Argument to make:

- Candidate-label methods encode labels as inputs, not only output-head indices. This enables inference over labels that were not supervised during training.

Project-specific links:

- Candidate-aspect DistilBERT selector scores `(review, candidate aspect)`.
- Joint pair scorer would score `(review, candidate aspect, candidate sentiment)`.
- Qwen/Gemini produce structured candidate-label pair outputs.

Missing literature:

- NLI-based zero-shot classification.
- Label-description-based classification.
- SetFit or efficient few-shot classification if comparing small models.
- Calibration and thresholding for zero-shot/candidate-label multi-label classification.

Search strings:

```text
NLI zero-shot text classification candidate labels
Benchmarking zero-shot text classification datasets evaluation entailment
label description text classification zero-shot
SetFit efficient few-shot learning without prompts
calibration zero-shot multi-label text classification
```

### 7. Structured-Output LLMs And Deployment Trade-Offs

Purpose:

- Position Qwen/Gemini without turning the dissertation into a generic LLM paper.
- Explain why structured output reliability, latency, and cost matter.

Current local anchors:

- Yu et al., 2023.
- Ventirozos et al., zero-shot ACSA.
- Lim et al., Qwen2.5 PEFT for ABSA.
- Aji's Gemini guidance.

Argument to make:

- LLMs are attractive for evolving taxonomies because they can consume natural-language candidate labels and output structured pairs. However, they must be evaluated on schema reliability, hallucinated labels, latency, token usage, and reasoning-token cost.

Project-specific links:

- Qwen indexed zero-shot fixed held-out-aspect test pair samples F1: `0.5374`.
- Gemini pilot should use JSON mode if supported.
- Gemini cost comparison must include reasoning/thinking tokens.

Missing literature:

- Structured-output / constrained decoding reliability.
- LLM classification benchmarking.
- LLM cost and latency evaluation.
- Privacy/governance trade-offs for hosted APIs.

Search strings:

```text
large language model structured output JSON reliability classification
constrained decoding JSON large language models
LLM classification cost latency evaluation
LLM structured output schema validity
hosted LLM privacy governance industrial NLP
```

## Priority Reading Order

1. FABSA paper.
   - Extract dataset, annotation, domain, and baseline framing.
2. Ding et al., 2023.
   - Extract open-domain topic classification and user-defined taxonomy framing.
3. Tang et al., 2016.
   - Extract aspect-level sentiment rationale.
4. Hu and Liu, 2004.
   - Extract broad customer review mining motivation.
5. Yu et al., 2023.
   - Extract local/open/closed model comparison framing.
6. Ventirozos et al. zero-shot ACSA.
   - Extract structured prompting and Qwen/Gemini ABSA relevance.
7. Lim et al. Qwen2.5 PEFT.
   - Extract Qwen/LoRA feasibility background.
8. Missing multi-label, zero-shot/NLI, domain-generalisation, and structured-output papers.

## Literature Gap Backlog

| Priority | Gap | Why It Matters | Search Target |
| ---: | --- | --- | --- |
| 1 | Multi-label evaluation metrics | Needed to justify samples F1, micro F1, macro F1, thresholding. | Multi-label text classification survey / evaluation metrics. |
| 1 | NLI-style zero-shot classification | Needed to ground candidate-label classification with label text. | Yin et al. zero-shot text classification and related entailment methods. |
| 2 | Cross-domain ABSA / domain generalisation | Needed to justify held-out organisation as a separate axis. | Cross-domain ABSA and NLP domain generalisation surveys. |
| 2 | Instruction-tuned / generative ABSA | Needed to position Qwen/Gemini and structured outputs. | InstructABSA, unified generative ABSA, prompt-based ABSA. |
| 2 | Structured-output LLM reliability | Needed for JSON mode, parse failures, schema validity. | JSON/constrained decoding/structured output LLM papers. |
| 3 | LLM cost/latency evaluation | Needed if Gemini becomes central. | LLM classification cost latency benchmark papers. |
| 3 | Dynamic taxonomy / taxonomy expansion | Would strengthen the taxonomy-shift framing. | Dynamic taxonomy classification / taxonomy evolution NLP. |

## Draft Related-Work Argument

The related-work chapter should argue:

1. Customer feedback has long been studied as review mining: extracting what users discuss and how they feel.
2. ABSA formalises this into aspect-specific sentiment, making global sentiment insufficient.
3. FABSA turns this into a realistic multi-domain, multi-label customer-feedback dataset with aspect+sentiment pair labels.
4. Standard fixed-taxonomy classifiers can perform well when train/test labels match, but this does not answer the industrial problem of evolving taxonomies.
5. Held-out organisation shift and held-out aspect shift are distinct; the current evidence suggests taxonomy shift is harder.
6. Candidate-label and zero-shot methods are natural for taxonomy shift because labels are represented as inputs.
7. Structured-output LLMs are attractive candidate-label models, but must be evaluated for reliability and cost, not just F1.
8. This dissertation therefore studies structured candidate-label aspect+sentiment prediction under taxonomy shift, using local baselines, LOAO robustness, and planned Qwen/Gemini comparisons.

## Notes For Writing

- Do not over-emphasise Qwen/Gemini in the literature review before the experiments are complete.
- Do not make "open vocabulary" sound like free-form topic invention.
- Use "taxonomy shift", "new-aspect generalisation", and "candidate-label classification" as the more precise terms.
- Use Qwen/Gemini literature to motivate future/next-stage model families, not to replace the evaluation spine.
- Make the strongest local baseline the empirical anchor until LLM experiments justify changing the centre of gravity.

