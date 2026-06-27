# Pre-Meeting Brief For Dr Kaan Aksit

Prepared by: Jialin (Lester) Liu  
Project: UCL MSc Data Science and Machine Learning industrial project with Chattermill  
Date: 18 June 2026  
Repository: <https://github.com/jl701/msc-project-chattermill-topic-classification>

## Purpose Of This Brief

This brief summarises the current state of my MSc project before our next supervision meeting. It is written as a short orientation note, assuming that the project details may not be fresh in memory. I would especially appreciate feedback on whether the current research direction, evaluation protocol, and planned next experiments are academically well framed.

## Project Context In Plain Terms

Chattermill works with large collections of customer feedback, such as app reviews, surveys, support tickets, and public reviews. A key product capability is to assign useful business topic labels to each piece of feedback, so that companies can track recurring issues and opportunities.

This project asks whether modern NLP models, including small language models, can assign these labels reliably when the data changes. The practical challenge is not only to perform well on a familiar fixed label set, but also to generalise to feedback from new companies and to new topic labels that may be introduced later.

The task is not ordinary single-label sentiment classification. One review can mention several topics, and each topic can have its own sentiment.

## Project Summary

The project studies multi-label topic classification for customer feedback in an industrial setting with Chattermill. Each review may express multiple aspect and sentiment labels, such as:

```text
Online experience: App website | negative
Staff support: Email | negative
```

The working project direction is:

> Open-vocabulary topic classification with LLMs for customer feedback, with emphasis on generalisation to new companies and new topic labels.

The current public experimental dataset is FABSA, a Chattermill-related aspect-based sentiment analysis dataset of user reviews. It contains 10,574 reviews, 14 anonymised organisations, 12 aspect categories, and aspect+sentiment labels.

Terminology used below:

- Aspect / topic: a business label such as `Online experience: App website`.
- Sentiment: positive, negative, or neutral sentiment for that aspect.
- Aspect+sentiment pair: the final prediction unit, such as `Online experience: App website | negative`.
- Candidate labels: a supplied list of allowed labels that the model must choose from, instead of inventing new label names.

## Current Research Questions

The project is now focused on two generalisation axes:

1. Cross-organisation generalisation: can a model trained on some organisations generalise to feedback from unseen organisations?
2. Open-topic / held-out-aspect generalisation: can a model handle topic labels that were not supervised during training, when those candidate labels are provided at inference time?

The second axis is the more novel part of the project. A fixed classifier head is not sufficient for labels that are unseen during training, so the current direction is to evaluate label-aware and LLM-based candidate-label methods.

## Evaluation Protocol

Headline metric:

- Sample-level F1 computed over aspect+sentiment pair labels.

Supporting metrics:

- Pair micro F1.
- Pair macro F1.
- Aspect-only diagnostic F1.

Current splits:

- Closed-topic benchmark: the original FABSA train/validation/test split.
- Held-out organisation split: train excludes selected organisations; validation uses organisation `600`; test uses organisations `369` and `727`. The label taxonomy remains fixed, so this isolates company/domain shift.
- Held-out aspect split: the held-out aspects are `Account management: Account access`, `Company brand: Competitor`, and `Value: Discounts promotions`. Validation and test evaluate only held-out aspect+sentiment labels, with candidate labels provided at inference.

For held-out aspect training, I tested two variants:

- Label-masked: keep training rows but remove held-out aspect labels from them.
- Example-filtered: remove any training row that contains a held-out aspect.

## Current Results

All headline values below are sample-level F1 over aspect+sentiment pairs.

| Setting | Model | Test F1 |
| --- | --- | ---: |
| Closed-topic provided split | Word+char TF-IDF + Linear SVM | 0.7090 |
| Closed-topic provided split | DistilBERT | 0.7803 |
| Held-out organisation | Word+char TF-IDF + Linear SVM | 0.7026 |
| Held-out organisation | DistilBERT | 0.7575 |
| Held-out aspect, label-masked | Lexical candidate-label lower bound + global sentiment | 0.4698 |
| Held-out aspect, example-filtered | Lexical candidate-label lower bound + global sentiment | 0.4626 |
| Held-out aspect, label-masked | Candidate-aspect DistilBERT cross-encoder + global sentiment | 0.5595 |
| Held-out aspect, example-filtered | Candidate-aspect DistilBERT cross-encoder + global sentiment | 0.5816 |
| Held-out aspect | Qwen3-4B-Instruct indexed candidate-label zero-shot | 0.5374 |

Important clarification: Qwen has not yet been fully fine-tuned for the held-out-aspect protocol. So far I have implemented the indexed candidate-label prompt, zero-shot evaluation, error analysis, and GPU-ready SFT/evaluation data preparation. Full Qwen fine-tuning should wait for stronger GPU access.

## Current Interpretation

The initial evidence suggests:

- Closed-topic classification is relatively well handled by a fine-tuned DistilBERT baseline.
- Cross-organisation/domain shift is manageable but still reduces long-tail performance.
- Held-out aspect/open-topic generalisation is substantially harder.
- Candidate-label methods are more appropriate than fixed classifier heads for unseen labels.
- Qwen-style instruction models are promising for the open-topic setting, but the current result is only zero-shot; the next meaningful experiment is full Qwen candidate-label fine-tuning.

## Methodological Grounding

The open-topic Qwen design is not free-form topic generation. It is a constrained candidate-label prediction setup: the model receives a list of canonical candidate aspects and outputs selected aspect IDs and sentiments in JSON.

This direction is motivated by:

- Instruction-tuned ABSA, such as InstructABSA.
- Few-shot instruction tuning for ABSA.
- Unified generative ABSA.
- Entailment-style zero-shot text classification, where candidate labels are represented as text.
- Structured-output prompting to reduce invalid labels and parsing errors.

## Planned Next Work

1. Confirm with Chattermill that the held-out organisation and held-out aspect protocols match the industrial objective.
2. Run full Qwen candidate-label fine-tuning on stronger GPU access.
3. Add at least one more non-LLM label-aware baseline, such as a bi-encoder or NLI-style candidate-label model.
4. Extend error analysis, especially around over-prediction and sentiment errors for held-out aspects.
5. Start turning the experimental design into dissertation structure and report figures.

## Questions For The Meeting

1. Does the current framing of the academic contribution make sense: comparing fixed-taxonomy, cross-organisation, and open-topic generalisation?
2. Is the held-out aspect protocol academically defensible, given that evaluation is held-out-label-only with candidate labels supplied at inference?
3. Should the dissertation prioritise the open-topic/new-aspect axis over further tuning of closed-topic baselines?
4. What practical metrics should be included beyond F1, such as inference latency, fine-tuning cost, and deployment complexity?
5. Is there a publication-oriented angle worth shaping early, or should the work remain primarily dissertation-focused?
