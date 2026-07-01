# Project Overview: Open-Vocabulary Topic Classification With LLMs

Last updated: 2026-06-27  
Local workspace: `C:\Msc_DSML\Msc_Project`  
GitHub repo: `https://github.com/jl701/msc-project-chattermill-topic-classification`  

## 1. One-Sentence Summary

This project aims to build and evaluate a system for assigning multiple topic labels to customer feedback, with a focus on robust generalisation to new or flexible topics and to data from different companies/organisations.

## 2. People And Context

- Student: Jialin (Lester) Liu
- Company: Chattermill
- External supervisor: Aji Ghose
- Additional Chattermill ML contact: Diego Piccinotti
- UCL internal supervisor: Kaan Aksit

The project is an industrial MSc project with Chattermill. Chattermill analyses large volumes of customer feedback, such as surveys, reviews, support tickets, and chats. A key product capability is assigning topic labels to each piece of feedback so that companies can monitor trends and extract actionable insights.

## 3. Latest Project Summary

Working title:

**Open-vocabulary Topic Classification with LLMs**

The project aims to develop and evaluate an LLM-based system for multi-label topic classification of customer feedback in an industrial setting. Each feedback text may be assigned multiple topic labels.

The main challenge is robust generalisation along two axes: **cross-organisation/domain-shift generalisation** and **open-topic/new-aspect generalisation**. The contrast between these two axes is itself a core project result. If one axis must be prioritised, the open-topic/new-aspect setting is likely the more novel and impactful direction.

The dataset is expected to be human-labelled, enabling evaluation against ground-truth annotations.

The work should start with strong baselines, followed by model development and fine-tuning strategies to improve performance. Evaluation should be rigorous and should include error analysis.

Model selection should consider:

- OOD / open-topic classification performance, for example micro F1 and macro F1.
- Cost per 1M responses.
- Time per 1M responses.
- Fine-tuning cost per iteration.

Expected deliverables:

- Reproducible experimental pipeline.
- Clear evaluation protocol.
- Results summary.
- Dissertation-style report with methods, ablations, error analysis, trade-offs, and recommendations.

### Current Working Assumptions

The first implementation phase established the **closed-topic FABSA task**. The model predicts labels from the fixed FABSA taxonomy using the provided train/validation/test splits. The current work has now moved on to the two generalisation settings: held-out organisation and held-out aspect.

The prediction target is the full **aspect + sentiment** pair. For example, `Online experience: App website | negative` and `Online experience: App website | positive` are different labels. If the aspect is correct but the sentiment is wrong, the pair-level prediction is counted as incorrect.

The current headline metric should be:

- Sample-level F1.

The main supporting metrics are:

- Aspect+sentiment pair-level micro F1.
- Aspect+sentiment pair-level macro F1.

Useful diagnostic metrics are:

- Aspect-only micro/macro F1.
- Per-label precision, recall, and F1.
- Sentiment analysis for cases where the aspect is predicted correctly.

The open-topic setting should not be treated as free-form topic generation. The confirmed direction is a **candidate-label setting**: given a feedback text and a list of candidate topics, the model selects the relevant canonical topics and assigns sentiment. This avoids synonym and label-canonicalisation issues such as `App usability` versus `Online experience: App website`.

The provided FABSA split remains the closed-topic benchmark for comparability. New splits should be constructed for strict held-out-organisation evaluation and held-out-aspect evaluation.

## 4. Current Understanding Of The Prediction Task

### Input

One customer feedback text, for example:

```text
The app keeps crashing and customer support never replied.
```

Depending on the final project setup, the model may also receive:

- A fixed topic taxonomy.
- A set of candidate topic names.
- Topic descriptions.
- A user-defined/open-vocabulary topic set.

Current confirmed interpretation: the important open-topic scenario is generalisation to **new topic labels**.

### Output

Multiple aspect + sentiment labels, for example:

```text
Online experience: App website | negative
Staff support: Email | negative
```

This is a **multi-label classification** task, because one feedback item can have more than one relevant aspect/sentiment label.

Confirmed after kickoff:

The output should be **aspect + sentiment**, not aspect/topic only.

### Key Challenge

The model should not only memorise topics or language patterns from known companies. It should perform well when:

- It sees feedback from a new or different company.
- The topic definitions/taxonomy change.
- Customer wording differs across organisations or industries.

## 5. Important Concept: Cross-Company Evaluation

Cross-company evaluation means training on feedback from some companies and evaluating on feedback from different companies.

Example:

```text
Train: companies A, B, C, D
Test: companies E, F
```

This is stricter than a random train/test split. A random split may place examples from the same company in both train and test, which can make the model look stronger than it really is.

Why it matters:

Chattermill needs models that work for new customers. If a model performs well only on companies it has already seen, it may not be useful in real deployment.

## 6. Important Concept: Open Topic / Open Vocabulary

Meeting note from 2026-06-15:

> Aji emphasised that the most important part is the open topic setting and achieving good generalisation when facing new topics.

Confirmed interpretation after kickoff:

The model should generalise to **new topic labels** rather than only performing well on topic labels seen during training.

Possible forms of the open-topic setting:

- Given feedback text and a list of candidate topic names, predict relevant topics.
- Given feedback text and topic descriptions, predict relevant topics.
- Given a new taxonomy from a new customer, adapt without full retraining.
- Generalise to topic labels or topic wording not heavily represented during training.

The first held-out-aspect evaluation protocol has now been defined using candidate labels and held-out aspects. This protocol may still be refined as stronger label-aware baselines are added.

## 7. FABSA Dataset

FABSA is a public Chattermill-related dataset:

- Paper: `FABSA: An aspect-based sentiment analysis dataset of user reviews`
- Local paper path: `Project_Preparation/Recent_Paper/1-s2.0-S0925231223009906-main.pdf`
- Hugging Face dataset: `https://huggingface.co/datasets/jordiclive/FABSA`
- Local exported data path: `Project_Preparation/Public_Datasets/FABSA/`

### Dataset Role

Confirmed after the kickoff discussion:

**FABSA is the main dataset for this project.**

This means the project should be designed around the public FABSA data and its taxonomy, while exploring methods that improve open-topic/open-vocabulary behaviour and generalisation.

FABSA should be treated as:

- The main experimental dataset.
- The primary source for data exploration.
- The primary benchmark for baselines and model development.
- A public dataset that supports reproducibility.

### FABSA Fields

Local exported files:

- `summary.json`
- `train.csv`
- `validation.csv`
- `test.csv`
- `train.jsonl`
- `validation.jsonl`
- `test.jsonl`

Main columns:

- `id`: review identifier.
- `org_index`: anonymised organisation/company identifier.
- `data_source`: source of review, such as Google Play, Apple Store, Trustpilot.
- `industry`: industry/domain.
- `text`: customer feedback text.
- `labels_json`: human-readable aspect + sentiment labels.
- `label_codes`: machine-readable label codes.

### FABSA Size And Distribution

From local `summary.json`:

```text
train: 7,930
validation: 1,057
test: 1,587
total: 10,574
unique organisations: 14
```

Data sources:

```text
Google Play: 6,130
Apple Store: 2,524
Trustpilot: 1,920
```

Industries:

```text
Fashion
Price Comparison
Groceries
Trading
Travel Booking
Banking
Ride Hailing
Information Technology
Consulting
Streaming
```

Common aspects:

```text
Online experience: App website
Company brand: General satisfaction
Purchase booking experience: Ease of use
Staff support: Attitude of staff
Value: Price value for money
Logistics rides: Speed
Company brand: Competitor
Account management: Account access
Value: Discounts promotions
Staff support: Phone
Company brand: Reviews
Staff support: Email
```

Important distinction:

FABSA labels are **aspect + sentiment** labels, such as:

```text
Online experience: App website | negative
Company brand: General satisfaction | positive
```

Confirmed after kickoff:

The project should predict **aspect + sentiment** labels.

### Current Split Interpretation

The local FABSA export has standard train/validation/test splits:

```text
train: 7,930 rows
validation: 1,057 rows
test: 1,587 rows
```

Local inspection showed that all three splits contain the same 12 aspects. Therefore, the provided splits are suitable for closed-topic baselines but do not directly evaluate unseen-topic generalisation.

The split is also not a held-out-company split. The test organisations overlap with the training organisations. A strict held-out-organisation split should be constructed separately.

### Related Demo Tool

Aji's FABSA Tagger demo appears to expose the 12-aspect taxonomy and a Qwen3-4B LoRA tagging interface. This is useful as a reference for the expected task shape and future model direction, but the repository should still keep experiments reproducible from the local FABSA files.

## 8. Literature And Existing Resources

### Background Papers

1. Hu and Liu, 2004: `Mining and Summarizing Customer Reviews`
   - Early feature/aspect-based customer review mining.
   - Useful for understanding why customer feedback should be structured into features/topics.

2. Tang et al., 2016: `Aspect Level Sentiment Classification with Deep Memory Network`
   - Shows that one text can express different sentiment about different aspects.
   - Useful background for fine-grained feedback analysis.

### Recent / More Relevant Papers

1. FABSA paper
   - Public Chattermill-related customer feedback dataset.
   - Multi-domain, multi-label, human-labelled aspect + sentiment data.

2. Ding et al., 2022: `Towards Open-Domain Topic Classification`
   - Local path: `Project_Preparation/Recent_Paper/2306.17290_Towards_Open_Domain_Topic_Classification.pdf`
   - Note: the local PDF is the 2023 arXiv copy; the formal citation should use the NAACL 2022 System Demonstrations version.
   - Important for the open-topic/open-vocabulary direction.
   - Focuses on classifying text into user-provided or open-domain topic categories.

3. `Open, Closed, or Small Language Models for Text Classification?`
   - Useful for comparing closed LLMs, open LLMs, and smaller supervised models.
   - Supports the idea that smaller/fine-tuned models may be competitive and cheaper.

4. Qwen / LoRA / PEFT ABSA paper
   - Useful for possible Qwen fine-tuning strategy.

5. Zero-shot ACSA with CoT / UMR paper
   - Useful for prompt-based LLM baselines and zero-shot classification approaches.

6. Scaria et al., 2024: `InstructABSA: Instruction Learning for Aspect Based Sentiment Analysis`
   - Supports reformulating ABSA subtasks as instruction-following generation tasks.
   - Relevant to the Qwen candidate-label prompt and future Qwen SFT direction.

7. Varia et al., 2023: `Instruction Tuning for Few-Shot Aspect-Based Sentiment Analysis`
   - Supports instruction fine-tuning for ABSA in low-resource/few-shot settings.
   - Relevant to fine-tuning Qwen on seen FABSA aspects before evaluating held-out aspects.

8. Zhang et al., 2021: `A Unified Generative Framework for Aspect-Based Sentiment Analysis`
   - Supports structured generative formulations for ABSA outputs.
   - Relevant to JSON-style aspect/sentiment output parsing.

9. Yin et al., 2019: `Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach`
   - Supports candidate-label and unseen-label classification by representing labels as text rather than fixed classifier outputs.
   - Relevant to held-out aspect evaluation and candidate-aspect scoring.

## 9. Baseline And Model Roadmap

Meeting note from 2026-06-15:

Baseline/model candidates mentioned:

1. Bag-of-Words / simple lexical baseline
   - A phrase was initially heard as "back and forth"; likely this was **Bag-of-Words** or BoW.
   - Should confirm if needed.

2. TF-IDF plus classical classifier
   - TF-IDF + Logistic Regression.
   - TF-IDF + XGBoost was mentioned.

3. BERT-style model
   - Possibly BERT or another encoder model.
   - Status: maybe / not fully confirmed.

4. Small language model
   - Qwen3-4B.
   - Fine-tuning Qwen3-4B appears important.
   - This fine-tuning is expected to be done as part of the project, rather than relying only on a pre-existing model.

5. Expected fine-tuned Qwen performance
   - Approximate internal reference: `0.74-0.75 F1`.
   - This should not be treated as a precise target.
   - Headline reporting should use sample-level F1, with micro and macro F1 alongside it.

Additional meeting notes:

- XGBoost was mentioned as an example/suggestion for a classical baseline, not necessarily a hard requirement.
- BERT-style baselines are worth doing if feasible. The general advice was to do as many useful baselines as possible early; extra baselines can be dropped later if they become less relevant.

## 10. Metrics

### Micro F1

Micro F1 pools all label decisions together before computing F1. It is strongly influenced by frequent labels.

Useful for:

- Overall system performance.
- High-volume business performance.

Risk:

- A model can get strong micro F1 while performing poorly on rare topics.

### Macro F1

Macro F1 computes F1 per label and then averages over labels. Every label gets equal weight.

Useful for:

- Long-tail topic performance.
- Whether the model treats rare labels reasonably.

Risk:

- Can be unstable when some labels have very few examples.

### Other Useful Metrics

Possible future metrics:

- Sample-level F1 as the headline metric.
- Per-label F1.
- Per-company F1.
- Per-topic F1.
- Aspect-only diagnostic F1.
- Sentiment correctness when the aspect is correct.
- Precision/recall at different thresholds.
- Invalid output rate for LLM prompting.
- Cost per 1M responses.
- Time per 1M responses.
- Fine-tuning cost per iteration.

## 11. Practical Deployment Metrics

The project should not select models based on F1 alone.

Practical metrics:

- Inference cost per 1M responses.
- Time per 1M responses.
- Training/fine-tuning cost per iteration.
- Complexity of deployment.
- Whether external APIs are allowed with company data.

This is important because a slightly better LLM may be too slow or too expensive for high-volume customer feedback processing.

## 12. Proposed Workflow

### Phase 0: Repository And Environment Setup

Current immediate task:

- Create private GitHub repository.
- Invite Aji and Diego.
- Keep repo clean and avoid committing internal data.
- Add basic README, requirements, and project overview.

Potential Python environment:

```text
pandas
numpy
scikit-learn
datasets
transformers
torch
matplotlib / seaborn
```

Later if needed:

```text
peft
accelerate
sentence-transformers
xgboost
```

### Phase 1: Clarify Data And Task Definition

Clarify:

- Is the main dataset FABSA or internal Chattermill data?
- Is the task topic-only or topic + sentiment?
- Is the label set fixed, open-vocabulary, or user-defined?
- Is there a company/organisation identifier?
- Can the data be used with external LLM APIs?
- What output format should the model produce?

### Phase 2: Data Exploration

For whichever dataset is used:

- Count examples.
- Count companies/organisations.
- Count topics.
- Count labels per example.
- Inspect topic frequency / long-tail distribution.
- Inspect company/topic distribution.
- Inspect short texts, long texts, ambiguous texts.
- Inspect examples where multiple labels appear.

### Phase 3: Evaluation Protocol

Design splits that reflect the project goal.

Possible settings:

1. Random split baseline.
2. Cross-company split:
   - Train on some companies.
   - Test on held-out companies.
3. Open-topic split:
   - Train with some topic definitions.
   - Evaluate on new/flexible topic definitions.
4. Combined setting:
   - New company plus new/flexible topic definitions.

The first version of this protocol has now been confirmed and implemented for the provided split, held-out organisation, and held-out aspect settings. Future work may add a combined held-out organisation plus held-out aspect setting.

### Phase 4: Baselines

Start simple:

- Bag-of-Words / TF-IDF + Logistic Regression.
- TF-IDF + XGBoost.

Then stronger:

- BERT-style encoder classifier.
- Possibly sentence-transformer embeddings + classifier.

LLM baselines:

- Zero-shot prompt.
- Few-shot prompt, if allowed.
- Candidate-label prompting.

### Phase 5: Model Development

Potential routes:

- Fine-tune BERT/encoder model for multi-label classification.
- Fine-tune Qwen3-4B.
- Use LoRA / PEFT for parameter-efficient fine-tuning.
- Explore topic descriptions or label-aware inputs for open-topic generalisation.
- Consider calibration/threshold tuning.

### Phase 6: Error Analysis

Classify errors into categories:

- Confusion between similar topics.
- Missed secondary labels.
- Rare topic failures.
- Company-specific jargon.
- Short or vague feedback.
- Long multi-issue feedback.
- LLM output format errors.
- Over-prediction vs under-prediction.

### Phase 7: Report And Recommendations

Final report should answer:

- Which method performs best overall?
- Which method performs best under domain shift?
- Which method works best for open-topic generalisation?
- What is the cost/performance trade-off?
- Is Qwen3-4B fine-tuning worth it compared with simpler baselines?
- What would be recommended for Chattermill deployment?

## 13. Current Local Preparation Already Done

In the parent workspace `C:\Msc_DSML\Msc_Project`:

- FABSA public dataset loaded and exported.
- `summary.json`, CSV, and JSONL files created.
- Basic environment/setup scripts created.
- TF-IDF + Logistic Regression baseline tested on FABSA validation.

Baseline warm-up result on FABSA validation:

```text
Micro F1: approximately 0.656
Macro F1: approximately 0.414
```

This is only a warm-up result, not a final project result.

First traditional baseline sweep:

```text
Best validation model: Linear SVM with word+character TF-IDF
Validation pair micro F1: 0.719
Validation pair macro F1: 0.462
Test pair micro F1: 0.704
Test pair macro F1: 0.421
```

First BERT-style encoder sweep:

```text
Best validation-selected model: DistilBERT with square-root positive-class weighting
Validation pair micro F1: 0.779
Validation pair macro F1: 0.545
Test pair micro F1: 0.766
Test pair macro F1: 0.538

Best observed DistilBERT test pair micro F1: 0.774
```

See `docs/closed_topic_baselines.md` for details.

First Qwen3-4B feasibility pilot:

```text
Model: Qwen/Qwen3-4B-Instruct-2507
Method: 4-bit QLoRA with rank-8 LoRA adapters
Local GPU: RTX 5050 Laptop GPU, 8 GB VRAM
Result scope: first 100 validation rows only
Zero-shot pair micro F1: 0.541
Best local LoRA pilot pair micro F1: 0.762
```

This confirms that local QLoRA is feasible, but the laptop GPU is slow. Full Qwen training and full validation/test evaluation should preferably run on a stronger GPU. See `docs/qwen_feasibility.md` for details.

Current generalisation split protocols:

```text
Held-out organisation:
  validation organisation: 600
  test organisations: 369, 727
  train rows: 7,020
  validation rows: 1,533
  test rows: 2,021

Held-out aspect:
  held-out aspects:
    Account management: Account access
    Company brand: Competitor
    Value: Discounts promotions
  strategies:
    A. label-masked training
    B. example-filtered training
```

See `docs/evaluation_protocol.md` for details.

First generalisation baseline results:

```text
Held-out organisation, refined word+char TF-IDF + Linear SVM:
  test pair samples F1: 0.703
  test pair micro F1:   0.692
  test pair macro F1:   0.363

Held-out organisation, DistilBERT:
  test pair samples F1: 0.758
  test pair micro F1:   0.760
  test pair macro F1:   0.404

Held-out aspect, candidate-label lexical TF-IDF + global sentiment:
  label-masked test pair samples F1:     0.470
  example-filtered test pair samples F1: 0.463

Held-out aspect, candidate-label lexical TF-IDF + DistilBERT aspect-conditioned sentiment:
  label-masked test pair samples F1:     0.484
  example-filtered test pair samples F1: 0.480

Held-out aspect, candidate-aspect cross-encoder + global sentiment:
  label-masked test pair samples F1:     0.560
  example-filtered test pair samples F1: 0.582

Held-out aspect, candidate-aspect cross-encoder + DistilBERT aspect-conditioned sentiment:
  label-masked test pair samples F1:     0.541
  example-filtered test pair samples F1: 0.607

Held-out aspect, Qwen3-4B indexed zero-shot:
  validation pair samples F1: 0.575
  test pair samples F1:       0.537
  valid JSON rate:            1.000

LOAO held-out aspect, Qwen3-4B indexed zero-shot all-row mean:
  validation pair samples F1 mean: 0.119
  validation pair micro F1 mean:   0.329
  test pair samples F1 mean:       0.121
  test pair micro F1 mean:         0.338
  test pair macro F1 mean:         0.241
  test valid JSON rate:            1.000
  test schema-valid rate:          0.995
  interpretation: strong positive-row semantic matching, weak empty-gold absence calibration

Held-out aspect, Gemini 2.5 Flash indexed JSON-schema:
  validation pair samples F1: 0.615
  test pair samples F1:       0.607
  test pair micro F1:         0.654
  test pair macro F1:         0.555
  test valid JSON rate:       1.000
  test schema-valid rate:     1.000

Held-out aspect, Gemini 2.5 Flash-Lite indexed JSON-schema:
  validation pair samples F1: 0.588
  test pair samples F1:       0.552
  test pair micro F1:         0.587
  test pair macro F1:         0.488
  test mean latency:          0.372 seconds/example
  validation + test cost:     about $0.017

Held-out aspect, Gemini 2.5 Pro indexed JSON-schema:
  validation pair samples F1: 0.727
  test pair samples F1:       0.714
  test pair micro F1:         0.742
  test pair macro F1:         0.629
  test mean latency:          5.651 seconds/example
  validation + test cost:     about $2.978

Held-out aspect, local-to-Gemini selective cascade:
  local -> Flash-Lite test pair samples F1: 0.668
  local -> Flash test pair samples F1:      0.746
  local -> Pro test pair samples F1:        0.810
  policy selection: validation-only local reliability features
  best Pro cascade call rate:               90.0% of test rows
```

See `docs/generalisation_baselines.md` for details.

The strongest current local non-LLM open-topic baseline is documented separately in `docs/non_llm_open_topic_baseline.md`. The detailed Qwen zero-shot LOAO analysis and thesis-use framing are recorded in `docs/qwen_loao_experiment_analysis.md`. The paused next-stage modelling plan and literature-review framing are recorded in `docs/next_stage_and_literature_review_plan.md`, with the first local-paper scoping pass in `docs/literature_review_scoping_2026_06_29.md`, the current internal dissertation spec in `docs/dissertation_internal_spec.md`, the literature matrix in `docs/literature_review_matrix.md`, the latest Aji guidance in `docs/aji_updates_2026_06_29.md`, the Gemini hosted baseline implementation status in `docs/gemini_candidate_label_baseline.md`, the consolidated experiment reproducibility audit in `docs/experiment_reproducibility_register.md`, and the current thesis completion plan in `docs/thesis_completion_roadmap.md`.

Qwen held-out-aspect status:

```text
Completed:
  indexed candidate-label prompt design
  zero-shot validation/test evaluation
  zero-shot 12-aspect all-row LOAO validation/test evaluation
  row-level error analysis
  GPU-ready SFT/evaluation JSONL preparation script
  strongest current non-LLM fixed held-out-aspect baseline:
    candidate-aspect DistilBERT selector + DistilBERT aspect-conditioned sentiment

Not yet completed:
  full Qwen held-out-aspect fine-tuning
  full Qwen validation/test fine-tuned evaluation
  full Qwen LoRA all-row LOAO robustness evaluation
```

The current Qwen held-out-aspect and LOAO results are therefore zero-shot prompt baselines, not fine-tuned Qwen results. The full all-row LOAO test mean is `0.1212` pair samples F1 and `0.3378` pair micro F1. Positive-gold rows are much stronger (`0.8194` pair samples F1 mean and `0.8659` pair micro F1 mean), so the main Qwen zero-shot weakness is empty-gold absence calibration. The full analysis shows a 6/6 per-aspect split against the preferred DistilBERT LOAO row: Qwen gains on broader semantic aspects such as app/website, general satisfaction, speed, and staff attitude, while DistilBERT remains better on narrower support-channel, account, and value aspects. Full Qwen candidate-label fine-tuning should run later on stronger GPU access using the indexed SFT/evaluation data generation workflow, and it should be evaluated with the same all-row LOAO protocol.

Gemini hosted candidate-label status:

```text
Completed:
  OpenAI-compatible Gemini runner
  indexed candidate-label prompt support
  JSON schema / JSON object / plain JSON response-format modes
  schema-validity and invalid-output diagnostics
  latency, token-usage, reasoning-token, and optional cost accounting
  dry-run request construction check
  real hosted Gemini API smoke test
  validation prompt/JSON-mode sweep
  full fixed held-out-aspect validation/test evaluation
  targeted fixed held-out-aspect error analysis
  50-row Gemini Pro fixed-split subset comparison
  full Gemini Pro fixed validation/test evaluation
  full Gemini Flash-Lite fixed validation/test evaluation
  local-to-Gemini uncertainty cascade over Flash-Lite, Flash, and Pro
  Gemini-generated aspect-description ablation

Final fixed held-out-aspect test:
  model: vertex_ai/gemini-2.5-flash
  prompt: indexed JSON-schema
  max tokens: 2048
  pair samples F1: 0.6071
  pair micro F1:   0.6541
  pair macro F1:   0.5547
  aspect samples F1: 0.6747
  valid JSON rate: 1.0000
  schema-valid rate: 1.0000
```

The Gemini implementation initially blocked on missing credentials, but the fixed held-out-aspect run was completed after Aji's endpoint/key details were supplied. The key was not committed. This is still a fixed three-aspect result, not Gemini LOAO robustness evidence.

The Gemini fixed-split error analysis is now documented in `docs/gemini_error_analysis.md`. The main finding is that Gemini Flash ties the strongest local non-LLM pair samples F1 through a different error profile: it predicts fewer labels per row, has higher pair precision and fewer aspect over-predictions, but returns more empty predictions and is conservative on `Company brand: Competitor`. The fixed-split hosted Pareto table is now complete: Flash-Lite is cheapest and fastest, Flash is the balanced hosted baseline, and Pro is clearly strongest but slower and more expensive. The local-to-Gemini cascade in `docs/local_gemini_cascade.md` is the strongest fixed-split system result so far, reaching `0.7459` pair samples F1 with Flash escalation and `0.8102` with Pro escalation. A Pro deep-dive shows that the cascade also beats pure Pro (`0.7141`) because `gemini_nonempty_else_local` recovers Pro's 28 empty prediction rows, reduces pair false negatives from 67 to 40, and increases exact-match rows from 167 to 194 while keeping false positives nearly flat. Tasks 1-3 of the Gemini follow-up are consolidated for thesis and Task 4 handoff in `docs/tasks_1_to_3_thesis_prep.md`.

Task 4, Gemini-generated aspect descriptions, is now complete and documented in `docs/gemini_aspect_descriptions.md`. The headline positive result is for Flash-Lite: label-only descriptions improve test pair samples F1 from `0.5516` to `0.5925`, pair micro F1 from `0.5872` to `0.6263`, pair macro F1 from `0.4876` to `0.5691`, and aspect samples F1 from `0.6062` to `0.6625`. The result is mixed for stronger models: Flash descriptions improve validation but reduce test pair samples F1, and a Pro 50-row validation diagnostic does not justify a full Pro description run. Use this as a label-semantics and precision-recall trade-off result, not as a universal prompt improvement.

The current Tier 1 literature-review foundation is recorded in `docs/tier1_core_literature_review_matrix.md`. It covers customer review mining, ABSA, FABSA, multi-label evaluation, domain generalisation, candidate-label taxonomy shift, and structured-output LLM evaluation, while leaving detailed Qwen/Gemini fine-tuning and deployment-governance work as later optional branches.

The active completion roadmap is now `docs/thesis_completion_roadmap.md`. It treats full fine-tuned Qwen LoRA LOAO as the only remaining major compute-bound experiment. While GPU access is being negotiated, the project should complete thesis-ready tables, cascade score/margin uncertainty, Qwen LoRA runner readiness, and LaTeX skeleton alignment.

The Gemini-assisted qualitative error taxonomy is now complete and documented in `docs/qualitative_error_taxonomy.md`. It reuses existing outputs, keeps review-text packets and Gemini drafts under ignored `outputs/`, and manually consolidates seven thesis-facing categories: semantic boundary ambiguity, competitor-positive recall bottleneck, generative over-prediction, cautious abstention, neutral sentiment under-recall, prompt-induced precision-recall shift, and cascade complementarity. A small Gemini Pro draft pass was used only as an assistant for category wording and cross-checking, with reasoning tokens counted in the recorded usage. These categories define the main Qwen fine-tuning targets: abstention calibration, hard-negative aspect boundaries, competitor-positive recall, neutral sentiment coverage, stable label semantics, and cascade-ready uncertainty signals.

## 14. Thesis LaTeX Source And Writing Rule

The dissertation source now lives in `thesis/`, created from the UCL MSc thesis template zip at `C:\Msc_DSML\Msc_Project\UCL_Msc_Thesis.zip`.

Use these files as the thesis source of truth:

- `thesis/main.tex`
- `thesis/chapters/02_literature_review.tex`
- `thesis/references.bib`
- `thesis/notes/pro_literature_review_feedback_2026_06_30.tex`

Future dissertation-writing work should edit the LaTeX source directly. Markdown files under `docs/` should continue to record experiment logs, protocol notes, planning, and handoff summaries, but they should not become parallel final-dissertation drafts. The detailed workflow is recorded in `docs/thesis_latex_workflow.md`.

## 15. Current Open Questions

1. Are tagger guidance notes available for FABSA aspects?
   - There are no formal descriptions, but Aji may be able to find human scoring guidance.

2. What GPU environment should be used for full Qwen experiments?
   - Local QLoRA works, and local zero-shot Qwen LOAO is complete, but full fine-tuning is still better suited to stronger GPU access.
   - Indexed held-out-aspect Qwen prompt/evaluation data is now prepared for stronger GPU experiments.

3. Should the open-topic setting include only held-out aspects, or also combined held-out organisation plus held-out aspect evaluation?

4. How should practical deployment metrics be estimated for each model family?
   - Cost per 1M responses.
   - Time per 1M responses.
   - Fine-tuning cost per iteration.

5. Which LLM stage should follow the completed non-LLM held-out-aspect baseline?
   - Candidate-aspect DistilBERT selector plus DistilBERT aspect-conditioned sentiment is now the strongest fixed held-out-aspect non-LLM result.
   - The full all-row DistilBERT LOAO robustness check is also complete. Its preferred LR `3e-5` run reaches only `0.3128` mean pair micro F1 across the 12 held-out aspects, below the documented lexical global-sentiment LOAO lower bound. This should be treated as a robustness caveat, not a new headline improvement.
   - The DistilBERT LOAO result suggests the main remaining local-model bottleneck is unseen-aspect relevance detection and threshold calibration under taxonomy shift. Sentiment accuracy when the gold aspect is predicted is high, around `0.93`.
   - The Qwen zero-shot all-row LOAO baseline is also complete. It slightly improves the DistilBERT LOAO mean pair micro F1 (`0.3378` vs `0.3128`) and has much higher recall, but it over-predicts empty-gold rows. The positive-row diagnostic is strong, which supports Qwen fine-tuning/calibration rather than treating zero-shot as sufficient.
   - Gemini 2.5 Flash now matches the local non-LLM fixed held-out-aspect headline score and improves pair micro/macro F1, but has hosted-API cost, latency, and governance trade-offs.
   - Gemini error analysis shows higher precision and fewer aspect over-predictions than the local DistilBERT pipeline, but weaker recall for `Company brand: Competitor`.
   - Gemini Pro is the strongest fixed-split hosted baseline, but with substantially higher latency and cost.
   - Flash-Lite provides the cheapest and fastest hosted baseline.
   - The local-to-Gemini cascade is complete and is the strongest fixed-split system result, but it remains fixed three-aspect evidence rather than LOAO robustness evidence.
   - The Pro cascade beats pure Pro through error complementarity: Pro handles most uncertain rows, while the local fallback protects against Pro abstentions and some local-reliable rows.
   - The strongest next dissertation-oriented work is not more DistilBERT LOAO, more Gemini prompt sweeps, or full Gemini LOAO by default. The roadmap is recorded in `docs/thesis_completion_roadmap.md`.
   - Candidate-aspect descriptions and the Gemini-assisted qualitative error taxonomy are now complete. Recommended order after this point: thesis-ready result tables, cascade uncertainty improvement using local score/margin export, Qwen LoRA runner readiness, and full Qwen LoRA LOAO only after GPU access is confirmed.

## 16. Immediate Next Steps

1. Use `docs/thesis_completion_roadmap.md` as the current task ordering before starting new modelling work.

2. Keep the provided FABSA split as the closed-topic benchmark.

3. Treat `example_filtered` as the cleaner fixed held-out-aspect result and `label_masked` as an incomplete-label-noise ablation.

4. Treat the completed Gemini Flash fixed held-out-aspect result as a hosted baseline, but do not over-claim it as LOAO evidence.

5. Keep LOAO all-row evaluation as the robustness view and positive-row LOAO only as a sentiment diagnostic.

6. Treat the completed DistilBERT all-row LOAO result as the robustness caveat for the local non-LLM branch: strong fixed three-aspect performance, weak and highly variable full LOAO performance.

7. Treat the completed Qwen all-row LOAO result as the local open-weight zero-shot robustness baseline before Qwen fine-tuning. It is strong on positive rows but weak on all-row absence calibration, so do not treat it as a finished calibrated open-topic system.

8. Treat the fixed-split hosted Pareto comparison as complete: Flash-Lite, Flash, and Pro have all been run on validation/test.

9. Treat the local-to-Gemini cascade as complete fixed-split selective-deployment evidence; its Pro variant beats pure Pro through fallback recovery and lower false-negative count, not because local is globally stronger. Use `docs/local_gemini_cascade.md` and `docs/tasks_1_to_3_thesis_prep.md` as the clean evidence map.

10. Treat Gemini-generated aspect descriptions as complete Task 4 evidence. Do not rerun the same description API work unless a new thesis question requires it.

11. Treat the Gemini-assisted qualitative error taxonomy as complete and keep raw review text, Gemini prompts, and Gemini drafts under ignored `outputs/`.

12. Prepare thesis-ready evidence tables and figure inputs that keep fixed-split, all-row LOAO, and positive-gold diagnostic results conceptually separate.

13. Try the cascade score/margin uncertainty improvement without new Gemini calls if the local score export can be recovered or regenerated cleanly.

14. Prepare and smoke-test the final Qwen LoRA SFT/evaluation runner before any long GPU run.

15. Do not run full Gemini Pro LOAO unless the dissertation value justifies the added cost and latency.

16. Treat full fine-tuned Qwen LoRA LOAO as the major pending compute-bound experiment, not as a prerequisite for finishing the rest of the dissertation evidence package.

## 17. Notes For Repository Hygiene

Do not commit:

- Internal Chattermill data.
- API keys.
- Credentials.
- Large model checkpoints.
- Raw outputs containing confidential customer text.

Use ignored folders such as:

```text
data/raw/
data/internal/
outputs/
models/
checkpoints/
```

Only commit:

- Code.
- Documentation.
- Non-sensitive configs.
- Aggregated results if approved.

Experiment reproducibility:

- Use `docs/experiment_log.md` as the chronological log after every meaningful run.
- Use `docs/experiment_reproducibility_register.md` as the consolidated index of commands, parameters, outputs, and remaining reproducibility caveats.
- Future scripts should ideally save a run manifest with exact command, git commit, parsed CLI arguments, hardware/API metadata, and package versions.
- Every future experiment must be closed by documenting the exact parameters/configuration and safely committing/pushing the code and documentation updates to GitHub.
- Do not treat an experiment as complete if its results only exist in local `outputs/`; summarise the result in tracked documentation before stopping.
- Never push internal data, raw prediction outputs, credentials, checkpoints, model weights, or confidential material. If GitHub push is blocked, record the blocker in `docs/experiment_log.md` and push as soon as it is resolved.
