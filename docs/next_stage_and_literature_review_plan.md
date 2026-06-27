# Next Stage And Literature Review Plan

Last updated: 2026-06-27

This note records the project state after completing the strongest current local non-LLM open-topic baseline. The next major modelling phase is intentionally paused while the dissertation literature review and overall framework are developed.

## Current Pause Point

The local non-LLM open-topic baseline phase has reached a useful stopping point.

Completed:

- Closed-topic traditional and DistilBERT baselines.
- Held-out organisation traditional and DistilBERT baselines.
- Fixed held-out-aspect lexical candidate-label lower bounds.
- Fixed held-out-aspect candidate-aspect DistilBERT selector baseline.
- Lightweight TF-IDF aspect-conditioned sentiment ablation.
- DistilBERT aspect-conditioned sentiment ablation.
- Strongest current local non-LLM fixed held-out-aspect baseline:
  - candidate-aspect DistilBERT selector + DistilBERT aspect-conditioned sentiment
  - `example_filtered`
  - test pair samples F1: `0.6071`
- Qwen indexed zero-shot held-out-aspect baseline.
- Qwen held-out-aspect SFT/evaluation data preparation.
- Lexical leave-one-aspect-out robustness diagnostics.
- Held-out-aspect row-level error analysis.

Paused:

- Hosted Gemini candidate-label baseline.
- Full Qwen candidate-label fine-tuning/evaluation.
- Joint aspect+sentiment pair scoring.
- Full DistilBERT leave-one-aspect-out run.

## Why Pause Here

The project now has enough experimental structure to support the dissertation framework:

1. A closed-topic benchmark shows performance when the label taxonomy is fixed.
2. A held-out organisation benchmark shows cross-company/domain-shift generalisation.
3. A held-out aspect benchmark shows open-topic/new-label generalisation.
4. Non-LLM candidate-label baselines show how far smaller local models can go without hosted LLMs.
5. Qwen zero-shot provides an initial LLM reference point.

The next modelling steps are larger and more expensive. They should be started after the literature review and dissertation structure make clear which comparison is most important.

## Recommended Dissertation Framing

Working title:

```text
Open-Vocabulary Topic Classification for Customer Feedback
```

Possible central research question:

```text
How well can supervised encoders and language-model-based candidate-label methods generalise customer-feedback topic classification across companies and to unseen topic labels?
```

Suggested sub-questions:

1. How strong are conventional fixed-taxonomy baselines on the provided FABSA split?
2. How much performance is lost under cross-organisation/domain shift?
3. How much harder is open-topic/new-aspect generalisation than cross-organisation generalisation?
4. Do candidate-label methods improve unseen-aspect prediction compared with lexical lower bounds?
5. Does aspect-conditioned sentiment improve pair-level open-topic prediction compared with global sentiment?
6. Are instruction-following LLMs a practical next step for open-topic candidate-label prediction?

## Suggested Report Structure

1. Introduction
   - Industrial motivation: customer feedback topic classification at scale.
   - Need for models that adapt to new customers and evolving topic taxonomies.
   - Research gap: fixed-label classifiers do not naturally handle unseen topic labels.
   - Contributions:
     - reproducible FABSA experimental pipeline
     - cross-organisation and held-out-aspect protocols
     - local non-LLM candidate-label baselines
     - initial LLM candidate-label baseline and preparation for fine-tuning

2. Background And Literature Review
   - Customer feedback mining and aspect-based sentiment analysis.
   - Multi-label text classification.
   - Domain generalisation and cross-domain/customer shift.
   - Open-set, zero-shot, and open-vocabulary topic classification.
   - Candidate-label and entailment-style classification.
   - Instruction tuning and generative ABSA.
   - Practical trade-offs: cost, latency, privacy, and maintainability.

3. Data And Task Formulation
   - FABSA dataset.
   - Aspect+sentiment pair labels.
   - Candidate-label open-topic formulation.
   - Why free-form generation is not the main evaluation target.
   - Label-masked versus example-filtered training.

4. Evaluation Protocol
   - Closed-topic provided split.
   - Held-out organisation split.
   - Fixed held-out aspect split.
   - Leave-one-aspect-out diagnostic.
   - Metrics:
     - pair samples F1 as headline
     - pair micro F1
     - pair macro F1
     - aspect-only diagnostics
     - sentiment accuracy when the gold aspect is predicted

5. Methods
   - Classical TF-IDF/SVM/logistic baselines.
   - DistilBERT fixed-label baselines.
   - Candidate-label lexical baseline.
   - Candidate-aspect DistilBERT selector.
   - Global versus aspect-conditioned sentiment.
   - Qwen indexed zero-shot baseline.

6. Results
   - Closed-topic results.
   - Held-out organisation results.
   - Held-out aspect fixed three-aspect results.
   - Leave-one-aspect-out robustness results.
   - Qwen zero-shot comparison.

7. Error Analysis
   - Over-prediction and threshold behaviour.
   - Aspect selection versus sentiment errors.
   - Competitor and promotion aspect difficulty.
   - Label-masked incomplete-supervision noise.

8. Discussion
   - Why open-topic generalisation is the hardest axis.
   - Why candidate-label methods are appropriate for industrial topic taxonomies.
   - Local encoder versus LLM trade-offs.
   - Limitations of FABSA and fixed held-out-aspect design.

9. Conclusion And Future Work
   - Current strongest local non-LLM result.
   - Next step: hosted Gemini or Qwen fine-tuning.
   - Deployment considerations.

## Literature Review Search Themes

Prioritise papers around these themes.

### Aspect-Based Sentiment Analysis

Purpose:

- Ground the aspect+sentiment pair formulation.
- Explain why global document sentiment is insufficient.

Useful angles:

- Aspect extraction and aspect sentiment classification.
- Instruction-based ABSA.
- Unified generative ABSA.
- Multi-aspect reviews with different sentiments per aspect.

Already relevant in project notes:

- Hu and Liu, 2004.
- Tang et al., 2016.
- FABSA dataset paper.
- InstructABSA.
- Unified generative ABSA.

### Multi-Label Text Classification

Purpose:

- Explain the multi-label nature of customer feedback.
- Justify sample-level F1, micro F1, and macro F1.

Useful angles:

- Label imbalance.
- Long-tail labels.
- Threshold selection.
- Per-label versus per-sample metrics.

### Domain Generalisation

Purpose:

- Position the held-out organisation experiment.
- Explain why random/provided splits can overstate deployment readiness.

Useful angles:

- Cross-domain text classification.
- Customer/company shift.
- Distribution shift in industrial NLP.

### Open-Topic / Open-Vocabulary Classification

Purpose:

- Position held-out aspects as the novel axis.
- Explain why fixed classifier heads cannot handle unseen labels.

Useful angles:

- Open-domain topic classification.
- Zero-shot text classification.
- Label descriptions and candidate-label prompting.
- Entailment/NLI-style classification.

Already relevant in project notes:

- Ding et al., 2023, `Towards Open-Domain Topic Classification`.
- Yin et al., 2019, zero-shot text classification benchmark.

### Instruction-Following LLMs For Classification

Purpose:

- Motivate Qwen/Gemini as next-stage baselines.
- Compare open and hosted LLMs with smaller supervised models.

Useful angles:

- Instruction tuning for ABSA.
- Structured JSON output.
- Candidate IDs instead of free-form label names.
- Fine-tuning small LLMs with LoRA/QLoRA.
- Cost and latency trade-offs.

## Next Modelling Stage

The next major modelling stage should start only after the literature review/framework checkpoint.

### Option A: Hosted Gemini Candidate-Label Baseline

Goal:

- Test whether a strong hosted LLM can outperform the best local non-LLM baseline under the same fixed held-out-aspect candidate-label protocol.

Recommended first run:

- Model: `vertex_ai/gemini-2.5-flash`
- Prompt: indexed candidate aspects, JSON output with `aspect_id` and `sentiment`.
- Evaluation:
  - validation first
  - then test if validation behaviour is stable
- Compare against:
  - candidate-aspect DistilBERT + DistilBERT aspect-conditioned sentiment: `0.6071`
  - Qwen indexed zero-shot: `0.5374`

Risks:

- API cost and budget.
- Need enough `max_tokens` because Gemini 2.5 may spend tokens thinking.
- External API data-governance concerns for non-public data.
- Prompt sensitivity and possible over-prediction.

### Option B: Qwen Candidate-Label Fine-Tuning

Goal:

- Fine-tune a local/open instruction model for candidate-label ABSA and compare against local DistilBERT baselines and Gemini.

Recommended setup:

- Use the already prepared indexed held-out-aspect SFT/evaluation JSONL workflow.
- Keep output constrained to JSON objects with candidate IDs.
- Start with a small smoke run, then a full validation/test run on stronger GPU access.

Risks:

- Local 6-8 GB GPUs are too slow for serious full experiments.
- Fine-tuned results need strict train/validation/test separation.
- Output parsing must be deterministic and invalid outputs must be counted.

### Option C: Joint Aspect+Sentiment Pair Scoring

Goal:

- Replace the two-stage selector + sentiment setup with a pair scorer:

```text
(review text, candidate aspect, candidate sentiment) -> relevance
```

Reason to defer:

- It is a useful non-LLM extension but less central than establishing the LLM comparison.
- The current two-stage pipeline is already strong enough to serve as the non-LLM reference.

## Immediate Non-Modelling Tasks

While the next modelling phase is paused, the best use of time is:

1. Build the literature review matrix.
2. Draft the dissertation outline using the structure above.
3. Convert current results into one or two clean result tables.
4. Decide which metrics and baselines deserve main-text tables versus appendix tables.
5. Prepare diagrams:
   - task formulation
   - split protocols
   - candidate-label pipeline
   - result comparison across closed-topic, held-out organisation, and held-out aspect settings
6. Re-read `docs/non_llm_open_topic_baseline.md`, `docs/generalisation_baselines.md`, and `docs/loao_heldout_aspect.md` before writing the methods section.

## What Not To Do Next

- Do not spend more time on small fixed held-out-aspect DistilBERT tuning unless a specific dissertation gap appears.
- Do not run Gemini or full Qwen fine-tuning without explicitly deciding the budget, model, prompt format, and evaluation scope.
- Do not treat fixed three-aspect results and LOAO all-row results as directly comparable headline numbers.
- Do not report Qwen zero-shot as a fine-tuned result.

