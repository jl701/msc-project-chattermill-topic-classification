# Dissertation Internal Spec

Last updated: 2026-06-29

This internal spec freezes the current dissertation spine and the experiment rules that future modelling work should follow. It is deliberately concise and operational: future Qwen, Gemini, LOAO, or joint-pair-scoring runs should conform to this document unless the protocol is explicitly revised.

## Working Dissertation Spine

The dissertation should be framed around:

```text
Structured candidate-label aspect-sentiment modelling under taxonomy shift in customer feedback.
```

The core question is:

```text
When customer-feedback taxonomies evolve, how should aspect+sentiment labels be predicted and evaluated when new candidate labels are supplied at inference time?
```

This keeps the project robust if LLMs underperform, while still allowing Qwen/Gemini to become central if structured-output LLM experiments are completed properly.

## Candidate Titles

Primary working title:

```text
Structured Candidate-Label Aspect-Sentiment Classification under Taxonomy Shift in Customer Feedback
```

Alternative title if Qwen/Gemini become central:

```text
Local and Generative Candidate-Label Models for Evolving Customer-Feedback Taxonomies
```

Avoid titles that imply:

- free-form topic discovery;
- arbitrary open-vocabulary generation;
- production-ready deployment;
- LLM superiority before evidence exists.

## Safe Claims

The dissertation can safely claim:

- FABSA supports a realistic multi-label aspect+sentiment customer-feedback task.
- Closed-topic classification is easier than held-out-aspect taxonomy shift in the current experiments.
- Held-out organisation shift is a separate axis from taxonomy/aspect shift.
- Candidate-label modelling is more appropriate than fixed classifier heads when inference-time labels can differ from training labels.
- Aspect-conditioned sentiment is more appropriate than global document sentiment for ABSA-style pair labels.
- LOAO is needed to avoid over-interpreting one fixed held-out-aspect choice.
- Structured-output LLMs should be evaluated on accuracy, schema reliability, latency, and token cost, not only F1.

Claims to avoid:

- "The model solves open-vocabulary topic classification."
- "The system discovers new topics."
- "The results generalise to arbitrary taxonomies or all customer-feedback datasets."
- "LLMs are better/worse in general."
- "The system is production-ready."
- "Gemini is cheap" unless reasoning/thinking tokens are counted.

## Dataset And Label Space

Dataset:

- FABSA public export.
- Approximately 10,574 reviews.
- 14 anonymised organisations.
- 10 domains.
- 12 aspect categories.
- Aspect+sentiment pair labels.

Sentiment set:

```text
positive
negative
neutral
```

Pair label format:

```text
{canonical aspect} | {sentiment}
```

Example:

```text
Online experience: App website | negative
```

One review may have zero, one, or multiple evaluated pair labels depending on the protocol and label scope.

## Task Definitions

### Closed-Topic Classification

Input:

- review text.

Output:

- one or more aspect+sentiment pair labels from the fixed FABSA taxonomy.

Purpose:

- Measures standard fixed-taxonomy performance.

### Held-Out Organisation Classification

Input:

- review text from an unseen organisation.

Output:

- one or more aspect+sentiment pair labels from the fixed FABSA taxonomy.

Purpose:

- Measures cross-organisation/domain-shift generalisation.

### Held-Out Aspect Candidate-Label Classification

Input:

- review text;
- a supplied list of candidate aspects.

Output:

- one or more selected candidate aspects, each with exactly one sentiment.

Purpose:

- Measures taxonomy/aspect shift where candidate labels are supplied at inference time.

This is not free-form topic generation.

### Leave-One-Aspect-Out

Input:

- review text;
- one held-out candidate aspect in the current lexical LOAO setup, or a candidate set in future expanded setups.

Output:

- labels involving the held-out aspect only.

Purpose:

- Measures robustness of the open-topic/new-aspect claim across all FABSA aspects.

## Candidate-Label Input Format

Candidate aspects must be canonical FABSA aspect strings.

Preferred indexed format for Qwen/Gemini:

```text
Candidate aspects:
A1. Account management: Account access
A2. Company brand: Competitor
A3. Value: Discounts promotions
```

Model outputs should refer to `aspect_id`, not copied aspect strings. This avoids near-miss labels such as `Account management`.

## Structured JSON Output Schema

Canonical output shape:

```json
[
  {"aspect_id": "A1", "sentiment": "negative"},
  {"aspect_id": "A3", "sentiment": "positive"}
]
```

Empty prediction:

```json
[]
```

Rules:

- Top-level value must be a JSON array.
- Each item should be an object.
- `aspect_id` must match one supplied candidate ID.
- `sentiment` must be one of `positive`, `negative`, `neutral`.
- Duplicate aspect+sentiment pairs should be deduplicated.
- Multiple aspects may be returned for one review.
- A candidate aspect should receive at most one sentiment in ideal outputs. If a model returns conflicting sentiments for the same aspect, keep only valid unique pair labels for scoring and record the conflict as a schema/semantic issue in LLM diagnostics.

For Gemini:

- Use `response_format` / JSON mode if the endpoint supports it.
- Record whether JSON mode was used.

## Invalid Output Handling

For LLM outputs:

- Invalid JSON: count as invalid and score as an empty prediction.
- Top-level non-list JSON: count as invalid and score as an empty prediction.
- Unknown `aspect_id`: drop that item and record as invalid candidate reference.
- Aspect name instead of `aspect_id`: accept only if the parser can map it unambiguously to an allowed candidate; otherwise drop it.
- Invalid sentiment: drop that item.
- Duplicate valid pair: keep one copy.
- Extra prose around valid JSON: parser may extract the JSON array, but still record this as non-strict output if using a stricter future diagnostic.

Diagnostics to record:

- valid JSON rate;
- schema-valid output rate if available;
- parse failures;
- hallucinated / out-of-candidate labels;
- invalid sentiments;
- duplicate labels;
- empty predictions;
- latency;
- token usage.

## Evaluation Protocols

### Closed-Topic

- Use official FABSA train/validation/test split.
- All 12 aspects are available in training and evaluation.
- Main model-selection split: validation.
- Final split: test.

### Held-Out Organisation

- Train: all except selected validation/test organisations.
- Validation organisation: `600`.
- Test organisations: `369`, `727`.
- Same taxonomy remains available.
- Purpose: organisation/domain shift.

### Fixed Held-Out Aspect

Held-out aspects:

- `Account management: Account access`
- `Company brand: Competitor`
- `Value: Discounts promotions`

Primary strategy:

- `example_filtered`, because it avoids censored held-out-aspect supervision and produced the strongest local result.

Secondary strategy:

- `label_masked`, as an incomplete-label-noise ablation.

Evaluation:

- held-out labels only;
- candidate labels supplied at inference;
- containing-heldout row scope for the fixed three-aspect benchmark unless explicitly stated otherwise.

### Leave-One-Aspect-Out

Primary goal:

- report robustness across all 12 FABSA aspects.

Required summary:

- per-aspect metrics;
- mean;
- median;
- standard deviation;
- minimum;
- maximum;
- support/frequency.

Use all-row LOAO when measuring unseen-aspect detection with negative rows. Use positive-row LOAO only as a sentiment diagnostic.

## Metrics

Headline metric:

- pair samples F1.

Always report:

- pair samples F1;
- pair micro F1;
- pair macro F1;
- aspect samples F1;
- aspect micro F1 where useful;
- sentiment accuracy when the gold aspect is predicted.

For all-row LOAO diagnostics, also report:

- pair micro precision;
- pair micro recall;
- false-positive rows per 100 reviews;
- false-positive labels per 100 reviews;
- false-negative rows per 100 reviews;
- exact-match rate.

For LLMs, additionally report:

- valid JSON rate;
- schema-valid rate if applicable;
- parse-failure rate;
- latency per example;
- input tokens;
- visible output tokens;
- reasoning/thinking tokens when reported;
- total billable tokens;
- estimated cost.

## Model Families

### Local Two-Stage Pipeline

Current strongest local baseline:

```text
Candidate-aspect DistilBERT selector
+ DistilBERT aspect-conditioned sentiment
```

Best fixed held-out-aspect result:

| Metric | Value |
| --- | ---: |
| Pair samples F1 | 0.6071 |
| Pair micro F1 | 0.5917 |
| Pair macro F1 | 0.4890 |
| Aspect samples F1 | 0.6651 |

Use this as the local anchor.

### Structured LLMs

Qwen/Gemini should be treated as structured generative candidate-label classifiers:

```text
review + indexed candidate aspects -> JSON aspect_id/sentiment pairs
```

They can become central only if:

- evaluated under the same candidate-label protocol;
- JSON/schema reliability is recorded;
- latency and token cost are recorded;
- comparison against local baselines is fair and clearly framed.

### Joint Pair Scorer

Potential future local model:

```text
(review text, candidate aspect, candidate sentiment) -> pair applicability score
```

This is the local discriminative counterpart to structured LLM pair outputs.

## Experiment Ordering

1. Finish Stage 1 internal spec.
2. Build Stage 2 literature matrix.
3. Complete LOAO for the strongest local baseline.
4. Run Gemini pilot.
5. Decide whether to scale LLMs or implement joint pair scoring.
6. Strengthen Qwen under the selected branch.
7. Implement joint pair scoring if needed.
8. Write dissertation.

## Primary Open Questions

- Can the strongest local two-stage pipeline remain strong under LOAO?
- Does Gemini beat or complement the local baseline when JSON mode and reasoning-token cost are measured?
- Can Qwen be strengthened beyond zero-shot without derailing the schedule?
- Does direct pair scoring outperform the two-stage local pipeline?
- Which result should become the main modelling contribution versus discussion/future work?

