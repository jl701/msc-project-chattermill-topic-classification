# Taxonomy two-stage validation v1: results

Date: 17 August 2026

Status: complete and audited

## Scope and interpretation boundary

This report records the preregistered validation-only B and C studies. No
official test data were loaded and no held-out validation result was used to
change a feature, prompt, threshold, representation, or model.

Study B and Study C answer different questions and their scores must not be
subtracted from one another:

- Study B is a decoder-only cross-fit diagnostic over existing **seen-aspect**
  validation score shards.
- Study C is a genuine no-transformer-fine-tuning two-stage pipeline evaluated
  on **held-out-aspect** validation partitions.

Consequently, Study C does not by itself establish a direct improvement over
the existing one-stage held-out system. A matched, separately preregistered
held-out comparison would be required for that causal claim.

## Study B: hierarchical decoding of existing pair scores

The hierarchical decoder uses the maximum of the three sentiment-specific
pair scores as the aspect score. It thresholds aspect presence first and then
assigns one argmax sentiment to each selected aspect. Values below are means
over twelve cross-fitted Level 2 folds.

| Method | Independent pair F1 | Hierarchical F1 | Delta | Fold wins/ties/losses |
|---|---:|---:|---:|---:|
| Strict train-only TF-IDF | 0.2772 | 0.2919 | **+0.0147** | 12 / 0 / 0 |
| Frozen E5-base-v2 | 0.2837 | 0.3237 | **+0.0400** | 12 / 0 / 0 |
| DistilBERT cross-encoder | 0.4275 | 0.4283 | +0.0009 | 7 / 2 / 3 |
| Frozen Qwen candidate-pair | 0.4598 | 0.4094 | **-0.0504** | 0 / 0 / 12 |
| QLoRA candidate-pair | 0.5442 | 0.5450 | +0.0008 | 9 / 0 / 3 |

The forced hierarchy is consistently useful for TF-IDF and E5 on this
seen-aspect diagnostic. It is effectively neutral for DistilBERT and QLoRA.
It is consistently harmful when retrofitted to the frozen-Qwen pair scores,
mainly because recall falls. Pair scores trained or prompted as independent
claims are therefore not automatically calibrated as aspect scores.

## Study C: genuine aspect-then-sentiment scoring

Each fold removes all training reviews containing its held-out aspect. The
aspect threshold is selected only on seen-aspect validation evidence. The
held-out aspect labels are then used once for evaluation. Values below are
unweighted means over the twelve held-out-aspect folds.

| Method | Held-out representation | Pair P | Pair R | Pair F1 | Aspect F1 | Conditional sentiment accuracy |
|---|---|---:|---:|---:|---:|---:|
| Strict train-only TF-IDF | Name + description | 0.2744 | 0.3706 | **0.2780** | 0.4110 | 0.6362 |
| Strict train-only TF-IDF | Description only | 0.2689 | 0.3601 | 0.2705 | 0.3980 | 0.6296 |
| Strict train-only TF-IDF | Name only | 0.3029 | 0.2156 | 0.2113 | 0.3377 | 0.6188 |
| Frozen E5-base-v2 | Name + description | 0.2273 | 0.3702 | 0.2246 | 0.2906 | 0.7851 |
| Frozen E5-base-v2 | Description only | 0.2191 | 0.2279 | 0.1761 | 0.2284 | 0.7888 |
| Frozen E5-base-v2 | Name only | 0.2303 | 0.4408 | **0.2419** | 0.3012 | 0.7615 |
| Frozen Qwen | Name + description | 0.3913 | 0.6401 | 0.4513 | 0.4971 | **0.8690** |
| Frozen Qwen | Description only | 0.3975 | 0.6652 | **0.4645** | **0.5168** | 0.8376 |
| Frozen Qwen | Name only | 0.4629 | 0.4718 | 0.4359 | 0.4599 | 0.8621 |

### Paired representation findings

- TF-IDF benefits clearly from the description: name plus description exceeds
  name only by +0.0667 pair F1 on average and wins 9 of 12 folds. Description
  only is close to the complete representation (-0.0075; 1/0/11), suggesting
  that the semantic description carries most of the transferable evidence.
- E5 is heterogeneous across aspects. Name only has the highest mean, but name
  plus description versus name only splits 6 wins and 6 losses. There is no
  consistent representation winner across the twelve held-out aspects.
- Frozen Qwen is the strongest of the three true two-stage, no-fine-tuning
  methods in every representation condition. Description only has the highest
  mean pair F1 (0.4645), exceeding name plus description by +0.0132 with 7 wins
  and 5 losses. This is descriptive evidence, not a newly selected prompt or a
  basis for post-hoc tuning.
- Restricting the analysis to aspect instances with one gold sentiment changes
  pair F1 only minimally. The occasional multiple-sentiment labels therefore
  do not explain the main ranking.

## Diagnostic event retained as evidence

Frozen Qwen's `l2-a02` name-only condition selected only one held-out
prediction and obtained zero pair F1. The preselected seen-aspect threshold was
0.999999940395, whereas nearly all name-only held-out aspect scores fell below
it. Prompt construction, candidate identity, finite probabilities, and cache
identity all passed audit. Other folds did not show a global collapse. This is
therefore retained as an aspect-specific representation/calibration shift,
not repaired using held-out labels.

During the initial Qwen cache fill, byte-identical prompts caused by duplicate
review text produced 36 duplicate cache records. Execution was stopped before
any fold result was emitted. The cache was deduplicated and within-chunk prompt
hash deduplication was added before resuming. The final cache contains 75,024
unique finite probability records under one prompt contract; no review text is
stored.

## Safety and completeness audit

- Study B: 5 methods × 12 folds complete.
- Study C: TF-IDF, E5, and frozen Qwen × 12 folds complete.
- Failed jobs: 0.
- Official-test artifacts or accesses: 0.
- Non-finite scores: 0.
- Qwen cache: 75,024 unique rows, with probability vectors summing to one.
- Qwen cache SHA-256:
  `32d0b82e417803dec5a7cdc620f4c7e947b133adfda43992a8cc4491562506dc`.
- Peak observed GPU memory was approximately 6.7 GB of 8.15 GB. No OOM or
  thermal-throttling stop condition occurred.

Large caches, raw scores, and review-level outputs remain local under the
ignored output root. The code, frozen configuration, tests, and this aggregate
report are safe to track.

## Decision

The supervisor's proposed factorisation is technically viable and useful, but
its benefit is method-dependent. The strongest current result is the genuine
two-stage frozen-Qwen pipeline, particularly when a held-out description is
available. The next scientifically clean step, if a direct replacement claim
is needed, is a matched held-out one-stage versus two-stage comparison under an
unchanged representation and selection protocol. The separately approved
unknown-aspect anomaly-detection study remains deferred and is not mixed into
these conclusions.
