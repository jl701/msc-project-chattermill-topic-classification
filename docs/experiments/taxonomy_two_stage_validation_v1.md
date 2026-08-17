# Taxonomy two-stage validation v1

Date: 17 August 2026

Status: preregistered before execution; execution completed on 17 August 2026

The audited results and interpretation are recorded in
`docs/experiments/taxonomy_two_stage_validation_v1_results.md`.

## Research question

Does separating aspect presence from conditional sentiment improve the current
independent 36-pair decision rule without paying for new transformer
fine-tuning?

This validation-only study contains two parts. Study B changes only the decoder
and reuses immutable final score shards. Study C changes the scorer into a true
aspect-then-sentiment pipeline for strict TF-IDF, frozen E5 and frozen Qwen.
Open-set anomaly detection is a separately approved fourth study and is not
mixed into these results.

## Safety boundary

- Only official train and validation data may be loaded.
- `test.csv` and `--include-official-test` remain forbidden.
- All aspect thresholds are selected from seen-aspect validation evidence.
- Held-out validation labels are evaluation-only and cannot change prompts,
  features, thresholds or model choices.
- No new DistilBERT or QLoRA training is permitted in this study.

## Study B: decoder-only comparison

Study B uses all five registered methods and all twelve Level 2 outer folds.
The source score shards are the final seen-calibration validation shards bound
to the existing threshold-transfer artifacts.

The old decoder thresholds each of the 36 pair scores independently. The new
decoder sets an aspect score to the maximum of its three sentiment scores,
thresholds the aspect score, and assigns the selected aspect the sentiment with
the largest score. Threshold selection and evaluation are separated by a
deterministic five-fold cross-fit over `row_uid`.

This is an in-distribution decoder diagnostic. It does not claim held-out
taxonomy generalisation because the reused formal validation shards contain
seen candidates only.

## Study C: true no-transformer-fine-tuning pipeline

Study C covers all twelve Level 2 outer folds. Seen candidates always receive
their canonical name and frozen minimal definition. The held-out candidate is
evaluated in three locked variants:

1. canonical name plus description;
2. canonical name only;
3. description only, with the canonical name absent from model input.

Stage 1 predicts aspect presence. Strict TF-IDF fits a cheap balanced logistic
regression over four aspect-only lexical interactions: word TF-IDF cosine,
character TF-IDF cosine, aspect-name token coverage, and complete
aspect-candidate token coverage. The vectorisers and calibration layer are
fitted separately inside every filtered training fold. E5 uses frozen cosine
similarity between the review and the aspect-only text. Frozen Qwen uses a
locked binary aspect-presence prompt with single-token `Y`/`N` verbalizers.

Stage 2 predicts sentiment only for a selected aspect. TF-IDF is trained only
on present seen aspects and their wrong-sentiment negatives, using the existing
six transferable lexical interaction features. E5 compares the three
sentiment-conditioned claims. Frozen Qwen receives the review and aspect-only
candidate and makes one locked three-way next-token decision: `A` is negative,
`B` is neutral, and `C` is positive. There is no transformer weight update.

The frozen-Qwen prompt/encoding/verbalizer contract is
`ed6eff449c4396e46ce11c17875997772feced1ac2701e82092a4c66e1ba162b`.
Its local resume cache stores only prompt hashes and probability vectors; it
does not store review text.

The aspect threshold is chosen by end-to-end pair micro-F1 on seen-aspect
validation evidence. Primary reporting uses the held-out aspect partition.

## Multiple sentiments for one aspect

An audit before execution found 72 train rows and 12 validation rows containing
more than one sentiment for the same aspect. No row or label will be removed or
rewritten. The primary analysis retains all rows and tests the supervisor-style
single-sentiment argmax decoder as specified. A sensitivity result restricted
to aspect instances with exactly one gold sentiment quantifies the structural
penalty introduced by that exclusivity constraint.

## Outputs and stopping rules

Aggregate summaries without review text may be tracked. Raw scores, caches and
review text stay under the ignored output root. Execution stops on any test
access, source-artifact mismatch, duplicate identity, non-finite score,
held-out calibration leakage, prediction collapse, OOM or repeated thermal
throttling.

An execution-time cache audit found byte-identical prompts caused by duplicate
review texts. The first partial Qwen cache was stopped before any fold result
was emitted, backed up, deduplicated from 6,816 rows to 6,780 unique hashes,
and resumed after adding within-chunk hash deduplication. This changes neither
model inputs nor scores; it makes resume identity explicit and auditable.
