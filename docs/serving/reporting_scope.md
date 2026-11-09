# Reporting scope for the simulated Chattermill service

This document fixes interpretation and reporting denominators before the final
engineering results are summarised. It contains no new model-quality claim.
Read it alongside `artifact_audit.md` and the executed run configurations.

## What this experiment establishes

The experiment packages one historical Qwen model fold as a local API and
measures complete-review inference, request handling and replay monitoring.
The fold is `l2-a01`, trained without reviews mentioning
`Account management: Account access`. The server's candidate taxonomy contains
all twelve aspects. Each request performs presence detection for all candidates,
then sentiment classification for the selected aspects. A request is a complete
review, not one candidate prompt or one forward pass.

The three variants differ in which scorer performs each stage: frozen few-shot
for both stages, the selected QLoRA adapter for both stages, or frozen few-shot
presence followed by QLoRA sentiment. All comparisons must state the actual
batching, token-length limits, precision, device and selected thresholds. Their
different prompt lengths are part of the evaluated systems, not a controlled
comparison of model weights alone.

The original validation split was divided deterministically into 100 engineering
development reviews and 957 evaluation reviews. The IDs do not overlap, but both
subsets belonged to validation used in historical selection. Therefore this is
**a disjoint engineering replay partition, not an independent or newly blind
model test**. Development may inform serving choices. The remaining 957 reviews
are frozen for reporting after those choices; they must not become a further
prompt, threshold or model tuning set. The official test split is not used.

## Audited quality semantics

`evaluate_quality()` joins labels after inference using `sample_id`, falling
back to `request_id`. Every unique sample contributes at most one prediction:
the successful attempt earliest by request start time. Later successful attempts
do not enlarge the quality sample; disagreements with that first prediction are
reported separately. This is not a majority vote or a best-of-repeats selection.

Presence micro-F1 counts `(review, aspect)` assignments. Pair micro-F1 counts
`(review, aspect, sentiment)` assignments. Predicting two sentiments for one
aspect contributes one presence assignment but two pair assignments. The two
metrics cannot be called accuracy or substituted for each other. Raw F1 values
are on the 0–1 scale; if displayed on a 0–100 scale, label the columns `F1 × 100`
and report absolute differences as F1 points.

Two views are deliberately separate:

1. **Successful unique quality:** only unique samples with at least one valid
   successful response. It measures the predictions available among successful
   deliveries and excludes samples that never succeeded.
2. **Delivered unique quality, failures as empty:** all unique attempted samples,
   with no prediction for samples that never succeeded. A sample that failed
   once and later succeeded uses its successful prediction. This is an
   eventual-delivery view for the replay, not a per-attempt quality or reliability
   score. More retries can improve this view without improving any individual
   request's reliability.

Unattempted labelled samples are excluded from both views and explicitly
counted. Client-capacity drops remain failures in the attempt-level performance
denominator, even when no HTTP request was sent. Keep client attempt success
rate beside both quality views. Never pool variants or model versions. When
comparing variants that succeeded on different subsets, additionally state the
common successful sample count or compare complete delivery views; conditional
F1 values on different subsets are not a paired comparison.

The analysis must receive the frozen artifact manifest to define the complete
taxonomy and held-out aspect. An absent held-out definition must not be reported
as a genuine held-out F1 of zero. Zero-denominator F1 in the helper uses zero;
always display support so this convention is distinguishable from observed
failure on positive examples.

## Ground-truth support in the frozen replay

These counts come from the independent label files, without rereading review
text or using predictions. “Independent” here means labels are stored and joined
separately from API inputs, not that the data form an independent statistical
test sample.

| Partition | Unique reviews | Overall gold presence assignments | Overall gold pairs | Held-out positive reviews / gold pairs | Held-out negative / neutral / positive |
|---|---:|---:|---:|---:|---|
| Development | 100 | 177 | 180 | 8 / 8 | 6 / 1 / 1 |
| Evaluation | 957 | 1669 | 1678 | 73 / 73 | 48 / 20 / 5 |

The 957-review evaluation contains only 73 gold-positive reviews for a01, and
only five of those have positive sentiment. All 957 reviews still matter when
counting false-positive a01 predictions. Report positive support separately from
total reviews; neither 957 requests nor repeated attempts can increase the 73
independent held-out positive reviews. The held-out support of a partial run is
the support of its actual unique sample subset, not automatically 73.

Seen scope includes the remaining eleven aspects. Overall scope includes all
twelve. Report gold support and TP/FP/FN for presence and pairs in each scope.
Held-out exact-set match is especially easy to inflate with true negatives and
should not replace held-out micro-F1 or recall.

The historical value `0.5075` was the equal-weight mean of held-out pair micro-F1
across twelve separately trained/evaluated leave-one-aspect-out folds. It is
neither overall twelve-aspect F1 nor this single deployed fold's expected F1.
Do not claim improvement over it by comparing the current single-fold replay
score, and do not carry it into the service table as a service quality result.

## Descriptive review-cluster intervals

`scripts/serving_quality_intervals.py` accepts repeated `--run NAME=DIR` arguments,
the frozen `--labels` and `--artifacts` files, and an aggregate `--output` file.
Its defaults are 2,000 percentile-bootstrap replicates and seed 20261109. Each
draw resamples complete unique reviews, retaining all aspect and sentiment
assignments together. Repeated HTTP attempts are deduplicated using the same
first-success rule as the monitoring module before any resampling.

The script validates the exact evaluation-label file hash and replay identities,
checks successful response fold IDs, and requires all systems to have identical
attempted-review sets. Matching partial coverage is allowed but explicitly
reported as partial; different attempted sets are rejected instead of silently
intersected. At least one successful response per run is necessary to verify its
fold identity. No inference or GPU operation occurs.

Each system receives presence/pair micro-F1 intervals for overall, seen and
held-out scopes in both quality views. Failure-as-empty differences are paired
on the common attempted reviews. Successful-only differences use each pair of
systems' common successful reviews, so their point estimates can differ from
the per-system intervals based on that system's own successful subset. Both
systems use the same resampled review multiplicities for a paired contrast.
Difference direction is always explicitly left minus right.

Zero-denominator micro-F1 follows the monitoring convention of zero, and the
number of such resamples is reported. Sparse held-out support can make this
convention visible. The intervals describe this fixed validation replay and
do not restore blindness, account for historical model selection, estimate
between-fold variation, provide p-values or establish statistical significance.
The output includes aggregate counts and scores only, with no sample identities,
review text or machine-specific input paths.

## Recommended result tables

Use separate tables for configuration, performance and quality; one wide table
would obscure their different denominators.

**Configuration table, one row per executed configuration:** variant; fold and
model/adapter identity; device and precision; request-batch limit and wait;
candidate microbatch and token budget; maximum prompt lengths; queue capacity;
server/client timeout; response-cache policy; warm-up exclusion; workload source
and character/token-length mix; concurrency or scheduled arrival rate.

**Performance table, one row per configuration and workload:** attempts; HTTP
requests sent; successful responses; unique review IDs; elapsed wall time
including drain; successful complete-review operations/second; success rate;
429 count; server deadline count; client timeout count; client-capacity drops;
successful client end-to-end P50/P95/P99; scheduled-arrival P50/P95/P99; scheduling
lag; peak measured GPU memory. Keep model load/cold start separate from steady
state, and label memory as measured peak rather than theoretical weight size.

**Quality table, one row per variant and scope:** unique attempted reviews;
unique successful reviews; unattempted label count; gold-positive review count;
gold presence and pair support; presence TP/FP/FN and micro-F1; pair TP/FP/FN and
micro-F1; both quality views; duplicate successes omitted; repeat disagreement
count and denominator. The headline quality comparison should use the fixed
957-review evaluation, with support adjusted visibly if coverage is incomplete.

Successful latency percentiles exclude failed attempts. They must never be
shown without the corresponding attempt success rate: rejecting difficult or
queued requests can reduce the observed successful latency while making the
service worse. Throughput counts completed review operations, including repeats;
it is not unique labelled reviews per second. Server-side inference completion
counts may include work finishing after the client timed out, so only valid
client responses count as successful delivery.

## Network and monitoring limits

The client and server communicate over loopback on one machine. Report these as
local end-to-end API results. They include local HTTP, queueing, prompt creation,
tokenization and model work, but do not establish public-network latency,
multi-tenant reliability, production availability or real customer usage. Shared
host CPU/GPU contention is part of this local measurement environment.

Input-length and prediction-distribution changes are descriptive monitoring
signals. They cannot establish declining accuracy or F1 without labels. Post-run
analysis assigns latency, errors and delayed-label quality to scheduled-arrival
cohorts, while completion throughput uses actual completion-time windows. The
separate live input-shift observer uses completed client outcomes available at
emission time and records its actual emission timestamp, late attachment,
revisions and partial windows. Queueing can move a signal into a later completion
window; these two time bases must not be conflated. State the reference workload,
window size and threshold; a threshold crossing is not a statistical significance
result. Small windows and repeatedly sampled source reviews need explicit limits.

Random audit queues are sampled from successful unique replay samples, so any
quality estimate from them targets that successful population. They do not
represent failures or unattempted reviews. The separate slow/empty-output queue
is deliberately targeted for diagnosis and is not an unbiased quality estimate.

## Permitted conclusion pattern

“On the declared local device and frozen a01 fold, the [variant] service processed
complete twelve-candidate reviews with [success rate] over [attempts] attempts
covering [unique reviews] review IDs. Successful end-to-end P50/P95 were [values]
under [load and batching]. On the frozen validation replay, unique-review
presence/pair micro-F1 were [values] for [scope and support], with [failure-view
results] when never-delivered samples were treated as empty output. These are
single-fold engineering replay measurements over loopback. They do not establish
a newly blind model result, a twelve-fold mean or public production performance.”
