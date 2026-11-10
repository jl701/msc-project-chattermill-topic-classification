# Dynamic demonstration retrieval and review routing

The user approved local execution on 11 November 2026. This campaign compares fixed, random, BM25, E5 and fused demonstration selection for the frozen l2-a01 two-stage Qwen service. The tracked configuration is `configs/experiments/serving_rag_20261111.json`.

The first execution track uses the existing 100 development and 957 historical validation replay reviews. These samples were exposed in historical selection; this track supplies regression and engineering evidence, not new blind generalisation results. A separate 800-review acquisition and human annotation workflow will provide independent development, calibration and evaluation records. New-source gold metrics cannot be produced until actual human annotations are available.

The preflight began from clean commit `6da0148392cb254292a58eccbc2de661069b6ef2`, fetched the remote and created branch `codex/chattermill-rag-20261111`. Existing experiment outputs are preserved. GPU jobs run serially. Raw text, scores, labels and paths remain under ignored `outputs/rag_20261111`.

For offline selection, all twelve sentiment vectors are computed with the same frozen adapter once per review and reused across detection variants. This gives identical Stage 2 evidence for rethresholding and avoids unnecessary repeated computation. Offline matrix elapsed time is not API latency. The final service comparison executes actual retrieval and conditional sentiment scoring and includes queueing, failures and fallbacks.

Strict held-out-theme methods exclude Account access examples and reviews containing that theme. Threshold selection and directed improvements use seen-theme labels only. A later optional support-example onboarding branch is separate from strict unseen evaluation. The official test split is outside this campaign.

All substantive configuration changes and their timing relative to outcomes will be recorded here. No resume is edited as part of execution.

Before new quality outcomes, the CPU-only index encoding attempt was stopped for runtime efficiency. Offline E5 index encoding runs on the otherwise idle GPU, then releases it before Qwen loads; online query encoding remains on CPU. Existing base-weight hashes match the previous serving fingerprints. Three development reviews passed the historical/full-grid/conditional/last-token decoding gates; conditional sentiment scores can differ numerically, as previously observed.

After public collection, all 800 new input texts were added to index duplicate exclusions before any new quality scoring. The original unencoded index is preserved; the final index uses 6,954 training reviews and 76,494 review-aspect examples. It excludes 473 held-out-theme reviews, 157 internal exact duplicates and 346 matches to evaluation/development inputs. All four original fixed demonstrations remain eligible. No evaluation label or model outcome informed these exclusions.

## Prompt budget implementation before new inference

The historical segment-aware encoder truncates the longest text among both
demonstrations and the current review. The approved new campaign instead
prioritises preserving the current review: all seven retrieval configurations,
including the fixed-example comparator, use `demonstration_review_first_left_character_trim_v1`.
It trims the longest demonstration review first, preserving task instructions,
candidate cards and answer labels. Only if all demonstration text is exhausted
may it trim the current review. Per-prompt original/final token counts,
demonstration characters removed, and query characters removed are recorded;
query removal exceeding 20% is flagged severe. No new prediction outcomes were
inspected when recording this implementation choice. The default historical
service without an attached retriever retains the original encoder.

The cleaned index uses official training records only. It excludes all reviews
with the held-out aspect, canonical exact matches against the declared replay
inputs, and normalised character-5-gram Jaccard matches of at least 0.90 for
texts of length at least 20. Canonical duplicates within training retain the
first stable row ID. Evaluation records are never dropped by this operation.
The initial index audit retained all four frozen aspect demonstrations, so the
fixed comparator does not need replacement. Dynamic examples bypass the old
fixed-prefix KV cache; fixed examples use its existing exact-token compatibility
check. Retrieval exceptions fail offline runs; the API may explicitly record a
fallback to the validated fixed examples.

Before performance measurements, the unused public-review reserve is designated
as a separate capacity/benchmark population. Reserve texts matching the legal
training pool or any quality input are excluded using the registered exact/near
rule. Within each app, stable SHA256 ordering alternates capacity and benchmark
allocation. Three 30-request C1 pilots per method calibrate common absolute
loads; closed/open/soak/burst comparisons use the disjoint benchmark portion.
No labels or classification outcomes inform this allocation. Burst peak is
2x the dynamic candidate's own C1 pilot capacity, while baseline/recovery use
0.5x the lower method capacity, ensuring a genuine overload even if the dynamic
method unexpectedly has higher capacity.

Before any RAG service run, the burst normal-phase rule is registered as
`max(requested_baseline_seconds, ceil(20 / baseline_rate) + 1)`, with the requested
duration defaulting to60 seconds and `baseline_rate = 0.5 * minimum_C1_capacity`.
The capacity-derived duration and planned arrivals are frozen in
`common_rates.json` before burst execution; all three cycles use that same
duration. This guarantees at least20 planned normal-phase arrivals even on a
slow local service. For example, capacity0.3 reviews/s gives a normal-phase
rate0.15 reviews/s, a135-second duration and21 planned arrivals. The burst and
recovery durations remain30 and180 seconds. No label or observed burst outcome
is used to extend the normal phase.

Planned arrivals do not replace successful observations. Recovery still requires
at least10 actual successful baseline requests with measured timing; otherwise
its status is `not_evaluable_insufficient_baseline_requests`. This RAG study
supplies no historical service fallback: missing usable normal-phase evidence is
labelled `baseline_source=no_baseline`, with original phase successes/failures
retained. An explicitly supplied historical fallback remains labelled as such
for the separate historical resilience analysis. The recovery-analysis source
hash is included in the immutable RAG service plan.

The initial 20-review seven-configuration smoke completed in 603.81 seconds,
with all140 scoring records successful. Presence-only mean time ranged from
3.03 to3.72 seconds/review across configurations; these are offline component
times, not end-to-end service latency. Full development plus historical
regression is provisionally around9 GPU-hours; final resource estimates will
include service tests and optional new-input scoring separately. No new gold
outcomes have been inspected at this checkpoint.

## Dynamic retrieval waiting limit before service measurements

The API applies a 10-second default waiting budget to dynamic retrieval through
`TimedDemonstrationRetriever`; `--retrieval-timeout-s` can explicitly set another
budget. Fixed demonstrations and offline quality scoring keep their existing
execution paths. Retriever construction, index loading and cold E5 encoder
initialisation are part of service startup, outside the per-request budget.

One background CPU worker performs query preparation and all review/candidate
selections for the request batch. It has at most one active future and no queued
batches. When waiting exceeds the budget, the engine either explicitly records
fixed-example fallback or fails according to `--retrieval-fallback`. While the
timed-out computation is still running, later requests immediately take the same
fallback/error path instead of piling up. Python threads do not safely support
hard model-compute cancellation; this bounds request waiting, not background
compute time. Shutdown likewise does not forcibly terminate running CPU work.

Completed values stay local to their original future. Late completions are
discarded, and each successful batch validates exact query/card keys, its batch
identity, the index version and fingerprint, and the selected demonstration
identities and 2Y2N answers. No query-result cache survives into another batch.
The worker's selection work time is reported separately; engine preparation
waiting and subsequent lookup time must not be summed with that work again.
CPU fault tests cover timeout, repeated busy fallback, late-result isolation,
exception recovery, metadata mismatch and explicit engine fallback. Real GPU
fallback checks and service measurements remain separate execution gates.
