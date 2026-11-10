# Registered retrieval failure and recovery checks

This record precedes real-model fault outcomes. These checks use three
development reviews selected at the minimum, middle and maximum character-length
ranks, with request ID breaking ties. No gold labels or evaluation outcomes are
used. They verify resilience and reproducible inference, not classification
quality or HTTP latency.

The approved fallback is a complete fixed-example inference policy. Dynamic
service artifacts must identify the fixed comparator artifact by absolute path
and SHA256 and copy its aspect and runner-up thresholds. The fixed artifact must
be the calibrated fixed method and share the same model, adapter, prompt limits,
taxonomy and demonstrations. Validate these claims before loading the model;
arbitrary unverified thresholds are rejected. The fixed artifact has no
self-referential fallback hash.

All retrieval selections for a request batch are resolved before prompt
encoding. If any selection or preparation fails, the whole current request batch
uses the fixed policy. This avoids mixing retrieved and fixed aspect decisions
inside a review. Fallback also uses the fixed thresholds for both the presence
gate and sentiment decoding. If any query canonically matches a fixed labelled
example, fail closed before any model scoring instead of exposing the answer as
its own demonstration. Normal dynamic inputs and their thresholds remain
unchanged, and dynamic paths continue to bypass the fixed-prefix cache.

Responses explicitly report `retrieval_fallback`, `effective_retrieval_method`,
`effective_thresholds` and `fallback_policy_source_sha256` in diagnostics.

`scripts/check_rag_faults_gpu.py` will execute the following serially with one
resident real Qwen model and its frozen adapter. The independently configured
fixed engine shares the weights sequentially, not concurrently. Prefix caching
is off throughout this gate to compare identical prompt inputs.

Before fault injection, compare the legacy `predict_batch(full_grid=True)`
sentiment grid with the additive `score_full_sentiment` method. The latter calls
the existing adapter stage over the same twelve candidates and microbatches,
omitting the unused presence stage. Require identical encoded sentiment inputs,
candidate order, truncation audit and per-aspect decoded sentiment labels; report
all probability differences. Only after this gate may new population runners
reuse sentiment-only anchors. This removes redundant computation without changing
the classifier or its training data.

1. Compare raw dynamic retrieval with the timed wrapper in normal operation.
2. Inject a retrieval exception and compare the complete fallback result with
   the fixed comparator.
3. Inject a retrieval wait deadline with a controlled blocking event. The
   0.05-second fault budget is deliberately shorter than the normal 10-second
   service budget; it is a fault fixture, not a proposed production setting.
4. Keep the timed-out worker blocked and submit another request batch. Require
   immediate busy fallback and no second queued retrieval job.
5. Release the worker, discard its late result, restore the normal budget and
   require recovery to the normal dynamic predictions.
6. Reject a query identical to a fixed example on the fallback path, as well as
   a tampered fixed-source hash or fallback threshold, before model scoring.
7. Switch to an isolated legal training subset with a new pool fingerprint and
   matching saved embedding rows. Require the response and model version to
   identify the new index, reject a deliberately tampered index manifest, then
   restore the original index and normal predictions. This subset is only a fault
   fixture and never enters quality selection.

Success requires exact final label sets and matching hashes of the encoded
model inputs for each normal/fallback equivalence comparison. All floating-point
score differences are reported separately without inventing a tolerance gate.
The harness records source/input hashes before model loading, preserves failures,
and verifies the original index is unchanged. The ordinary 10-second service
deadline bounds waiting only; it does not cancel a running CPU model thread.

For the final registered example-policy iteration, `--method e5_negative_seen`
uses the shared method resolver: E5, seed13, `same_aspect_negative_seen`. Optional
`--retrieval-selection-policy` asserts that mapping rather than overriding it.
The method ID and resolved settings must agree with the service clone before GPU
code is imported. Normal, restored and version-switched retrievers all use the
same resolved settings. Response checks require the new policy for seen
candidates, global selection for the heldout candidate, and complete global fixed
policy on fallback. The model-version and example-identity gates remain in force.

The isolated version-switch fixture now retains the first eight legal Y rows,
the first eight N rows for each candidate aspect, and every fixed demonstration.
This preserves per-aspect negative support for the new policy, with embedding
rows kept aligned; it is a resilience fixture and does not enter quality results.

The replacement service's `service_execution.sentiment_execution` and
`sentiment_group_contract` are bound into both clones. The fault harness resolves
that same mode for fixed and dynamic engines and rejects a CLI override, candidate
order or batching-parameter mismatch before GPU loading. With
`canonical_full_grid_groups`, it additionally compares the three input-only
reviews under full-grid and conditional execution for each engine: selected
sentiment vectors and presence scores must be exactly equal, and final label
sets must agree. Skipped groups need not appear in the conditional trace. This is
a grouping-correctness check, not additional quality selection. Failures retain
their response evidence, and no threshold is adjusted to hide a mismatch.
