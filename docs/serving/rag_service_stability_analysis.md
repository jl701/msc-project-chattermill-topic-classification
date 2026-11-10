# RAG service stability analysis

This CPU analysis is specified before the new RAG service measurements. It
uses only completed runs from the `rag_http_service_study_v1` campaign supplied
on the command line. It does not load a model, call a GPU, read gold labels or
import any earlier service results. It does not modify frozen service code,
plans, raw responses or telemetry.

```powershell
python scripts/analyse_rag_service_stability.py --campaign PATH_TO_THIS_RAG_SERVICE_CAMPAIGN --output-dir PATH_TO_NEW_PRIVATE_ANALYSIS
```

The output directory must not already exist. The default selects `soak__` runs.
Optional repeated `--run-prefix open__` or `--run-prefix burst__` arguments select
other independently scheduled runs. Closed-loop pilots are deliberately excluded
from this analysis. `stability.json` and `windows.csv` contain aggregates and
file hashes, without input text, individual sample/request IDs, prediction sets,
machine paths or process IDs. Raw evidence and these working aggregates remain
in ignored local outputs until separately reviewed for publication.

## Completion and denominators

Each selected run must have a completion marker whose summary hash matches. The
summary must declare an uninterrupted run and cover the entire arrival period.
The analyser reconstructs every planned arrival from the registered rate and
duration, or the registered continuous phase plan. Raw record count, unique
attempt indices, scheduled times, response identities and summary delivery counts
must agree. Missing, duplicated or unpersisted attempts prevent promotion to a
completed stability result. The replay fingerprint must match, and referenced
replay/phase files must remain inside this campaign.
Stored HTTP/scheduled latency and scheduling lag must also be finite,
non-negative and agree with the recorded request timestamps; corrupt derived
latency cannot improve a response-budget fraction.

An existing incomplete run is listed separately and is not treated as successful
or silently filled with invented outcomes. A completed subset is not a declaration
that every stage of the wider campaign has finished.

Five-minute arrival cohorts use `scheduled_s`. Their denominators retain all
planned attempts, including HTTP 429/503/504, client timeouts and unsent client
capacity drops. Successful HTTP latency percentiles use actual send to complete
response. Scheduled latency also includes client scheduling delay. The two-second
completion fraction uses scheduled latency and every planned attempt. No
contractual SLA or automatic business acceptance decision is inferred.

Completion-window throughput separately counts successful responses whose
`finished_s` falls in each wall-clock interval, including work arriving in an
earlier window. Whole-run throughput includes client drain. Per-cohort throughput
is not summed. A shorter final arrival interval is `partial_arrival_tail`.
Completion after the declared arrival horizon is reported as `drain_only_tail`,
with no newly planned arrivals. Neither is described as a complete five-minute
arrival window. A tiny drain interval can have a numerically large completion
rate; use whole-run throughput for capacity claims.

## Fallbacks and resource sampling

Fallback counts use the actual response's `diagnostics.retrieval_fallback` when
available, with the batch retrieval records as a compatibility source. Missing
diagnostics are explicitly unobserved, rather than assumed to mean no fallback.
A successful fixed-example fallback remains an HTTP success and is reported
separately from ordinary dynamic retrieval. Failed requests remain in delivery
denominators regardless of whether fallback diagnostics are available.
Retrieval-timeout fallbacks are also counted separately from HTTP timeout
failures. Multiple affected candidate aspects within one response count as one
review-level occurrence of each fallback error category.

Only a completed server session whose launch-to-close interval encloses the run
can supply resource observations. Samples are approximately aligned by the
client's UTC start, written just before its monotonic epoch. Missing or ambiguous
association is unavailable; earlier campaigns and whole-session fallback values
are not substituted. Both whole-run sampled memory min/max and the range of
complete five-minute window memory maxima are retained as different quantities.

GPU measurements are device-wide samples, not continuous per-process peaks.
Process-tree RSS sums the owned server and descendants and can count shared pages
more than once. Memory trends can support a bounded description of this run;
they do not prove the absence of all possible memory leaks. Driver limit counters
and sampled flags remain distinct measurements, with their existing units and
measurement notes preserved.

## Repeat consistency and interpretation

Exact frozen replay text is hashed internally to identify repeated inputs; hashes
are not published per input. Within a single run, comparisons are further grouped
by the response's model version, fold, variant, effective thresholds, effective
retrieval method, effective retrieval selection policy and fallback policy.
`e5_negative_seen` still reports underlying retrieval method `e5`; the additional
`diagnostics.effective_retrieval_selection_policy` distinguishes
`same_aspect_negative_seen` from E5 `global` selection. The response-level policy
defines the group: its seen candidates use the new selection policy and held-out
candidates intentionally retain global selection. These per-entry differences
do not create artificial configuration changes. A fixed fallback reports effective
method `fixed` and selection `global` and remains separate from successful dynamic
retrieval. The same input under different thresholds or selection/fallback
configurations is never pooled into one consistency claim. Successful responses
lacking this policy identity, including a missing selection field, remain in
delivery counts but are excluded from qualified consistency comparisons; missing
selection is never inferred to be `global`.

For each policy group, the analyser counts successful repetitions after the first,
inputs with different final aspect-sentiment sets and repetitions differing from
the first successful set. Order of aspects or labels does not create a false
difference. Numeric score identity is not tested. Repeatedly returning the same
set does not show that the set is correct or create new independent quality
samples.

This analyser reports latency, errors, fallback use, queueing, resources and
repeat consistency. It does not define a drift alarm or compute F1. A future
unlabelled drift signal and a later evaluation with independent gold labels are
separate tasks with separate evidence and denominators.
