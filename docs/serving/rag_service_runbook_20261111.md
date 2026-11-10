# Frozen-classifier service validation

This runbook records the invocation order, not completed GPU measurements.
Complete the shared-margin development matrix and any final authorised
example-policy iteration first. Freeze the final classifier selection separately
from later review-routing selection. No957-review evaluation labels are needed
for artifact creation, API correctness, or the fault gate.

Run from the repository root. Set `RAG_PYTHON` to the approved GPU environment's
Python executable and `RAG_E5_SNAPSHOT` to the local pinned E5 snapshot at revision
`f52bf8ec8c7124536f0efb74aca902b2995e5bcd`. Replace the three explicit final-path
placeholders below only after the classifier is frozen. The score directory must
contain the exact per-method development JSONL files hashed in that selection;
the final directory may differ from `development_margin` if the last iteration
is used. Do not substitute a preliminary probability-space selection.

```powershell
$FrozenSelection = 'outputs/rag_20261111/<final_selection>/selection.json'
$DevelopmentScores = 'outputs/rag_20261111/<final_development_scores>'
$ServiceOutput = 'outputs/rag_20261111/<new_final_service_output>'
$ServiceArgs = @(
  'scripts/run_rag_service_study.py',
  '--artifacts', 'outputs/serving_20261109/artifacts.json',
  '--index', 'outputs/rag_20261111/index_all_populations',
  '--encoder-path', $env:RAG_E5_SNAPSHOT,
  '--selection', $FrozenSelection,
  '--development-replay', 'outputs/serving_20261109/replay_development.jsonl',
  '--development-score-root', $DevelopmentScores,
  '--correctness-reviews', '100',
  '--capacity-replay', 'outputs/rag_20261111/performance_pool/replay_capacity.jsonl',
  '--replay', 'outputs/rag_20261111/performance_pool/replay_benchmark.jsonl',
  '--output', $ServiceOutput,
  '--port', '8766', '--retrieval-timeout-s', '10',
  '--sentiment-execution', 'canonical_full_grid_groups'
)
& $env:RAG_PYTHON @ServiceArgs --plan-only
```

`--plan-only` loads no model and starts no subprocess. It freezes inputs and
source hashes, creates calibrated fixed/dynamic artifact clones, and saves
`plan.json`. The dynamic clone contains the complete fixed fallback policy,
including score space, separate margin threshold, runner-up threshold, fixed
clone path and SHA256. Original artifacts are unchanged. The fixed comparator
uses prefix caching; the chosen dynamic method bypasses it. Legacy probability
values remain present alongside the explicit raw-margin policy.

The registered final-iteration method ID `e5_negative_seen` resolves to underlying
retriever `e5`, seed13 and `same_aspect_negative_seen`. This mapping is stored in
the clone and plan and passed through `--retrieval-selection-policy`; the method
ID remains distinct in output paths and calibration identity. The fixed fallback
retains `global` selection and its separately frozen thresholds. Existing E5,
BM25 and RRF IDs continue to resolve to `global`. Correctness checks also compare
requested/effective selection policies, treating absent legacy metadata as global.

After inspecting that immutable plan, run the100-development-review API gate:

```powershell
& $env:RAG_PYTHON @ServiceArgs --stage correctness
```

It compares ordinary conditional HTTP responses with the corresponding frozen
offline full-grid outputs using the active score space. Exact complete label
sets and retrieval/prompt contract must agree for every selected review, with
all probability/margin differences reported. A failed request or mismatch
blocks timing promotion and must be audited before proceeding. This gate reads
input text and score outputs, not gold labels. It is distinct from classification
accuracy and its elapsed time is excluded from service latency results.

Use the resulting clones for the separate registered real-model fault gate:

```powershell
$ServicePlan = Get-Content -LiteralPath "$ServiceOutput/plan.json" -Raw | ConvertFrom-Json
$DynamicMethod = $ServicePlan.dynamic_candidate
$DynamicPolicy = $ServicePlan.methods.$DynamicMethod.retrieval_selection_policy
$SentimentExecution = $ServicePlan.server.sentiment_execution
$DynamicArtifacts = $ServicePlan.methods.$DynamicMethod.artifacts
$FixedArtifacts = $ServicePlan.methods.fixed.artifacts
& $env:RAG_PYTHON scripts/check_rag_faults_gpu.py `
  --artifacts $DynamicArtifacts --fixed-artifacts $FixedArtifacts `
  --index 'outputs/rag_20261111/index_all_populations' `
  --encoder $env:RAG_E5_SNAPSHOT --method $DynamicMethod `
  --retrieval-selection-policy $DynamicPolicy `
  --sentiment-execution $SentimentExecution `
  --development-replay 'outputs/serving_20261109/replay_development.jsonl' `
  --output-dir "$ServiceOutput/fault_gate" --normal-timeout-s 10
```

That harness uses three input-only length ranks for explicit failures, timeouts,
busy fallback, recovery and index-version checks. The100-review correctness gate
and three-review fault harness are separate denominators. Only after both pass
may the final957-review replay and separately calibrated business routing be
evaluated. Final evaluation outcomes must not change cache behaviour, classifier
thresholds or retrieval policy. A necessary implementation change after a gate
requires a fresh immutable study output and rerun of the affected gate.

Performance stages use the disjoint label-free capacity and benchmark pools:

```powershell
& $env:RAG_PYTHON @ServiceArgs --stage capacity
& $env:RAG_PYTHON @ServiceArgs --stage closed
& $env:RAG_PYTHON @ServiceArgs --stage open
& $env:RAG_PYTHON @ServiceArgs --stage soak
& $env:RAG_PYTHON @ServiceArgs --stage burst
```

Use these separate stages so the external fault-gate result is checked before
timing. The orchestrator enforces API correctness but does not itself invoke the
separate fault harness. `--stage all` should therefore not be used to bypass this
execution sequence. Resume only unchanged completed evidence; interrupted
partial runs are preserved and require a fresh output directory.

Capacity and comparison stages alternate method order within three paired
repeats. Cold startup and three warmups per model load are recorded separately.
Open-loop schedules retain timeout, rejection and client-capacity-drop attempts.
Burst normal duration is frozen from the capacity pilot as
`max(requested60, ceil(20 / (0.5 * minimum_capacity)) + 1)` for all three cycles.
At least20 arrivals are planned; fewer than10 actual baseline successes remains
unevaluable, with no historical fallback. Full outputs and local absolute command
paths remain ignored under `outputs/rag_20261111/`.

CPU validation for the burst update:50 focused synthetic tests passed across
service scheduling, resilience analysis and margin integration. The CLI help was
also checked without loading a model. No GPU service run was executed while
preparing this record.

The first conditional API gate is preserved under `service_final`: fixed matched
100/100 and dynamic matched99/100, with a Stage2 full-grid/selected-only numerical
batching discrepancy. The replacement study uses a fresh `service_final_v2`
directory and explicitly selects `canonical_full_grid_groups` for both methods.
It keeps the frozen classifier, thresholds, taxonomy and examples. Complete
original full-grid microbatches containing a selected aspect are forwarded,
including companion aspects; groups with no selected aspect are skipped. All
actual companion compute remains in measured service latency and prompt counts.

The service plan and both artifact clones freeze the execution policy and the
shared grouping contract: ordered-candidate SHA256, pair batch size, padded token
budget, logits mode, grouping rule and contract SHA256. Every API correctness
response must match that contract and account for actual/companion/full-grid
prompt counts. The replacement reruns all100 development inputs for both methods,
not only the one failed record. The separate fault harness checks exact selected
sentiment vectors against full-grid execution for both fixed and dynamic methods,
then retains the existing failure/fallback/recovery checks. Legacy QLoRA comparator
plans remain unchanged and use the default `selected_only` mode.
