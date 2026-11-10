# Frozen production-configuration quality replay

The100-review API gate preserved all fixed-method label sets, but its cached
presence scores differed numerically from offline scores. The old957-review
service evidence also used different retrieval/truncation/scoring/execution
settings. Neither source establishes that the new fixed cache-on deployment
matches this campaign's uncached offline fixed classifier on all957 reviews.

This additional check is deployment evidence, not another algorithm iteration.
It keeps the frozen classifier, thresholds and service configuration. No gold is
read by the runner. Default scope is the fixed cache-on method, with one C1 HTTP
attempt for each evaluation input, no retry, and the existing three warmups saved
separately. An explicitly named dynamic method can be run in its own new output
if required; the default registration does not silently add that cost.

The new runner reuses `run_rag_service_study.server_session` and `run_benchmark`
without editing them. It preserves the exact original service plan bytes and
binds its source hashes, fixed clone, execution/grouping contract, evaluation
replay SHA and frozen index exclusion evidence. It refuses modified inputs or
source files. Before model startup it also requires the full main offline matrix
to be complete and binds all its method hashes plus the fixed comparator's
957-review population. The registration pins the SHA of
`configs/experiments/serving_rag_negative_20261111.json` and its exact eight-method
list. Every method must have all 957 unique request IDs, successful response
identities, complete ordered candidate probability/margin and sentiment grids,
valid finite scores, and a matching successful count and score-file SHA. A
self-consistent fixed-only matrix or a file of placeholder rows cannot satisfy
this gate. Registration can precede the long inference run.

The initial plan-only `deployment_fixed957` directory is preserved. It was never
executed; the strengthened dependency gate is registered separately as
`deployment_fixed957_v2`.

Run from the repository root, with `RAG_PYTHON` set to the approved local GPU
environment's Python executable. These are separate registration and execution
commands; preparing this tool did not start GPU work.

```powershell
& $env:RAG_PYTHON scripts/run_rag_deployment.py `
  --service-plan outputs/rag_20261111/service_final_v2/plan.json `
  --replay outputs/serving_20261109/replay_evaluation.jsonl `
  --offline-root outputs/rag_20261111/evaluation_final `
  --offline-config configs/experiments/serving_rag_negative_20261111.json `
  --output outputs/rag_20261111/deployment_fixed957_v2 `
  --method fixed --expected-reviews 957 --plan-only
```

After the main eight-method offline matrix completes:

```powershell
& $env:RAG_PYTHON scripts/run_rag_deployment.py `
  --service-plan outputs/rag_20261111/service_final_v2/plan.json `
  --replay outputs/serving_20261109/replay_evaluation.jsonl `
  --offline-root outputs/rag_20261111/evaluation_final `
  --offline-config configs/experiments/serving_rag_negative_20261111.json `
  --output outputs/rag_20261111/deployment_fixed957_v2 `
  --method fixed --expected-reviews 957
```

The main request journal remains ordinary benchmark JSONL at
`runs/deployment_quality__fixed/requests.jsonl`. `sample_id` identifies the
original review; `request_id` identifies the HTTP attempt. All failures and
timeouts remain in the journal. Completion records coverage and hashes of the
request journal, summary, client configuration and dependency receipts. A failed
or interrupted execution preserves partial evidence and is not silently retried.
Warmup requests never enter the957-review denominator.

`analyse_rag_deployment.py` is a separate post-completion join. It uses actual API
predicted sets and the offline scores decoded at already-frozen thresholds;
it never calls the classifier selector. It reports attempted/successful/failed/
missing/duplicate coverage, per-review output changes, presence versus
sentiment-only changes, and numerical presence probability/margin deltas.
Complete-reviewed-label pair and presence counts/F1 cover overall, seen and
heldout scopes, with API-minus-offline paired complete-review bootstrap intervals
using2000 draws and seed20261111. Failed or missing reviews count as empty
predictions. Coverage anomalies are explicit and cannot produce an unqualified
clean deployment claim. HTTP times are not divided by offline component times
to invent an acceleration result. Raw inputs, attempts and individual labels
stay in ignored outputs; public reports contain aggregates only.

After inference has completed, run the CPU-only analysis against a fresh output
directory. The example uses the strengthened deployment registration (`_v2`):

```powershell
& $env:RAG_PYTHON scripts/analyse_rag_deployment.py `
  --deployment-root outputs/rag_20261111/deployment_fixed957_v2 `
  --offline-scores outputs/rag_20261111/evaluation_final/fixed.jsonl `
  --labels outputs/serving_20261109/labels_evaluation.jsonl `
  --output outputs/rag_20261111/deployment_fixed957_v2_analysis
```

The analysis opens gold only after validating completed inference receipts,
the frozen replay/clone/plan, the complete registered offline method matrix,
all score hashes, the HTTP journal and effective runtime policy. Model and
threshold selection are never invoked. It reports the actual API outputs;
offline scores are decoded using the deployed clone's frozen threshold.
Probability deltas and raw-margin deltas are kept separate.

`coverage.json` is written before a coverage-related refusal. Missing or
duplicated review coverage blocks the primary gold report by default. The
explicit `--allow-incomplete-audit` option permits a separately labelled audit:
the earliest `attempt_index` is used even if a later retry succeeded, missing
reviews receive empty predictions, and unexpected inputs never enter the
registered population. Ordinary failed HTTP attempts do not invalidate
one-attempt coverage; they remain empty predictions in every quality denominator
and are explicitly counted as failures. Exact-set accuracy also counts a failed
request as incorrect when its gold set happens to be empty.

`quality.json` contains only aggregate results and evidence hashes.
`private_review_differences.json` contains the changed review identifiers and
predicted sets, with no raw review text or gold. Keep it in ignored outputs.
Complete-review paired bootstrap uses identical resampled review indices for API
and offline outputs, including failures, for 2,000 draws with seed 20261111.
Differences are API minus offline; F1 differences are percentage points, while
TP/FP/FN differences are counts. Seen and heldout scopes include negative reviews
from the whole registered population. This is a
`historically_exposed_validation_replay`, not a new blind test, regardless of
the current run's classifier-freeze separation. No threshold, primary model,
or routing decision may be changed using these results.
