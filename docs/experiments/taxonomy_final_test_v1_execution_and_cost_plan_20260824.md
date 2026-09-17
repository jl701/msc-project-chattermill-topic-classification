# Taxonomy final test v1 — execution and cost plan

Date: 24 August 2026

Status: **pre-release plan; not an authorisation to open the official test**

## 1. What will run

The final evaluation is inference-only. It reuses the 15 selected DistilBERT
checkpoints, 15 selected QLoRA adapters, 15 Frozen-Qwen few-shot demonstration
contracts, and every frozen validation-selected threshold. It does not retrain
on train plus validation and does not choose any new parameter on test.

The immutable job graph contains:

- 84 Level-2 base score jobs: seven methods x twelve LOAO folds, each producing
  D and N together with one canonical seen-score grid;
- 12 Level-2 fixed-composition jobs;
- 9 Level-4 D-only base score jobs: three methods x three parent groups;
- one all-score seal/backup barrier; and
- one analysis/reveal job.

This is 93 base score jobs, 12 composition jobs, and 107 nodes including the
seal and single reveal. The exact IDs and dependencies are frozen in
`docs/experiments/taxonomy_final_test_v1_job_plan.json`.

## 2. Three-pod assignment

The recommended deployment uses three RTX 4090 Community Cloud pods at the
previously observed price of USD 0.75 per GPU-hour. Job IDs always come from the
frozen plan; operators do not pass an ad-hoc method, fold, condition, threshold,
or metric.

| Worker | Exact scoring responsibility | Expected 25%-margin time |
|---|---|---:|
| A — few-shot | Frozen Qwen few-shot, all 12 L2 folds and all 3 L4 groups | about 8.0 h |
| B — adapted/supervised | QLoRA, all 12 L2 folds and all 3 L4 groups; then DistilBERT for the same 15 scopes | about 4.0--5.0 h |
| C — remaining base systems | Frozen Qwen zero-shot for 12 L2 folds; then TF--IDF, E5, and DCWT for their 12 L2 folds | about 4.0--5.5 h |

The 12 fixed-composition bundles are deterministic score recombinations after
the few-shot and QLoRA sources have been collected; they require no new model
inference. Final metrics are not computed on any pod while scoring is in
progress.

## 3. Runtime basis

For 1,587 held-out reviews, the canonical two-stage runner makes two Qwen calls
per review--aspect candidate: one aspect-presence call and one three-way
sentiment call. Sharing unchanged seen scores across N and D by construction
gives:

- 495,144 prompt calls for each L2-only Qwen system;
- 114,264 additional prompt calls for a method included in all three L4
  groups; and
- 609,408 calls for each few-shot or QLoRA L2+L4 workload.

Using the measured formal-validation rates of 26.611 prompts/s for few-shot,
57.327 prompts/s for QLoRA/short frozen prompting, and 1,159.647 candidate
calls/s for DistilBERT gives the following 25%-margin estimates:

- few-shot: 7.95 GPU-hours;
- QLoRA: 3.69 GPU-hours;
- zero-shot Qwen: 3.00 GPU-hours;
- DistilBERT: 0.18 GPU-hours.

The classical/E5/DCWT jobs, environment start-up, transfer, and sealing are
budgeted separately as approximately 1--3 additional pod-hours. The expected
total is therefore about 16--18 GPU-hours, with a conservative operational
ceiling of 20 GPU-hours. At USD 0.75/hour this is approximately USD 12--13.50
expected and USD 15 at the conservative ceiling. With all three pods active the
instantaneous burn rate is USD 2.25/hour. These are planning estimates, not a
guaranteed bill; current RunPod availability and price must be checked before
launch.

The likely wall-clock time is 8--10 hours because the few-shot worker is the
critical path. A cold model-cache or network delay may extend wall time without
changing the scientific protocol.

## 4. Score, backup, and reveal order

1. On a clean commit that exactly equals its upstream, create one authorised
   release record. This is the first permitted operation that reads and hashes
   the official test.
2. Materialise one unlabelled worker file and a separate private label vault.
   Copy and verify both at the registered immutable backup root.
   GPU workers receive only `unlabelled_reviews.csv`, `data_manifest.json`,
   and the authorised release record. They do not receive, open, hash, or
   otherwise access `label_vault.csv`; only the final local analysis process
   requires and verifies that private file.
3. Run the three frozen worker assignments. Each completed score bundle is
   immutable and contains no review text or labels.
4. Continuously copy completed bundles to the local collection root and verify
   their file and payload SHA-256 values before considering a remote copy safe.
5. Stop an individual pod only after its exact assigned set is complete and the
   final local receipt matches. Stop never means terminate/delete until the
   complete project backup is independently confirmed.
6. After all 93 base bundles are collected, generate all 12 fixed-composition
   bundles from the frozen source scores and thresholds.
7. Require all 105 bundles, zero failures, finite/non-collapsed scores, zero
   third-sentiment violations, and 210 verified score/manifest backup receipts.
8. Seal the score graph while outcomes remain hidden.
9. Run the one-time registered analysis against the label vault and reveal all
   confirmatory and descriptive outputs together.
10. Report the result regardless of direction. No post-test candidate,
    threshold, prompt, batch-size, metric, or fold change is permitted.

## 5. Failure and spending controls

- A data, schema, row-identity, artifact, or hash mismatch aborts before
  metrics.
- There is no post-release batch-size fallback. OOM, non-finite scores,
  prediction collapse, a third sentiment, or a resume conflict is a protocol
  failure, not permission to tune.
- A pure operational interruption may be repeated only before any reveal and
  only with byte-identical scientific configuration and preserved failure
  evidence.
- Low performance or a failed confidence interval is a scientific result and
  is never rerun.
- No pod may remain active merely to wait for another worker. Completed,
  locally verified workers are stopped immediately to cap spending.

## 6. Launch boundary

Phase 6 can declare the implementation ready to request one-time release only
when the full preflight, validation mirror, synthetic 107-node dry-run, complete
test suite, thesis build, secret/test-contract scan, scientific freeze, and Git
push have all passed. That readiness verdict is not itself authorisation. A new
explicit user instruction is required before release creation or any official-
test access.
