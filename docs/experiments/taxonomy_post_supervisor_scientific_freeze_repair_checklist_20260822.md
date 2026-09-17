# Taxonomy post-supervisor scientific-freeze repair checklist

Date opened: 2026-08-22

Status: **PASS - validation-only scientific freeze complete**

Decision boundary: validation-only; the official test remains sealed.

## Objective

Close the scientific-analysis and reporting gaps jointly identified by the
independent GPT Pro audit and the repository-level verification, without
discarding valid formal-v2 computation or using official-test outcomes.

## Frozen safety boundary

- Formal protocol: `taxonomy_two_stage_formal_v2`.
- Formal execution code boundary: commit
  `aa84212976a652d62cfca31ed8bf0516a216c485`.
- Analysis repair branch start: commit
  `bc098caa9278e77a04ca0acb0618b61b39508c84`.
- Allowed partitions: train and validation only.
- `include_official_test=false` and `test_contract_count=0` throughout.
- Bootstrap: 20,000 synchronized paired `row_uid` cluster draws, seed 13.
- Primary L2 estimand: the aspect-balanced mean of the 12 held-out-aspect
  pair micro-F1 values.
- Sensitivity estimand: pooled held-out pair micro-F1 across all 12 folds.
- Primary L2 representation family: matched `D-N` effects across all seven
  registered methods, with Holm adjustment across the seven methods for each
  declared primary metric.
- Existing single neural training realisations are retained. Multi-seed
  training is a disclosed limitation, not silently represented as completed.
- Large checkpoints, raw score shards, private data and credentials remain
  outside Git.

## Existing user-owned worktree changes

The following pre-existing thesis changes must be preserved and integrated,
not reverted: `report_notes.md`, Chapters 1 and 2, `thesis/main.tex`, and the
new Chapters 3 and 4 working files.

## Checklist

### A. Pre-flight and source of truth

- [x] Fetch the remote and confirm the analysis branch is synchronized.
- [x] Record and protect the pre-existing thesis worktree changes.
- [x] Confirm the formal campaign and replicated artifact hashes previously
  passed, with zero failures and zero test contracts.
- [x] Correct the stale plan statement that says formal-v2 had not launched.

### B. QLoRA `N/D` seen-score invariant

- [x] Compare all 12 L2 QLoRA `N/D` raw seen-score grids.
- [x] Confirm exact equality for a01-a06 and a08-a11.
- [x] Isolate score drift to a07 and a12 and preserve the diagnostic counts.
- [x] Identify the deterministic cause or establish a bounded canonical repair.
- [x] Restore one shared seen-score grid per affected fold without averaging or
  using held-out outcomes to choose a source.
- [x] Recompute affected validation-only metrics and seal new hashes.
- [x] Re-run the invariant audit and require exact `N/D` equality for all seen
  rows in all 12 QLoRA L2 folds.

### C. Statistical analysis repair

- [x] Replace fold-mean percentile bootstrap intervals with synchronized paired
  `row_uid` cluster bootstrap intervals that recompute nonlinear metrics.
- [x] State explicitly that intervals are conditional on the observed trained
  model realisations and fixed 12-aspect taxonomy.
- [x] Rename the enumerated fold sign-flip result as a sensitivity diagnostic,
  not a design-exact randomization test.
- [x] Apply Holm correction across the seven registered L2 `D-N` method effects
  for each primary metric.
- [x] Report model-by-representation heterogeneity conservatively; do not use a
  causal interaction claim without an explicit interaction analysis.
- [x] Add pooled held-out pair micro-F1 as a sensitivity result while retaining
  the aspect-balanced mean as the primary estimand.

### D. Comparability, reuse and evidence provenance

- [x] Extend the L3 reuse audit to field-by-field compatibility: code/config,
  model revision, tokenizer/prompt, training scope, fold/candidate identities,
  row identities, representation, decoder, thresholds and score hashes.
- [x] Reclassify the six-pair L3 study as post-hoc targeted robustness and keep
  it appendix-only; do not claim outcome-blind selection.
- [x] Audit all 12 historical L3 dual-unseen folds and report the selected six
  beside the alternate cyclic matching as an appendix sensitivity.
- [x] Remove stale generic `R` rows from the formal comparison table and admit
  only the selected `R2_positive_concat` TF-IDF/E5 confirmation evidence.
- [x] Add capped-two-sentiment and matched one-stage-versus-two-stage evidence
  and their audits to the scientific-freeze evidence index.
- [x] Audit the Frozen-Qwen zero-shot versus QLoRA interface before making any
  task-adaptation claim; otherwise report a descriptive system comparison.

### E. Formal result tables and figures

- [x] Produce an exact comparability matrix for all headline contrasts.
- [x] Rebuild L2 model/condition, `D-N`, stage-decomposition and pooled
  sensitivity tables from audited inputs.
- [x] Remove the duplicated L2 held-out presence-F1 display where it equals the
  held-out aspect-F1 by construction.
- [x] Move conditional sentiment accuracy to the appendix and rename the oracle
  measure `oracle-gated sentiment-set micro-F1`.
- [x] Report L4 groups separately with support counts and review-cluster
  intervals; do not report cross-group significance claims.
- [x] Add fold-level train rows, removed rows and held-out support diagnostics.
- [x] Generate the agreed thesis figure-data tables and negative-findings table.

### F. Thesis, manifest and final verification

- [x] Update the plan, experiment record and thesis claim boundaries without
  overwriting unrelated user edits.
- [x] Describe L1 as a conventional development reference rather than an
  unbiased generalisation-loss estimate.
- [x] Update Chapters 1, 3 and 4 for corrected uncertainty, L3 provenance,
  metric names and RQ3 claim boundaries.
- [x] Produce a scientific-freeze manifest with code/config/input/output hashes,
  environment, commands, exclusions and test-seal evidence.
- [x] Run unit tests, analysis regeneration, schema checks, non-finite checks,
  prediction-collapse checks, secret scan and official-test contract audit.
- [x] Review every checklist item, mark the final gate PASS/NO-GO, then commit
  and push only safe tracked artifacts to the current GitHub branch.

## Final gate

**PASS.** The repair closes every P0 item shared by the independent audit and
the repository-level verification. The immutable manifest binds clean analysis
commit `61ab6b35fab1966c11d68acb6b2d1a088de427bd`, 35 registered inputs,
26 reporting outputs and 10 audit files. Hash re-verification passed with zero
missing or mismatched files. The official test remains sealed and blocked;
this freeze does not authorise test execution.

Final verification evidence:

- formal evidence: 81 result payloads, 648 score shards and 3,082,212 score
  rows, with zero failures, non-finite values, resume conflicts or test
  contracts;
- local replay: 48 exact fold results and 96 row-evidence files, with every
  prediction-collapse check non-empty and non-saturated;
- inference: 20,000 synchronized paired `row_uid` bootstrap draws, seed 13,
  with seven-method fold-sign sensitivity Holm family;
- implementation: 458 repository tests passed, Python compilation passed,
  staged-content secret scan passed, and no raw data, checkpoints, score shards
  or credentials were admitted to Git;
- thesis: 59 pages compiled with resolved references and citations, zero
  overfull boxes, zero LaTeX errors, and complete rendered-PDF visual review.

## QLoRA invariant diagnostic already established

All affected score grids contain 34,881 seen pair rows with no missing keys.

| Fold | Aspect-score rows changed | Sentiment-score rows changed | Maximum aspect delta | Maximum sentiment delta |
|---|---:|---:|---:|---:|
| `l2-a07` | 4,710 | 9,062 | 0.0156199336 | 0.0276356339 |
| `l2-a12` | 2,472 | 4,051 | 0.0156199038 | 0.0162736177 |

The `N` and `D` results use the same selected training contract in each fold.
The repair must therefore treat this as inference-artifact drift, not as two
independently trainable conditions.
