# Human re-review and adaptation feedback

This workflow prepares the planned 20% blind re-review, records human adjudication, and exports a separate adaptation-only demonstration candidate pool. It never overwrites the live first-round label store or updates the live retrieval index. All comments, labels and intermediate versions belong under ignored `outputs/` directories.

## Complete a defined first-round scope

Use the main local annotation page to complete a scope such as the 50-review pilot, the adaptation partition, or all reviews. Every item in that scope must be confirmed by a human before the blind sample can be prepared. Ambiguous or non-English items remain pending rather than becoming gold automatically.

The current gold policy requires genuine English input and human English confirmation. Save a non-English or unresolved-language item as `needs_review`, leave English confirmation unchecked, and explain the language issue in the note. Understanding a translation does not make the original input English. Such items stay pending until a separate explicit policy is registered; they cannot be released as the current English gold or exported as feedback examples. A pending item blocks release of its whole chosen scope rather than silently shrinking the denominator.

Avoid treating overlapping scopes as independent repeat audits. For example, a pilot audit and a later all-review audit must each retain their own sampled IDs; do not add their counts as if all sampled comments were different.

## Prepare a blind sample

Run:

```powershell
python scripts/review_annotation_workflow.py prepare-blind --data PATH_TO_NEW_REVIEWS --output PATH_TO_BLIND_QUEUE_V1 --group pilot --mode independent
python scripts/serve_blind_review.py --queue PATH_TO_BLIND_QUEUE_V1 --port 8771
```

The queue samples `ceil(20% × scope size)` IDs by a fixed SHA256 ordering with seed 20261111, without using labels or model mistakes. For the 50-review pilot this produces 10 items. The new page has an empty, separate annotation store and displays neither first-round labels, first-round notes, ratings nor model predictions. The original first-round page can remain running.

An independent second reviewer must use a different real reviewer identity. If only one reviewer is available, prepare the queue explicitly with `--mode delayed_same_reviewer --delay-hours 24` instead. The comparison step checks the same reviewer identity and elapsed time against the recorded first- and second-round timestamps. It reports delayed single-person review, not independent double annotation. Reviewer identity is recorded rather than externally verified.

The private first-round snapshot supports later reconciliation; it is not served by the blind-review page. Reviewers must not consult it while completing the blind pass.

## Compare and resolve disagreements

After all sampled items have a second-round record:

```powershell
python scripts/review_annotation_workflow.py compare --queue PATH_TO_BLIND_QUEUE_V1 --output PATH_TO_COMPARISON_V1
python scripts/serve_review_adjudication.py --data PATH_TO_COMPARISON_V1/adjudication --port 8772
```

The comparison treats changes in the complete aspect–sentiment set or annotation flags as disagreements. A second-round item saved as needing review also requires adjudication. Missing second-round items block comparison. The adjudication page shows the original comment and both human judgements, including status and all five annotation flags, but begins with no selected final labels. A human chooses the final labels and must explain the decision in the note field.

The comparison freezes an adjudication manifest with hashes of the exact displayed comments, taxonomy and first/second-round reference records. The adjudication page checks that version before startup and every save. Each saved decision is bound to the same manifest through a separate `adjudication_bindings.json` file. Edited, unbound or copied decisions from a different input version cannot enter a release. Keep this binding file alongside the adjudication store; use the adjudication page to revise a decision so its revision history and binding update together. Do not edit the copied comments or reference judgements in place.

If there are no disagreements, the comparison emits an empty `decisions_template.json`; no adjudication page is needed. If a reviewer identity or delay rule fails, correct the review procedure instead of relabelling its metadata as independent.

## Release a new label version

For a comparison with disagreements:

```powershell
python scripts/review_annotation_workflow.py release --data PATH_TO_NEW_REVIEWS --comparison PATH_TO_COMPARISON_V1 --decisions PATH_TO_COMPARISON_V1/adjudication/human_labels/annotations.json --output PATH_TO_LABEL_RELEASE_V1
```

With no disagreements, pass the empty `decisions_template.json` as `--decisions`. An incomplete decision, missing justification or missing decision binding blocks release. Release rechecks the adjudication manifest and all three displayed input files against the comparison. Changes to the frozen source data or the primary labels since queue creation also block release, requiring an appropriately updated audit. Comparisons created before this version binding was introduced must be prepared under a new directory; do not retrofit their manifest or overwrite their records.

Each release preserves source hashes, reviewer mode, agreement/adjudication counts, prior and resolved label revisions, and correction reasons. It exports separate label and input files for each original split. The original first-round store remains unchanged; the release is the new reviewed label authority for the chosen scope.

New releases also freeze per-split primary labels and revision histories, resolved labels, corrections and adjudication decisions. The release manifest hashes these files and the label/replay files against the original snapshot manifest. Consumers can validate calibration or adaptation without opening locked gold, and later edits in the live first-round store do not replace an earlier release. A missing or broken primary revision history blocks release. The new-source analysis entry point requires this evidence and full registered calibration/locked coverage; see [new-source analysis](rag_new_source_analysis.md).

## Prepare example feedback from adaptation only

```powershell
python scripts/review_annotation_workflow.py export-feedback --data PATH_TO_NEW_REVIEWS --release PATH_TO_LABEL_RELEASE_V1 --output PATH_TO_FEEDBACK_V1 --adaptation-use-policy PATH_TO_PRIVATE_POLICY --adaptation-use-policy-sha256 REGISTERED_POLICY_SHA256 --policy-source-root PATH_TO_SOURCE_REPOSITORY
```

Eligibility is checked against the frozen dataset, never an annotation-supplied split field. Calibration and locked-evaluation comments are excluded. The strict-unseen export additionally excludes every review containing the Account access theme, excludes that theme as a demonstration candidate, and removes exact/near duplicates of non-adaptation inputs. Confirmed reviews flagged as outside scope or incomplete are also excluded from feedback.

The required purpose policy is checked before reviewed adaptation gold is opened.
It excludes registered nonempty retrieval-normalised protected matches from this
use while preserving every annotation/review/release record. The expected hash
must come from the registered policy, not a hash recomputed from a changed file.
The policy and all source-bound files must remain available. See
[purpose-bound adaptation use](adaptation_use_policy_20261111.md) for the guarded
error-analysis/model-improvement entry point and migration requirements.

The output contains the five-field demonstration schema (`row_uid`, `candidate_aspect`, `review_text`, `aspect_candidate`, `answer`) plus a separate provenance record. Its real origin remains **new-source adaptation data**; it is not relabelled as official training data. This is a proposed feedback pool, not an automatic live-index replacement. Integrating it requires a separately versioned index, encoder identity, and regression evaluation, preserving the previous index for comparison.

All output directories are immutable: use a new version name for each preparation, comparison, release or feedback export. Do not commit the private queues, raw labels, review text or correction records.
