# From reviewed adaptation examples to an immutable retrieval index

This integration is implemented and covered by synthetic tests. No real human
annotation release has been promoted into an index yet, and no feedback quality
gain is claimed. Actual use still requires the human-reviewed adaptation release.

The original `prepare_rag_index.py` deliberately accepts only official training
records. Reviewed new-source adaptation examples use a separate command:

```text
python scripts/prepare_rag_feedback_index.py append
  --parent <existing-index>
  --data <frozen-annotation-dataset>
  --release <reviewed-human-release>
  --feedback <export-feedback-directory>
  --artifacts <frozen-model-artifacts.json>
  --parent-exclusion-replay <historical-development-inputs.jsonl>
  --parent-exclusion-replay <historical-evaluation-inputs.jsonl>
  --parent-exclusion-replay <all-new-inputs.jsonl>
  --output <new-unencoded-index-version>
```

Each `--parent-exclusion-replay` must match one of the parent's registered input
hashes, and all parent exclusion sources must be supplied. Additional protected
inputs can be supplied with repeatable `--extra-protected-replay`. The append
operation re-exports feedback from its reviewed human source through the current
annotation validation gates and compares the manifest, demonstrations and
provenance. It verifies the frozen split and taxonomy, complete seen-aspect grids,
held-out-review exclusion and eligibility of the original fixed examples.

The original all-800-input exclusion certificate is a statement about the
official-train pool. Once adaptation examples are added, that complete input file
can no longer be certified as disjoint from the new index. The new manifest
retains the parent audit as base-only provenance, removes only dataset-bound
adaptation IDs/texts from that certificate, and rechecks the remaining historical,
calibration and locked inputs. Exact normalised matches and registered near
matches are excluded. The new index includes separate input-only calibration and
locked replay files and their disjointness hashes, as well as the combined
protected replay. Adaptation evaluation is explicitly not certified disjoint.

`origin_by_review.json` distinguishes `official_train` from
`human_reviewed_adaptation` and binds annotation revisions and release hashes.
An existing review with changed labels requires an explicit replacement workflow;
it is not silently overwritten. The parent directory is unchanged. The appended
index contains no stale embedding file and is marked as requiring re-encoding.
BM25/fixed/random retrieval can read this validated pool immediately.

E5/RRF use a second new version:

```text
python scripts/prepare_rag_feedback_index.py encode
  --source <new-unencoded-index-version>
  --output <new-encoded-index-version>
  --encoder-path <pinned-E5-snapshot>
  --encoder-device cpu
```

The encoder defaults to CPU. A coordinated offline GPU job can explicitly select
`cuda` after other GPU jobs release the device. The pinned E5 revision is checked,
all combined-pool vectors are rebuilt, row count and finite values are validated,
and a failed build retains a failure receipt instead of a completed manifest.
This command never overwrites its source or an existing output version.
