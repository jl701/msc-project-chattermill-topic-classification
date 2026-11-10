# Validate a reviewed new-source population before analysis

`new_source_locked` requires the original frozen annotation snapshot and a new
reviewed label release containing split-specific hashes and primary revision
history. A scope flag or an arbitrary labels JSONL file is insufficient.

```powershell
python scripts/analyse_rag_campaign.py `
  --scope new_source_locked `
  --artifacts PATH_TO_FROZEN_MODEL_ARTIFACTS `
  --annotation-data PATH_TO_ORIGINAL_ANNOTATION_SNAPSHOT `
  --reviewed-release PATH_TO_REVIEWED_RELEASE `
  --development-root PATH_TO_CALIBRATION_SCORES `
  --evaluation-root PATH_TO_LOCKED_SCORES `
  --development-labels PATH_TO_REVIEWED_RELEASE/labels_calibration.jsonl `
  --evaluation-labels PATH_TO_REVIEWED_RELEASE/labels_locked.jsonl `
  --output PATH_TO_NEW_ANALYSIS_VERSION
```

The original `annotation_manifest.json` and its hashed input snapshot determine
the registered IDs, splits and counts. The approved collection targets 200
adaptation, 200 calibration and 400 locked reviews. A recorded collection
shortfall is not silently presented as the target count. The reviewed release
must cover every registered calibration and locked ID. Pending or missing gold
blocks final analysis; it cannot reduce the denominator.

The analysis verifies the taxonomy against the model's candidate definitions,
label/replay file hashes, frozen primary revision chains, reviewed corrections,
and their comparison/adjudication bindings. It reads immutable release snapshots,
not the live annotation page's changing store. Old releases lacking these
integrity fields require a newly named release; do not retrofit old evidence.

Calibration validation opens only calibration gold/history. Classification
selection uses seen labels only. Routing is a separate business simulation that
may use complete calibration labels after the classifier is frozen. Both
selections are written before locked gold/history is opened. `--classification-only`
and `--selection-only` therefore never validate or inspect locked gold.

Every registered review remains in final quality calculations. Explicit failed
attempts and absent score records are separately counted and decoded as empty
outputs; a complete successful score file is not required. Extra or duplicate
score IDs are rejected. Removing both a gold row and its score row cannot make
the missing review disappear, because gold coverage is checked against the
original frozen snapshot. Historical replay keeps its existing failure-inclusive
population and does not require a new-source reviewed release.

Report all planned, calibration-frozen routing policies. If one policy is needed
for a later operational decision, select and freeze it using calibration data.
Do not choose an evaluation winner and describe its workload reduction or
Clopper–Pearson bound as a prospectively selected result. Manual corrections
remain offline gold substitutions, not measured human labour or combined-path
service latency.
