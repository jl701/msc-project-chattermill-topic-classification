# Seen-only development error analysis

`scripts/analyse_rag_seen_badcases.py` is a CPU-only diagnostic helper. It does
not load a model, modify the retrieval implementation, choose an improvement or
read the original/full development labels or any evaluation labels.

Inputs are the explicitly sanitized `development_seen_labels.jsonl`, the
label-free development replay, scoring artifacts and completed method files.
Held-out candidate scores, predictions and per-candidate diagnostics are removed
before calibration and analysis. All development reviews remain in the
population, including those with no remaining seen labels.

The helper reproduces the registered quantile threshold budget and tie rules
using only seen scores and labels. Synthetic tests compare its threshold with
the registered selector and poison held-out values to verify their exclusion.
Per-theme reports distinguish presence FP/FN from sentiment-set errors after a
theme is detected. Demonstration alignment, repeated review identities and
truncation are descriptive diagnostics, not causal estimates.

Each score file must match its completed-file digest. Replay/artifact digests
and all review IDs must match the scoring registration. An explicit
`--method fixed` can produce a partial fixed profile; full-matrix mode requires
the state to be complete and every registered method to be present. No partial
profile selects an improvement. Use a fresh output directory for a new snapshot.

```powershell
python scripts/analyse_rag_seen_badcases.py `
  --artifacts outputs/serving_20261109/artifacts.json `
  --replay outputs/serving_20261109/replay_development.jsonl `
  --development-root outputs/rag_20261111/development `
  --training-conflicts outputs/rag_20261111/training_duplicate_audit/private_conflict_groups.json `
  --output outputs/rag_20261111/badcase_matrix_profile
```

Outputs include an aggregate `summary.json`, `observations.md`, and separate
private JSON/Markdown files containing the worst eight seen-error reviews for
each method. Raw review text and labels occur only in those private case files.
The CLI requires output beneath ignored `outputs/`; never publish case files.
The comparison supports choosing at most two preregistered follow-up rounds,
but only after the full development matrix and seen-only selection are ready.

The optional training-conflict audit reports exposure to the retained duplicate
groups. It distinguishes a selected example whose own aspect label is ambiguous
across duplicate rows from an example that belongs to a conflicting group but
whose displayed aspect label is consistent. Counts and errors are stratified by
query aspect; review-level cohorts and private retained-ID exposures are also
saved. These are descriptive associations, not evidence that quarantine would
cause a quality improvement. No labels or index entries are changed by this tool.
