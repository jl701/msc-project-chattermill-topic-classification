# Descriptive diagnostics alongside the frozen main comparison

Registered while the957 inference matrix was still computing its common
sentiment anchor, before opening this run's evaluation gold. No classifier,
threshold, routing or runtime source is changed.

`profile_rag_scores.py` supplements the primary paired-bootstrap analysis with
predefined review character-length bands:0–100,101–300,301–1000,over1000. For each
method it reports raw presence/pair counts and F1 when historical gold is
available, executed prompt-token distributions, demonstration/query truncation
counts, retrieval fallback, per-aspect counts and offline presence-stage elapsed
time. Empty bands remain visible. Explicit failed predictions remain in the
registered population; missing completed-run records are rejected. Subgroup differences are descriptive, not additional
selection criteria or significance claims. The primary whole-review confidence
intervals remain in `analyse_rag_campaign.py`.

New-source use omits gold and reports only input/prompt/operational and predicted
topic distributions under the already frozen historical thresholds. These
distributions are not quality metrics or evidence of concept drift. This helper
rejects a new-source label argument; reviewed new-source quality must follow the
existing reviewed-release workflow.

Inference must be complete and its artifact/replay/config identity, exact method
set, per-method completion hashes and population coverage must match. Every
successful full score grid and frozen threshold is validated before any gold
file is read. The original historical replay manifest binds the historical
evaluation input and label hashes. A malformed or interrupted matrix cannot be
silently profiled as a completed run. Explicit failed records remain in quality
denominators; corrupted or missing completion evidence is rejected.

Stage1 demonstration/query trimming and Stage2 full-grid/selected-topic left
trimming have separate denominators. Elapsed-time summaries contain successful
offline presence calls only, with failures separately counted. Output contains
aggregate statistics and source hashes, without review text, individual IDs,
labels or predictions.

After the historical matrix is complete, run with the approved local Python:

```powershell
& $env:RAG_PYTHON scripts/profile_rag_scores.py `
  --artifacts outputs/serving_20261109/artifacts.json `
  --replay outputs/serving_20261109/replay_evaluation.jsonl `
  --scores outputs/rag_20261111/evaluation_final `
  --selection outputs/rag_20261111/final_selection/selection.json `
  --scope historical_replay `
  --historical-labels outputs/serving_20261109/labels_evaluation.jsonl `
  --historical-manifest outputs/serving_20261109/replay_manifest.json `
  --output outputs/rag_20261111/historical_descriptive_profile.json
```

For the completed800-input new-source matrix, use its input replay and score
directory with `--scope new_source_labelblind`, omit both historical arguments,
and write a separate `new_source_labelblind_profile.json`. This uses the frozen
historical thresholds and does not replace later human-labelled calibration.
