# Repository quality work — 16 September 2026

## Scope and baseline

Improve the repository's research explanation, installation, CPU examples,
core analysis interfaces, tests and contributor workflow. No new model selection,
training run or official evaluation is part of this work.

The starting commit is `27cf9cd`. Existing local changes include the completed
training-policy extension and thesis revisions. They belong to the ongoing
project and must be retained. A local source snapshot, including those changes,
was captured before editing. Historical execution commits remain the provenance
of published results, even when maintained code is reorganised.

## Work checklist

- [x] Establish and record the existing test baseline.
- [x] Replace the obsolete README with the current question, evidence and entry points.
- [x] Make the Python package installable and add a CPU-only synthetic walkthrough.
- [x] Provide a result-table command using checked-in aggregate evidence only.
- [x] Extract shared statistical analysis from executable scripts.
- [x] Audit source and scripts, document current and historical entry points.
- [x] Add regression tests for the extracted behaviour and package entry points.
- [x] Configure reproducible development checks and CPU CI.
- [x] Verify installation and documented commands in a clean environment.
- [x] Deliver a change review with reasons, tests and remaining limitations.

## Acceptance

A new reader can understand the task without knowing project abbreviations,
run a demonstration without credentials or model downloads, locate each important
scientific operation, and distinguish the locked official result from later
validation evidence. Core refactors preserve established outputs on fixed
fixtures. CI must test executable behaviour rather than count files or badges.

## Publication boundary

Prepare a reviewable branch locally. The thesis manuscript, source data,
checkpoints, row-level predictions and cloud credentials are outside this
engineering change set. Do not rewrite historical commits or present newly
reorganised code as the original execution revision. Public data and model
licensing must be described from their actual terms; no repository-wide licence
is chosen on the author's behalf.
