# Documentation and evidence index

Start with the [project README](../README.md). This index separates explanations
for a new reader from the dated records of how the research developed.

## Read the project

| Need | Document |
|---|---|
| Understand the methods and training examples | [Methods in practical terms](methods.md) |
| Interpret the main results | [Results and their evidential roles](results.md) |
| Find the implementation and its tests | [Architecture guide](architecture.md) |
| Install, test or reproduce an analysis | [Reproduction guide](reproducibility.md) |
| Change the code responsibly | [Contributor guide](../CONTRIBUTING.md) |
| Understand what is intentionally not public | [Public release boundary](github_upload_scope.md) |
| Understand earlier directions | [History and provenance](history.md) |

## Current scientific evidence

1. **Original validation and locked official evaluation:** the
   [dated scientific authority](taxonomy_current_authority_20260824.md) records
   the registered comparison and historical execution revisions. Its original
   protocol removes training reviews containing the held-out aspect.
2. **Completed training-policy extension:** the
   [completed results index](results/training_policy_completed/README.md) adds
   retained-review QLoRA and complete, same-policy compositions. This is later
   validation-only evidence. It does not replace the official result.
3. **Writing direction:** the
   [10 September narrative decision](thesis_narrative_decision_20260910.md)
   prioritises stage diagnosis, complementary composition and training-policy
   sensitivity. Working thesis sources are not a final submission release.

The date in an older filename does not make it the latest account of every
subsequent study. In particular, the early training-policy extension's
mixed-component diagnostic is not the completed retained-review composition.

## Repository engineering

[Quality plan](engineering/quality_plan.md) records the boundaries of the
September repository cleanup. Engineering changes do not retroactively change
the code revision that produced a historical scientific result.

Most dated files under `experiments/` are provenance records, not a sequence of
commands that a new reader should execute. Use the reproduction guide first.
