# Final-test TF-IDF inference-contract recovery (2026-08-25)

## Status before repair

- Release commit: `eb460c31e97e22e0352e746d7b214ea5fc958455`.
- Worker B failed on its first registered TF-IDF score job before any score
  bundle or performance result was produced.
- The failure was operational and label-free: the final-test sentiment grid
  omitted the scorer contract's required `negative_type` metadata column.
- Workers A and D were stopped after preserving 8 and 7 completed immutable
  score bundles respectively. Their in-progress jobs were terminated and are
  excluded. Outcomes have not been revealed.
- No official-test label, fold metric, aggregate metric, threshold, prediction,
  or score value was inspected to diagnose or design this repair.

## Pre-registered repair

The final-test unlabelled sentiment grid will add the constant metadata value
`negative_type = "eval_candidate"` to every candidate row. This is the same
evaluation-only sentinel already used by `build_unified_pair_eval_grid`.
It is not a target, is constant across rows, and is not used by the registered
six TF-IDF numerical features.

The repair must not change:

- review rows, folds, candidate ordering, candidate text, representations,
  methods, checkpoints/adapters, prompts, thresholds, decoder, hypotheses, or
  output metrics;
- the label-free worker payload boundary;
- any already completed A/D score value.

## Validation and reuse rule

Before resuming any worker:

1. add a regression test proving the unlabelled sentiment grid satisfies the
   shared scorer interface while remaining target-free;
2. run the focused final-test/scorer tests and the full test suite;
3. run a label-free equivalence audit showing that the new column is constant
   and does not enter TF-IDF feature transformation;
4. create a new versioned execution commit and release manifest;
5. do not promote the 15 completed A/D bundles into the replacement release.
   Although the added metadata is unused by their scorers, their manifests are
   bound to the superseded execution commit. Rerun all three workers under one
   replacement release rather than creating a mixed-commit score graph.

No performance-guided adjustment or test-result inspection is permitted.

## Validation result

- Focused final-test and scorer tests: `17 passed`.
- Full repository suite: `506 passed`.
- A dedicated equivalence test confirms that changing only `negative_type`
  leaves all six TF-IDF features and the resulting probabilities bitwise
  unchanged.

