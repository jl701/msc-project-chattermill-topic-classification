# Strict similarity LOAO with aspect-conditioned sentiment

## Status

Completed on 23 July 2026. The design was written before the new validation and test rescoring. This is nevertheless a post-hoc component analysis because the source test artifacts and their original results already existed.

After the subsequent meeting with Aji, the validation-reselected aspect-conditioned systems below became the active representation baselines. Global sentiment, the shallow sentiment classifier, and fixed-threshold component-audit values are not thesis-facing alternatives.

## Question

The source similarity baselines assigned one document-level polarity after detecting the held-out aspect. That design cannot represent different polarities for different aspects in the same review. This experiment replaces it with sentiment predicted from `(review, held-out aspect)`.

## Locked design

The source presence predictions are the completed 12-fold strict similarity artifacts. The replacement sentiment predictions come from the completed 12-fold DistilBERT score export. Only `sentiment_features[heldout_aspect].predicted_sentiment` is used from that run; its candidate-selector prediction is ignored.

Two analyses are locked:

1. **Fixed-threshold component swap.** Preserve every original present/absent decision and replace only the sentiment label on predicted-present rows. This is the clean comparison for the effect of the sentiment component.
2. **Validation-reselected system.** Re-run the registered threshold sweep on validation with aspect-conditioned sentiment, freeze one threshold per method and aspect, and then score test once. This estimates the best complete baseline after the component change without test-guided calibration.

Exact row and gold-label alignment is a hard gate. Full details, stopping rules, metrics, limitations, and artifact paths are recorded in `configs/experiments/loao_similarity_aspect_conditioned_sentiment_rescore_v1.json`.

## Active results

All 48 validation artifact pairs aligned exactly by split, row order, `row_uid`, held-out aspect, and gold pair labels. This covered 50,736 method-row comparisons. The validation stage then froze all 48 registered method-aspect thresholds before test was loaded. The same audit passed for all 48 test artifact pairs and 76,176 method-row comparisons.

Each complete system reselects its presence threshold on validation after fixing the shared aspect-conditioned sentiment component:

| Presence method | Test mean pair micro-F1 | Thesis role |
|---|---:|---|
| Count BoW | 0.3144 | Sanity control |
| Strict TF-IDF | 0.3856 | Main classical baseline |
| MiniLM | 0.3889 | Repository/appendix |
| E5 | 0.4013 | Main frozen sentence-embedding baseline |

The shared oracle-presence aspect-conditioned sentiment accuracy is `0.9103`, averaged over the twelve folds.

## Interpretation

Aspect conditioning is fixed by the task definition rather than selected according to whether a shallow comparator happens to score higher. The representation comparison now isolates candidate-presence modelling while every method uses the same candidate-specific sentiment component.

The fixed-threshold swap remains an internal correctness audit showing that only the sentiment label changed while presence decisions stayed identical. It is not an additional baseline row.

This experiment remains target-calibrated LOAO, uses one existing DistilBERT score export rather than multiple random seeds, and evaluates one held-out candidate per fold rather than making a joint 12-aspect inference call.

## Verification

- 139 repository tests passed.
- The source baseline metrics were exactly reproduced before replacement.
- For the fixed-threshold comparison, all per-fold presence counts and presence metrics are exactly unchanged.
- Test scoring loaded the complete validation selection manifest containing all 48 frozen thresholds.

The tracked active-system numeric results are stored under `docs/thesis_figure_data/`. Review-level prediction artifacts remain ignored.
