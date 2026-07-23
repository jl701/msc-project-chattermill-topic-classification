# Thesis Result Tables — Active Interim Registry

Last updated: 23 July 2026

## Authority

This file contains only the currently active target-calibrated all-row LOAO rows. Method selection, exclusions, protocol language, and remaining work are governed by `docs/dissertation_loao_mainline_lock_2026_07_23.md`.

The previous mixed-protocol table pack has been retired. Closed-topic, held-out-organisation, fixed held-out-aspect, positive-gold, hosted Gemini, component-audit, and historical router results must not be added to the table below.

## Aspect-conditioned sentiment policy

Every active method predicts sentiment conditional on the supplied candidate aspect:

- Count, strict TF-IDF, MiniLM, and E5 share the frozen DistilBERT aspect-conditioned sentiment component and use validation-reselected presence thresholds.
- The DistilBERT cross-encoder uses its existing DistilBERT aspect-conditioned sentiment component.
- Frozen Qwen and QLoRA predict candidate-conditioned aspect–sentiment pairs directly.

Global document sentiment and the shallow TF-IDF sentiment classifier are not active alternatives.

## Active target-calibrated all-row LOAO table

The primary metric is the unweighted mean of twelve fold-level test pair micro-F1 values.

| Method | Sentiment formulation | Mean pair micro-F1 | Placement |
|---|---|---:|---|
| Count BoW | Shared DistilBERT aspect-conditioned sentiment | 0.3144 | Sanity control |
| Strict train-only character TF-IDF | Shared DistilBERT aspect-conditioned sentiment | 0.3856 | Main classical baseline |
| MiniLM-L6-v2 | Shared DistilBERT aspect-conditioned sentiment | 0.3889 | Appendix/repository |
| E5-base-v2 | Shared DistilBERT aspect-conditioned sentiment | 0.4013 | Main frozen sentence-embedding baseline |
| DistilBERT review–candidate cross-encoder | DistilBERT aspect-conditioned sentiment | 0.3158 | Fine-tuned contextual baseline |
| Frozen candidate-pair Qwen | Joint candidate-conditioned pair prediction | 0.337804 | Matched QLoRA control |
| Candidate-pair QLoRA | Joint candidate-conditioned pair prediction | 0.483158 | Main adaptation contribution |

## Current evidence boundaries

- The first four rows form the strict representation comparison because their sentiment component and outer protocol are shared.
- Frozen Qwen versus QLoRA is the matched adaptation comparison.
- Cross-family point estimates share the outer benchmark but do not imply identical internal training or calibration.
- The strict zero-label and multi-seed QLoRA rows remain pending and must be reported separately when complete.
- A strict TF-IDF-to-Qwen router row is permitted only after it is rebuilt from the aspect-conditioned TF-IDF endpoint under validation-only policy selection.

## Retired result guard

Do not add or regenerate:

- global-sentiment baseline rows;
- shallow aspect-conditioned sentiment rows;
- legacy TF-IDF `0.3780`;
- legacy router `0.4549`;
- refined legacy router `0.4560`;
- derivative legacy-router figures or tables;
- fixed-threshold sentiment component-audit values as complete systems.

The final publication-quality table remains a checklist item because presence, conditional-sentiment, uncertainty, seeds, and strict zero-label fields still need to be frozen.
