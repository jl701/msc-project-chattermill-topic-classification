# Dissertation claim–evidence register

Created: 31 August 2026

Status: working authority for the final dissertation rewrite. This register does not authorise new experiments or post-test tuning.

## Evidential hierarchy

1. **Confirmatory:** the locked Level 2 `D` official-test comparisons in `official_confirmatory_results.csv`.
2. **Official descriptive:** the remaining predeclared Level 2 `N/D`, stage, fold and Level 4 outputs from the same single reveal.
3. **Validation confirmatory or matched:** the formal-v2 description contrasts and matched one-stage/two-stage controls frozen before the official test.
4. **Validation exploratory:** the fixed composition and bounded extensions selected or tested after inspecting formal validation diagnostics.
5. **Post-hoc descriptive:** influence and error analyses of already frozen predictions. These may explain but cannot redefine the endpoint or candidate.
6. **Historical/contextual:** Level 1, Level 3 and superseded protocols. These cannot be used as confirmatory evidence for the revised protocol.

## Claims admitted to the main text

| ID | Claim | Evidence and value | Role | Required wording boundary |
| --- | --- | --- | --- | --- |
| C1 | The task is a supplied-candidate taxonomy shift rather than open-world aspect discovery. | `configs/experiments/taxonomy_final_test_v1.json`; supplied 12-candidate, 36-pair grid. | Protocol fact | State explicitly that candidate names are supplied and no aspect name is generated. |
| C2 | Level 2 uses strict example-filtering. | `l2_fold_support_and_filtering.csv`: 120–3,600 of 7,930 training reviews removed by fold. | Protocol fact | Holding out an aspect also removes co-occurring supervision for seen aspects; it is not a pure label-index intervention. |
| C3 | A true two-stage factorisation is beneficial in matched cheap-model controls. | Formal validation: TF–IDF two-stage capped-two minus one-stage `+0.0577`; E5 `+0.0362`. | Matched validation evidence | Limit the claim to the tested TF–IDF and E5 implementations under the common protocol; do not claim universal architectural superiority. |
| C4 | Minimal definitions have model-dependent effects. | Formal validation and official `official_l2_description_effects.csv`; official held-out `D-N`: QLoRA `+0.051616`, TF–IDF `+0.051156`, few-shot `-0.018653`, E5 `-0.013429`. | Validation + official descriptive | For trainable methods, `D-N` includes representation-format matching as well as semantic content; no universal benefit claim. |
| C5 | Richer descriptions did not improve the frozen bounded study. | `taxonomy_rich_description_validation_confirmation_v1_results.md`: R–D held-out F1 TF–IDF `-0.029352`, E5 `-0.010227`, with higher false-positive rates. | Validation negative result | Do not claim that rich descriptions are intrinsically harmful; only one frozen interface and two cheap representations were confirmed. |
| C6 | Stage diagnostics expose complementary operating-point strengths. | Official `official_l2_stage_diagnostics.csv`: few-shot thresholded presence F1 `0.559198`; QLoRA `0.516239`; QLoRA presence AP `0.570746` versus few-shot `0.481819`; oracle Stage 2 QLoRA `0.897727` versus few-shot `0.780809`. | Official diagnostic | Say few-shot has the stronger **thresholded gate at the frozen operating point**, while QLoRA has the stronger ranking AP and conditional sentiment decoder. |
| C7 | The fixed stage-wise composition is the best registered system on the primary official endpoint. | `official_confirmatory_results.csv`: composition `0.507538`; few-shot `0.495398`; QLoRA `0.486437`. | Confirmatory | Call it a validation-selected fixed composition, not a learned router, MoE or jointly optimised ensemble. |
| C8 | H1 passed. | Composition minus few-shot `+0.012140`, synchronized review-cluster 95% interval `[+0.001407,+0.022439]`. | Primary confirmatory | The interval is tied to the registered aspect-balanced endpoint and fixed 12-aspect taxonomy. |
| C9 | H2 passed after H1 opened the gate. | Composition minus QLoRA `+0.021101`, interval `[+0.000184,+0.041118]`. | Gate-kept confirmatory | Always state the hierarchical order; the small lower bound and aspect influence limit the strength of the conclusion. |
| C10 | The official ordering transfers the validation ordering, but gains narrow. | `official_validation_test_comparison.csv`: composition validation/test `0.525931/0.507538`; few-shot `0.504883/0.495398`; QLoRA `0.479903/0.486437`. | Official descriptive | Do not present validation and test as independent repeated samples of taxonomies. |
| C11 | Overall full-grid F1 and held-out primary F1 answer different questions. | Official composition overall `0.582950`; QLoRA overall `0.588388`, while composition leads held-out primary. | Official secondary | Do not replace the preregistered held-out endpoint with the overall result. |
| C12 | Performance is heterogeneous across aspects. | `official_l2_confirmatory_fold_results.csv`: H1 improves/ties/worsens `7/2/3`; H2 `7/0/5`. | Official descriptive | Do not claim universal per-aspect dominance. |
| C13 | H2 is sensitive to App/website. | Post-hoc `official_l2_confirmatory_leave_one_fold_influence.csv`: omitting `l2-a06` changes the 11-fold H2 mean from `+0.021101` to `-0.006895`; no H1 omission reverses sign. | Post-hoc influence | This does not invalidate the registered 12-fold result; label it post hoc and use it to bound generalisation. |
| C14 | The composition reduces conditional sentiment-set errors relative to few-shot but inherits the same gate. | Post-hoc `official_l2_confirmatory_error_decomposition.csv`: both have 831 missed-aspect and 1,411 spurious-aspect rows; wrong sentiment-set rows fall from 229 to 180. | Post-hoc diagnostic | Counts aggregate each review once per held-out-aspect fold and do not expose review text. |
| C15 | QLoRA's principal trade-off is a more conservative frozen gate and a stronger conditional decoder. | Post-hoc official counts: QLoRA misses 1,215 gold-positive rows versus 831 for few-shot/composition, but has 74 wrong sentiment-set rows versus 229/180. | Post-hoc diagnostic | Interpret jointly with AP and thresholded F1; do not call QLoRA's representation universally weaker. |
| C16 | Multi-sentiment cases are rare and remain difficult. | Frozen official rows contain 21 held-out gold multi-sentiment cases across 19,044 fold-review units; none of the three confirmatory systems predicts the exact two-sentiment set on those 21. | Post-hoc diagnostic | Report rarity and failure honestly; do not infer a reliable population rate from 21 cases. |
| C17 | Level 4 favours few-shot on held-out pairs across all three fixed groups. | Official group F1: few-shot `0.504405/0.597496/0.599018`; QLoRA `0.454140/0.590837/0.580729`; DistilBERT `0.081395/0.210762/0.253145`. | Official descriptive | The three groups are non-exchangeable stress cases; no population-level significance or monotonic level claim. |
| C18 | More adaptive combinations were not promoted. | Validation-only router, relative calibration, smooth fusion and retrieval records; smooth fusion `-0.012220` and retrieval `-0.008422` versus the fixed composition, intervals crossing zero. | Validation negative evidence | Use as a stopping rationale, not proof that adaptive routing can never work. |
| C19 | The official result followed a sealed, single-reveal execution. | 93/93 base jobs, 12/12 compositions, 105/105 bundles, 210/210 copied score/manifest files, zero integrity failures, exactly one reveal. | Governance fact | Treat as reproducibility strength, not the main scientific contribution. |
| C20 | Historical test familiarity weakens a pristine-blindness claim but does not erase the revised-protocol check. | Current authority and preregistration disclosure. | Limitation/governance | Use “protocol-locked held-out evaluation of the revised protocol”, never “first completely blind test”. |

## Contextual claims

| ID | Claim | Evidence | Boundary |
| --- | --- | --- | --- |
| X1 | The closed-taxonomy task is learnable with conventional classifiers. | Historical validation-selected DistilBERT official-split pair micro-F1 `0.7663`; TF–IDF/SVM `0.7042`. | Level 1 is a compact development reference. It must not be subtracted from Level 2 as a pure taxonomy-generalisation penalty. |
| X2 | DCWT did not synthesise a competitive unseen classifier. | Formal validation D held-out F1 `0.1453`; nearest-description control `0.1534`; official D `0.1425`. | A bounded negative method result, not a general impossibility theorem. |
| X3 | Level 3 is historical dual-unseen robustness. | Formal validation records for selected and alternate matchings. | Appendix only, post hoc, no inferential p-values and no official test. |

## Claims explicitly prohibited

- Minimal or rich descriptions universally improve unseen-label learning.
- QLoRA is causally better or worse than prompting as an adaptation mechanism.
- The fixed composition is a learned router, mixture-of-experts architecture or new neural model.
- Level 1, Level 2 and Level 4 form a directly comparable monotonic difficulty ladder.
- The study discovers arbitrary unknown aspects or validates open-world extraction.
- Results on twelve fixed aspects or three parent groups generalise statistically to all taxonomies.
- Single-seed neural results include training-realisation uncertainty.
- The official evaluation was the project's first untouched test access.
- A descriptive winner or post-hoc subgroup may replace the frozen primary candidate.

## Remaining thesis checks

- Every number used in LaTeX must point to one row or derivation above.
- Every literature claim must be paired with a separately verified primary reference.
- Post-hoc diagnostics must be visibly labelled as such in tables, captions and prose.
- The final PDF must contain the historical-test disclosure, example-filtering limitation, fixed-taxonomy limitation, single-seed limitation and non-exchangeable Level 4 limitation.
