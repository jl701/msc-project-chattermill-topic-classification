# Level 2 description-to-classifier weight transfer

## Decision

This method is added to the revised Level 2 single-unseen LOAO work as a
validation-only, high-risk classical zero-shot transfer experiment. It is not a
new taxonomy level and it does not replace the main TF-IDF, E5, DistilBERT,
Frozen-Qwen or QLoRA comparisons.

The experiment asks whether an unseen aspect-presence classifier can be
constructed without target-labelled examples. In each outer LOAO fold, eleven
binary logistic-regression aspect classifiers are learned for the eleven seen
aspects. A frozen embedding is also computed for each seen aspect's canonical
name and minimal definition. The relationship between the eleven description
embeddings and eleven classifier parameter vectors is then used to synthesise
the missing classifier parameters for the twelfth aspect from its label-side
text alone.

The registered machine-readable contract is
`configs/experiments/taxonomy_description_to_classifier_weight_transfer_v1.json`.

## Why it is worth trying

The proposal has a clear scientific role. The ordinary strict TF-IDF method
uses lexical interactions between a review and a candidate description. This
new method asks a different question: can label semantics predict what a
supervised classifier for that label would have learned? A positive result
would show that descriptions transfer not only through direct semantic
similarity but also through the geometry of classifiers learned for related
seen aspects. A negative result would still be useful because it identifies a
limit of description-based transfer when only eleven source tasks exist.

The general idea has precedent in zero-shot classifier synthesis. Earlier work
learned classifiers or classifier weights for unseen categories from textual
descriptions, including *Write a Classifier* and *Predicting Deep Zero-Shot
Convolutional Neural Networks using Textual Descriptions*. More recent
classifier-injection work also reconstructs unseen classifier weights from
semantic descriptors and existing weights. Those papers are mostly in vision;
the present study is a small, explicit transfer of the principle to
aspect-presence classification:

- Elhoseiny, Saleh and Elgammal, *Write a Classifier: Zero-Shot Learning Using
  Purely Textual Descriptions*, ICCV 2013:
  <https://openaccess.thecvf.com/content_iccv_2013/papers/Elhoseiny_Write_a_Classifier_2013_ICCV_paper.pdf>
- Ba et al., *Predicting Deep Zero-Shot Convolutional Neural Networks using
  Textual Descriptions*, ICCV 2015:
  <https://openaccess.thecvf.com/content_iccv_2015/papers/Ba_Predicting_Deep_Zero-Shot_ICCV_2015_paper.pdf>
- Christensen et al., *Image-free Classifier Injection for Zero-Shot
  Classification*, 2023: <https://arxiv.org/abs/2308.10599>

## Why a naive implementation is not acceptable

Each outer fold supplies only eleven description-to-weight training pairs. A
direct unrestricted mapping from a 768-dimensional description embedding to
hundreds of thousands of raw word and character TF-IDF coefficients would be
severely underdetermined. A neural hypernetwork would make this worse and could
produce an impressive-looking validation result through selection noise.

Classifier weights are also meaningful only within a shared coordinate system.
Every seen classifier must therefore use the same outer-fold train-only review
representation, feature scaling and regularisation. Intercepts and coefficient
scales must be handled consistently. No target-aspect review, target classifier
or target-labelled validation outcome may enter representation fitting,
generator fitting, hyperparameter selection or threshold selection.

## Registered implementation

For each of the twelve outer LOAO folds:

1. Remove the held-out aspect under the existing `example_filtered` contract.
2. Fit word-plus-character TF-IDF only on permitted outer-fold training
   reviews, then reduce it to a registered low-dimensional SVD space and
   standardise the latent coordinates.
3. Train eleven independent balanced logistic regressions for aspect presence,
   all in this identical coordinate system.
4. Encode the eleven canonical names and frozen minimal definitions with the
   existing frozen E5-base-v2 label encoder.
5. Learn a centred linear-kernel ridge mapping from description embeddings to
   the logistic coefficient-plus-intercept vectors.
6. Predict the held-out classifier parameters from the held-out descriptor.
7. Score the held-out aspect and pass selected aspects to the unchanged Level 2
   aspect-conditioned sentiment component and capped-two decoder.

The generated classifier is deliberately limited to the affine span of the
seen classifiers. This is a regularised interpolation/extrapolation experiment,
not an unconstrained parameter generator.

## Strict selection without target labels

The method needs a calibration procedure that resembles its deployment state.
Using the performance of the eleven original supervised classifiers would be
optimistic because the twelfth classifier is synthesised rather than directly
trained.

Selection therefore uses leave-one-seen-aspect-out pseudo-unseen
meta-validation. For each of the eleven outer-seen aspects in turn, its weight
is excluded, a pseudo-unseen weight is synthesised from the remaining ten, and
that synthesised classifier is evaluated on validation labels for the
outer-seen aspect. The eleven pseudo-unseen predictions are pooled to select
the registered review projection, logistic regularisation, generator
regularisation and presence threshold. The true outer-held-out aspect remains
evaluation-only.

The same generator and threshold are then used when the target descriptor is:

- the canonical name only;
- the canonical name plus frozen minimal description; or
- the pre-registered rich guidance.

This permits paired description-minus-name-only analysis without changing the
target checkpoint or threshold.

## Mandatory baselines

With only eleven source aspects, a learned mapping must beat simpler transfer
rules before it is credited with learning description-to-weight structure:

- the mean of all seen classifier weights, which ignores descriptions;
- copying the weight of the most description-similar seen aspect; and
- a cosine-similarity-weighted average of seen classifier weights.

The primary kernel-ridge generator and all three baselines must be reported.
The primary result cannot be chosen after inspecting the held-out aspects.

## Sentiment boundary

This experiment generates only Stage 1 aspect-presence parameters. An aspect
description alone does not identify an aspect-specific positive, negative and
neutral decision boundary reliably, and attempting to synthesise all sentiment
heads would triple an already underdetermined meta-learning problem. Stage 2
therefore remains the exact registered shared aspect-conditioned sentiment
component with the capped-two decoder. This isolates the proposed intervention
and keeps the output compatible with the main two-stage architecture.

## Expected outcome and thesis placement

The prior expectation is modest. Eleven source classifiers are very few;
description similarity may not align with lexical review cues; and an unseen
aspect may require vocabulary directions absent from every seen classifier.
The method should not be described as expected to outperform E5, Qwen or QLoRA.

It nevertheless belongs in the thesis if the protocol remains strict:

- a positive result supports classifier synthesis from label semantics;
- performance no better than the mean/nearest-weight baselines shows that the
  learned mapping contributes little;
- performance below direct E5 similarity shows that predicting a classifier is
  less effective than comparing review and label in a shared semantic space;
  and
- collapse on semantically isolated aspects provides an interpretable failure
  mode.

If it is competitive and stable across aspects, it receives a short Level 2
subsection in the main results. If it clearly fails, the thesis should still
record the hypothesis, strict design, result and failure interpretation in a
brief negative-result paragraph or the appendix.

## Execution gate

No official test data may be opened. Implementation must first pass synthetic
weight-recovery tests, target-isolation tests, deterministic hash/resume tests
and one real validation-fold smoke. The full twelve-fold validation run is
cheap enough for local CPU/GPU execution and does not require a new RunPod
campaign.

## Validation result — 20 August 2026

The implementation and all twelve Level 2 outer folds completed under the
registered train-plus-validation contract. Synthetic recovery, target
isolation, deterministic selection, finite-score and resume tests passed. The
held-out aspect was never used to fit the review representation, source
classifiers, weight generator, hyperparameters or threshold. Every ledger
reported `failure_count=0`, `non_finite_value_count=0`,
`resume_conflict_count=0` and `test_contract_count=0`.

Unweighted means over the same twelve held-out aspects are:

| Weight generator | Presence F1 N | Presence F1 D | Presence F1 R | Held-out pair F1 N | Held-out pair F1 D | Held-out pair F1 R |
|---|---:|---:|---:|---:|---:|---:|
| Mean seen weight | 0.214963 | 0.214963 | 0.214963 | 0.138323 | 0.139573 | 0.138150 |
| Nearest description weight | **0.249665** | **0.245320** | **0.230273** | **0.157193** | **0.153412** | **0.140800** |
| Cosine-barycentric weight | 0.224260 | 0.225079 | 0.220755 | 0.141358 | 0.142614 | 0.136642 |
| Registered kernel ridge | 0.232849 | 0.236820 | 0.223756 | 0.144329 | 0.145314 | 0.132310 |

For the registered kernel-ridge method, `D-N` is small but positive for
held-out presence F1 (`+0.003971`, 9/12 folds improved), average precision
(`+0.007440`, 10/12) and held-out pair F1 (`+0.000985`, 10/12). Rich guidance
does not improve on the minimal definition: `R-D` is `-0.013064` for presence
F1 and `-0.013004` for held-out pair F1.

The learned mapping also does not beat the nearest-description control. This
is a scientifically useful negative result: with only eleven source
classifiers, most of the transferable signal appears to come from selecting a
semantically related seen classifier rather than learning a stable global
description-to-weight map. The result should therefore be reported briefly in
the thesis without presenting DCWT as a competitive main method.

Protocol SHA-256:
`5f8b89cb5eff6adefcee6f7d530329d0a2df6190db10bee3ae9c5ad019bf2618`.
The machine-readable fold results and aggregate remain under
`outputs/experimental/taxonomy_post_supervisor_local_v1/dcwt`.
