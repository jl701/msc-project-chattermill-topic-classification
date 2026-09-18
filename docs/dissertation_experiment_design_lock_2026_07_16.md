# Dissertation Experiment Design Lock — 16 July 2026

> **Superseded for planning on 23 July 2026.** This document is retained only
> as historical design provenance. The approved taxonomy-generalisation
> difficulty ladder, description governance, stress tests, and live checklist
> are defined in `docs/dissertation_loao_mainline_lock_2026_07_23.md`. Do not
> use this file to decide or authorise future experiments.

## Status and Authority

This document froze the supervisor-aligned dissertation experiment design after
the 16 July 2026 discussion with Aji. It was superseded by the user-approved
23 July taxonomy-generalisation mainline.

The lock changes the organisation of evidence, not the provenance of completed experiments. Existing results remain available in their recorded scopes.

No experiment was run to create this design lock.

## Core Empirical Question

The dissertation is centred on one deployment question:

> When an organisation supplies a new candidate aspect with no task-specific training examples, can a model scan all reviews, reject reviews in which the candidate is absent, and assign sentiment when it is present? Can selective Qwen use improve the resulting quality–call-rate trade-off?

The system is given the canonical candidate text. It does not discover or name an unrestricted topic.

## Evidence Architecture

The dissertation contains one primary benchmark and two supporting blocks.

| Evidence block | Role | Required comparison | Headline status |
|---|---|---|---|
| Primary: twelve-fold all-row LOAO | Evaluate supplied unseen-candidate presence, sentiment, and end-to-end pair prediction across every FABSA aspect | Common folds, candidate text, all-row scope, projected gold, information regime, and metrics | Sole primary benchmark |
| Support A: fixed-taxonomy representation | Establish what increasingly rich representations add when every label is observed during training | BoW, TF–IDF, LSA, frozen sentence embeddings, and DistilBERT on the official split | Supporting supervised reference |
| Support B: strict zero-label calibration | Test whether thresholds and routing policies transfer without any target-aspect validation labels | Target-calibrated cold start versus leave-one-aspect-out cross-aspect selection | Supporting robustness check |

Calibration and selective deployment share the third evidence block but answer different research questions. A strict transfer check measures information dependence; a router measures the predictive-quality–Qwen-call-rate frontier.

## Primary All-Row LOAO Contract

For each of the 12 FABSA aspects in turn:

1. hold out the target aspect from task-specific supervised training evidence;
2. use example_filtered as the primary construction for trainable methods;
3. fit every vocabulary, IDF weighting, SVD transform, classifier, encoder, and other trainable component on the permitted training partition only;
4. supply the same singleton canonical candidate text to every candidate-conditioned method;
5. retain every official validation and test row;
6. project gold labels to the current held-out aspect;
7. permit an empty predicted pair set when the candidate is absent;
8. use the same metric implementation and unweighted aggregation across the 12 folds.

The label_masked construction is an incomplete-label-noise sensitivity condition. It must not be averaged with, or placed as a co-equal row beside, the primary example_filtered result without an explicit sensitivity label.

The required prediction state is absent, positive, negative, or neutral only if the annotation audit confirms that one review cannot contain conflicting sentiments for the same aspect. Until that invariant is confirmed, the pair-set representation is authoritative and the four-state form is a protocol assumption.

Predicting absent is a task decision. It is not selective abstention. Abstention or deferral refers only to passing a decision to Qwen or a human rather than issuing the local prediction.

## Method Ladder

All rungs address the same primary candidate question. A rung may remain pending, but it must not silently change the row scope, candidate text, sentiment component, calibration information, or metric.

| Level | Method | Theoretical purpose | Current evidence status |
|---:|---|---|---|
| 0 | Always-absent and exact-overlap sanity controls | Expose target prevalence, class imbalance, and the task floor | Required consolidation; not a headline model |
| 1 | BoW or unweighted lexical overlap | Test whether exact observed terms are sufficient | Protocol-matched candidate-conditioned count BoW complete |
| 2 | Word and character TF–IDF | Test weighted lexical and spelling evidence | Primary all-row LOAO evidence complete; strict train-only reference rerun complete |
| 3 | LSA/SVD | Test corpus-derived low-rank co-occurrence semantics | Closed-topic sweep exists; candidate-conditioned LOAO evidence is not yet frozen |
| 4 | Frozen sentence embeddings | Test pretrained semantic similarity and paraphrase transfer | Primary protocol-matched MiniLM and E5 LOAO evidence complete |
| 5 | DistilBERT review–candidate cross-encoder | Test joint contextual interaction between review and candidate | Primary all-row LOAO evidence complete |
| 6 | Zero-shot structured Qwen | Test instruction following, label semantics, and generative sentiment assignment | Primary all-row LOAO evidence complete |
| 7 | Local-to-Qwen router | Test whether complementary local and generative errors improve quality under a call budget | TF–IDF-to-Qwen and DistilBERT-to-Qwen evidence complete |

LSA is mandatory in the fixed-taxonomy support ladder. Candidate-conditioned LSA must remain explicitly pending unless a protocol-matched row is produced. It must not be implied by the existing closed-topic sweep.

Only two frozen sentence encoders are needed: one compact general-purpose model and one stronger retrieval-oriented model. Their exact identifiers, pooling, normalisation, truncation, candidate text, threshold grid, and model-selection rule must be registered before execution. Test data must not select the encoder.

Completion note, 17 July 2026: the protocol-matched Count BoW, strict train-only character TF-IDF, MiniLM, and E5 experiment is recorded in `docs/experiments/loao_bow_sentence_embedding_v1.md`. E5 has the highest mean test pair micro F1 (`0.3791`), but its paired-aspect advantage over strict TF-IDF and MiniLM is not statistically resolved; strict TF-IDF is the only method with a Holm-adjusted stable improvement over Count BoW in the post-run six-pair analysis.

The core Level 6 method is zero-shot Qwen under the frozen prompt and parser. Gemini fixed-split evidence and Qwen QLoRA evidence are secondary development or deployment cases, not additional primary ladder rungs.

## Measurement Contract

### Candidate Presence

Collapse positive, negative, and neutral into present. Report:

- positive-class precision, recall, and F1;
- target prevalence;
- false-positive rows per 100 reviews;
- false-negative rows per 100 reviews;
- PR-AUC where a protocol-matched continuous presence score exists.

PR-AUC is not available from a single discrete Qwen JSON decision. It must be reported as not applicable unless a pre-specified continuous score is genuinely available; a self-reported confidence or one operating point must not be converted into a fabricated curve.

### Conditional Sentiment

Keep two denominators distinct:

1. oracle-presence sentiment evaluates a modular sentiment component on all gold-present rows, where such a component can be run independently;
2. sentiment given successful detection evaluates sentiment only where the gold candidate is present and the system predicted it as present.

Report three-class sentiment accuracy and macro F1 where available. The second view must be accompanied by its denominator, detection coverage, and candidate recall. It must not be compared as if it shared the first denominator.

The positive-gold Qwen view is a diagnostic for present-candidate understanding and sentiment. It is not evidence for candidate rejection or the all-row headline.

### End-to-End Pair Prediction

The primary score is the unweighted mean of the twelve fold-level pair micro-F1 values. Also report:

- pair micro precision and recall;
- pair samples precision, recall, and F1;
- exact match;
- per-aspect scores and support;
- mean, median, standard deviation, minimum, and maximum across aspects.

Accuracy is not a headline metric because always-absent predictions can exploit target-absence prevalence.

## Calibration Information Regimes

Target-calibrated cold start permits the held-out aspect’s validation labels to select a local threshold or routing policy.

Strict zero-label transfer permits no target-aspect training or validation label to influence:

- feature fitting or model training;
- candidate-presence thresholds;
- checkpoint or encoder selection;
- prompt or parser selection;
- router boundaries or policy choice;
- hyperparameters or tie-breaking.

For target aspect a, selection must use only the other 11 validation aspects and then apply the frozen rule to target-aspect test predictions.

The existing leave-one-aspect router-policy diagnostic with mean pair micro F1 0.4525 is not yet a full strict zero-label result. It transfers router-policy selection, but the underlying local presence threshold was selected on the target aspect’s validation labels. It remains a useful partial transfer diagnostic.

## Frozen Existing Evidence

Under the comparable all-row LOAO score-export and routing evidence:

| System | Mean pair micro F1 | Qwen call rate | Evidence role |
|---|---:|---:|---|
| TF–IDF | 0.3780 | 0.000 | Primary lexical component |
| DistilBERT score-export rerun | 0.3158 | 0.000 | Primary contextual component; earlier 0.3128 run retained as reproducibility sensitivity |
| Qwen zero-shot | 0.3378 | 1.000 | Primary generative component |
| DistilBERT-to-Qwen router | 0.3900 | 0.291 | Lower-call-rate selective comparison |
| TF–IDF-to-Qwen router | 0.4549 | 0.360 | Primary pre-registered router result |
| Refined TF–IDF-to-Qwen boundary | 0.4560 | 0.362 | Sequential sensitivity only |

The 0.4549 router remains the primary result. No further test-informed boundary tuning is permitted. The 0.4560 value does not replace it.

The existing evidence is consistent with complementary errors, but any final claim must cite the paired aspect-level comparison and changed false-positive/false-negative behaviour. Aggregate improvement alone is not proof of a mechanism. Superiority over DistilBERT-to-Qwen remains statistically unresolved at the aspect level.

## Placement of Other Experiments

| Existing evidence | Dissertation placement |
|---|---|
| Held-out organisation | Short source-shift context or appendix; not a generalisation leaderboard |
| Fixed held-out aspect | Development-stage capability evidence showing why one fixed split is insufficient |
| Positive-gold LOAO | Conditional-sentiment and present-candidate diagnostic within the primary LOAO analysis |
| Gemini fixed and cascades | Deployment, cost, and router-origin case study; not a primary LOAO comparison |
| Qwen fixed QLoRA | Fixed-split adaptation evidence |
| Single-fold QLoRA validation gate | Negative stopping-rule evidence for the tested recipe |
| label_masked | Incomplete-label-noise sensitivity |
| Anomaly detection | Related-work and task-boundary discussion only |

Full Qwen LoRA LOAO is no longer a default future experiment. The failed validation gate justifies stopping the tested recipe; it does not establish that the entire LoRA or QLoRA family is ineffective.

## Experiment Admission Rule

A new experiment is admitted only if it answers at least one of the following:

1. Does it improve or explain recognition of a supplied unseen candidate under the primary all-row LOAO contract?
2. Does it test whether performance transfers without target-aspect validation labels?
3. Does it improve or explain the predictive-quality–Qwen-call-rate trade-off?

It must also close a named evidence gap, use a pre-registered selection rule, and preserve test isolation.

The next permitted order is:

1. complete the two sentence-embedding protocols, including only the cheap sanity/BoW/LSA consolidations needed to close their protocol-matched comparison tables;
2. audit and execute strict zero-label threshold and routing transfer using existing score exports where possible;
3. freeze unified presence, conditional-sentiment, end-to-end, and paired statistical tables;
4. attempt one sentence-embedding-to-Qwen router only if validation evidence shows clear, pre-defined complementarity.

No further router test-boundary search, full Gemini LOAO, full Qwen LoRA LOAO, anomaly-detection branch, or unrelated model sweep is authorised by the current dissertation design.
