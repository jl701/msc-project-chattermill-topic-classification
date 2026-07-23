# Dissertation LOAO Mainline Lock — 23 July 2026

## Status and authority

This is the sole operational source of truth for the dissertation's LOAO experiments, active results, remaining work, and stopping decisions.

It incorporates:

- the user's 23 July review of the strict representation and QLoRA evidence;
- the completed strict similarity-baseline sentiment rescore; and
- Aji's subsequent instruction that all active baselines must use aspect-conditioned sentiment, regardless of whether a shallow baseline becomes numerically weaker.

It supersedes:

- `docs/dissertation_experiment_design_lock_2026_07_16.md`;
- the old completion order in `docs/thesis_completion_roadmap.md`;
- the global-sentiment and shallow-sentiment directions in older baseline notes;
- the legacy TF-IDF-to-Qwen router as an active dissertation result; and
- any thesis table or handoff prompt that still presents a global-sentiment baseline as current.

Older experiment logs remain provenance only. They do not define the current method, result table, or next task.

## Fixed modelling decision: aspect-conditioned sentiment everywhere

Every active method must predict sentiment conditional on the supplied candidate aspect.

For modular representation baselines:

```text
review + candidate aspect
        |
        +--> candidate-presence score
        |
        +--> DistilBERT aspect-conditioned sentiment
        |
        +--> absent, or candidate aspect | candidate-specific sentiment
```

The following rules are now fixed:

1. Count BoW, strict TF-IDF, MiniLM, and E5 use the same frozen DistilBERT aspect-conditioned sentiment component.
2. Each complete system selects its own presence threshold on validation after the sentiment component has been fixed.
3. The DistilBERT candidate cross-encoder retains its existing DistilBERT aspect-conditioned sentiment component.
4. Frozen candidate-pair Qwen and candidate-pair QLoRA are aspect-conditioned by construction because both presence and sentiment are predicted from `(review, candidate aspect)`.
5. No active baseline may assign one document-level global sentiment to every detected aspect.
6. The shallow TF-IDF aspect-conditioned sentiment classifier is retired. Its negative result is not a reason to use global sentiment and is not part of the thesis argument.

This decision makes the task formulation consistent with reviews that contain different sentiments for different aspects. The primary LOAO evaluation remains singleton-candidate per fold; applying the same method separately to different candidates permits different candidate-specific sentiments.

## Locked primary evaluation scope

The sole headline benchmark remains twelve-fold all-row LOAO:

- `example_filtered` permitted training construction;
- each of the twelve FABSA aspects held out from task-specific training in turn;
- all official validation and test rows retained;
- gold labels projected to the held-out aspect;
- raw canonical singleton candidate supplied;
- empty predictions permitted;
- pair-set output retained because a review–aspect instance may contain more than one gold sentiment;
- primary metric: unweighted mean of twelve fold-level test pair micro-F1 values;
- presence, conditional sentiment, false-positive/false-negative rows per 100, and per-aspect spread reported as diagnostics;
- target-calibrated and strict zero-label results labelled as different information regimes.

Every fitted vocabulary, IDF weight, classifier, encoder, adapter, threshold, checkpoint, and router policy must obey its declared training and calibration information regime.

## Locked thesis argument

The dissertation will make the following empirical argument:

> Under supplied unseen-aspect taxonomy shift, stronger representations do not automatically solve candidate-absence detection. Fair baselines must nevertheless predict sentiment conditional on the supplied aspect. Under that task-correct formulation, E5 is the strongest retained non-fine-tuned representation baseline, while absence-aware candidate-pair QLoRA provides the largest matched improvement over frozen Qwen. Selective routing is retained only if it can be rebuilt from the strict aspect-conditioned TF-IDF endpoint without test-guided tuning.

The results progression is:

1. establish a simple Count sanity control and a strict sparse TF-IDF baseline;
2. test frozen semantic transfer with E5;
3. test a fine-tuned DistilBERT candidate cross-encoder;
4. diagnose candidate absence and false-positive control;
5. test frozen candidate-pair Qwen;
6. test candidate-pair QLoRA as task-specific adaptation;
7. test zero-label calibration and seed robustness before freezing the final claim.

## Active result registry

Only the rows below are eligible for the primary all-row LOAO result table.

| Method | Sentiment formulation | Test mean pair micro-F1 | Thesis role |
|---|---|---:|---|
| Count BoW | Shared DistilBERT aspect-conditioned sentiment; validation-reselected presence threshold | 0.3144 | Compact sanity control |
| Strict train-only character TF-IDF | Shared DistilBERT aspect-conditioned sentiment; validation-reselected presence threshold | 0.3856 | Main classical baseline |
| MiniLM-L6-v2 | Shared DistilBERT aspect-conditioned sentiment; validation-reselected presence threshold | 0.3889 | Repository/appendix only |
| E5-base-v2 | Shared DistilBERT aspect-conditioned sentiment; validation-reselected presence threshold | 0.4013 | Main frozen sentence-embedding baseline |
| DistilBERT review–candidate cross-encoder | DistilBERT aspect-conditioned sentiment | 0.3158 | Representative fine-tuned contextual baseline |
| Frozen candidate-pair Qwen | Joint candidate-conditioned pair prediction | 0.337804 | Matched QLoRA control |
| Candidate-pair QLoRA | Joint candidate-conditioned pair prediction | 0.483158 | Main task-adaptation contribution |

The strict similarity rescore exactly aligned all 48 validation and 48 test method-fold artifact pairs. It froze all 48 validation-selected thresholds before test scoring. The completed component audit showed that the stronger aspect-conditioned sentiment head improved all four similarity methods when presence decisions were held fixed, but those fixed-threshold numbers are diagnostic rather than the registered complete-system rows above.

## Evidence removed from the active direction

The following results and claims are retired. They must not appear in the dissertation prose, tables, figures, captions, abstract, or new-task instructions.

| Retired evidence | Treatment |
|---|---|
| Any global-document-sentiment Count, TF-IDF, MiniLM, E5, or fixed-split baseline | Historical experiment record only |
| Strict similarity results with global sentiment: Count `0.3225`, TF-IDF `0.3667`, MiniLM `0.3699`, E5 `0.3791` | Internal component reference only; not an active baseline |
| Fixed-threshold sentiment-swap results: Count `0.3351`, TF-IDF `0.3847`, MiniLM `0.3862`, E5 `0.3968` | Internal causal diagnostic only; not the complete-system table |
| Shallow TF-IDF aspect-conditioned sentiment and its `0.3576` LOAO result | Retired; no thesis claim and no rerun |
| Legacy lexical TF-IDF `0.3780` | Retired; candidate text participated in vectorizer fitting and sentiment was global |
| Legacy TF-IDF-to-Qwen router `0.4549` and refined `0.4560` | Retired; must not be presented or regenerated as current evidence |
| Any derivative legacy-router Pareto, per-aspect, uncertainty, boundary-search, or leave-one-aspect result | Retired from the thesis |
| The claim that aspect conditioning is harmful because a shallow classifier was weaker | Deleted from the active argument |
| The direction “keep global sentiment because it scores better for a shallow baseline” | Prohibited |

Historical logs may retain these values solely to explain what was run. No future task may select, tune, compare against, or build a thesis claim from them.

## Comparison language

Three comparison levels must remain explicit:

1. **Strict representation comparison.** Count, strict TF-IDF, MiniLM, and E5 share the same outer LOAO data, candidate text, aspect-conditioned sentiment component, validation-selection rule, output representation, and metric implementation. Their main difference is the presence representation.
2. **Matched adaptation comparison.** Frozen candidate-pair Qwen versus candidate-pair QLoRA holds the candidate-pair task and evaluation fixed while changing task-specific adapter training.
3. **Common outer-benchmark comparison.** DistilBERT, representation baselines, Qwen, QLoRA, and any router share the outer twelve-fold all-row LOAO benchmark but differ in internal architecture, training, or calibration. Their point estimates may share a table, but only the two matched blocks support intervention-level claims.

Do not describe every retained method as internally identical.

## Locked remaining-work checklist

This checklist is the authority for subsequent LOAO work. Complete it in order. Change `[ ]` to `[x]` only after the named evidence, validation, and documentation are complete.

### A. Governance and baseline consolidation

- [x] Approve the evidence-driven LOAO thesis mainline.
- [x] Record Aji's decision that every active baseline must use aspect-conditioned sentiment.
- [x] Complete the strict Count/TF-IDF/MiniLM/E5 aspect-conditioned sentiment rescore.
- [x] Verify exact validation/test row alignment and freeze 48 validation-selected thresholds before test.
- [x] Replace the active global-sentiment baseline numbers with the validation-reselected aspect-conditioned numbers.
- [x] Remove shallow-sentiment, global-sentiment, legacy TF-IDF, and legacy-router results from the active thesis direction.
- [ ] Integrate the completed aspect-conditioned rescore code, config, tests, report, and tracked numeric exports from `agent/bow-sentence-embedding-baselines` into the final target branch after user review.
- [ ] Rebuild one authoritative baseline table from the active result registry and verify that no retired row is regenerated.

### B. Rebuild the optional strict TF-IDF router

This block is permitted only with the strict TF-IDF presence representation and the shared DistilBERT aspect-conditioned sentiment component.

- [ ] Record a strict aspect-conditioned router protocol/config before running the replacement analysis.
- [ ] Add and test a compatibility path exposing strict TF-IDF `presence_score`, its validation-reselected threshold, present/absent prediction, and candidate-specific sentiment to the router analyser without changing the frozen source artifacts.
- [ ] Verify exact validation/test row alignment between strict aspect-conditioned TF-IDF and reused Qwen predictions for all twelve aspects.
- [ ] Select one global router policy from validation only; do not inspect or tune against router test results.
- [ ] Freeze the policy and evaluate it once on all twelve test folds.
- [ ] Report strict local-only and router pair F1, presence precision/recall/F1, FP/FN rows per 100, Qwen call rate, per-aspect deltas, and paired descriptive uncertainty.
- [ ] Decide from the registered result whether the router remains a supporting deployment method or is dropped entirely.

The legacy global-sentiment router is not a baseline, fallback, or selection reference for this block.

### C. Strengthen the QLoRA claim

- [ ] Register a strict zero-label calibration analysis for frozen candidate-pair Qwen and QLoRA using only non-target-aspect validation evidence for each target fold.
- [ ] Execute the strict zero-label analysis from saved validation/test scores without retraining or changing predictions.
- [ ] Compare target-calibrated and strict zero-label results and freeze the permitted generalisation wording.
- [ ] Register additional QLoRA seeds with unchanged configuration, aggregation, stopping rule, runtime budget, and output locations.
- [ ] Run the registered additional full twelve-fold QLoRA seed replication.
- [ ] Aggregate seed-by-aspect results and determine whether the single-seed `12/12` improvement and mean gain remain stable.

The current thesis claim treats descriptions, contrastive negatives, balancing, and adapter training as one QLoRA candidate-pair package. A factorial component ablation is not required.

### D. Freeze the final evidence set

- [ ] Produce one authoritative table containing only the active rows listed above.
- [ ] Produce a separate matched-effect table for the strict representation block and frozen-Qwen-to-QLoRA adaptation.
- [ ] Produce a quality–call-rate table or figure only if the rebuilt strict aspect-conditioned router passes its registered gate.
- [ ] Verify that validation pilots, fixed-split results, positive-only diagnostics, post-hoc sensitivities, and component audits are not mixed with headline all-row results.
- [ ] Search thesis prose, tables, figures, captions, abstract, handoff prompts, and generated CSVs for every retired value and model name.
- [ ] Freeze final source paths, protocol labels, seeds, uncertainty wording, and limitations.

### E. Write and validate the dissertation

- [ ] Update Methods around candidate-conditioned sentiment, information regimes, retained methods, matched comparisons, and common outer-benchmark comparisons.
- [ ] Update Results in the locked progression: baselines, absence diagnosis, optional strict router, QLoRA, and robustness.
- [ ] Update Discussion around task-correct sentiment conditioning, non-monotonic representation gains, absence calibration, QLoRA limitations, and any router quality–cost trade-off.
- [ ] Keep discarded development experiments out of the main dissertation; cite repository provenance only if necessary.
- [ ] Compile and visually inspect the complete thesis PDF.
- [ ] Run final citation, cross-reference, metric-definition, retired-result, secret, and reproducibility checks.
- [ ] Commit and push the final safe tracked evidence and thesis changes after user review.

## Stop rules

- Do not run or report global-document-sentiment baselines.
- Do not rerun the shallow TF-IDF aspect-conditioned sentiment classifier.
- Do not use legacy TF-IDF or legacy-router outputs for any new analysis.
- Do not tune router boundaries after observing test.
- Do not add another sentence encoder unless a named thesis gap cannot be answered by E5.
- Do not add Gemini, anomaly detection, unrelated model sweeps, or fixed-split experiments to the LOAO mainline.
- Do not start QLoRA component ablations unless an approved component-level claim requires them.
- If strict zero-label or multi-seed evidence weakens QLoRA, narrow the claim rather than selecting a favourable post-hoc protocol.
