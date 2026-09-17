# Two-Stage Global Router Validation Study v1

Date frozen before router outcome inspection: 2026-08-23

## Objective

Design exactly one deployable global router for the formal two-stage taxonomy-generalisation pipeline. The router may combine two existing formal-v2 systems, but it must share one model direction and one pair of routing bands across all aspects. Its primary goal is to improve aspect-balanced held-out pair micro-F1 while retaining a directly comparable overall pair micro-F1, precision, recall, presence and call-rate record.

## Safety boundary

- Validation only. The official test partition, labels, predictions and artifacts are prohibited.
- The only raw inputs are locally verified formal-v2 score shards with `split=validation` and `test_contract_count=0`.
- The study must reject incomplete shards, hash mismatches, non-finite scores, duplicate pair identities, row-set mismatches and any non-validation split.
- No result from this study authorises an official-test run by itself.

## Eligible router inputs

The ordered base/expert search is restricted to the three systems with complete, content-addressed, fully matched formal-v2 raw score shards:

1. DistilBERT two-stage cross-encoder;
2. Frozen Qwen few-shot;
3. QLoRA Qwen.

All six ordered pairs are eligible. TF-IDF, E5, description-to-weight transfer and Frozen Qwen zero-shot remain mandatory comparison rows, but they are not router components because the scientific-freeze package does not retain equivalent formal-v2 score shards for them.

Before routing, the existing post-audit QLoRA canonicalisation is reproduced for L2-D folds a07 and a12: unchanged seen candidates inherit their same-fold N scores while every held-out D score remains bitwise unchanged. This is required for exact agreement with the scientific-freeze comparison tables; it is not a router outcome-dependent operation.

## Router

Each component first retains its frozen fold-local aspect threshold, runner-up sentiment threshold and capped-two decoder. Scores are monotonically mapped so that the component's own threshold becomes 0.5. This alignment changes neither that component's binary aspect decision nor its sentiment rank/runner-up decision.

Routing happens once per review-candidate-aspect instance:

- if the base predicts absence and its aligned distance from 0.5 is inside the rescue band, use the expert's complete aspect/sentiment decision for that instance;
- if the base predicts presence and its aligned distance from 0.5 is inside the confirmation band, use the expert's complete decision;
- otherwise retain the base decision.

The rescue and confirmation bands independently use the frozen grid `[-1, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5]`, where `-1` disables that side. This gives 81 policies per ordered pair and 486 policies in total. Gold labels, aspect identity, condition identity and test-derived information are not router features.

## Selection and estimation

The D condition is the primary deployment interface and the only condition used for router selection. Candidate policies are ranked lexicographically by:

1. mean held-out pair micro-F1 over aspect folds;
2. mean overall pair micro-F1;
3. lower route rate;
4. lexical policy ID.

A twelve-way leave-one-aspect-fold-out meta-evaluation selects a policy on eleven folds and evaluates it on the omitted fold. This cross-fitted result is the principal estimate of whether the router generalises across unseen aspects. Pair/policy stability is reported rather than hidden.

After cross-fitting, exactly one final router is selected on all twelve D validation folds. Its ordered pair and bands are transferred unchanged to N. The all-fold selected validation score is reported as a fitted descriptive result, not as the cross-fitted estimate.

## Fair comparison

The router uses the same 1,057 validation reviews, 12 L2 folds, full 36-pair grid, N/D interfaces and maximum-two-sentiment contract as the seven formal systems. The primary display must include:

- held-out pair micro-F1, reported as the aspect-balanced fold mean;
- overall pair micro-F1;
- pooled held-out and overall pair micro-F1;
- held-out precision and recall;
- held-out presence F1;
- route rate and component identities.

The success rule is demanding: the cross-fitted D held-out pair F1 must exceed both selected components and the current best single-system D value. Failure is retained as a negative result; no new policy family is added after inspection.

## Reproducibility outputs

The analysis must write the pre-registered configuration hash, every verified input hash, all 486 policy/fold records, cross-fitted selections, the final selected router, method comparison tables, row-level sufficient statistics, and an audit manifest. Raw reviews and official-test material remain excluded.

## Completion pointer

The registered study was completed without changing its policy family or
selection rule. The cross-fitted success rule did not pass. Results, bootstrap
intervals, interpretation and integrity counts are recorded in
`docs/experiments/taxonomy_two_stage_global_router_validation_v1_results.md`.
