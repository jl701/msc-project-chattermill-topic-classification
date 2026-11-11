# Execution completion: 2026-11-11

All 13 registered serial stages completed successfully. The supervisor exited with code 0; its final guard check passed before reporting completion. Classification selection remains fixed, with the final dynamic method retained as a comparator. No historical-evaluation tuning was performed after freezing.

| Stage | Exit | Wall seconds |
|---|---:|---:|
| historical_analysis | 0 | 13.8832 |
| historical_profile | 0 | 5.6818 |
| fixed_deployment957 | 0 | 1221.0156 |
| fixed_deployment_analysis | 0 | 3.1469 |
| new800_labelblind | 0 | 22679.4939 |
| new800_profile | 0 | 4.6631 |
| new800_qlora_comparator | 0 | 1049.1657 |
| service_capacity | 0 | 653.7753 |
| service_closed | 0 | 2006.6390 |
| service_open | 0 | 1306.0797 |
| service_soak | 0 | 3641.0882 |
| service_burst | 0 | 1118.4465 |
| service_stability | 0 | 1.0596 |

Stage times include their immediate process management. The new-source quality matrix itself records 22,675.2365 seconds; its enclosing job records 22,679.4939 seconds. The historical matrix completed before the listed dependent stages and records 26,705.1609 seconds.

## Conservative budget

The registered ceiling is 30 hours with 5.5 prior reserved hours. Final conservative accounting is **22.2836 hours**. It adds historical runtime and subsequent queue wall time, including CPU analysis and some management overhead. This is not an exact GPU-utilisation integral. Planning estimates are retained separately from completed durations; no paid resources were used.

## Completed commands

These are the registered stage arguments, using the pinned local serving Python environment. The machine-specific E5 directory is replaced by PATH_TO_E5_BASE_V2_SNAPSHOT; its pinned revision is f52bf8ec8c7124536f0efb74aca902b2995e5bcd. Model weights, datasets, private output directories and raw predictions are not distributed. Reproduction requires those matching artifacts and a new destination; completed results must not be overwritten.

### historical_analysis

```text
python scripts/analyse_rag_campaign.py --artifacts outputs/serving_20261109/artifacts.json --development-root outputs/rag_20261111/development_final --development-labels outputs/serving_20261109/labels_development.jsonl --evaluation-root outputs/rag_20261111/evaluation_final --evaluation-labels outputs/serving_20261109/labels_evaluation.jsonl --output outputs/rag_20261111/final_selection --presence-score-space logit_margin --comparator-development outputs/rag_20261111/comparator_development/runs/qlora__quality_development/requests.jsonl --comparator-evaluation outputs/serving_20261109/primary/runs/qlora__quality_evaluation/requests.jsonl --routing-config configs/experiments/review_routing_20261111.json --compare e5_negative_seen:e5
```

### historical_profile

```text
python scripts/profile_rag_scores.py --artifacts outputs/serving_20261109/artifacts.json --replay outputs/serving_20261109/replay_evaluation.jsonl --scores outputs/rag_20261111/evaluation_final --selection outputs/rag_20261111/final_selection/selection.json --scope historical_replay --historical-labels outputs/serving_20261109/labels_evaluation.jsonl --historical-manifest outputs/serving_20261109/replay_manifest.json --output outputs/rag_20261111/historical_descriptive_profile.json
```

### fixed_deployment957

```text
python scripts/run_rag_deployment.py --service-plan outputs/rag_20261111/service_final_v2/plan.json --replay outputs/serving_20261109/replay_evaluation.jsonl --offline-root outputs/rag_20261111/evaluation_final --offline-config configs/experiments/serving_rag_negative_20261111.json --output outputs/rag_20261111/deployment_fixed957_v2 --method fixed --expected-reviews 957
```

### fixed_deployment_analysis

```text
python scripts/analyse_rag_deployment.py --deployment-root outputs/rag_20261111/deployment_fixed957_v2 --offline-scores outputs/rag_20261111/evaluation_final/fixed.jsonl --labels outputs/serving_20261109/labels_evaluation.jsonl --output outputs/rag_20261111/deployment_fixed957_v2_analysis
```

### new800_labelblind

```text
python scripts/run_rag_quality.py --artifacts outputs/serving_20261109/artifacts.json --index outputs/rag_20261111/index_all_populations --encoder-path PATH_TO_E5_BASE_V2_SNAPSHOT --replay outputs/rag_20261111/new_reviews/replay_new_inputs.jsonl --config configs/experiments/serving_rag_negative_20261111.json --output outputs/rag_20261111/new_source_labelblind --methods fixed,random_13,random_37,random_73,bm25,e5,rrf,e5_negative_seen
```

### new800_profile

```text
python scripts/profile_rag_scores.py --artifacts outputs/serving_20261109/artifacts.json --replay outputs/rag_20261111/new_reviews/replay_new_inputs.jsonl --scores outputs/rag_20261111/new_source_labelblind --selection outputs/rag_20261111/final_selection/selection.json --scope new_source_labelblind --output outputs/rag_20261111/new_source_labelblind_profile.json
```

### new800_qlora_comparator

```text
python scripts/run_serving_study.py --plan outputs/rag_20261111/comparator_new_source_plan.json --output outputs/rag_20261111/comparator_new_source
```

### service_capacity

```text
python scripts/run_rag_service_study.py --artifacts outputs/serving_20261109/artifacts.json --index outputs/rag_20261111/index_all_populations --encoder-path PATH_TO_E5_BASE_V2_SNAPSHOT --selection outputs/rag_20261111/final_selection/selection.json --development-replay outputs/serving_20261109/replay_development.jsonl --development-score-root outputs/rag_20261111/development_final --correctness-reviews 100 --capacity-replay outputs/rag_20261111/performance_pool/replay_capacity.jsonl --replay outputs/rag_20261111/performance_pool/replay_benchmark.jsonl --output outputs/rag_20261111/service_final_v2 --port 8766 --retrieval-timeout-s 10 --sentiment-execution canonical_full_grid_groups --stage capacity
```

### service_closed

```text
python scripts/run_rag_service_study.py --artifacts outputs/serving_20261109/artifacts.json --index outputs/rag_20261111/index_all_populations --encoder-path PATH_TO_E5_BASE_V2_SNAPSHOT --selection outputs/rag_20261111/final_selection/selection.json --development-replay outputs/serving_20261109/replay_development.jsonl --development-score-root outputs/rag_20261111/development_final --correctness-reviews 100 --capacity-replay outputs/rag_20261111/performance_pool/replay_capacity.jsonl --replay outputs/rag_20261111/performance_pool/replay_benchmark.jsonl --output outputs/rag_20261111/service_final_v2 --port 8766 --retrieval-timeout-s 10 --sentiment-execution canonical_full_grid_groups --stage closed
```

### service_open

```text
python scripts/run_rag_service_study.py --artifacts outputs/serving_20261109/artifacts.json --index outputs/rag_20261111/index_all_populations --encoder-path PATH_TO_E5_BASE_V2_SNAPSHOT --selection outputs/rag_20261111/final_selection/selection.json --development-replay outputs/serving_20261109/replay_development.jsonl --development-score-root outputs/rag_20261111/development_final --correctness-reviews 100 --capacity-replay outputs/rag_20261111/performance_pool/replay_capacity.jsonl --replay outputs/rag_20261111/performance_pool/replay_benchmark.jsonl --output outputs/rag_20261111/service_final_v2 --port 8766 --retrieval-timeout-s 10 --sentiment-execution canonical_full_grid_groups --stage open
```

### service_soak

```text
python scripts/run_rag_service_study.py --artifacts outputs/serving_20261109/artifacts.json --index outputs/rag_20261111/index_all_populations --encoder-path PATH_TO_E5_BASE_V2_SNAPSHOT --selection outputs/rag_20261111/final_selection/selection.json --development-replay outputs/serving_20261109/replay_development.jsonl --development-score-root outputs/rag_20261111/development_final --correctness-reviews 100 --capacity-replay outputs/rag_20261111/performance_pool/replay_capacity.jsonl --replay outputs/rag_20261111/performance_pool/replay_benchmark.jsonl --output outputs/rag_20261111/service_final_v2 --port 8766 --retrieval-timeout-s 10 --sentiment-execution canonical_full_grid_groups --stage soak
```

### service_burst

```text
python scripts/run_rag_service_study.py --artifacts outputs/serving_20261109/artifacts.json --index outputs/rag_20261111/index_all_populations --encoder-path PATH_TO_E5_BASE_V2_SNAPSHOT --selection outputs/rag_20261111/final_selection/selection.json --development-replay outputs/serving_20261109/replay_development.jsonl --development-score-root outputs/rag_20261111/development_final --correctness-reviews 100 --capacity-replay outputs/rag_20261111/performance_pool/replay_capacity.jsonl --replay outputs/rag_20261111/performance_pool/replay_benchmark.jsonl --output outputs/rag_20261111/service_final_v2 --port 8766 --retrieval-timeout-s 10 --sentiment-execution canonical_full_grid_groups --stage burst
```

### service_stability

```text
python scripts/analyse_rag_service_stability.py --campaign outputs/rag_20261111/service_final_v2 --output-dir outputs/rag_20261111/service_final_v2/stability --run-prefix soak__ --run-prefix open__ --run-prefix burst__
```

## Completion and audit evidence

- Historical eight-method and new-source eight-method matrices have full registered coverage.
- Service coverage is 46 runs: 1,786 attempts, 1,780 successes and six burst HTTP 429 rejections. Warmups are excluded.
- Stability covers 16 open/soak/burst runs with no incomplete matching run. The one-hour soak has twelve complete five-minute arrival windows.
- Independent final service review passed 1,613 assertions, inspecting 383 JSON sources and 479 hash references. Raw request/telemetry files were hashed without parsing review text or labels during this review.
- Aggregate extraction verified 185 source metadata fingerprints before post-run feedback-policy integration. It did not independently recompute raw gold metrics.
- Focused feedback-policy/workflow/analysis CPU validation: 77 passed in 24.64 seconds; Ruff and whitespace checks passed. Real input-only policy validation passed all four purposes and retained original annotation/evaluation populations.

## Key identities

| Artifact | SHA256 |
|---|---|
| serial plan | `e842d9636ee236218c49ffd5dede9cc9dc8dcce97dc7267891291af558641f65` |
| final queue state | `c80a7d419ce3f0f9c59d8780a008ade5b68fb58a585da01c55b07e8e292905ce` |
| service plan | `c7e4069ea2a72983ad4dfd8f6bde5dd4bcdade1eb616d336b0b5b85113f60a57` |
| stability aggregate | `0e431fc8cfea4fa8875eed731e1bf6d8cfa4c81b55c84919fac73226d8f611e7` |
| published aggregate | `c9231ee5ed0ac07a642b6bfd82f55cf6cf584dceddc313f756cee0e78b5bd1cd` |

Measured campaign code: `3633f87`. Post-run source-bound adaptation policy integration: `6c1b3779a01d218d2caafaf5390f071aed677950`. Final reporting changes do not modify frozen model predictions. See [results](results_20261111.md) for interpretation and [policy integration](adaptation_use_policy_20261111.md) for the separate input-only safeguards.

## Human-dependent work

At 2026-11-11 22:16:39 Asia/Shanghai, the annotation interface reports 0/800 confirmed and 0/50 pilot confirmed. Remaining work is reviewed gold release, new-source quality/routing evaluation, actual eligible adaptation feedback and separately registered target-support comparisons. No CV wording or resume metrics are changed by this release.
