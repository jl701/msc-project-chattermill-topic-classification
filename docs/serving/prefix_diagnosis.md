# Prefix reuse numerical diagnosis

The frozen training-prefix cache is evaluated as a **separate runtime configuration**. It is not described as a lossless or numerically equivalent acceleration. The initial 100-review development gate preserved the final predictions for both few-shot and hybrid systems, but some probability changes were substantial. Subsequent quality and latency comparisons must identify cache-on and cache-off separately.

## What changed and what stayed fixed

Cache-on reuses the system instruction and frozen training demonstrations before the query payload. The complete encoded query must match the cached prefix exactly. Each call receives an independent copy of the prefix KV tensors; query states never become persistent cache entries. Model weights, tokenizer, demonstrations, candidate descriptions, truncation, thresholds and the capped-two sentiment decoder remain fixed.

The execution changes from a complete prompt forward to a prefix forward followed by a suffix forward. The prefix is computed once with batch size one, whereas the measured candidate microbatches contain four prompts with different lengths. These differences in computation shape can change low-precision numerical results even when token identities and absolute positions agree.

## Evidence from the development gate

For the full few-shot system, 9 of 100 reviews had a maximum aspect-probability difference greater than 0.01. Four reviews had a selected-sentiment probability difference greater than 0.01. The largest aspect difference was **0.120684**, and the largest common-selected sentiment difference was **0.044019**. These are not merely tiny output rounding errors.

The 1,200 aspect decisions nevertheless retained identical selected labels under the frozen threshold. The largest differences occurred in scores well below that threshold. A selected aspect close to the threshold moved by one output-probability step and remained only approximately **5.96e-8** above the decision boundary. Passing this finite sample does not establish that all future decisions will be unchanged. The same 100 reviews evaluated under two variants are not 200 independent quality samples.

## Controlled diagnosis

Two development reviews were selected before the diagnostic run: one with the largest observed aspect-score change and one nearest the frozen aspect threshold. Each diagnostic used the same first four candidate prompts. No weights or thresholds were changed.

| Control | Observation | Supported interpretation |
|---|---|---|
| Repeat the complete forward twice | Scores were identical. | Ordinary run-to-run nondeterminism did not explain these examples. |
| Complete forward with `use_cache=True` | Scores matched the complete uncached forward. | Merely requesting returned KV states was not the source of the discrepancy. |
| Build the prefix at batch four, retain its first row, then process the same suffix batch | The maximum presence difference in the first group fell from about 0.120684 to 0.066254, but did not disappear. | Prefix construction batch shape contributes to the result. |
| Extract prefix KV directly from a complete forward, then recompute only the suffix | The first group still differed by about 0.066254; the threshold-near group's target score matched the complete forward. | Prefix states and suffix execution shape both matter. Reusing a prefix from the complete computation does not universally remove the difference. |
| Force the SDPA math backend for both complete and split execution | The first group's target presence score matched between the two math paths, but another target near the threshold crossed the frozen boundary. | Attention backend choice is relevant, but switching to math is not a general numerical repair. This diagnostic-only backend is not used in the service. |
| Actual two-layer Qwen3 on CPU with FP32 eager attention and unequal suffix padding | Maximum final-logit difference was about 5.96e-8. | Absolute positions, full prefix-plus-suffix masks and suffix-relative final-token selection work with the actual Qwen implementation. This small-model check does not certify quantized 4B numerical equivalence. |

The KV comparison also found differences in the **first layer**, before accumulated attention effects from later layers: the maximum value-state difference was about 0.000488 and the maximum key-state difference about 0.25. Differences continued into later layers. This is consistent with shape-sensitive low-precision projections and normalization as well as attention execution, rather than an error introduced only by the final softmax.

The installed Transformers SDPA wrapper takes different paths when an explicit attention mask is present: it expands grouped KV heads for masked attention, while mask-free causal attention may use native grouped-query attention. The unpadded prefix and the right-padded complete/suffix batches therefore need not take the same attention path. The diagnostic supports contributions from numerical execution shape and backend; it does not isolate every contributing kernel or prove that no other issue could exist.

## Decision and reporting limits

1. Keep cache-off as the reference configuration and measure cache-on as a separately identified candidate configuration.
2. Compare both configurations on the same 957-review evaluation replay, including quality, changed aspect/sentiment labels, latency and resource use. Do not select or tune thresholds against those evaluation labels.
3. Recommend a serving configuration only after considering the measured quality–latency trade-off. Do not infer losslessness from the 100-review development gate.
4. Preserve the default SDPA implementation and frozen thresholds. The math-backend diagnostic is evidence, not a deployment change.
5. Report the two targeted cases as a bounded numerical diagnostic, not a population-level attribution study. The aggregate record contains no review text, request identifiers, host paths or raw score grids.

Reproducible commands and source are provided by `scripts/check_serving_prefix_gpu.py` and `scripts/diagnose_serving_prefix.py`. Public aggregates are in `results/prefix_diagnosis.json`; detailed scores and identifiers remain in ignored local outputs.
