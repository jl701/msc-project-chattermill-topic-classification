# Extended training-policy sensitivity

> Historical intermediate report. Its six-family results remain evidence, but
> its mixed-component diagnostic is not the final same-policy composition.
> QLoRA and both complete compositions are now covered by the
> [completed policy analysis](../results/training_policy_completed/README.md).

Validation-only, proposed after official test. No official-test outcomes were reopened.

Six original non-QLoRA families are covered. E5 and zero-shot Qwen are invariant references, not independent reruns.
The mixed-component diagnostic retains the original review-filtered QLoRA Stage 2. It is not a label-masked QLoRA result.

All twelve folds and both N/D conditions are reported. Kernel-ridge DCWT is the registered primary generator. The three original controls are retained, not selected using held-out outcomes.
Intervals are nominal conditional review-cluster intervals, not confirmatory tests or training-seed uncertainty.

| Method | D masked-minus-filtered F1 | Nominal 95% interval |
|---|---:|---|
| dcwt_cosine_barycentric_weight | -0.0165 | [-0.0204, -0.0127] |
| dcwt_kernel_ridge | -0.0114 | [-0.0165, -0.0063] |
| dcwt_mean_seen_weight | -0.0219 | [-0.0259, -0.0181] |
| dcwt_nearest_description_weight | -0.0004 | [-0.0049, +0.0040] |
| distilbert | -0.0062 | [-0.0256, +0.0128] |
| e5_base_v2 | +0.0000 | [+0.0000, +0.0000] |
| fewshot | -0.0322 | [-0.0507, -0.0127] |
| frozen_qwen_candidate_pair | +0.0000 | [+0.0000, +0.0000] |
| mixed_component_diagnostic | -0.0254 | [-0.0348, -0.0162] |
| tfidf | -0.0291 | [-0.0363, -0.0223] |
