# Genuine two-stage taxonomy generalisation: local pre-cloud completion

Date: 17 August 2026

Status: **local pre-cloud gate passed**. The next authorised action is the
single-scope measured cloud benchmark. The full cloud schedule remains blocked
until that benchmark passes. Official test remains sealed.

## 1. What was completed

The new mainline is a genuine aspect-first pipeline. Stage 1 assigns a
presence probability to each supplied aspect candidate and selects every
candidate above a seen-only threshold. Stage 2 assigns conditional sentiment
probabilities only for selected aspects. The primary decoder emits one
sentiment and may emit a second when a separate seen-only runner-up threshold
is met; it can never emit three.

The following local work is complete:

- deterministic, balanced train-only manifests for both stages;
- deterministic Frozen-Qwen few-shot demonstrations, with held-out aspects
  excluded;
- L3 NN, DN, ND, DD and RR representation handling with shared checkpoint,
  demonstrations and thresholds, plus exact seen-score identity checks;
- strict reuse classification for old one-stage and new two-stage artifacts;
- real-model DistilBERT, Frozen-Qwen few-shot and QLoRA GPU smoke tests;
- validation-only L1, L3 and L4 runs for TF-IDF, E5 and Frozen-Qwen zero-shot;
- exact reuse of the completed matched L2 TF-IDF/E5 comparison and the
  compatible Frozen-Qwen zero-shot cache; and
- a machine-validated cloud gate whose upload whitelist contains only
  `train.csv` and `validation.csv`.

## 2. Validation-only results available now

The table reports the mean held-out pair micro-F1 across registered folds.
These values are descriptive evaluation evidence. They were not used to alter
the model, prompt, candidate representation, threshold rule or search grid.

| Method | Level | Condition | Folds | Held-out pair micro-F1 mean | SD |
|---|---|---|---:|---:|---:|
| TF-IDF | L1 | D | 12 | 0.2773 | 0.1336 |
| TF-IDF | L1 | N | 12 | 0.2108 | 0.1388 |
| TF-IDF | L3 | DD | 12 | 0.2749 | 0.1182 |
| TF-IDF | L3 | DN | 12 | 0.2453 | 0.1298 |
| TF-IDF | L3 | ND | 12 | 0.2529 | 0.1276 |
| TF-IDF | L3 | NN | 12 | 0.2136 | 0.1223 |
| TF-IDF | L3 | RR | 12 | 0.2321 | 0.1133 |
| TF-IDF | L4 | D | 3 | 0.2314 | 0.0889 |
| E5-base-v2 | L1 | D | 12 | 0.2222 | 0.1406 |
| E5-base-v2 | L1 | N | 12 | 0.2415 | 0.1667 |
| E5-base-v2 | L3 | DD | 12 | 0.2404 | 0.1315 |
| E5-base-v2 | L3 | DN | 12 | 0.2528 | 0.1455 |
| E5-base-v2 | L3 | ND | 12 | 0.2481 | 0.1462 |
| E5-base-v2 | L3 | NN | 12 | 0.2597 | 0.1554 |
| E5-base-v2 | L3 | RR | 12 | 0.2417 | 0.1320 |
| E5-base-v2 | L4 | D | 3 | 0.1636 | 0.0696 |
| Frozen Qwen zero-shot | L1 | D | 12 | 0.4513 | 0.1821 |
| Frozen Qwen zero-shot | L1 | N | 12 | 0.4359 | 0.2276 |
| Frozen Qwen zero-shot | L3 | DD | 12 | 0.4751 | 0.1106 |
| Frozen Qwen zero-shot | L3 | DN | 12 | 0.4725 | 0.1121 |
| Frozen Qwen zero-shot | L3 | ND | 12 | 0.4833 | 0.1188 |
| Frozen Qwen zero-shot | L3 | NN | 12 | 0.4829 | 0.1156 |
| Frozen Qwen zero-shot | L3 | RR | 12 | 0.4309 | 0.1259 |
| Frozen Qwen zero-shot | L4 | D | 3 | 0.4802 | 0.0369 |

The matched L2 comparison gives the clearest direct architecture result so
far because rows, candidates, representations and seen-only selection are
identical:

| Method | Independent 36-pair F1 | Genuine two-stage capped-two F1 | Delta | Fold wins/ties/losses |
|---|---:|---:|---:|---:|
| TF-IDF | 0.2196 | 0.2773 | +0.0577 | 11 / 0 / 1 |
| E5-base-v2 | 0.1860 | 0.2222 | +0.0362 | 9 / 0 / 3 |

This supports continuing the two-stage design, but it is not yet evidence that
every trainable or prompted model improves. Frozen-Qwen few-shot, genuine
two-stage DistilBERT and genuine two-stage QLoRA still require their full
validation runs.

## 3. Integrity and safety audit

The final local audit passed with:

- 261 method × level × condition fold records;
- zero failed folds, non-finite values, resume conflicts or test contracts;
- all registered conditions present;
- one or two sentiments per selected aspect, never three;
- no method-level prediction collapse;
- exact L3 seen-score hashes across representation conditions;
- 26 reproducible unique training scopes, each with 2,048 aspect-presence and
  2,048 conditional-sentiment training instances; and
- 100,032 finite, conflict-free Frozen-Qwen zero-shot cache records after the
  missing RR prompts were added to a copied cache.

The augmented zero-shot cache SHA-256 is
`08d276ac8915a929c3849dc92a54130df53a9835a8f0b36a0040ee7dbfad9db3`.
Raw caches, predictions and model weights remain outside Git.

## 4. Real-model local GPU gates

| Gate | Result | Elapsed | CUDA peak memory |
|---|---|---:|---:|
| DistilBERT separate Stage-1/Stage-2 heads | Pass | 1.55 s | 1.91 GB |
| Frozen Qwen deterministic few-shot | Pass | 11.61 s | 3.15 GB |
| Shared two-task QLoRA adapter, one update | Pass | 16.91 s | 4.22 GB |

The QLoRA smoke exposed 2,949,120 trainable parameters, completed one finite
optimizer step and produced finite two-class aspect and three-class sentiment
probabilities. This smoke is an execution/memory gate only, not a formal
result.

## 5. Exact next boundary

The first cloud action is limited to the three commands in
`taxonomy_two_stage_cloud_gate_v1.json`, all on `l2-a01`. It measures the cloud
runtime and memory of Frozen-Qwen few-shot, genuine two-stage DistilBERT and
genuine two-stage QLoRA. It produces no formal dissertation result.

Only after this gate has zero failures, zero test contracts, finite losses and
scores, no OOM, no repeated thermal throttling and no prediction collapse may
the full validation-only schedule be released. The order is Frozen-Qwen
few-shot, DistilBERT, matched controls where exact reuse fails, one-scope QLoRA,
remaining L2 QLoRA, then L1/L3/L4 QLoRA.

## 6. Authoritative artifacts

- Scientific preregistration:
  `configs/experiments/taxonomy_two_stage_precloud_v2.json`
- Cloud gate:
  `configs/experiments/taxonomy_two_stage_cloud_gate_v1.json`
- Human-readable preregistration:
  `docs/experiments/taxonomy_two_stage_precloud_v2_preregistration.md`
- Local audit:
  `outputs/experimental/taxonomy_two_stage_precloud_v2/artifact_audit.json`
- Cloud gate manifest:
  `outputs/experimental/taxonomy_two_stage_precloud_v2/cloud_gate_manifest.json`

No document in this set authorises opening official test.
