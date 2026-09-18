# Frozen Candidate-Pair Qwen Validation and Cache Completion

Date: 2026-07-25

Status: complete for official train/validation; official test remains sealed.

## Frozen method contract

- Method: `frozen_qwen_candidate_pair`
- Thesis role: matched no-adapter LLM control
- Model: `Qwen/Qwen3-4B-Instruct-2507`
- Model revision:
  `cdbee75f17c01a7cc42f958dc650907174af0554`
- Task format: one rendered review-candidate sentiment claim with a one-token
  `Y`/`N` answer
- Score: `softmax(Y, N)` probability assigned to `Y`
- Load: NF4 4-bit double quantisation with FP16 model compute
- Probability calculation: FP32 softmax over the two verbalizer logits
- Maximum length: 384
- Evaluation batch size: 6
- Scientific-parameter SHA-256:
  `c6c21a724464096fcc4ab4cc34ae976e666c887945fcac71a749504d322ae918`
- Parameter-selection file SHA-256:
  `341d54bfcde0e6d21cc591d0e8ec7ecffa639e35b4563493f8d97209283ef1dc`

The prompt, verbalizers, parser, model revision, quantisation, maximum length,
and batch size were fixed before execution. No model, prompt, or parameter
search was performed.

## Execution audit

The guarded through-validation plan completed `93/93` Frozen Qwen jobs with
zero failures and no running jobs:

| Artifact | Count |
| --- | ---: |
| Frozen task-scope checkpoint contracts | 26 |
| Validation score contracts | 27 |
| Score CSV shards | 216 |
| Matching score manifests | 216 |
| Threshold-transfer artifacts | 39 |

Every checkpoint:

- pins the same model ID, revision, method specification, scientific
  parameters, and protocol;
- has `task_specific_fit_performed: false`;
- contains an intact hashed `reload.json`;
- records one of the 26 registered outer training scopes without modifying
  model weights.

Every score contract has exactly eight shards. All 216 CSV content hashes
match their manifests. Every manifest binds the same frozen scientific
protocol and description-resource hash:
`fcf546d227ad2ac52ccfa9682fc3685d2e396069ed911016f0bc12bf205f1367`.
All declare the `validation` split.

All 39 threshold artifacts:

- have valid self-hashes;
- bind the fixed scientific-parameter hash;
- reference existing validation score contracts;
- comprise 12 L1, 12 L2, 12 L3, and 3 L4 folds;
- give every L3 fold one shared threshold for `NN/DN/ND/DD/RR`.

No official-test job, score, analysis, cache, or test-use ledger entry was
created.

Total executor wall time was `01:13:10.93`.

Execution-state SHA-256:
`2100a964d2f03f5d87e48d548792d3e9ff803ee48a2d52316c0c64522751dace`

SHA-256 over the newline-joined sorted 27 score-contract identities:
`bb8a16a70313577f83545af393ebcba1d0a74d84d9e45739dfcfdcc5d70cd92a`

SHA-256 over the newline-joined sorted 39 threshold-artifact identities:
`8f37922f9c15f1096b5f83af73374eff2894afe12f9653e6d854e49c311f6d4d`

## Exact validation raw-score cache

The cache contract is split-specific and model-specific. Its input key is the
SHA-256 of the exact review text, rendered candidate text, canonical candidate
aspect, candidate sentiment, and representation variant. It stores only that
content hash and raw `P(Y)`: no review text, labels, threshold, prediction, or
metric is stored.

| Cache quantity | Value |
| --- | ---: |
| Logical requested pair scores across 27 scopes | 887,880 |
| Persisted unique model inputs | 37,512 |
| Repeated forward passes avoided | 850,368 |
| Avoided fraction | 95.7751% |
| Logical-to-unique ratio | 23.67x |

The first large scope filled most of the cache in approximately 50 minutes.
The second scope added the final missing minimal-aspect inputs in approximately
5 minutes. Every remaining score scope was a complete cache hit and therefore
did not load Qwen.

SQLite `integrity_check` returned `ok`. The cache has 37,512 distinct
64-character input hashes, zero out-of-range probabilities, and metadata that
matches its filename contract. Cache database SHA-256:
`8d6ec3b62cc1eb57e5ffb0fa9d0365bfb63030910fbae15e26f64f0f3f876e4f`

This cache is lossless memoisation. It changes runtime, not prompts, logits,
probabilities, thresholds, predictions, or metrics.

## Seen-validation calibration diagnostics

These values describe only seen-aspect evidence used to freeze the transferred
threshold. They are not official test performance and do not measure unseen
taxonomy generalisation.

| Level | Folds | Mean seen pair micro F1 | Min-max | Mean presence F1 | Mean conditional sentiment accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| L1 | 12 | 0.459815 | 0.435462-0.492645 | 0.888981 | 0.976615 |
| L2 | 12 | 0.459815 | 0.435462-0.492645 | 0.888981 | 0.976615 |
| L3 | 12 | 0.460002 | 0.429605-0.481413 | 0.868986 | 0.976722 |
| L4 | 3 | 0.461354 | 0.460502-0.462170 | 0.873081 | 0.976336 |

L1 and L2 are identical by registered calibration reuse. Cross-level means
must not be read as a difficulty curve because the seen-aspect subsets differ.

## Probability saturation diagnostic

All 39 folds selected the same transferred threshold:
`0.999999940395`.

The 37,512 cached inputs contain 1,619 distinct probabilities; 2,842 inputs
are exactly `1.0`, and the selected threshold retains those same 2,842 inputs.
The implementation already converts the two verbalizer logits to FP32 before
softmax. The saturation therefore reflects very large frozen-Qwen `Y` versus
`N` logit gaps under the pre-registered probability interface, not an FP16
softmax implementation bug.

Changing the score after observing validation to an unbounded logit margin
would change the scientific interface and break the registered Frozen
Qwen/QLoRA comparison. The probability score and threshold are therefore kept
frozen. The limited resolution near one must be reported as a calibration
characteristic and possible limitation when interpreting the sealed test
results.
