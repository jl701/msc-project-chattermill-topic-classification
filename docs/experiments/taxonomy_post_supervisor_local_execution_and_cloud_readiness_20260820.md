# Post-supervisor local execution and revised cloud readiness

Date: 20 August 2026
Status: **local work complete; revised cloud campaign prepared but not launched**

## Boundary and outcome

All work used only the official `train` and `validation` splits. The official
test remained sealed; every result and audit reports `test_contract_count=0`.
No RunPod Pod or volume was started during this work, so this phase incurred no
new cloud charge.

The local phases P1--P5 in the approved post-supervisor plan are complete:

1. exact reuse versus missing-work matrix;
2. new Level 1 closed-taxonomy reference;
3. implementation and twelve-fold validation of description-to-classifier
   weight transfer (DCWT);
4. twelve-fold Level 2 `N/D/R` evidence for TF-IDF, E5 and Frozen-Qwen
   zero-shot, including both L2-S and L2-E views; and
5. a reduced, dependency-closed three-GPU cloud plan for the remaining
   Frozen-Qwen few-shot, DistilBERT and QLoRA validation.

## Exact data and resource identity

| Item | Value |
|---|---|
| Train rows | 7,930 |
| Validation rows | 1,057 |
| Train SHA-256 | `3ffcc7407cc8077fe9efc5ca06532847589a59464ea929d7f23d1f732b9efc7c` |
| Validation SHA-256 | `3a32afda9c4ba1d59e76ee8fcfb480011c7178d8201d0025b254a40c2f81203c` |
| Minimal-description resource SHA-256 | `fc93cf27efdb64ad335f39f4a0dbdbd3dad13b1aae5de0010280931867af7d4c` |
| Rich-guidance resource SHA-256 | `289ba3238cb6772f9bfda98eb73ac8a1108ba4b2bb3d23eecf72826881f415b9` |
| Reuse-matrix SHA-256 | `2783c1f5bfe6d9554d0afe797d8f398946e1ca4e343368b79cea4a567ace8101` |

The exact-reuse audit rejected approximate reuse. In particular, the new
Level 1, the unified Level 2 `N/D/R` TF-IDF/E5 score bundles, DCWT, and every
remaining trainable cloud method require new result contracts. Frozen-Qwen
zero-shot alone could reuse its raw score cache exactly: cache SHA-256
`08d276ac8915a929c3849dc92a54130df53a9835a8f0b36a0040ee7dbfad9db3`
and prompt-contract SHA-256
`ed6eff449c4396e46ce11c17875997772feced1ac2701e82092a4c66e1ba162b`.

## Level 1 closed-taxonomy reference

The validation-only word-plus-character TF-IDF one-vs-rest Logistic
Regression sweep selected balanced Logistic Regression with `C=4` and
threshold `0.544508595979`.

| Metric | Result |
|---|---:|
| Pair micro-F1 | **0.705078** |
| Pair macro-F1 | 0.465019 |
| Aspect micro-F1 | 0.759466 |
| Sentiment accuracy when the gold aspect was predicted | 0.953737 |

The selected score SHA-256 is
`5896e581246574bee8bfb95654218adc45d6cc3a83d202dd3ea14a265e48417d`.
This is a conventional in-distribution reference, not a novel result and not
an official-test estimate.

## Level 2 local `N/D/R` results

Values below are unweighted means over the same twelve single-held-out-aspect
folds. “Presence” is the L2-S held-out aspect-presence F1; “pair” is the L2-E
held-out aspect-sentiment micro-F1. The same fold checkpoint/frozen model,
validation rows, seen-only thresholds and seen-candidate scores were held
fixed across `N`, `D` and `R`.

| Method | Presence N | Presence D | Presence R | Pair N | Pair D | Pair R |
|---|---:|---:|---:|---:|---:|---:|
| Strict train-only TF-IDF | 0.337745 | **0.411028** | 0.342993 | 0.210762 | **0.277251** | 0.228107 |
| Frozen E5-base-v2 | **0.301210** | 0.290552 | 0.274035 | **0.241473** | 0.222184 | 0.222266 |
| Frozen Qwen zero-shot | 0.459919 | **0.497101** | 0.460553 | 0.435860 | **0.451298** | 0.406751 |
| DCWT kernel ridge | 0.232849 | **0.236820** | 0.223756 | 0.144329 | **0.145314** | 0.132310 |

### Paired description contrasts

| Method | Presence `D-N` | Improved folds | Pair `D-N` | Improved folds | Presence `R-D` | Pair `R-D` |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF | **+0.073283** | 9/12 | **+0.066489** | 9/12 | -0.068035 | -0.049144 |
| E5 | -0.010658 | 7/12 | -0.019290 | 6/12 | -0.016517 | +0.000082 |
| Frozen Qwen zero-shot | **+0.037183** | 7/12 | **+0.015439** | 6/12 | -0.036549 | -0.044547 |
| DCWT kernel ridge | **+0.003971** | 9/12 | **+0.000985** | 10/12 | -0.013064 | -0.013004 |

These validation results support a nuanced claim. Minimal definitions help
TF-IDF clearly and improve the F1 operating point for Frozen Qwen, but they do
not improve E5 F1. Rich guidance is usually worse than the minimal definition.
No significance claim should be made until the registered paired bootstrap
analysis is run after the cloud methods are complete.

DCWT is a useful negative result: the registered kernel-ridge generator does
not beat the nearest-description-weight control (for example, D pair F1
0.145314 versus 0.153412). With only eleven source tasks, a learned global
description-to-weight map adds little beyond nearest semantic transfer.

All 36 local Level 2 fold results per method passed the final audit. Frozen
Qwen used the exact raw cache in read-only mode with zero cache misses and zero
new inference. Local summary hashes are:

- TF-IDF: `dd3f31d1c38f1d893bfb0ffd23fac01886e3708817cb196872bbfb5bdbcceb4e`;
- E5: `dba97a15be4f6fb07959783cb64748619dfdc3a55d16d55166a055eb8bc904d0`;
- Frozen Qwen zero-shot:
  `9c719b5d3c9346bf268c5812e0a582bbd1e1492842debb5f8d2c5c23514257bb`;
- DCWT: `bed71de424bbdfdc12b593265006b06a45a107d0a6234c7f1f0bfd390ae9981f`;
- Level 1:
  `37e7746577e4f5bbf9765196cab224e81f46ddb7da289c355fe17a111de6b76e`.

## Revised formal cloud plan v2

The old 26-scope/297-result v1 launch plan remains revoked. Formal v2 contains
only mandatory Level 2 and Level 4 validation:

| Quantity | Formal v2 |
|---|---:|
| Unique training scopes | 15 |
| Level 2 fold-conditions per method | 24 (12 × N/D) |
| Level 4 fold-conditions per method | 3 (3 × D) |
| Fold-condition results per method | 27 |
| Results across three cloud methods | 81 |
| Candidate-training jobs | 90 |
| Seen-only selection jobs | 30 |
| Selected-checkpoint scoring jobs | 30 |
| Total dependency-closed trainable jobs | 150 |

The worker partition is isolated:

- `worker-qlora-a`: 8 scopes, 40 jobs;
- `worker-qlora-b`: 7 scopes, 35 jobs; and
- `worker-distil-frozen`: 15 DistilBERT scopes (75 jobs), followed by 15
  Frozen-Qwen few-shot scopes.

The offline union audit reports 150/150 jobs, 81/81 result assignments, zero
missing jobs, zero duplicate jobs, `failure_count=0` and
`test_contract_count=0`. Safety-config SHA-256:
`0f7b64f0532c6bb1506a511e172634eeaf219478ef0a77d360fe9660ac96acc0`.

The final merge audit is also implemented. It will accept cloud output only if
all candidate checkpoints, selections, eight-part score shards, immutable sync
units, result payloads and SHA-256 identities verify across all three local
worker copies. Level 2 results must contain both L2-S and L2-E views.

## Runtime, storage and cost forecast

The new forecast uses the exact 204 full candidate contracts, 160 seen-only
selection contracts, 44 selected-checkpoint additions, 15 training scopes and
1,042 unique rendered validation texts. It applies the previously qualified
Secure RTX 4090 throughputs and a 25% operational margin.

| Component | Planned GPU hours | Cost at $0.75/h |
|---|---:|---:|
| Frozen-Qwen few-shot | 5.55 | $4.16 |
| DistilBERT | 1.41 | $1.06 |
| QLoRA | 18.12 | $13.59 |
| **Total** | **25.08** | **$18.81** |

With the registered three-worker split, the longest worker is estimated at
9.65 hours, so ideal parallel wall time is about 9.65 hours including margin.
This is capacity planning, not a guaranteed duration or scientific result.
The 100 GiB isolated volume per worker remains more than sufficient under the
registered 30 GiB growth ceiling and 20 GiB final-headroom requirement.

## Verification and release boundary

- Repository-wide test suite: **441 passed** after the final source and
  documentation update.
- Local completion audit: pass; 12/12 folds for each local Level 2 method.
- Cloud graph/wrapper focused suites: pass.
- Failures, resume conflicts and non-finite values: zero.
- Third sentiment per aspect: prohibited; capped-two remains enforced.
- Official-test access and test contracts: zero.
- No cloud resource was started in this phase.

Formal execution is deliberately not released yet. It should wait for the
industrial-supervisor response (unless the user explicitly decides otherwise),
then requires the reviewed and pushed final commit, three actual isolated
workers, fresh per-host startup/CUDA/thermal/storage checks, a live verified
local sync receiver, and confirmation that no unneeded Pod or volume is
billing. Any mismatch, OOM, non-finite value, collapse, third sentiment,
thermal slowdown, official-test evidence or replication failure stops the
campaign rather than changing the scientific protocol.
