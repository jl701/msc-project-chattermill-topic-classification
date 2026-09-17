# Genuine two-stage taxonomy: cloud benchmark handoff

Date: 17 August 2026

Status: **the Secure-Cloud RTX 4090 replacement host is qualified; formal
validation has not yet been released**. Official test was not opened and is
not part of this handoff.

## Measured outcome

The three original execution/memory smokes passed on a Community RTX 3090.
The subsequent representative throughput benchmark also produced finite,
non-collapsed outputs with no OOM and `test_contract_count=0`.

The safe batch-2/4/6/8 run passed its independent audit:

- Frozen-Qwen few-shot: batch 8, 12.36 prompts/s;
- QLoRA scoring: batch 8, 25.43 prompts/s;
- QLoRA training at the formal length: 3.13 examples/s;
- DistilBERT: 367.12 training examples/s and 594.23 scoring prompts/s;
- peak sampled temperature 78 C, peak memory 6,776 MiB, and zero thermal,
  hardware-slowdown or power-brake samples.

An administrative extension to batches 12, 16 and 24 was then attempted
because batch 8 had substantial memory headroom. It did not improve throughput
and it triggered `SW Thermal Slowdown=Active` in 3 of 70 two-second telemetry
samples, with a peak of 84 C. There was no OOM, hardware slowdown, power brake,
non-finite loss or collapsed prediction. Nevertheless, repeated thermal
throttling is a preregistered stop condition, so the Pod was stopped and no
formal validation job was released.

Batch sizes above 8 are now rejected by the benchmark command. Batch 8 remains
the measured throughput optimum, but the full run stays blocked until the
thermal/power boundary is explicitly resolved and batch 8 is revalidated on
the selected host.

Evidence:

- safe benchmark SHA-256:
  `a3146832040a035935f836329b9e89abe29be6d5b35f46e45ec8a05f744d3418`;
- safe audit SHA-256:
  `6c830cd7ab55d384b12fada5d990958efeb02e2a51ee3c35270695b2d7ca2ec9`;
- extended benchmark SHA-256:
  `56f39159c50187542fde0e9193d3ab00481d7ca6e62e8afbf17ed9238a7ce550`;
- extended telemetry SHA-256:
  `fdc3747565178d717072bab493710a8339ad2337799c654b265ab0ea2046d2b1`.

The exact prompt-hash-deduplicated workload is 775,248 prompts for one frozen
arm and 1,900,608 scoring prompts plus 319,488 training examples for QLoRA.
Using the safe batch-8 measurements and a 25% operational margin gives a first
planning estimate of 21.77 h / $5.23 for Frozen-Qwen few-shot, 2.02 h / $0.48
for DistilBERT and 61.40 h / $14.74 for QLoRA on this $0.24/h host. These are
capacity estimates, not formal experimental results.

## RTX 4090 replacement-host gate

The next permitted action is a replacement-host gate on one Runpod Community
RTX 4090. The machine-readable plan is
`configs/experiments/taxonomy_two_stage_4090_hardware_gate_v1.json`.

Changing from RTX 3090 to RTX 4090 does not invalidate completed scientific
results because model revisions, prompts, data, folds and score semantics stay
unchanged. It does require fresh hardware evidence. Before any formal fold, the
4090 must pass the original three real-model smokes, the batch-2/4/6/8
formal-length throughput benchmark, a sustained batch-8 Frozen-Qwen scoring
soak and one complete 4,096-example QLoRA training soak. The scoring batch is
capped at 8; the rejected 12/16/24 probe will not be repeated.

Formal validation remains blocked unless every loss and score is finite,
`test_contract_count=0`, no OOM or resume conflict occurs, predictions remain
non-constant and the named NVIDIA telemetry contains zero thermal,
hardware-slowdown and power-brake events throughout both the benchmark and
sustained soak.

### Replacement-host result

The complete gate was run on Runpod Pod `zotpukvqqkcb05`, named
`fabsa-two-stage-4090-gate-l2-a01`. Runpod identified the allocated machine as
a **Secure Cloud** RTX 4090 in `EU-RO-1`, not as the preregistered Community
Cloud target. The distinction is retained in the result record rather than
silently relabelling the host. Runtime source was commit
`0c57911afd356b1794cce4e01452141f2b649e1a`; the remote artifact manifest was
verified locally with 18 files checked and zero mismatches.

All three real-model smokes passed. The representative benchmark and sustained
soak both passed their fail-closed audits with `failure_count=0` and
`test_contract_count=0`. The 789 sustained telemetry samples reached 100% GPU
utilisation, 6,930 MiB memory, 74 C and 451.91 W, with zero software-thermal,
hardware-thermal, hardware-slowdown or power-brake samples. The informational
software power-cap flag was active in 292 samples at the 450 W board limit;
this was not accompanied by thermal or hardware slowdown.

Frozen-Qwen measured 26.61 prompts/s at batch 6 and 26.53 prompts/s at batch 8.
Batch 8 then remained stable for 602 seconds and 15,744 prompts at 26.15
prompts/s. Because the registered short benchmark made batch 6 faster by
0.30%, the formal throughput setting is frozen at **batch 6**, while batch 8
remains verified as a safe upper bound. QLoRA is frozen at **batch 8**, with
57.33 scoring prompts/s; the full 4,096-example training soak completed at
5.60 examples/s with finite loss 1.0525. No scientific model, data, prompt or
score parameter changed through this administrative batch selection.

The measured 25%-margin capacity estimate is 10.12 h for Frozen-Qwen, 2.45 h
for DistilBERT and 31.46 h for QLoRA: approximately 44.02 GPU-hours in total.
At the observed Secure-host price of about $0.75/h, this is approximately
$33.02. These are planning estimates rather than formal experimental results.

The Pod was stopped after artifact retrieval and is no longer consuming GPU
compute. A Community RTX 4090 appeared in the catalogue at $0.34/GPU-h, but
two allocation attempts both failed with Runpod's resource-unavailable error.
It is therefore not qualified. If Community capacity becomes provisionable,
that newly allocated host must pass its own hardware gate before its formal
artifacts are accepted; evidence from the Secure host must not be presented as
Community-host evidence.

Machine-readable resolution:
`docs/experiments/taxonomy_two_stage_4090_hardware_gate_v1_result.json`.

## Upload boundary

Upload the repository commit containing the v2 two-stage implementation and
only these two data files to the directory supplied through `FABSA_DATA_DIR`:

| File | SHA-256 |
|---|---|
| `train.csv` | `3ffcc7407cc8077fe9efc5ca06532847589a59464ea929d7f23d1f732b9efc7c` |
| `validation.csv` | `3a32afda9c4ba1d59e76ee8fcfb480011c7178d8201d0025b254a40c2f81203c` |

Do not upload `test.csv`, any test-derived manifest, or any test-derived
scores, predictions, labels, metrics or thresholds.

Raw local result caches are not needed for the first gate. If the exact
Frozen-Qwen zero-shot cache is transferred later, keep it outside Git and
verify its declared SHA-256 before use.

## Environment setup

From the repository root:

```bash
python -m pip install -r requirements-taxonomy-cloud.txt
export PYTHONPATH="$PWD/src"
export FABSA_DATA_DIR=/absolute/path/to/train-and-validation-only
python scripts/build_taxonomy_two_stage_cloud_gate.py \
  --output outputs/experimental/taxonomy_two_stage_cloud_v1/cloud_gate_manifest.json
```

The manifest must report `status=pass`, `include_official_test=false`,
`test_contract_count=0`, 26 unique training scopes and three benchmark
commands. Run the commands exactly as recorded in its `benchmark.commands`
array, sequentially.

## Gate review

After the three commands finish, check:

- process exit status and artifact `status` are `pass`;
- every loss and probability is finite;
- CUDA peak memory, elapsed time and GPU telemetry are recorded;
- no OOM or repeated thermal throttling occurred;
- predictions did not collapse to all-positive or all-negative behaviour;
- each demonstration and training manifest remains train-only and excludes
  the held-out aspect; and
- test-contract count remains zero.

Do not release the full run merely because the commands completed. Use the
measured throughput to produce the final wall-time/cost estimate and freeze the
batch sizes. Batch-size changes are administrative only if all scientific
inputs, examples, prompt text, max length, checkpoint recipe and score
semantics remain unchanged.

## Full validation order after the thermal block is cleared

1. Frozen-Qwen few-shot, all registered L1-L4 folds and conditions.
2. Genuine two-stage DistilBERT, with learning rate selected independently
   inside each exact training scope from seen-aspect validation evidence.
3. Matched one-stage controls only where the strict exact-reuse audit rejects
   an old artifact.
4. Genuine two-stage QLoRA on `l2-a01` as the first formal scope.
5. Remaining L2 QLoRA scopes.
6. L1, L3 and L4 QLoRA.

At every boundary, stop on any failed or missing artifact, incompatible resume,
non-finite score, prediction collapse, OOM, repeated thermal throttling or L3
cross-condition identity mismatch.

Official test remains sealed after validation completes. Opening it requires a
separate, explicit post-validation freeze and user approval.
