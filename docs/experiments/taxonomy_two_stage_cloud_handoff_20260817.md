# Genuine two-stage taxonomy: cloud benchmark handoff

Date: 17 August 2026

Status: **ready for the measured benchmark gate only**. Full validation jobs
remain blocked until the measured gate passes. Official test is not part of
this handoff.

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

## Full validation order after approval

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
