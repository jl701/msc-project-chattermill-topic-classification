# QLoRA First Cloud Gate Handoff

Date: 2026-07-25

Status: code, telemetry, dependency lock and dry-run complete; no cloud
experiment has run.

## Purpose and stopping point

The next mainline action is not the complete QLoRA sweep. It is one measured,
seen-validation-only tuning scope:

- method: `qwen_candidate_pair_qlora`;
- outer scope: Level 1 fold `l1-a01`, whose training scope is
  `heldout-a01`;
- seed: `13`;
- registered candidates: the three frozen QLoRA learning-rate recipes;
- work: train, score seen-aspect validation, transfer one threshold for each
  candidate, then select the registered candidate;
- target job:
  `select-tuned-qwen_candidate_pair_qlora-heldout-a01`;
- exact job count: `10`; and
- official test: excluded.

This gate measures whether the rented GPU is technically compatible and
cost-efficient before admitting the remaining QLoRA validation work. It does
not produce a thesis result or authorise the official test.

## Frozen scientific contract

The gate delegates every model job to the existing dependency plan and
`run_taxonomy_generalisation.py`. The monitoring wrapper changes no model
input, prompt, label representation, pair identity, optimiser setting,
checkpoint rule, probability, threshold rule, or metric.

Required identities:

- scientific protocol SHA-256:
  `d7ccded514ac1cbccf337e496e039ac418698566be0c0ca0c21e18608cc40f85`;
- model: `Qwen/Qwen3-4B-Instruct-2507`;
- model revision:
  `cdbee75f17c01a7cc42f958dc650907174af0554`;
- description bundle SHA-256:
  `fcf546d227ad2ac52ccfa9682fc3685d2e396069ed911016f0bc12bf205f1367`;
- plan job count: `2,611`;
- guarded QLoRA through-validation job count: `398`; and
- first measured scope: `10` dependency-closed jobs.

The plan creation timestamp is administrative and may differ when the ignored
plan file is regenerated on the cloud machine. The scientific protocol hash,
job counts, commands, model registry and resources must match.

## Data transfer and test sealing

For this first gate, upload only `train.csv` and `validation.csv` to a private
cloud directory. Do not upload `test.csv`. This makes accidental test access
physically impossible during the benchmark.

Expected source files:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `train.csv` | 2,520,919 | `3ffcc7407cc8077fe9efc5ca06532847589a59464ea929d7f23d1f732b9efc7c` |
| `validation.csv` | 338,210 | `3a32afda9c4ba1d59e76ee8fcfb480011c7178d8201d0025b254a40c2f81203c` |

Raw rows, model weights, adapters, checkpoints, outputs and credentials remain
outside Git. The cloud data directory is supplied through `FABSA_DATA_DIR`.

## Reproducible environment

The tracked `requirements-taxonomy-cloud.txt` freezes the successfully tested
local package versions, including PyTorch `2.10.0+cu128`, Transformers
`4.57.6`, bitsandbytes `0.49.2`, PEFT `0.19.1` and TRL `1.6.0`.

Use a Linux CUDA image whose NVIDIA driver supports CUDA 12.8. Clone the latest
GitHub main and verify that
`scripts/benchmark_taxonomy_cloud_gate.py` is present before installing:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-taxonomy-cloud.txt
python -m pytest -q
nvidia-smi
```

Set the private data directory and verify the two file hashes before any model
run:

```bash
export FABSA_DATA_DIR=/root/autodl-tmp/FABSA
sha256sum "$FABSA_DATA_DIR/train.csv" "$FABSA_DATA_DIR/validation.csv"
test ! -e "$FABSA_DATA_DIR/test.csv"
```

## Cloud compatibility smoke

Run the bounded synthetic QLoRA smoke first. It downloads the pinned model if
necessary but opens no official split:

```bash
python scripts/smoke_taxonomy_generalisation.py \
  --method qwen_candidate_pair_qlora \
  --output-root outputs/experimental/cloud_qlora_smoke_v1 \
  --allow-download
```

The smoke must complete checkpoint reload, sharding and resume before the
measured gate proceeds.

## Generate and inspect the plan

The large plan is intentionally ignored by Git. Regenerate it from the tracked
frozen configs:

```bash
python scripts/build_taxonomy_cloud_plan.py \
  --output outputs/experimental/taxonomy_hybrid_execution_plan_cloud.json \
  --output-root outputs/experimental/taxonomy_generalisation_formal_v1 \
  --selection-dir outputs/experimental/taxonomy_generalisation_formal_v1/_parameter_selections \
  --shard-count 8
```

The printed plan must report `jobs_total: 2611` and
`formal_execution_blocked: false`. Inspect the exact first scope without
executing it:

```bash
python scripts/benchmark_taxonomy_cloud_gate.py \
  --plan outputs/experimental/taxonomy_hybrid_execution_plan_cloud.json \
  --dry-run
```

The dry-run must report `official_test_included: false`, ten jobs, and the
registered target job above.

## Execute the first measured scope

```bash
python scripts/benchmark_taxonomy_cloud_gate.py \
  --plan outputs/experimental/taxonomy_hybrid_execution_plan_cloud.json
```

The wrapper calls the guarded executor with `--stop-after 1` until the exact
parameter-selection target completes. Each call is resumable. The telemetry is
written atomically after each completed job to:

```text
outputs/experimental/taxonomy_generalisation_formal_v1/
  _runtime_telemetry/qwen_candidate_pair_qlora/first_validation_gate.json
```

The artifact records:

- exact plan hash, Git commit, Python/platform and GPU inventory;
- job ID, stage, command, start/end time and wall seconds;
- peak sampled device memory and GPU utilisation;
- newly persisted training pairs, validation score pairs and bytes;
- aggregate training time;
- aggregate scoring time and pairs per wall second;
- peak GPU memory, adapter/checkpoint size and score-output size;
- complete `10/10` telemetry coverage; and
- an explicit `official_test_included: false` field.

If a job fails, the wrapper stops immediately and retains the failed attempt.
Rerunning the same command uses the existing executor state and immutable
checkpoint/shard resume guards. It refuses to claim a measured gate when jobs
were previously completed without matching telemetry.

## Decision after the gate

Do not launch the remaining 388 QLoRA through-validation jobs immediately.
First copy back the small telemetry JSON and parameter-selection summary, then
record:

1. actual peak VRAM and whether the GPU has safe memory headroom;
2. mean candidate training time;
3. validation pairs per wall second, including model load;
4. adapter/checkpoint storage per candidate;
5. projected validation-only and full admitted QLoRA hours;
6. projected rental cost with a retry/storage reserve; and
7. whether the current GPU remains cheaper than an alternative after measured
   throughput, not advertised TFLOPS.

Only after that evidence is reviewed may the remaining QLoRA validation
selection be admitted. Official test remains sealed until every method,
threshold and comparison rule is frozen.
