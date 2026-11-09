# Frozen artifacts and validation replay audit — 9 November 2026

The serving experiment reuses the historical Qwen two-stage system and one
explicitly selected leave-one-aspect-out fold, `l2-a01` / `heldout-a01`. It does
not train a new model or use the later DeBERTa D/E implementation. The selected
adapter jointly learned presence and conditional sentiment for the eleven seen
aspects. It is not an ensemble of twelve aspect-specific adapters.

## Preparation performed

Run from the serving worktree:

```text
python scripts/prepare_serving_replay.py --all-folds
```

The command completed successfully. It reads the existing official train and
validation exports, frozen validation-era selections and checkpoint files. It
does not open the official test split, download files, load model weights into
PyTorch, or start a service. All generated files reside in the already ignored
`outputs/serving_20261109/` directory. Absolute machine paths, review text, gold
labels and demonstration text are private output contents, not tracked files.

| Check | Observed result |
|---|---|
| L2 selected artifacts | All 12 folds passed for both frozen few-shot and QLoRA: 24 successful checks, zero failures |
| QLoRA checkpoint trees | All 12 selected trees matched the SHA-256 values in the frozen inventory |
| Selections | File hashes and internal selection payload seals matched |
| Selected fold demonstrations | Reconstructed from eligible train rows; both demonstration hashes matched the frozen selection |
| Selected fold prompt code | Few-shot length-1024 and QLoRA length-384 contracts matched the frozen records |
| Base snapshot | Exact revision directory, tokenizer/config files and every indexed weight shard present; all three shard SHA-256 fingerprints recorded separately; no trusted upstream digest comparison or GPU load performed in this audit |
| Replay | 100 development plus 957 evaluation reviews, disjoint IDs, label-free request records |
| Official test | Not opened |

The checkpoint sources are the pre-existing `taxonomy_two_stage_formal_v2_r2`
backup's `worker-qlora-a` and `worker-qlora-b` roots. The few-shot selections are
from its `worker-distil-frozen` root. No backup or historical output was modified.
The precise private paths are recorded in `artifacts.json` and the file-level
provenance is anchored to the public inventory below.

## Frozen identities

- Inventory: `docs/experiments/taxonomy_final_test_v1_artifact_inventory.json`,
  file SHA-256 `814df702c3c9b952295bcf4c9ff247229d4bf01493b79f81e2741ed9d58954a4`.
- Preregistration: `configs/experiments/taxonomy_final_test_v1.json`,
  file SHA-256 `8109bebd7a32a83762efa1a31d2ac2110729ffeaf7a41a330a1503cbd13665ee`.
- Base model: `Qwen/Qwen3-4B-Instruct-2507`, revision
  `cdbee75f17c01a7cc42f958dc650907174af0554`.
- Selected QLoRA checkpoint: relative to its worker root,
  `checkpoints/qwen_candidate_pair_qlora/heldout-a01/lr-1e-5/7f8967bd8a16fca6`.
- Selected checkpoint tree SHA-256:
  `175b5e7a1e249ec66283c7c92f4788191a26b2bc2b8fc2f78dce05a84075269e`.
- Frozen taxonomy: `fabsa_aspect_descriptions_minimal_v2`; condition D uses
  `name_and_description` for all twelve candidate aspects. Held-out aspect:
  `Account management: Account access`.
- Adapter configuration: rank 4, alpha 8, dropout 0.05, targeting
  `q_proj`, `k_proj`, `v_proj`, `o_proj`. One shared adapter serves both tasks.

The selected adapter's base-model `revision` field is null in its PEFT config;
the required exact revision is supplied by the independently verified training
manifest and few-shot scope contract. A runtime must preserve that revision.

### Base-weight fingerprints

A subsequent read-only audit hashed every byte of the three safetensors shards,
totalling 8,044,982,000 bytes. Each is an ordinary file in the exact revision's
snapshot directory, not a symlink resolving to a content-addressed HF blob
filename. Consequently the hashes below identify the local bytes; they do not
establish agreement with a trusted upstream digest. No upstream metadata was
fetched. The separate private record is
`outputs/serving_20261109/base_weight_integrity.json`; `artifacts.json` was left
unchanged so running services retain their original configuration identity.

| Shard | Bytes | Local SHA-256 |
|---|---:|---|
| `model-00001-of-00003.safetensors` | 3,957,900,840 | `75311d91bb08cf0b882913da464a1e722a31fb44db35208663487efb7a3d8ed6` |
| `model-00002-of-00003.safetensors` | 3,987,450,520 | `0b48adbb1f60e901153d91907ba11ce63bd4b8b584482e730f48808d055dfba1` |
| `model-00003-of-00003.safetensors` | 99,630,640 | `7dd39ccca5e4de123c74c14af44c9bf2eb75df33b4614382af0134528e060d5d` |

| Variant | Presence threshold | Runner-up sentiment threshold | Prompt lengths |
|---|---:|---:|---|
| `few_shot` | 0.99999922514 | 0.99 | 1024 for both stages |
| `qlora` | 0.789099752903 | 0.402159124613 | 384 for both stages |
| `hybrid` | 0.99999922514 | 0.402159124613 | Presence 1024; sentiment 384 |

These are historical frozen per-fold thresholds. They are not fitted again on
the new replay. The runner-up decoder retains at most two sentiments per
selected aspect. Presence probabilities are restricted-label scores, not
calibrated guarantees of real-world correctness; the very high few-shot
threshold also makes precision and batch-equivalence checks material.

## Few-shot reconstruction

The official loader `load_official_fabsa_splits(data_dir, ("train",))` supplies
stable `train:<id>` row identities. `example_filtered_train_rows()` removes
every training review mentioning the held-out aspect, leaving 7,457 of 7,930
rows for this fold. `_assert_few_shot_contract()` then reconstructs the fixed
seed-13 demonstrations using the original selection code and compares hashes.

- Four presence examples: two Y and two N. Their combined hash is
  `48fe7ad26bbe051a281a069b62eb5dce811c7abfeee9c93b8b23d640923bed9f`.
- Three sentiment examples: one A/negative, one B/neutral, one C/positive.
  Their combined hash is
  `3f01b8dbe60b12bcf47335c51155f9b43c9ec2d7909aab8d1afddd998e580e1b`.

`artifacts.json` contains the original `TwoStageDemonstration` fields, ready for
reconstruction with that dataclass. No dynamic example retrieval is introduced.
The aspect-diversity preference applies within each answer class, not globally
across all demonstrations.

## Replay selection and limits

The official validation export has 1,057 reviews. Sampling uses only stable row
identity: sort by `sha256(salt + "|" + row_uid)`, then `row_uid`. The fixed salt is
`chattermill-serving-validation-replay-20261109-v1`. The first 100 reviews form
the engineering development replay; the remaining 957 form the frozen
engineering evaluation replay. Labels and model predictions do not affect this
selection. Both subsets were historically part of validation used for model
and threshold selection, so **neither is a newly blind test set**.

Development reviews may inform serving implementation, resource choices and
pressure sampling. The 957-review evaluation is for reporting after those
choices are fixed, not further threshold, prompt or model tuning. Results from
one fold cannot be reported as the historical twelve-fold held-out F1 mean.

Original length is measured as Unicode code-point count before tokenization or
truncation. Decile boundaries are computed on all validation review lengths
without labels. These are character-length deciles, not token-length deciles.
Every one of the ten length deciles is represented in the 100 development
reviews, with counts `9, 9, 14, 6, 10, 11, 7, 9, 13, 12`.

| Replay | Minimum | Median | P90 | P95 | P99 | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| Development, characters | 6 | 63 | 290.2 | 413.75 | 1045.45 | 1981 |
| Evaluation, characters | 3 | 60 | 271.8 | 407.8 | 656.6 | 1888 |

Pressure sampling should select by the label-free metadata or measured token
length, and preserve source `request_id` as `sample_id` when generating a new
request ID for each attempt. Report the workload's length mix and never count
repeated requests as independent labelled examples.

Private file contracts:

- `replay_{development,evaluation}.jsonl`: `{request_id, text}` only.
- `labels_{development,evaluation}.jsonl`:
  `{request_id, row_uid, pairs: [{aspect, sentiment}], aspects: [...]}`.
- `metadata_{development,evaluation}.jsonl`:
  `{request_id, row_uid, sampling_hash, character_count, length_decile, text_sha256}`.
- `replay_manifest.json`: source split, sampling rule, file hashes, ID hashes,
  counts, length distributions and the non-blind evaluation limitation.
- `artifact_verification.json`: the 24 frozen-artifact verification outcomes.
- `artifacts.json`: model/taxonomy/adapter identities, exact private paths,
  demonstrations, thresholds, provenance and replay manifests.

The preparation completed structural checks for exact request keys, nonempty
review strings, unique and disjoint IDs, matching replay/label/metadata order,
valid taxonomy and sentiment labels, and label-free metadata. The preparation
script passed Python compilation. The base-model runtime, numerical agreement,
end-to-end latency, concurrent service behaviour and monitoring are separate
checks; this artifact audit alone does not establish them.
