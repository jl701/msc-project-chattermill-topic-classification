# Reproduction guide

## Choose the right level

The public repository supports an offline mechanics demo and inspection of
saved aggregate results. Complete model/score-level reproduction additionally
requires artifacts not distributed in Git. A successful demo is not a rerun of
the dissertation experiments.

| Level | Inputs | Expected output |
|---|---|---|
| Mechanics walkthrough | Included synthetic examples | Filtering/masking example and decoded pairs |
| Aggregate inspection | Included reporting CSVs | Recorded official and validation headline tables |
| Implementation verification | Synthetic fixtures, development dependencies | Test/lint reports |
| Policy analysis reproduction | Author-held scored fold bundles, train/validation export, source manifest | Fold/aggregate tables and paired bootstrap evidence |
| Full experimental reproduction | Dataset, model revisions, prompts, adapters or training compute | New score artifacts from an explicitly recorded run |

## 1. Offline entry points

Follow the [README installation](../README.md#try-it-without-a-gpu), then:

```bash
taxonomy-demo
taxonomy-results
taxonomy-demo --output outputs/demo.json
taxonomy-results --output outputs/headline-results.md
```

The output path must be new. Running again against an existing file returns an
error rather than overwriting evidence. `taxonomy-results --repo-root PATH`
works when the current directory is outside the checkout. The package wheel
contains the demo code, not the repository's aggregate files.

## 2. Verify the implementation

The [contributor guide](../CONTRIBUTING.md) describes the pinned lightweight
environment, full-suite dependencies and quality commands. Tests use synthetic
data and tiny fake neural models where needed. They do not retrain the reported
models or open the author's official-test label vault.

## 3. Recompute the completed policy analysis

The entry point is manifest-driven:

```bash
python scripts/analyse_taxonomy_policy_qlora_completion.py --help
python scripts/analyse_taxonomy_policy_qlora_completion.py inventory --manifest PATH_TO_SOURCE_MANIFEST
python scripts/analyse_taxonomy_policy_qlora_completion.py analyse --manifest PATH_TO_SOURCE_MANIFEST --output NEW_ANALYSIS_DIRECTORY
```

The source manifest must identify the few-shot, filtered-QLoRA and masked-QLoRA
bundles plus the train/validation export. Inventory checks must pass before
analysis. The runner requires the complete registered fold/condition set and
validates source identity. An aggregate CSV cannot reconstruct the row-level
paired bootstrap. Request the underlying shareable artifacts from the author
subject to data rights, or generate a new set using the registered protocol.

## 4. Training and scoring

Inspect commands before allocating compute:

```bash
python scripts/run_taxonomy_training_policy_sensitivity.py --help
python scripts/run_taxonomy_policy_qlora_completion.py --help
```

The QLoRA completion runner requires an explicit registered configuration,
data directory and new output root. It exposes plan, preparation, candidate
training, selection and scoring phases. Its configuration locks the original
learning-rate candidates, budgets, allowed partitions and canonical N/D score
reuse. Do not change these and call the outcome a reproduction of the same run.

Original cloud dependencies are recorded in
[`requirements-taxonomy-cloud.txt`](../requirements-taxonomy-cloud.txt). They are an
execution record with CUDA-specific packages, not a universal installation
recipe. The current `pyproject.toml` separates CPU, neural and QLoRA dependencies.
Model access and compatible CUDA drivers are separate prerequisites.

The data loader's expected FABSA CSV export has training, validation and test
partitions of 7,930, 1,057 and 1,587 reviews in the recorded study. Verify schema,
row identities and the registered dataset checks before a new run. Respect
the dataset publisher's terms and do not commit private exports.

## 5. Preserve the historical comparison

Use the [history index](history.md) for execution commits. The official result
uses train-only checkpoints, frozen demonstrations, thresholds and decoder.
The final-test runner's scoring/reveal workflow is not part of the quick start
and is not invoked by CI. New model development must not tune against those
official outcomes or silently replace the existing result.

For the later policy extension, identical seen prompts share canonical scores
within each registered N/D contract. Both components of a reported composition
must come from the same policy. Few-shot demonstration identity is recorded,
not assumed equal merely because the selection rule is the same.

## Known portability boundaries

Some historical orchestration and reporting scripts contain author-machine
paths. They are retained as execution records and may require explicit path
arguments or adapters on another machine. The CPU entry points and maintained
quality commands are portable. The wheel is not a self-contained distribution
of all experiment scripts, resources, data and checkpoints.
