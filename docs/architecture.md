# Code guide

The repository contains both reusable modelling code and the recorded runners
of a developing research project. Start with the modules below, not the oldest
script in the directory. Historical scripts remain available for provenance.

## Data flow and contracts

1. Load normalised review rows with stable identities and aspect–sentiment labels.
2. Construct a training fold by review filtering or held-out-supervision masking.
3. Build candidate-conditioned training prompts or features, then fit on training data.
4. Score the complete evaluation candidate grid.
5. Apply seen-only selected thresholds and the capped-two decoder.
6. Aggregate pair counts, report fold metrics and compute paired uncertainty.

Grid identities join scores by review, aspect and sentiment, not row position
alone. The policy bootstrap additionally requires identical review order across
all methods and folds. Its low-level numeric function cannot infer identities
from an array, so callers audit the joins before constructing the tensor.

## Module and test map

| Responsibility | Implementation | Tests |
|---|---|---|
| Normalised FABSA rows | [`data/fabsa.py`](../src/msc_project/data/fabsa.py) | [`test_fabsa_loader.py`](../tests/test_fabsa_loader.py) |
| Training-policy transformation | [`taxonomy_training_policy_sensitivity.py`](../src/msc_project/experiments/taxonomy_training_policy_sensitivity.py) | [`test_taxonomy_training_policy_sensitivity.py`](../tests/test_taxonomy_training_policy_sensitivity.py) |
| CPU candidate TF–IDF | [`candidate_tfidf.py`](../src/msc_project/baselines/candidate_tfidf.py) | [`test_candidate_tfidf_cpu.py`](../tests/test_candidate_tfidf_cpu.py) |
| Two-stage grid and runtime | [`taxonomy_two_stage_runtime.py`](../src/msc_project/experiments/taxonomy_two_stage_runtime.py) | [`test_taxonomy_two_stage.py`](../tests/test_taxonomy_two_stage.py) |
| QLoRA training manifests | [`taxonomy_two_stage_training.py`](../src/msc_project/experiments/taxonomy_two_stage_training.py) | [`test_taxonomy_two_stage_training.py`](../tests/test_taxonomy_two_stage_training.py) |
| Decoding and thresholds | [`taxonomy_two_stage.py`](../src/msc_project/experiments/taxonomy_two_stage.py) | [`test_taxonomy_two_stage.py`](../tests/test_taxonomy_two_stage.py) |
| Shared policy bootstrap | [`resampling.py`](../src/msc_project/evaluation/resampling.py) | [`test_resampling.py`](../tests/test_resampling.py) |
| Official evaluation boundary | [`taxonomy_final_test_analysis.py`](../src/msc_project/experiments/taxonomy_final_test_analysis.py) | [`test_taxonomy_final_test_analysis.py`](../tests/test_taxonomy_final_test_analysis.py) |
| Safe public walkthrough/results | [`demo.py`](../src/msc_project/demo.py), [`reporting.py`](../src/msc_project/reporting.py) | [`test_repository_interfaces.py`](../tests/test_repository_interfaces.py) |

## Entry points

`taxonomy-demo` is a synthetic mechanics walkthrough. `taxonomy-results` reads
aggregate CSVs. Neither is a replacement for an experiment runner.

The training-policy runner is
[`run_taxonomy_training_policy_sensitivity.py`](../scripts/run_taxonomy_training_policy_sensitivity.py).
The completed QLoRA arm uses
[`run_taxonomy_policy_qlora_completion.py`](../scripts/run_taxonomy_policy_qlora_completion.py)
and its registered configuration. Paired completion analysis is driven by
[`analyse_taxonomy_policy_qlora_completion.py`](../scripts/analyse_taxonomy_policy_qlora_completion.py)
with a source manifest. It checks all required folds before publishing tables.

The official-test runner has a separate release, score-sealing and reveal
contract. Repository setup and CI do not invoke its evaluation modes.

## Engineering boundaries

- Core installation has no PyTorch dependency. Neural dependencies are optional.
- Candidate TF–IDF is separated from the neural cross-encoder module. Public
  names in `unified_pair_scorers.py` remain aliases for older callers/artifacts.
- The shared validation bootstrap was extracted without changing valid-input
  arithmetic, seed or RNG call sequence. The frozen official implementation
  remains separate.
- Current public interfaces receive stricter lint/format checks. All source,
  scripts and tests receive the repository-wide correctness lint checks.
- Dated launch scripts and large orchestration modules remain research tooling.
  They are not presented as a production service with deployment SLOs.

Large-scale renaming of historical runners would create more migration risk
than value immediately before submission. Further decomposition should be
driven by a new use case and protected by the relevant regression tests.
