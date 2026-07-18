# Unified Candidate-Pair LOAO Experiment v1

## Status

The protocol was written locally on 17 July 2026 before executing any new model or inspecting any new validation/test result. It was not committed to version control until 18 July, after pilot execution had begun, so it is an internally time-stamped prospective protocol rather than an externally immutable preregistration.

This is an isolated exploratory experiment requested after the dissertation design lock. It does not modify the frozen main benchmark, headline result tables, report narrative, or thesis. Admission requires explicit user review and approval after the results are complete.

The machine-readable authority is `configs/experiments/loao_unified_candidate_pair_experimental_v1.json`. The frozen taxonomy descriptions are in `configs/experiments/fabsa_aspect_descriptions_v1.json`.

## Why the first four-state idea was rejected before execution

A read-only audit found 106 review–aspect instances with more than one annotated sentiment: 73 train, 12 validation, and 21 test. A single `absent/negative/neutral/positive` target would therefore discard valid pair labels. No model was run under that invalid assumption.

The unified task is instead:

```text
(review, candidate aspect definition, candidate sentiment) -> applicable / not applicable
```

Each held-out candidate is scored independently for negative, neutral, and positive. Any subset may be emitted; emitting none means absent. This preserves the authoritative pair-set representation.

## Fixed comparison

Every model uses the same outer folds, `example_filtered` training construction, all official validation/test rows, singleton held-out candidate, projected gold pair set, empty-prediction option, 4,096-pair training budget, seed, threshold selection, and metric implementation.

Two data variants isolate the proposed package:

- Matched control: canonical name plus sentiment; four seeded random non-gold pair negatives per positive.
- Enhanced: frozen definition/cues/boundary plus sentiment definition; both wrong sentiments for the true aspect, one same-parent-or-semantic hard aspect with the same sentiment, and one random absent aspect with the same sentiment.

Both are deterministically reduced to 2,048 positive and 2,048 negative training pairs by aspect-balanced round-robin sampling. This deliberately leaves some local-model data unused so that TF–IDF, DistilBERT, and Qwen receive the same pair budget on the 8 GB machine.

The comparison estimates the effect of the complete unified package; it does not separately identify description, hard-negative, and balancing effects. If the package succeeds, those ablations would require a later registration.

## Selection and leakage boundary

The experiment retains the frozen primary benchmark's target-calibrated cold-start information regime so that local-model deltas can be compared with the historical TF–IDF and DistilBERT rows. For each fold, the target validation labels may select a threshold and, for DistilBERT, an epoch. Test labels never select a candidate text, negative policy, training budget, model configuration, checkpoint, or threshold.

This is not a strict zero-label calibration result. A later strict-transfer experiment would need protocol-matched controls and cannot reuse the historical local headline values as unconditional baselines.

## Execution gates

The three validation-only pilot folds are:

1. `Company brand: Competitor` — difficult semantic boundary;
2. `Company brand: General satisfaction` — high-prevalence broad label;
3. `Staff support: Email` — rare specific label.

For a local model to proceed to the twelve-fold test confirmation, enhanced must beat its matched control by at least `+0.02` mean validation pair micro-F1, win at least two of three folds, and retain at least half of control recall in every fold.

Frozen candidate-wise Qwen is evaluated before QLoRA. QLoRA proceeds only if that control is not more than `0.01` below historical JSON zero-shot mean on the same pilot and reduces false-positive rows by at least 15%. QLoRA must then add at least `+0.02`, win at least two folds, and retain at least half of frozen-control recall. Failure stops that branch without test evaluation.

Full confirmation success requires at least `+0.02` mean test pair micro-F1 over the protocol-matched control and wins on at least 8 of 12 aspects. Historical headline deltas are context, not a substitute for the matched comparison.

## Isolation and reporting

All manifests, predictions, logits, adapters, checkpoints, and summaries go under ignored `outputs/experimental/loao_unified_candidate_pair_v1/`. The final experimental note will report negative results as well as positive ones, all commands, runtime, package/hardware state, limitations, and gate decisions.

Until user approval, do not edit:

- `docs/thesis_result_tables.md`;
- `docs/dissertation_experiment_design_lock_2026_07_16.md`;
- any file under `thesis/`.

## Execution outcome (18 July 2026)

The local branches failed their registered validation-only pilot gate and were stopped. Frozen candidate-pair Qwen passed its pilot gate but did not improve full-test F1 relative to the historical JSON-prompt Qwen result. QLoRA passed both its pilot and the registered twelve-fold confirmation gate.

The primary matched result is therefore:

```text
frozen candidate-pair Qwen mean test pair micro-F1: 0.337804
QLoRA candidate-pair Qwen mean test pair micro-F1: 0.483158
paired mean delta:                              +0.145355
relative change of fold means:                    +43.03%
fold wins / ties / losses:                       12 / 0 / 0
```

This establishes that improvement was possible under the present target-calibrated LOAO regime. It does not establish strict zero-label transfer, nor that every model family benefits from the same semantic-description and hard-negative package.

### Gate and comparison summary

| Branch and comparison | Stage/split | Reference F1 | Candidate F1 | Delta | Wins | Decision |
|---|---:|---:|---:|---:|---:|---|
| Corrected TF-IDF enhanced vs matched control | 3-fold pilot/validation | 0.298484 | 0.235858 | -0.062626 | 1/3 | Gate failed; no twelve-fold test run |
| DistilBERT enhanced vs matched control | 3-fold pilot/validation | 0.334766 | 0.226227 | -0.108539 | 1/3 | Gate failed; no twelve-fold test run |
| Frozen candidate-pair Qwen vs historical JSON Qwen | 12-fold/test | 0.337788 | 0.337804 | +0.000016 | 6/12 | Descriptive context only; protocols are not matched |
| QLoRA vs frozen candidate-pair Qwen | 12-fold/test | 0.337804 | 0.483158 | **+0.145355** | **12/12** | Full registered gate passed |

The first TF-IDF pilot was invalidated after an implementation audit found that its engineered feature set did not exactly match the registered six features. The corrected TF-IDF pilot above was rerun from the frozen registration. The DistilBERT part of the earlier local run was unaffected by that feature mismatch.

Frozen candidate-pair Qwen first passed the registered three-fold pilot: mean validation F1 was 0.353691 versus 0.287205 for historical JSON-prompt Qwen, while false-positive rows per 100 fell by 52.30%. QLoRA then improved over the frozen candidate-pair control by 0.104329 on the pilot and won all three folds. These pilot results authorised, but are not substitutes for, the twelve-fold test confirmation.

### Twelve-fold QLoRA confirmation

The QLoRA and frozen rows below use identical candidate texts, validation/test rows, score interpretation, threshold-selection rule, metric code, and held-out folds. Only the QLoRA adapter differs.

| Held-out aspect | Frozen F1 | QLoRA F1 | Absolute delta |
|---|---:|---:|---:|
| Account access | 0.292011 | 0.464000 | +0.171989 |
| Competitor | 0.192053 | 0.222222 | +0.030169 |
| General satisfaction | 0.550158 | 0.576080 | +0.025922 |
| Reviews | 0.080844 | 0.177515 | +0.096671 |
| Speed | 0.523333 | 0.722222 | +0.198889 |
| App website | 0.428097 | 0.499316 | +0.071219 |
| Ease of use | 0.432990 | 0.630769 | +0.197780 |
| Attitude of staff | 0.493554 | 0.643979 | +0.150425 |
| Email | 0.235294 | 0.400000 | +0.164706 |
| Phone | 0.215488 | 0.431373 | +0.215884 |
| Discounts promotions | 0.212245 | 0.416667 | +0.204422 |
| Price/value for money | 0.397576 | 0.613757 | +0.216181 |
| **Unweighted fold mean** | **0.337804** | **0.483158** | **+0.145355** |

The paired median delta was +0.168347. A fold-resampling bootstrap gave a 95% interval of [0.105157, 0.181727], and the exact sign-flip value was 0.000488. These are descriptive uncertainty diagnostics because the folds reuse much of the same underlying dataset and should not be treated as twelve independent datasets.

### What changed in the errors

| Mean test diagnostic | Frozen Qwen | QLoRA | Change |
|---|---:|---:|---:|
| Pair micro-precision | 0.244003 | 0.501423 | +0.257420 |
| Pair micro-recall | 0.651851 | 0.499727 | -0.152124 |
| Presence F1 | 0.451565 | 0.500890 | +0.049325 |
| False-positive rows per 100 | 14.287965 | 7.146608 | -7.141357 (about -50%) |
| False-negative rows per 100 | 4.584121 | 6.070153 | +1.486032 |
| Conditional sentiment macro-F1 | 0.633782 | 0.643085 | +0.009303 |

The F1 gain is primarily a precision and boundary-control gain: QLoRA roughly halved false-positive rows, at the cost of lower recall and more false negatives. Conditional sentiment classification changed little. The result therefore supports the hypothesis that fine-tuning helped Qwen learn when the candidate claim should *not* be emitted, rather than merely improving polarity recognition.

Frozen candidate-pair Qwen also reduced false-positive rows relative to historical JSON-prompt Qwen (14.29 versus 34.42 per 100), but its recall was lower (0.652 versus 0.818) and full-test F1 was effectively unchanged. Because prompt, output space, and calibration differ, this historical comparison cannot isolate the cause of either change.

## Post-hoc duplicate-text sensitivity

The official split contains normalised review texts that also occur in the permitted training rows. A saved-score-only post-hoc analysis removed those rows independently within each fold, reselected each model's threshold on the retained validation rows, and evaluated the retained test rows.

```text
frozen retained-test mean F1: 0.347040
QLoRA retained-test mean F1:  0.489056
paired mean delta:            +0.142016
wins / ties / losses:         12 / 0 / 0
```

The near-identical delta indicates that duplicate-text rows do not explain the observed gain. This check was designed after seeing the main result, so it is descriptive sensitivity analysis only; no confirmatory inference is attached to it.

## Runtime and environment

- Base model: `Qwen/Qwen3-4B-Instruct-2507`.
- GPU: NVIDIA GeForce RTX 5050 Laptop GPU.
- Packages recorded in the run manifest: PyTorch 2.10.0+cu128, Transformers 4.57.6, PEFT 0.19.1, bitsandbytes 0.49.2, Accelerate 1.14.0.
- Twelve adapters: 10.63 recorded adapter-training wall-clock hours in total; mean 53.14 minutes per fold (range 50.30–58.07 minutes).
- QLoRA validation and test scoring: 3.75 recorded hours; training plus scoring: 14.38 hours.
- Frozen validation and test scoring: 3.36 recorded hours.
- Maximum recorded per-fold CUDA allocation: 6.05 GiB.

The first full QLoRA process accumulated GPU memory across completed folds and was stopped safely before training fold six. A garbage-collection and CUDA-cache release was added between folds, and execution resumed from verified manifests and adapters. Completed folds were not rerun. A final `--resume` audit deep-validated all twelve fold identities, manifests, adapters, row IDs, texts, gold labels, and scores before returning without model execution.

## Exact execution records

The run manifests under each ignored output directory are authoritative. The principal commands were:

```powershell
$env:PYTHONPATH='src'

python scripts/run_unified_candidate_pair_loao.py --stage pilot --model tfidf --resume --output-dir outputs/experimental/loao_unified_candidate_pair_v1/local_pilot_tfidf_prereg_corrected_20260718

python scripts/run_unified_candidate_pair_loao.py --stage pilot --model tfidf --model distilbert --variant control --variant enhanced --output-dir outputs/experimental/loao_unified_candidate_pair_v1/local_pilot_preregistered_20260717

python scripts/run_qwen_unified_candidate_pair_loao.py --mode frozen --stage pilot --variant enhanced --eval-batch-size 8 --output-dir outputs/experimental/loao_unified_candidate_pair_v1/qwen_frozen_pilot_preregistered_20260717

python scripts/run_qwen_unified_candidate_pair_loao.py --mode qlora --stage pilot --variant enhanced --frozen-summary outputs/experimental/loao_unified_candidate_pair_v1/qwen_frozen_pilot_preregistered_20260717/summary.json --eval-batch-size 4 --resume --output-dir outputs/experimental/loao_unified_candidate_pair_v1/qwen_qlora_pilot_preregistered_20260717

python scripts/run_qwen_unified_candidate_pair_full.py --mode frozen --pilot-frozen-root outputs/experimental/loao_unified_candidate_pair_v1/qwen_frozen_pilot_preregistered_20260717 --eval-batch-size 6 --resume --output-dir outputs/experimental/loao_unified_candidate_pair_v1/qwen_frozen_full_preregistered_20260718

python scripts/run_qwen_unified_candidate_pair_full.py --mode qlora --pilot-frozen-root outputs/experimental/loao_unified_candidate_pair_v1/qwen_frozen_pilot_preregistered_20260717 --pilot-qlora-root outputs/experimental/loao_unified_candidate_pair_v1/qwen_qlora_pilot_preregistered_20260717 --frozen-full-summary outputs/experimental/loao_unified_candidate_pair_v1/qwen_frozen_full_preregistered_20260718/summary.json --eval-batch-size 6 --resume --output-dir outputs/experimental/loao_unified_candidate_pair_v1/qwen_qlora_full_preregistered_20260718

python scripts/analyse_loao_duplicate_text_sensitivity.py --analysis-mode full-comparison --frozen-full-root outputs/experimental/loao_unified_candidate_pair_v1/qwen_frozen_full_preregistered_20260718 --frozen-pilot-root outputs/experimental/loao_unified_candidate_pair_v1/qwen_frozen_pilot_preregistered_20260717 --qlora-full-root outputs/experimental/loao_unified_candidate_pair_v1/qwen_qlora_full_preregistered_20260718 --qlora-pilot-root outputs/experimental/loao_unified_candidate_pair_v1/qwen_qlora_pilot_preregistered_20260717 --output-dir outputs/experimental/loao_duplicate_text_sensitivity_v1/full_comparison_20260718

python scripts/summarise_unified_candidate_pair_results.py --tfidf-pilot-summary outputs/experimental/loao_unified_candidate_pair_v1/local_pilot_tfidf_prereg_corrected_20260718/summary.json --distilbert-pilot-summary outputs/experimental/loao_unified_candidate_pair_v1/local_pilot_preregistered_20260717/summary.json --frozen-full-summary outputs/experimental/loao_unified_candidate_pair_v1/qwen_frozen_full_preregistered_20260718/summary.json --historical-qwen-summary outputs/llm/qwen_loao_heldout_aspect_all_rows_test_20260701/summary.json --qlora-full-summary outputs/experimental/loao_unified_candidate_pair_v1/qwen_qlora_full_preregistered_20260718/summary.json --output-dir outputs/experimental/loao_unified_candidate_pair_v1/final_improvement_summary_20260718
```

Final consolidated outputs are:

- `outputs/experimental/loao_unified_candidate_pair_v1/final_improvement_summary_20260718/unified_improvement_summary.json`;
- `outputs/experimental/loao_unified_candidate_pair_v1/final_improvement_summary_20260718/unified_improvement_per_fold.csv`;
- `outputs/experimental/loao_duplicate_text_sensitivity_v1/full_comparison_20260718/summary.json`.

All model outputs, pair scores, predictions, adapters, and checkpoints remain ignored local artifacts. Only protocol, implementation, tests, configuration, and this experimental record are tracked.

## Interpretation, limitations, and admission decision

Observed fact: the registered QLoRA branch improved the protocol-matched frozen candidate-pair control on every held-out aspect and exceeded both full-confirmation gates. Observed fact: the same enhanced candidate representation and negative policy did not improve TF-IDF or DistilBERT in their registered pilot.

The most plausible interpretation is that the shared candidate-pair structure created a learnable interface, while QLoRA had enough capacity to use the definitions and contrastive negatives to sharpen aspect boundaries. The local models could not exploit that representation under the same 4,096-pair budget. This is an interpretation, not an isolated causal ablation: description enrichment, negative construction, balancing, and QLoRA were not separately crossed.

Additional limitations are:

- the prospective protocol was locally recorded before execution but only committed after pilots had begun;
- one random seed and one QLoRA hyperparameter setting;
- target validation labels select a threshold, so this is target-calibrated LOAO rather than strict zero-label deployment;
- the twelve folds share most training data and are not independent datasets;
- the official split's duplicate texts were handled only in a post-hoc sensitivity check;
- model-specific token limits differ (DistilBERT 256, Qwen 384), even though task, pair budget, folds, selection rule, and metrics are unified;
- frozen Qwen probabilities often saturate near one, making its selected thresholds numerically extreme;
- the full comparison estimates the complete QLoRA candidate-pair package, not the separate contribution of descriptions, hard negatives, balancing, or adapter training.

Status remains `experimental_only_pending_user_approval`. No dissertation chapter, frozen headline table, main experiment log, or main-story document has been updated with these results. A later user decision may admit the result, request a stricter zero-label replication or ablation, or leave it permanently outside the dissertation narrative.
