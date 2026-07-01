# Gemini-Generated Aspect Descriptions

Last updated: 2026-07-01

This note records Task 4 of the Gemini follow-up work: adding Gemini-generated aspect descriptions to the fixed held-out-aspect candidate-label prompt. The purpose is to test whether richer label semantics improve hosted LLM classification without changing the split, candidate IDs, JSON-schema output mode, or evaluation metrics.

## Research Question

The fixed held-out-aspect prompt previously listed only canonical labels:

```text
A1. Account management: Account access
A2. Company brand: Competitor
A3. Value: Discounts promotions
```

Task 4 asks whether generated natural-language descriptions help the model interpret these candidate labels, especially for semantically ambiguous labels such as `Company brand: Competitor` and `Value: Discounts promotions`.

The experiment is deliberately scoped to the fixed three-aspect held-out protocol. It is not Gemini LOAO robustness evidence.

## Implementation

Code changes:

- Added `indexed_generated_descriptions` and `indexed_conservative_generated_descriptions` prompt variants in `src/msc_project/llm/candidate_label.py`.
- Added `--aspect-descriptions-json` support in `scripts/run_gemini_heldout_aspect.py`.
- The runner stores `aspect_descriptions_json` and the loaded descriptions in `summary.json`.
- Added prompt-construction unit tests in `tests/test_llm_candidate_label.py`.

Tracked description configs:

- `configs/gemini_aspect_descriptions_fixed_heldout.json`
- `configs/gemini_aspect_descriptions_decision_boundary_heldout.json`

Both configs were generated with `vertex_ai/gemini-2.5-pro` using only the canonical held-out aspect names and general label semantics. No validation or test review text was used.

The generation calls reported reasoning-token diagnostics:

| Config | Prompt Tokens | Completion Tokens | Reasoning Tokens | Total Tokens |
| --- | ---: | ---: | ---: | ---: |
| Label-only descriptions | 117 | 567 | 456 | 684 |
| Decision-boundary descriptions | 132 | 826 | 701 | 958 |

## Validation Selection Rule

Prompt variants were selected on validation pair samples F1. Pair micro F1 and pair macro F1 were used as supporting diagnostics. Test results were not used to select the prompt variant.

For Flash-Lite, the decision-boundary descriptions won validation pair samples F1, but the label-only descriptions generalised better on test. This split sensitivity is a useful finding, so both are recorded below.

For Flash, label-only descriptions improved validation, so they were evaluated on test. They did not improve test pair samples F1.

For Pro, a 50-row validation diagnostic showed no primary-metric gain, so no full Pro description run was performed.

## Commands

Dry-run check:

```powershell
python .\scripts\run_gemini_heldout_aspect.py --output-dir .\outputs\llm\gemini_generated_descriptions_dry_run_check --split validation --limit 2 --prompt-variant indexed_generated_descriptions --aspect-descriptions-json .\configs\gemini_aspect_descriptions_fixed_heldout.json --dry-run
```

Flash-Lite validation sweep:

```powershell
python .\scripts\run_gemini_heldout_aspect.py --output-dir .\outputs\llm\gemini_candidate_label_20260701_desc_flash_lite_validation_full --model vertex_ai/gemini-2.5-flash-lite --split validation --limit 10000 --prompt-variant indexed_generated_descriptions --prompt-variant indexed_conservative_generated_descriptions --aspect-descriptions-json .\configs\gemini_aspect_descriptions_fixed_heldout.json --response-format json_schema --response-format-fallback --max-tokens 2048 --input-cost-per-1m 0.10 --output-cost-per-1m 0.40
```

Flash-Lite decision-boundary validation:

```powershell
python .\scripts\run_gemini_heldout_aspect.py --output-dir .\outputs\llm\gemini_candidate_label_20260701_desc_boundary_flash_lite_validation_full --model vertex_ai/gemini-2.5-flash-lite --split validation --limit 10000 --prompt-variant indexed_generated_descriptions --aspect-descriptions-json .\configs\gemini_aspect_descriptions_decision_boundary_heldout.json --response-format json_schema --response-format-fallback --max-tokens 2048 --input-cost-per-1m 0.10 --output-cost-per-1m 0.40
```

Flash-Lite selected test runs:

```powershell
python .\scripts\run_gemini_heldout_aspect.py --output-dir .\outputs\llm\gemini_candidate_label_20260701_desc_flash_lite_test_full --model vertex_ai/gemini-2.5-flash-lite --split test --limit 10000 --prompt-variant indexed_generated_descriptions --aspect-descriptions-json .\configs\gemini_aspect_descriptions_fixed_heldout.json --response-format json_schema --response-format-fallback --max-tokens 2048 --input-cost-per-1m 0.10 --output-cost-per-1m 0.40

python .\scripts\run_gemini_heldout_aspect.py --output-dir .\outputs\llm\gemini_candidate_label_20260701_desc_boundary_flash_lite_test_full --model vertex_ai/gemini-2.5-flash-lite --split test --limit 10000 --prompt-variant indexed_generated_descriptions --aspect-descriptions-json .\configs\gemini_aspect_descriptions_decision_boundary_heldout.json --response-format json_schema --response-format-fallback --max-tokens 2048 --input-cost-per-1m 0.10 --output-cost-per-1m 0.40
```

Flash validation/test:

```powershell
python .\scripts\run_gemini_heldout_aspect.py --output-dir .\outputs\llm\gemini_candidate_label_20260701_desc_flash_validation_full --model vertex_ai/gemini-2.5-flash --split validation --limit 10000 --prompt-variant indexed_generated_descriptions --aspect-descriptions-json .\configs\gemini_aspect_descriptions_fixed_heldout.json --response-format json_schema --response-format-fallback --max-tokens 2048 --input-cost-per-1m 0.30 --output-cost-per-1m 2.50

python .\scripts\run_gemini_heldout_aspect.py --output-dir .\outputs\llm\gemini_candidate_label_20260701_desc_flash_test_full --model vertex_ai/gemini-2.5-flash --split test --limit 10000 --prompt-variant indexed_generated_descriptions --aspect-descriptions-json .\configs\gemini_aspect_descriptions_fixed_heldout.json --response-format json_schema --response-format-fallback --max-tokens 2048 --input-cost-per-1m 0.30 --output-cost-per-1m 2.50
```

Pro diagnostic:

```powershell
python .\scripts\run_gemini_heldout_aspect.py --output-dir .\outputs\llm\gemini_candidate_label_20260701_desc_pro_val50 --model vertex_ai/gemini-2.5-pro --split validation --limit 50 --sample --seed 13 --prompt-variant indexed_generated_descriptions --aspect-descriptions-json .\configs\gemini_aspect_descriptions_fixed_heldout.json --response-format json_schema --response-format-fallback --max-tokens 2048 --input-cost-per-1m 1.25 --output-cost-per-1m 10.00
```

## Flash-Lite Full Results

Validation:

| Prompt | Pair Samples F1 | Pair Micro F1 | Pair Macro F1 | Aspect Samples F1 | FP | FN | Empty Rows | Schema Valid | Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Indexed baseline | 0.5876 | 0.5973 | 0.4477 | 0.6783 | 99 | 83 | 14 | 1.0000 | $0.0075 |
| Label-only descriptions | 0.5896 | 0.6050 | 0.4565 | 0.7005 | 91 | 84 | 16 | 0.9953 | $0.0091 |
| Decision-boundary descriptions | 0.5986 | 0.6274 | 0.4705 | 0.6830 | 73 | 85 | 24 | 0.9906 | $0.0091 |
| Conservative label-only descriptions | 0.5558 | 0.6192 | 0.4805 | 0.6384 | 63 | 92 | 46 | 0.9953 | $0.0092 |

Test:

| Prompt | Pair Samples F1 | Pair Micro F1 | Pair Macro F1 | Aspect Samples F1 | FP | FN | Empty Rows | Schema Valid | Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Indexed baseline | 0.5516 | 0.5872 | 0.4876 | 0.6062 | 108 | 124 | 37 | 0.9964 | $0.0096 |
| Label-only descriptions | 0.5925 | 0.6263 | 0.5691 | 0.6625 | 97 | 113 | 31 | 0.9929 | $0.0118 |
| Decision-boundary descriptions | 0.5724 | 0.6199 | 0.5456 | 0.6340 | 85 | 121 | 45 | 0.9893 | $0.0119 |

The label-only description prompt gives the best observed Flash-Lite test result. Relative to the indexed baseline, it improves pair samples F1 by `+0.0409`, pair micro F1 by `+0.0391`, pair macro F1 by `+0.0814`, and aspect samples F1 by `+0.0563`. It also reduces false positives and false negatives by 11 each.

The decision-boundary prompt wins validation but is more conservative on test, reducing false positives while increasing empty predictions and false negatives. This is useful negative evidence: richer descriptions can help, but overly restrictive boundaries can harm recall.

## Flash Full Results

Validation:

| Prompt | Pair Samples F1 | Pair Micro F1 | Pair Macro F1 | Aspect Samples F1 | FP | FN | Empty Rows | Schema Valid | Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Indexed baseline | 0.6146 | 0.6446 | 0.4830 | 0.7129 | 89 | 72 | 25 | 0.9953 | $0.2352 |
| Label-only descriptions | 0.6363 | 0.6838 | 0.5291 | 0.7255 | 63 | 72 | 31 | 1.0000 | $0.2548 |

Test:

| Prompt | Pair Samples F1 | Pair Micro F1 | Pair Macro F1 | Aspect Samples F1 | FP | FN | Empty Rows | Schema Valid | Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Indexed baseline | 0.6071 | 0.6541 | 0.5547 | 0.6747 | 104 | 98 | 42 | 1.0000 | $0.2996 |
| Label-only descriptions | 0.5893 | 0.6555 | 0.5655 | 0.6676 | 72 | 113 | 61 | 1.0000 | $0.3390 |

Flash descriptions improved validation and reduced false positives on test, but they increased empty predictions and false negatives enough to reduce test pair samples F1. This suggests the description prompt shifts Flash toward a higher-precision, lower-recall operating point rather than a clear overall improvement.

## Pro 50-Row Diagnostic

The Pro diagnostic used the same sampled 50-row validation subset as earlier prompt sweeps.

| Prompt | Pair Samples F1 | Pair Micro F1 | Pair Macro F1 | Aspect Samples F1 | FP | FN | Empty Rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pro indexed baseline | 0.8000 | 0.8113 | 0.5985 | 0.8400 | 12 | 8 | 2 |
| Pro label-only descriptions | 0.7467 | 0.7921 | 0.6128 | 0.8067 | 10 | 11 | 6 |

The primary metric worsened on the Pro sample. A full Pro description run is therefore not justified by the observed validation signal.

## Interpretation For The Dissertation

This experiment is most useful as a label-semantics ablation, not as a new overall leaderboard headline.

The positive result is strongest for Flash-Lite: short Gemini-generated descriptions materially improve the cheap hosted baseline on test for both pair-level and aspect-level metrics, with a small cost increase. This supports the claim that low-cost hosted LLMs benefit from explicit label semantics in open-topic candidate-label settings.

The mixed result for Flash and Pro is equally important. Stronger Gemini models appear less consistently helped by descriptions: Flash improves validation but loses test pair samples F1, and Pro worsens on a 50-row validation diagnostic. Descriptions tend to reduce false positives but can increase empty predictions and false negatives. This gives the dissertation a more nuanced conclusion: label descriptions are a useful controllable prompt feature, but they are not monotonically beneficial across model capacity or prompt wording.

For thesis tables, the cleanest reporting is:

- include Flash-Lite label-only descriptions as the main positive Task 4 result;
- include Flash label-only descriptions as a mixed validation/test result;
- include decision-boundary and conservative variants as ablations showing precision-recall trade-offs;
- report the Pro 50-row diagnostic as the stopping rationale for not running full Pro descriptions;
- keep the local-to-Gemini cascade as the strongest fixed-split system result overall.

## Validation

Repository validation after the implementation and documentation updates:

| Check | Result |
| --- | --- |
| `python -m unittest discover -s tests` | 73 tests OK |
| `python -m compileall -q src scripts tests` | passed |
| Secret scan over tracked code/docs/configs | no matches |
| `git diff --check` | passed |

## Next Step

Task 4 is complete enough for thesis evidence. The next high-value experiments are:

1. Sampled Gemini LOAO diagnostic, if a small robustness signal is still needed.
2. Cascade uncertainty improvement using local selector score/margin features and existing Gemini predictions.
3. Qualitative error taxonomy, using Gemini Pro only as an assistant for category drafting and with manual review.
