# Strict TF-IDF to Frozen Qwen Router — Aspect-Conditioned LOAO

Date: 23 July 2026
Status: complete negative optional experiment; not admitted to the active baseline table

## Question

This experiment asked whether a selective router could improve the strict
train-vocabulary-only character TF-IDF system by replacing locally uncertain rows
with the active frozen candidate-pair Qwen endpoint. Both endpoints predict
sentiment conditional on the supplied aspect.

The experiment was registered before execution in
`configs/experiments/loao_strict_tfidf_aspect_qwen_router_v1.json`. It reused saved
predictions only: there was no training and no new model inference.

## Protocol

- Twelve-fold, all-row, singleton-candidate LOAO.
- Target-calibrated information regime.
- Local endpoint: strict TF-IDF presence score, its aspect-specific
  validation-reselected threshold, and the frozen DistilBERT aspect-conditioned
  sentiment prediction.
- Remote endpoint: frozen enhanced candidate-pair Qwen.
- Exact one-to-one `row_uid` and gold-label alignment for all twelve validation and
  twelve test folds. The artifacts were deterministically joined in strict TF-IDF
  order because their physical file order differs.
- Policy family: independently replace locally absent rows close below the
  threshold and locally present rows close above the threshold.
- Fraction grid: `0`, `0.05`, `0.10`, `0.15`, `0.20`, `0.30`, `0.40`, `0.50` for
  each side, giving 64 global policies.
- Fold-specific raw score cutoffs were derived from validation ranks only.
- Hard selection budget: mean validation Qwen call rate no higher than `0.50`.
- One global fraction pair was selected from the mean of all twelve validation
  folds. Test was read only after the selection manifest had been written.
- Test evaluated only strict local, frozen Qwen, and the frozen selected router.

This is a newly defined router over the current aspect-conditioned endpoints. No
legacy router result, boundary, prediction file, or historical JSON-prompt Qwen
output participated in selection.

## Validation selection

The best of all 64 registered policies was:

```text
rescue fraction:  0.00
confirm fraction: 0.00
mean Qwen calls:   0.0000
mean pair F1:      0.425221
```

This is exactly the strict local endpoint. Every policy that called Qwen had lower
mean validation pair F1. The closest non-zero-call policy used confirm fraction
`0.10`: F1 `0.424413`, Qwen call rate `0.079628`. The selected validation decision
therefore froze a zero-call router before test.

## Frozen test result

| Test system | Pair F1 | Precision | Recall | Presence F1 | FP rows/100 | FN rows/100 | Qwen calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| Strict TF-IDF + aspect-conditioned sentiment | 0.385640 | 0.415569 | 0.504221 | 0.417175 | 17.617 | 4.442 | 0.000 |
| Frozen candidate-pair Qwen | 0.337804 | 0.244003 | 0.651851 | 0.451565 | 14.288 | 4.584 | 1.000 |
| Validation-selected router | 0.385640 | 0.415569 | 0.504221 | 0.417175 | 17.617 | 4.442 | 0.000 |

The router-minus-local pair-F1 delta is exactly `0.000000`: `0` wins, `12` ties,
and `0` losses. The paired median is `0`, the seeded 100,000-resample interval is
`[0, 0]`, and the exact two-sided sign-flip value is `1.0`. These diagnostics are
degenerate because validation selected no routing.

After the result was fixed, the unchanged test command was replayed once solely to
write the tracked public CSVs from the analyser rather than by manual transcription.
It used the same frozen manifest, performed no selection, and reproduced the same
numeric result. This reporting replay is not a second model inference or a
test-informed policy evaluation.

Frozen Qwen exceeded strict TF-IDF on four individual aspects and lost on eight,
but local threshold distance did not identify that complementarity reliably on
validation. The candidate-pair Qwen endpoint also traded much higher recall for
substantially lower precision.

## Admission decision

The registered gate required at least `+0.01` test mean pair F1 over strict local
with no more than `0.50` mean Qwen calls. The observed gain was `0`, so the gate
failed.

The router is dropped from the dissertation mainline. It is not a baseline row,
does not justify a quality–call-rate figure, and does not replace the strict TF-IDF
result. If mentioned at all, it should receive one sentence as a negative
validation-selected deployment check.

## Reproduction

Run validation first:

```powershell
$env:PYTHONPATH='src'
python .\scripts\analyse_strict_tfidf_aspect_qwen_router.py --stage validation --strict-root .\outputs\baselines\loao_bow_sentence_embedding_v1 --sentiment-root .\outputs\baselines\loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_score_export_20260702\cross_encoder\example_filtered --qwen-full-root .\outputs\experimental\loao_unified_candidate_pair_v1\qwen_frozen_full_preregistered_20260718 --qwen-pilot-root .\outputs\experimental\loao_unified_candidate_pair_v1\qwen_frozen_pilot_preregistered_20260717 --similarity-selection-manifest .\outputs\analysis\loao_similarity_aspect_conditioned_sentiment_rescore_v1\validation\selection_manifest.json
```

Then pass the frozen router manifest to test:

```powershell
python .\scripts\analyse_strict_tfidf_aspect_qwen_router.py --stage test --strict-root .\outputs\baselines\loao_bow_sentence_embedding_v1 --sentiment-root .\outputs\baselines\loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_score_export_20260702\cross_encoder\example_filtered --qwen-full-root .\outputs\experimental\loao_unified_candidate_pair_v1\qwen_frozen_full_preregistered_20260718 --qwen-pilot-root .\outputs\experimental\loao_unified_candidate_pair_v1\qwen_frozen_pilot_preregistered_20260717 --similarity-selection-manifest .\outputs\analysis\loao_similarity_aspect_conditioned_sentiment_rescore_v1\validation\selection_manifest.json --router-selection-manifest .\outputs\analysis\loao_strict_tfidf_aspect_qwen_router_v1\validation\selection_manifest.json
```

Ignored detailed outputs are under
`outputs/analysis/loao_strict_tfidf_aspect_qwen_router_v1/`. Tracked aggregate and
per-aspect numeric exports are:

- `docs/thesis_figure_data/loao_strict_tfidf_aspect_qwen_router_v1_summary.csv`;
- `docs/thesis_figure_data/loao_strict_tfidf_aspect_qwen_router_v1_per_aspect.csv`.

## Limitations

- The endpoint test results already existed before this replacement router was
  designed; only the router policy and gate were registered prospectively.
- Target-aspect validation labels calibrate both endpoints and the router, so this
  is not strict zero-label LOAO.
- The twelve folds reuse much of the same dataset and are not independent studies.
