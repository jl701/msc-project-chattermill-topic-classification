# Report Notes

Last updated: 2026-07-02

This file is a compact evidence ledger for dissertation/report drafting. Detailed experiment records remain in `docs/`.

## 2026-07-01 - Gemini-Generated Aspect Descriptions

Task:

- Test whether Gemini-generated descriptions of the three fixed held-out FABSA aspects improve hosted candidate-label classification.

Protocol:

- Fixed held-out aspects:
  - `Account management: Account access`
  - `Company brand: Competitor`
  - `Value: Discounts promotions`
- Split strategy: `example_filtered`
- Evaluation label scope: held-out labels only
- Prompt base: indexed candidate IDs with `response_format=json_schema`
- Max tokens: `2048`
- Temperature: `0`
- Descriptions generated from canonical aspect names only; no validation/test review text used.

Tracked evidence:

- Implementation: `src/msc_project/llm/candidate_label.py`, `scripts/run_gemini_heldout_aspect.py`
- Tests: `tests/test_llm_candidate_label.py`
- Configs:
  - `configs/gemini_aspect_descriptions_fixed_heldout.json`
  - `configs/gemini_aspect_descriptions_decision_boundary_heldout.json`
- Detailed write-up: `docs/gemini_aspect_descriptions.md`
- Chronological log: `docs/experiment_log.md`

Observed results:

| Model / Prompt | Split | Pair Samples F1 | Pair Micro F1 | Pair Macro F1 | Aspect Samples F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Flash-Lite indexed | test | 0.5516 | 0.5872 | 0.4876 | 0.6062 |
| Flash-Lite label-only descriptions | test | 0.5925 | 0.6263 | 0.5691 | 0.6625 |
| Flash-Lite decision-boundary descriptions | test | 0.5724 | 0.6199 | 0.5456 | 0.6340 |
| Flash indexed | test | 0.6071 | 0.6541 | 0.5547 | 0.6747 |
| Flash label-only descriptions | test | 0.5893 | 0.6555 | 0.5655 | 0.6676 |
| Pro indexed | validation 50-row sample | 0.8000 | 0.8113 | 0.5985 | 0.8400 |
| Pro label-only descriptions | validation 50-row sample | 0.7467 | 0.7921 | 0.6128 | 0.8067 |

Interpretation:

- Label-only descriptions materially improve the cheap Flash-Lite hosted baseline.
- Decision-boundary descriptions reduce false positives but can increase empty predictions and false negatives.
- Flash descriptions improve validation but do not improve test pair samples F1.
- Pro descriptions do not justify a full run based on the validation diagnostic.
- Dissertation framing: this is a label-semantics ablation and precision-recall trade-off, not a universal prompt improvement.

Open follow-up:

- Sampled Gemini LOAO diagnostic if a lightweight robustness signal is needed.
- Cascade uncertainty improvement using local selector score/margin features.
- Qualitative error taxonomy with manual review.

## 2026-07-02 - Qualitative Error Taxonomy Pre-Registration

Task:

- Build a Gemini-assisted, manually verified qualitative error taxonomy before moving to Qwen fine-tuning.

Planned configuration:

- Use existing local ignored prediction outputs; do not rerun fixed-split or LOAO models.
- Fixed-split row-level sources: local DistilBERT, Qwen zero-shot, Gemini Flash-Lite/Flash/Pro, aspect-description variants, and Pro cascade.
- LOAO sources: existing Qwen-vs-DistilBERT comparison summaries and per-aspect CSVs.
- Output directory: `outputs/analysis/qualitative_error_taxonomy_20260702/`.
- Tracked documentation: `docs/qualitative_error_taxonomy.md`.
- Gemini Pro may assist taxonomy drafting from local ignored packets, but final categories must be manually reviewed.
- Raw review text must remain local-only and must not be committed.

Observed result:

- Implemented `scripts/analyse_qualitative_error_taxonomy.py`.
- Local output directory: `outputs/analysis/qualitative_error_taxonomy_20260702/`.
- Gemini Pro draft call succeeded with 19,494 prompt tokens, 6,362 completion tokens, 4,336 reasoning tokens, and 25,856 total tokens.
- Final tracked write-up: `docs/qualitative_error_taxonomy.md`.

Final taxonomy:

1. Semantic boundary bleed.
2. Competitor-positive recall bottleneck.
3. Generative over-prediction / fail-noisy behaviour.
4. Cautious abstention / fail-silent behaviour.
5. Sentiment polarity under-recall.
6. Prompt-induced precision-recall shift.
7. Cascade complementarity.

Qwen fine-tuning targets:

- abstention/no-label calibration;
- hard-negative aspect boundaries;
- competitor-positive recall;
- neutral sentiment coverage;
- stable label semantics;
- cascade-ready uncertainty signals.

## 2026-07-02 - Thesis Completion Roadmap

Decision:

- Keep LOAO as the dissertation's open-topic robustness spine.
- Treat full fine-tuned Qwen LoRA LOAO as the only major compute-bound unfinished experiment.
- Complete all non-major thesis work before GPU access is resolved.

Immediate non-major work:

- Finish qualitative error taxonomy from existing outputs.
- Build thesis-ready result tables and figure data.
- Try cascade score/margin uncertainty without new Gemini calls.
- Prepare and smoke-test the Qwen LoRA SFT/evaluation runner.
- Refresh the LaTeX thesis skeleton around the completed Gemini/Qwen/LOAO evidence.

Tracked roadmap:

- `docs/thesis_completion_roadmap.md`

## 2026-07-02 - Gemini-Assisted Qualitative Error Taxonomy

Command:

```powershell
python .\scripts\analyse_qualitative_error_taxonomy.py --output-dir .\outputs\analysis\qualitative_error_taxonomy_20260702

python .\scripts\analyse_qualitative_error_taxonomy.py --output-dir .\outputs\analysis\qualitative_error_taxonomy_20260702 --max-examples-per-category 10 --max-gemini-examples-per-category 2 --snippet-chars 220 --gemini-draft --gemini-model vertex_ai/gemini-2.5-pro --gemini-max-tokens 7000 --request-timeout 240
```

Observed Gemini-assisted packet:

- Rows aligned across fixed held-out-aspect test predictions: 281.
- Gemini Pro draft: successful, used only as an assistant for candidate taxonomy wording.
- Gemini usage: 19,494 prompt tokens, 6,362 completion tokens, including 4,336 reasoning tokens; 25,856 total tokens.
- Output directory: `outputs/analysis/qualitative_error_taxonomy_20260702/`.
- Tracked summary: `docs/qualitative_error_taxonomy.md`.

Key mechanism counts:

| Mechanism | Rows |
| --- | ---: |
| discounts/value boundary | 155 |
| account-access overprediction | 112 |
| empty abstention | 84 |
| competitor positive miss | 82 |
| description precision shift | 66 |
| Qwen overprediction | 56 |
| local overprediction | 50 |
| description recall loss | 45 |
| neutral under-recall | 25 |
| Pro-empty cascade recovery | 20 |

Interpretation:

- Qualitative evidence supports the quantitative story: local and Qwen over-predict, hosted Gemini is more conservative and can abstain, descriptions move precision/recall, and the Pro cascade works by recovering hosted abstentions.
- The final taxonomy remains manually consolidated: semantic boundary bleed, competitor-positive recall bottleneck, generative over-prediction, cautious abstention, sentiment polarity under-recall, prompt-induced precision-recall shift, and cascade complementarity.

## 2026-07-02 - Pre-Qwen LoRA Full LOAO Checklist Audit

Task:

- Re-check completed work and record the remaining work before full fine-tuned Qwen LoRA LOAO as a tickable checklist.

Tracked checklist:

- `docs/thesis_completion_roadmap.md`, section `Pre-Qwen LoRA Full LOAO Checklist`.

Completed before the checklist:

- Closed-topic, held-out organisation, fixed held-out-aspect, lexical/DistilBERT LOAO, Qwen zero-shot LOAO, Gemini Pareto, local-to-Gemini cascade, Gemini descriptions, qualitative taxonomy, and thesis skeleton refresh.

Remaining next work before full Qwen LoRA LOAO:

1. Thesis-ready result tables and figure data.
2. Cascade score/margin uncertainty improvement.
3. Final Qwen held-out-aspect LoRA SFT runner with manifest logging.
4. Resume/skip behaviour and focused tests for the runner.
5. Tiny local Qwen LoRA held-out-aspect smoke test.
6. Optional fixed held-out-aspect Qwen LoRA configuration before full LOAO.
7. Full 12-fold LOAO command templates and GPU environment confirmation.
