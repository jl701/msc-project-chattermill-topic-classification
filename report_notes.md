# Report Notes

Last updated: 2026-07-01

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
