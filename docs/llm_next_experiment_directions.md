# LLM Next Experiment Directions

Last updated: 2026-07-01

This note records the recommended next LLM-centred experiments after the completed local DistilBERT LOAO robustness run and the completed fixed-split Gemini Pareto/cascade experiments.

## Current Evidence Position

The project now has five complementary evidence blocks:

1. Strong local fixed-split non-LLM baseline: candidate-aspect DistilBERT selector plus DistilBERT aspect-conditioned sentiment reaches `0.6071` test pair samples F1.
2. Local LOAO robustness caveat: the same DistilBERT branch drops to `0.3128` mean pair micro F1 in full all-row LOAO, showing weak unseen-aspect relevance detection and threshold calibration under taxonomy shift.
3. Hosted Gemini fixed-split Pareto: Flash-Lite is cheap/fast, Flash matches local pair samples F1 with better pair micro/macro F1, and Pro is the strongest pure hosted fixed-split baseline.
4. Local-to-Gemini cascade: selective escalation is the strongest fixed-split system result so far, reaching `0.7459` with Flash and `0.8102` with Pro.
5. Gemini-generated aspect descriptions: label-only descriptions improve Flash-Lite test pair samples F1 from `0.5516` to `0.5925`, but stronger models show mixed precision-recall trade-offs rather than monotonic gains.

The dissertation story should therefore not be "run every model on every expensive protocol". It should be:

- use LOAO to expose robustness limits of the local candidate-label branch;
- use Gemini fixed-split and cascade experiments to study semantic candidate-label reasoning under cost, latency, and governance constraints;
- use targeted low-cost diagnostics rather than full Gemini LOAO unless a new dissertation-value argument appears.

## Why Full Gemini LOAO Is Not The Next Default

The completed fixed-split Gemini validation/test evaluation uses `493` API calls:

- validation rows containing the fixed held-out aspects: `212`
- test rows containing the fixed held-out aspects: `281`

Full all-row LOAO would rotate all 12 aspects and keep all official validation/test rows:

- all validation rows: `1,057`
- all test rows: `1,587`
- validation + test LOAO calls: about `31,728`
- test-only LOAO calls: about `19,044`

Using the observed fixed-split average cost per call, the rough public-rate cost estimate is:

| Model | Fixed Validation + Test Cost | Estimated Full All-Row LOAO Validation + Test Cost | Estimated Test-Only LOAO Cost |
| --- | ---: | ---: | ---: |
| Gemini 2.5 Flash-Lite | $0.0171 | about $1.10 | about $0.66 |
| Gemini 2.5 Flash | $0.5348 | about $34 | about $21 |
| Gemini 2.5 Pro | $2.9781 | about $192 | about $115 |

The time cost is also substantial. Using observed mean latency, full validation+test all-row LOAO would take roughly:

| Model | Estimated Wall-Clock Time |
| --- | ---: |
| Gemini 2.5 Flash-Lite | about 3 hours |
| Gemini 2.5 Flash | about 21 hours |
| Gemini 2.5 Pro | about 49 hours |

These estimates are approximate because one-aspect LOAO prompts may be shorter than the fixed three-aspect prompt, but Gemini 2.5 thinking/output tokens are a major part of the observed cost. Full Gemini LOAO should therefore remain a non-default option.

## Recommended Experiment Roadmap

### Completed: Candidate-Aspect Descriptions

Purpose:

- Test whether richer label semantics improve Gemini candidate-label prediction without changing the evaluation split.
- Directly probe whether LLMs help because they understand aspect meanings better than local encoders.

Status:

- Implemented in `src/msc_project/llm/candidate_label.py` and `scripts/run_gemini_heldout_aspect.py`.
- Tracked configs live in `configs/gemini_aspect_descriptions_fixed_heldout.json` and `configs/gemini_aspect_descriptions_decision_boundary_heldout.json`.
- Full results are recorded in `docs/gemini_aspect_descriptions.md`.

Main finding:

- Label-only descriptions help Flash-Lite substantially on test (`0.5516` to `0.5925` pair samples F1).
- Decision-boundary descriptions win Flash-Lite validation but generalise less well on test.
- Flash descriptions improve validation but reduce test pair samples F1, while improving micro/macro slightly.
- Pro descriptions do not justify a full run based on the 50-row validation diagnostic.

### 1. Sampled Gemini LOAO Diagnostic

Purpose:

- Get a small robustness signal without paying for full all-row Gemini LOAO.
- Check whether Gemini fixed-split gains plausibly carry to harder aspect rotations.

Recommended design:

- Do not call this full LOAO.
- Report it as a sampled Gemini LOAO robustness diagnostic.
- Use Gemini Flash first.
- Use Pro only on a very small confirmatory subset if needed.
- Select a representative aspect set:
  - fixed-split weak Gemini aspect: `Company brand: Competitor`
  - local LOAO weak aspect: `Company brand: Reviews`
  - medium-difficulty aspect: `Staff support: Email` or `Account management: Account access`
  - easier aspect: `Staff support: Phone` or `Purchase booking experience: Ease of use`
- Sample positive and negative rows per held-out aspect so that the diagnostic includes both detection and abstention behaviour.

Key questions:

- Does Gemini reduce the local DistilBERT LOAO failure on difficult aspects?
- Does Gemini also become conservative or empty on rare/ambiguous aspects?
- Are the same error modes visible as in the fixed split?

### 2. Cascade Uncertainty Improvement

Purpose:

- Improve the methodological strength of the local-to-Gemini cascade without new Gemini API calls.
- Replace validation-reliability proxy uncertainty with model-score uncertainty where possible.

Recommended design:

- Rerun or extend the local DistilBERT prediction export to save:
  - candidate-aspect relevance score
  - selected threshold
  - distance to threshold
  - top score
  - score margin
  - sentiment confidence if available
- Reuse existing Gemini prediction files.
- Rerun cascade policy search with score/margin features.
- Keep validation-only policy selection.

Key questions:

- Does calibrated or margin-based uncertainty select fewer Gemini calls for the same F1?
- Does it reduce the current 90% Flash/Pro escalation rate?
- Does it improve the dissertation defensibility of selective deployment?

### 3. Qualitative Error Taxonomy

Purpose:

- Turn model comparisons into an interpretable dissertation discussion.
- Explain why local, Gemini, and cascade systems fail differently.

Recommended design:

- Sample representative error rows from:
  - local-only errors
  - Gemini Flash errors
  - Gemini Pro errors
  - cascade wins and cascade losses
- Use row IDs and label sets in tracked documentation; do not commit raw review text.
- Gemini Pro may assist category drafting, but final taxonomy must be manually reviewed.

Candidate categories:

- missed competitor praise
- neutral versus positive confusion
- account/access overprediction
- promotion/value ambiguity
- hosted LLM abstention or empty prediction
- local over-prediction
- multi-aspect boundary ambiguity

Key questions:

- Which errors are caused by label semantics?
- Which errors are caused by sentiment ambiguity?
- Which errors are deployment-relevant, such as hosted abstention or local over-prediction?

### 4. Qwen Fine-Tuning And Evaluation

Purpose:

- Test whether an open/local LLM can close part of the gap to hosted Gemini while preserving privacy/control advantages.
- Move beyond zero-shot Qwen once stronger GPU access is available.

Recommended design:

- Reuse the indexed candidate-label protocol and prepared Qwen SFT data.
- Start with a small, reproducible QLoRA/SFT run on stronger GPU access.
- Evaluate on the fixed held-out-aspect split first.
- Only consider broader robustness diagnostics after the fixed-split pipeline is stable.
- Compare against:
  - Qwen zero-shot
  - local DistilBERT baseline
  - Gemini Flash
  - Gemini Pro
  - local-to-Gemini cascade

Key questions:

- Can Qwen fine-tuning beat zero-shot Qwen and approach Gemini Flash?
- How does Qwen compare on cost, privacy, latency, and deployability?
- Does it reduce the need for hosted escalation, or mainly provide another local baseline?

## Recommended Order

1. Sampled Gemini LOAO diagnostic, if a small robustness signal is still needed.
2. Cascade uncertainty improvement using local score/margin export.
3. Qualitative error taxonomy.
4. Qwen fine-tuning/evaluation once stronger GPU access is available.

This order maximises dissertation value per unit cost. It first tests the label-semantics hypothesis, then adds a lightweight robustness check, then strengthens the selective-deployment method without more hosted calls, then prepares the qualitative discussion, and finally moves to the larger open-LLM fine-tuning stage.
