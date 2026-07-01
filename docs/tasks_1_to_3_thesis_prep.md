# Tasks 1-3 Thesis Prep And Task 4 Handoff

Last updated: 2026-07-01

This note consolidates the completed Gemini/local follow-up tasks so the dissertation write-up and Task 4 can start from a clean evidence map. It summarises what has been implemented, which outputs support each claim, which caveats must be preserved, and what should be done next.

## Task Map

The task numbering follows the Gemini follow-up plan recorded on 2026-07-01:

1. Run full fixed-split Gemini Pro validation/test as the stronger hosted upper-bound baseline.
2. Run full fixed-split Gemini Flash-Lite validation/test as the cheapest hosted baseline.
3. Build and analyse a local-to-Gemini uncertainty cascade.
4. Test Gemini-generated candidate-aspect descriptions as label-representation support.
5. Use Gemini Pro as a qualitative error-taxonomy aid with manual review.

Tasks 1, 2, and 3 are complete, documented, and synchronised to GitHub. Task 4 is the next clean modelling task.

## Repository Status

Tracked evidence and implementation:

- Gemini runner: `scripts/run_gemini_heldout_aspect.py`
- Gemini prompt/schema utilities: `src/msc_project/llm/candidate_label.py`
- Cascade runner: `scripts/run_local_gemini_cascade.py`
- Cascade helpers: `src/msc_project/evaluation/cascade.py`
- Pro cascade deep-dive: `scripts/analyse_local_gemini_cascade.py`
- Tests:
  - `tests/test_llm_candidate_label.py`
  - `tests/test_cascade_evaluation.py`

Generated outputs remain ignored under `outputs/` because they may contain review text. API keys, credentials, raw outputs, and generated prediction files must not be committed.

## Task 1: Full Gemini Pro Fixed-Split Baseline

Purpose:

- Remove the caveat from the earlier 50-row Pro subset.
- Establish a stronger hosted-LLM upper-bound baseline for the fixed held-out-aspect protocol.
- Compare Pro against Flash and the strongest local non-LLM baseline with latency and cost diagnostics.

Protocol:

- Dataset: FABSA fixed held-out-aspect validation/test split.
- Held-out aspects:
  - `Account management: Account access`
  - `Company brand: Competitor`
  - `Value: Discounts promotions`
- Prompt: indexed candidate labels.
- Response format: `json_schema`.
- Max tokens: `2048`.
- Temperature: `0`.
- Model: `vertex_ai/gemini-2.5-pro`.

Command:

```powershell
python .\scripts\run_gemini_heldout_aspect.py --model vertex_ai/gemini-2.5-pro --split both --limit 10000 --prompt-variant indexed --response-format json_schema --response-format-fallback --max-tokens 2048 --input-cost-per-1m 1.25 --output-cost-per-1m 10.00 --output-dir .\outputs\llm\gemini_candidate_label_20260701_031040_pro_fixed_full
```

Result:

| Split | Pair Samples F1 | Pair Micro F1 | Pair Macro F1 | Aspect Samples F1 | Valid JSON | Schema Valid | Mean Latency | Approx Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 0.7270 | 0.7377 | 0.6092 | 0.7954 | 1.0000 | 0.9953 | 5.3514 s | $1.2640 |
| test | 0.7141 | 0.7425 | 0.6287 | 0.7746 | 1.0000 | 1.0000 | 5.6510 s | $1.7140 |

Dissertation claim supported:

- Pro is the strongest full hosted baseline on the fixed three-aspect protocol, improving clearly over Flash, but it is slower and more expensive.
- The hosted evaluation should include thinking tokens in output-token cost accounting.

Evidence locations:

- `docs/experiment_log.md`
- `docs/gemini_candidate_label_baseline.md`
- `docs/generalisation_baselines.md`
- `docs/local_gemini_cascade.md`

## Task 2: Full Gemini Flash-Lite Fixed-Split Baseline

Purpose:

- Add the cheapest and lowest-latency hosted point.
- Complete the fixed-split hosted Pareto comparison across Flash-Lite, Flash, and Pro.
- Avoid over-spending on LOAO before fixed-split trade-offs are understood.

Protocol:

- Same split, candidate labels, prompt format, JSON schema, max-token setting, and temperature as Task 1.
- Model: `vertex_ai/gemini-2.5-flash-lite`.

Command:

```powershell
python .\scripts\run_gemini_heldout_aspect.py --model vertex_ai/gemini-2.5-flash-lite --split both --limit 10000 --prompt-variant indexed --response-format json_schema --response-format-fallback --max-tokens 2048 --input-cost-per-1m 0.10 --output-cost-per-1m 0.40 --output-dir .\outputs\llm\gemini_candidate_label_20260701_034545_flash_lite_fixed_full
```

Result:

| Split | Pair Samples F1 | Pair Micro F1 | Pair Macro F1 | Aspect Samples F1 | Valid JSON | Schema Valid | Mean Latency | Approx Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 0.5876 | 0.5973 | 0.4477 | 0.6783 | 1.0000 | 1.0000 | 0.3605 s | $0.0075 |
| test | 0.5516 | 0.5872 | 0.4876 | 0.6062 | 1.0000 | 0.9964 | 0.3719 s | $0.0096 |

Dissertation claim supported:

- Flash-Lite is extremely cheap and fast, but it underperforms the strongest local fixed-split baseline and the stronger hosted models.
- It is best framed as the low-cost hosted corner of the Pareto table rather than as the main quality result.

Evidence locations:

- `docs/experiment_log.md`
- `docs/gemini_candidate_label_baseline.md`
- `docs/generalisation_baselines.md`
- `docs/local_gemini_cascade.md`

## Fixed-Split Hosted Pareto Summary

| System | Test Pair Samples F1 | Test Pair Micro F1 | Test Pair Macro F1 | Test Aspect Samples F1 | Mean Test Latency | Test Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Local DistilBERT pipeline | 0.6071 | 0.5917 | 0.4890 | 0.6651 | local | n/a |
| Gemini 2.5 Flash-Lite full | 0.5516 | 0.5872 | 0.4876 | 0.6062 | 0.3719 s | $0.0096 |
| Gemini 2.5 Flash full | 0.6071 | 0.6541 | 0.5547 | 0.6747 | 2.3721 s | $0.2996 |
| Gemini 2.5 Pro full | 0.7141 | 0.7425 | 0.6287 | 0.7746 | 5.6510 s | $1.7140 |

This table is ready for the dissertation results chapter, with the caveat that the costs are approximate public-rate calculations, not Chattermill billing statements.

## Task 3: Local-To-Gemini Uncertainty Cascade

Purpose:

- Test selective deployment rather than full hosted replacement.
- Use local predictions by default and escalate locally uncertain rows to Gemini.
- Compare Flash-Lite, Flash, and Pro as hosted escalators under validation-selected policies.
- Analyse why the Pro cascade beats pure Pro.

Implementation:

- `src/msc_project/evaluation/cascade.py`
- `scripts/run_local_gemini_cascade.py`
- `tests/test_cascade_evaluation.py`
- `scripts/analyse_local_gemini_cascade.py`

The cascade uses validation-derived local reliability features because the archived local prediction files do not store calibrated selector probabilities or margins. Test labels are not used for headline policy selection.

Commands:

```powershell
python .\scripts\run_local_gemini_cascade.py --gemini-dir .\outputs\llm\gemini_candidate_label_20260701_034545_flash_lite_fixed_full --input-cost-per-1m 0.10 --output-cost-per-1m 0.40 --output-dir .\outputs\analysis\local_gemini_cascade_flash_lite_grid1 --rank-rate-step 1

python .\scripts\run_local_gemini_cascade.py --gemini-dir .\outputs\llm\gemini_candidate_label_20260701_0145_fixed_full --input-cost-per-1m 0.30 --output-cost-per-1m 2.50 --output-dir .\outputs\analysis\local_gemini_cascade_flash_grid1 --rank-rate-step 1

python .\scripts\run_local_gemini_cascade.py --gemini-dir .\outputs\llm\gemini_candidate_label_20260701_031040_pro_fixed_full --input-cost-per-1m 1.25 --output-cost-per-1m 10.00 --output-dir .\outputs\analysis\local_gemini_cascade_pro_grid1 --rank-rate-step 1

python .\scripts\analyse_local_gemini_cascade.py
```

Headline validation-selected test results:

| Escalator | Policy Summary | Pair Samples F1 | Pair Micro F1 | Pair Macro F1 | Aspect Samples F1 | Calls | Call Rate | Test Gemini Cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| None, local only | n/a | 0.6071 | 0.5917 | 0.4890 | 0.6651 | 0 | 0.0000 | $0.0000 |
| Flash-Lite | weighted pair+sentiment reliability, 51%, Gemini non-empty else local | 0.6679 | 0.6579 | 0.5401 | 0.7259 | 143 | 0.5089 | $0.0052 |
| Flash | low minimum pair precision, 90%, Gemini non-empty else local | 0.7459 | 0.7348 | 0.6223 | 0.7993 | 253 | 0.9004 | $0.2789 |
| Pro | low minimum pair precision, 90%, Gemini non-empty else local | 0.8102 | 0.7955 | 0.6809 | 0.8493 | 253 | 0.9004 | $1.5421 |

Pro cascade deep-dive:

| System | Pair Samples F1 | Pair Micro F1 | TP | FP | FN | Empty Prediction Rows | Exact Rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Local only | 0.6071 | 0.5917 | 192 | 168 | 97 | 0 | 144 |
| Pure Pro | 0.7141 | 0.7425 | 222 | 87 | 67 | 28 | 167 |
| Local -> Pro cascade | 0.8102 | 0.7955 | 249 | 88 | 40 | 0 | 194 |

Key explanation:

- The cascade does not win because the local model is globally stronger; pure Pro is stronger than local on average.
- It wins because `gemini_nonempty_else_local` exploits complementary errors.
- Pure Pro has `28` empty prediction rows; the cascade recovers all `28` using local predictions.
- Cascade versus pure Pro: `29` better rows, `250` equal rows, `2` worse rows.
- Pair false negatives fall from `67` to `40`, while false positives remain nearly flat (`87` to `88`).
- The largest aspect-level gain is `Company brand: Competitor`, where cascade F1 rises from `0.7407` to `0.8487` with `21` fewer false negatives.

Dissertation claim supported:

- The strongest fixed-split system result is a selective local-hosted cascade, not a full hosted replacement.
- This supports a practical deployment argument: local models can handle routine rows, while hosted LLMs can be reserved for uncertain or high-value rows.
- The result should be reported as fixed three-aspect selective-deployment evidence, not as LOAO robustness evidence.

Evidence locations:

- `docs/local_gemini_cascade.md`
- `docs/experiment_log.md`
- `docs/gemini_candidate_label_baseline.md`
- `docs/generalisation_baselines.md`

## Dissertation-Ready Claims

The following claims are supported by current evidence:

- Structured-output Gemini prompting is reliable on this endpoint when using `response_format=json_schema`, indexed candidate IDs, and `max_tokens=2048`.
- Too-low `max_tokens` can produce empty or truncated content because Gemini 2.5 spends many completion tokens on thinking before visible JSON.
- Cost comparisons must count output tokens that include thinking tokens; reasoning tokens should also be reported separately when available.
- Full Pro is the strongest pure hosted fixed-split baseline, but it is slower and more expensive than Flash and Flash-Lite.
- Flash-Lite is the cheapest and fastest hosted point, but it is not strong enough to replace the local model.
- Flash is the balanced hosted baseline and improves pair micro/macro F1 over the local model while matching local pair samples F1.
- The Pro cascade is the best fixed-split system result so far because it combines Pro's semantic generalisation with local fallback protection.

Do not claim yet:

- That Gemini or the cascade has been validated under full leave-one-aspect-out robustness.
- That costs are exact billing values.
- That the 80% Pro budget diagnostic is the headline result, because it was selected as a test diagnostic rather than a validation-selected policy.
- That the current uncertainty policy is calibrated probability-based; it uses validation reliability proxies.

## Suggested Tables And Figures For The Dissertation

Ready tables:

- Fixed-split hosted Pareto table: local, Flash-Lite, Flash, Pro.
- Selective cascade table: local only, local -> Flash-Lite, local -> Flash, local -> Pro.
- Pro cascade explanation table: TP/FP/FN, empty rows, exact rows for local, pure Pro, cascade.
- Cost/latency table including thinking-token-aware output cost.

Useful figures to create later:

- Pair samples F1 versus test cost for hosted and cascade systems.
- Pair samples F1 versus mean latency.
- False-negative reduction from local, pure Pro, and cascade.
- Aspect-level F1 change for `Company brand: Competitor`, `Account management: Account access`, and `Value: Discounts promotions`.

## Task 4 Readiness

Task 4 should test candidate-aspect descriptions as label-representation support. The clean starting point is:

- Keep the same fixed held-out-aspect validation/test split.
- Keep indexed candidate IDs to avoid near-miss copied labels.
- Keep `response_format=json_schema`, `temperature=0`, and `max_tokens=2048`.
- Generate or define descriptions without using validation/test review texts or validation/test labels.
- Use validation only to select the description/prompt variant, then evaluate once on test.
- Start with Gemini Flash for cost efficiency; use Pro only if the Flash description result is promising or diagnostically ambiguous.
- Report token/cost changes because descriptions lengthen the prompt.

Recommended Task 4 comparison:

| Variant | Purpose |
| --- | --- |
| `indexed` | Current no-description baseline. |
| Existing simple descriptive prompt | Recheck the earlier lightweight description style if needed. |
| Gemini-generated aspect descriptions | Test whether richer label semantics improve unseen-aspect prediction. |
| Optional manual/tagger-guidance descriptions | Use only if Aji can provide approved guidance or if descriptions are created from aspect names alone. |

Key evaluation questions:

- Do descriptions improve pair samples F1, micro F1, or macro F1 over the indexed baseline?
- Do they help `Company brand: Competitor` recall without causing over-prediction?
- Do they help rare neutral labels?
- Do they increase cost or latency enough to offset any metric gain?
- Do they change JSON/schema reliability?

Task 4 should be logged separately in `docs/experiment_log.md` and should update this handoff note only after results are available.

## Validation Snapshot

Latest completed checks after Task 3:

```powershell
python .\scripts\analyse_local_gemini_cascade.py
python -m unittest discover -s tests
python -m compileall -q src scripts tests
git diff --check
rg -n "sk-[A-Za-z0-9_-]{12,}" . --glob '!outputs/**' --glob '!data/**' --glob '!models/**' --glob '!checkpoints/**' --glob '!.git/**'
```

Results:

| Check | Result |
| --- | --- |
| Pro cascade deep-dive | passed |
| Unit tests | 63 tests OK |
| Compile check | passed |
| Diff whitespace check | passed, with CRLF conversion warnings only |
| Secret scan | no real API key found |
