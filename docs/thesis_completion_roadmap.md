# Thesis Completion Roadmap

Last updated: 2026-07-03

This note freezes the remaining dissertation-oriented work around the Qwen LoRA LOAO boundary. The original aim was to keep the project moving while GPU access was being negotiated. A later single-fold all-row Qwen LoRA pilot showed that the current SFT recipe does not pass the validation gate, so the full 12-fold run should now wait for a revised absence-calibration objective rather than GPU access alone. The immediate modelling direction has shifted to local-to-Qwen hybrid routing: DistilBERT acts as a cheap calibrated gate, while Qwen is used as a semantic judge for uncertain or unfamiliar unseen-aspect cases.

## Current Thesis Spine

Working title:

```text
Structured Candidate-Label Aspect-Sentiment Classification under Taxonomy Shift in Customer Feedback
```

Central research question:

```text
When customer-feedback taxonomies evolve, how should aspect+sentiment labels be predicted and evaluated when new candidate labels are supplied at inference time?
```

The thesis should not be framed as unrestricted topic discovery. The defensible formulation is candidate-label aspect-sentiment classification: the model receives a review and a supplied set of canonical candidate aspects, then predicts relevant aspect+sentiment pairs.

## Evidence Blocks Already Available

The dissertation currently has enough evidence for a coherent taxonomy-shift story even before Qwen fine-tuned full LOAO:

| Evidence Block | Role In Thesis | Current Status |
| --- | --- | --- |
| Closed-topic baselines | Fixed-taxonomy reference point | Complete |
| Held-out organisation baselines | Domain/organisation-shift comparison | Complete |
| Fixed held-out-aspect local baselines | Candidate-label modelling under a controlled unseen-aspect split | Complete |
| Lexical and DistilBERT LOAO | Robustness evidence across all FABSA aspects | Complete |
| Qwen zero-shot fixed and full LOAO | Open-weight LLM zero-shot baseline and failure diagnosis | Complete |
| Gemini fixed Pareto | Hosted structured-output LLM comparison under cost/latency accounting | Complete |
| Local-to-Gemini cascade | Selective-deployment evidence from local/hosted error complementarity | Complete |
| Gemini aspect descriptions | Label-semantics ablation for hosted LLM prompting | Complete |
| Qualitative error taxonomy | Thesis-facing explanation of why systems fail differently | Gemini-assisted packet and tracked taxonomy complete |
| Qwen LoRA single-fold all-row pilot | Validation-gated check before spending 12-fold GPU time | Complete negative validation result |
| Local-to-Qwen hybrid diagnostic | Evidence that Qwen is more useful as a semantic complement than a direct replacement | Asymmetric global score-distance routing and Pareto analysis complete; sentiment-margin-only routing weak |
| Qwen LoRA fine-tuned full LOAO | Final open-weight LLM adaptation robustness check | Deferred until revised absence calibration passes validation |

## Main Dissertation Narrative

The results chapter should be organised as a ladder of increasing difficulty rather than a flat leaderboard.

1. Closed-topic FABSA classification is a strong fixed-taxonomy baseline. DistilBERT performs well when every evaluation label has been observed during training.
2. Held-out organisation evaluation shows that cross-organisation shift is real, but it is not the same as taxonomy shift because the label set remains fixed.
3. Fixed held-out-aspect evaluation introduces candidate labels and unseen aspects. The strongest local model is the candidate-aspect DistilBERT selector plus DistilBERT aspect-conditioned sentiment.
4. Full all-row LOAO is the core open-topic robustness view. It shows that fixed held-out-aspect scores overstate robustness and that unseen-aspect relevance detection is the bottleneck.
5. Qwen zero-shot full LOAO provides the open-weight LLM baseline. It has stable JSON and strong positive-gold recognition, but it over-predicts on empty-gold rows.
6. Gemini fixed-split and cascade results answer a different question: how hosted structured-output LLMs and local models can be combined under cost, latency, and governance constraints. They must not be over-claimed as full LOAO robustness evidence.
7. Qwen LoRA fine-tuning is motivated by the diagnosed zero-shot failure mode: the model needs task-specific calibration for when a supplied candidate aspect is absent.
8. A single-fold all-row Qwen LoRA pilot on `Company brand: Competitor` showed that simple singleton negative sampling changes calibration but does not beat same-fold zero-shot or local baselines. Full Qwen LoRA LOAO should therefore wait for a revised absence-aware objective before spending 12-fold GPU time.
9. The local-to-Qwen LOAO diagnostics now support a more promising model-division route: use DistilBERT for calibrated local gating and Qwen for selective semantic judgement. The completed asymmetric global score-distance router raises test mean pair micro F1 to `0.3900` with Qwen called on `29.1%` of rows, compared with the local rerun's `0.3158`, Qwen-only `0.3378`, and the previous symmetric score-distance gate's `0.3800`. This direction is documented in `docs/qwen_local_hybrid_direction.md`.

The short thesis argument is:

```text
Candidate-label modelling is necessary but not sufficient for evolving customer-feedback taxonomies. Local supervised models can be strong on fixed held-out aspects, but LOAO exposes weak robustness across aspect rotations. Qwen zero-shot understands present aspects but lacks absence calibration. Hosted Gemini and local-hosted cascades improve fixed-split performance, but their strongest evidence is selective deployment rather than LOAO robustness. The completed local-to-Qwen score-distance diagnostic shows that the strongest near-term Qwen role is selective semantic judgement under local uncertainty. Absence-aware Qwen LoRA fine-tuning remains a future open-weight robustness experiment, but the current SFT recipe has not yet passed the validation gate needed to justify a full 12-fold run.
```

After the Qwen LoRA validation gate failed, the main near-term modelling claim should become narrower and stronger:

```text
Qwen should be evaluated as a selective semantic judge in a hybrid local-to-Qwen system, not forced to replace the calibrated DistilBERT candidate-aspect pipeline in a traditional multi-label classifier role.
```

## Work To Finish Before Full Qwen LoRA LOAO

Use the checklist below as the operational source of truth before starting the full fine-tuned Qwen LoRA LOAO run. When a task is completed, update this file by changing `[ ]` to `[x]`, then record the command/result in `docs/experiment_log.md` and `report_notes.md` where relevant.

### Pre-Qwen LoRA Full LOAO Checklist

#### Completed Foundation

- [x] Freeze the dissertation spine as structured candidate-label aspect-sentiment classification under taxonomy shift.
- [x] Complete closed-topic FABSA baselines and record the fixed-taxonomy reference point.
- [x] Complete held-out organisation baselines and keep them separate from taxonomy-shift claims.
- [x] Complete the strongest local fixed held-out-aspect baseline: candidate-aspect DistilBERT selector plus DistilBERT aspect-conditioned sentiment.
- [x] Complete lexical and DistilBERT full all-row LOAO robustness baselines.
- [x] Complete Qwen indexed zero-shot fixed held-out-aspect validation/test evaluation.
- [x] Complete Qwen indexed zero-shot full all-row LOAO validation/test evaluation and positive-gold diagnostic.
- [x] Complete Gemini Flash-Lite, Flash, and Pro fixed held-out-aspect Pareto runs.
- [x] Complete local-to-Gemini cascade and Pro cascade deep-dive.
- [x] Complete Gemini aspect-description ablation.
- [x] Complete qualitative error taxonomy with Gemini-assisted drafting and manual consolidation.
- [x] Refresh the LaTeX thesis skeleton so it reflects completed Qwen/Gemini/LOAO evidence and the pending Qwen LoRA full LOAO experiment.
- [x] Keep `outputs/`, raw predictions, review-text packets, credentials, checkpoints, and model weights out of Git.

#### Still Required Before Starting Full Qwen LoRA LOAO

- [x] Build thesis-ready result tables and figure data that keep closed-topic, held-out organisation, fixed held-out aspect, all-row LOAO, and positive-gold diagnostics conceptually separate.
- [x] Run or document the cascade score/margin uncertainty improvement using existing Gemini predictions and local score exports where feasible.
- [x] Confirm or implement the final Qwen held-out-aspect LoRA SFT runner for indexed candidate-label JSONL, rather than relying on the closed-topic pilot runner.
- [x] Add run-manifest logging to the Qwen LoRA runner, including exact command, git commit, model name, quantisation, LoRA parameters, split/protocol, row scope, runtime, hardware, and output directory.
- [x] Add or confirm resume/skip-existing behaviour for Qwen LoRA training checkpoints, adapter outputs, and validation/test predictions.
- [x] Add focused tests for the final Qwen LoRA runner's data loading, manifest writing, resume/skip logic, and prediction normalisation.
- [x] Run a tiny local Qwen LoRA held-out-aspect smoke test on a few training/evaluation rows to verify model loading, loss masking, adapter saving, JSON parsing, and metrics.
- [x] Run one fixed held-out-aspect Qwen LoRA configuration before full LOAO if local/remote GPU time allows; use validation selection before test evaluation.
- [x] Define the final full 12-fold Qwen LoRA LOAO command templates, output directory pattern, checkpoint naming, and recovery plan.
- [x] Run a single-fold all-row Qwen LoRA validation-gated pilot before launching the full 12-fold run.
- [x] Develop and evaluate the local-to-Qwen score/margin LOAO routing diagnostic before returning to full Qwen JSON-SFT LOAO.
- [x] Complete the local-to-Qwen asymmetric global score-distance router.
- [x] Complete the local-to-Qwen F1/call-rate Pareto reporting from the same validation policy grid.
- [ ] Revise the Qwen absence-calibration/training objective after the single-fold pilot failed to beat same-fold baselines.
- [ ] Confirm the target GPU environment, storage budget, package versions, and data-transfer rules before launching any long full-LOAO run with a revised recipe.
- [ ] Run the standard validation and safety checks immediately before the full Qwen LoRA LOAO launch.

#### Optional Or Deferred

- [ ] Run sampled Gemini LOAO only if a supervisor specifically asks for hosted-LLM LOAO evidence or if Qwen LoRA full LOAO becomes infeasible.
- [ ] Run full Gemini LOAO only with an explicit dissertation-value and cost/latency justification.
- [ ] Add a new joint pair-scoring model only if the Qwen path becomes blocked and the thesis needs another local modelling contribution.
- [ ] Implement candidate-wise Qwen semantic judgement only if the thesis needs another Qwen experiment beyond the completed score-distance routing result.

### 1. Thesis Evidence Map And Result Tables

Purpose:

- Convert the many documented experiments into a compact set of thesis-ready tables and figure inputs.
- Prevent fixed-split, positive-row, and all-row LOAO results from being mixed as if they were the same benchmark.

Deliverables:

- A table mapping every headline claim to its evidence source and protocol.
- Main result table for the protocol ladder: closed-topic, held-out organisation, fixed held-out aspect, LOAO.
- Main LLM/cascade table: Qwen zero-shot, Gemini Flash-Lite/Flash/Pro, local-to-Gemini cascade.
- LOAO robustness table: lexical lower bound, DistilBERT LOAO, Qwen zero-shot LOAO, with precision/recall and false-positive rows.
- Figure-ready CSVs or Markdown tables for:
  - fixed split versus LOAO drop;
  - Qwen all-row versus positive-gold diagnostic;
  - local/Gemini/cascade cost-latency-quality trade-off.

Suggested tracked documentation:

- `docs/thesis_completion_roadmap.md`
- `report_notes.md`
- optional figure-data script under `scripts/` if manual table extraction becomes error-prone.

### Completed: Gemini-Assisted Qualitative Error Taxonomy

Purpose:

- Turn completed model comparisons into explainable thesis evidence.
- Explain why local DistilBERT, Qwen zero-shot, Gemini, descriptions, and cascades fail differently.

Starting point:

- Pre-registered in `docs/qualitative_error_taxonomy.md`.
- Uses existing local ignored outputs, plus one Gemini Pro draft pass for candidate category wording.
- Raw review text must remain under `outputs/`.

Completed output:

- `docs/qualitative_error_taxonomy.md` updated from pre-registration to completed analysis.
- Local ignored packets and Gemini draft under `outputs/analysis/qualitative_error_taxonomy_20260702/`.
- Categories should cover:
  - local over-prediction;
  - Qwen empty-gold over-prediction;
  - hosted LLM abstention;
  - Pro cascade recovery;
  - competitor positive recall;
  - account/access overprediction;
  - discount/value boundary ambiguity;
  - neutral sentiment under-recall;
  - description-driven precision/recall shifts.
- Final manually consolidated taxonomy:
  - semantic boundary bleed;
  - competitor-positive recall bottleneck;
  - generative over-prediction / fail-noisy behaviour;
  - cautious abstention / fail-silent behaviour;
  - sentiment polarity under-recall;
  - prompt-induced precision-recall shift;
  - cascade complementarity.

Thesis contribution:

- Strengthens the discussion chapter by linking metric patterns to concrete failure mechanisms without committing raw review text.

### 3. Cascade Uncertainty Improvement

Purpose:

- Improve the methodological defensibility of the local-to-Gemini cascade.
- Replace the current validation-reliability proxy with model score or margin features where feasible.

Preferred scope:

- Do not rerun Gemini API calls.
- Reuse existing Gemini prediction files.
- Rerun or extend the local fixed held-out-aspect baseline only if needed to export:
  - candidate-aspect score;
  - selected threshold;
  - distance to threshold;
  - top score;
  - score margin;
  - predicted sentiment confidence if available.

Deliverables:

- Score/margin export for the strongest local fixed held-out-aspect baseline.
- Cascade policy search using the new uncertainty features.
- Documentation update comparing reliability-proxy cascade versus score/margin cascade.

Stopping rule:

- If the score/margin cascade does not improve call efficiency or F1, keep the current cascade as the headline and report the score/margin attempt as a negative methodological check.

### 4. Qwen LoRA Pipeline Readiness

Purpose:

- Keep the final full Qwen LoRA LOAO experiment executable, but launch it only after a revised absence-calibration recipe passes validation and stronger GPU access is available.
- Avoid discovering runner, manifest, parser, or resume problems only after GPU time starts.

Work that should be completed before the full LOAO run:

- Confirm or implement the final Qwen SFT runner for indexed candidate-label JSONL.
- Add manifest logging:
  - exact command;
  - git commit;
  - model name;
  - quantisation;
  - LoRA parameters;
  - split/protocol;
  - train/eval row scope;
  - runtime;
  - hardware;
  - output directory.
- Support resume/skip-existing checkpoints and predictions.
- Run tiny local smoke training and evaluation.
- Completed one fixed held-out-aspect QLoRA configuration before full LOAO:
  - `example_filtered`;
  - LoRA rank `8`, alpha `16`, dropout `0.05`;
  - learning rate `1e-5`;
  - 1 epoch;
  - fixed held-out validation evaluation before test;
  - test pair samples F1 `0.5528`, pair micro F1 `0.5552`, pair macro F1 `0.4393`;
  - valid JSON rate `1.0000`, schema-valid rate `0.9964`.
- Record the expected cloud/local command templates for full 12-fold LOAO.

Boundary:

- The full 12-fold fine-tuned Qwen LOAO is deliberately excluded from this pre-GPU completion list.
- The current singleton SFT recipe failed a single-fold validation gate: best validation pair micro F1 `0.1900` on `Company brand: Competitor`, below same-fold Qwen zero-shot `0.2397` and local DistilBERT `0.2490`.
- Do not launch the full 12-fold run with this recipe; revise the absence-calibration objective first.

### 5. Thesis Skeleton Refresh

Purpose:

- Keep the LaTeX source aligned with the completed evidence.
- Avoid letting Markdown plans become a parallel dissertation draft.

Deliverables:

- Refresh `thesis/main.tex` placeholders so they mention:
  - completed Gemini fixed/cascade evidence;
  - completed Qwen zero-shot LOAO;
  - pending Qwen LoRA full LOAO.
- Keep the literature review focused on the hourglass structure:
  - review mining and ABSA;
  - multi-label metrics;
  - domain shift;
  - taxonomy shift and candidate labels;
  - structured-output LLMs and LoRA/QLoRA only as much as needed.
- Later, draft the results chapter directly in LaTeX after tables are frozen.

### 6. Repository Hygiene And Reproducibility

Purpose:

- Ensure every final claim has a tracked documentation record and every raw artifact stays local.

Required checks before each commit:

```powershell
python -m unittest discover -s tests
python -m compileall -q src scripts tests
git diff --check
rg -n "sk-[A-Za-z0-9_-]{12,}|OPENAI_API_KEY|OPENAI_BASE_URL|AIza|private-openai-compatible-endpoint" . --glob '!outputs/**' --glob '!data/**' --glob '!models/**' --glob '!checkpoints/**' --glob '!.git/**'
```

Never commit:

- `outputs/`;
- raw data;
- raw predictions;
- review text packets;
- model weights;
- adapter checkpoints;
- API keys;
- private endpoints.

## Explicitly Deferred Unless Requested

The following work should not be started by default while Qwen full LOAO GPU access is being negotiated:

- Full Gemini LOAO.
- More fixed-split Gemini prompt sweeps.
- More small DistilBERT LOAO hyperparameter tuning.
- More Qwen JSON-SFT ratio sweeps under the current singleton/grouped formulation.
- New model families such as joint pair scoring unless the Qwen path becomes blocked.
- Full 12-fold Qwen LoRA LOAO without a revised validation-passing recipe, confirmed GPU window, and resume plan.

## Decision Logic For Qwen LoRA Full LOAO

If a revised Qwen absence-calibration recipe passes validation and GPU access is granted:

- Run full 12-fold Qwen LoRA LOAO after the fixed SFT runner, smoke checks, and single-fold validation gate pass.
- Prefer one validated configuration over a wide sweep.
- Evaluate validation first, then test.
- Report all-row metrics, positive-gold diagnostics, valid JSON/schema rates, runtime, and per-aspect spread.

If the revised recipe does not pass validation or GPU access is not granted:

- Report the fixed Qwen SFT and single-fold all-row pilot as resource- and validation-gated adaptation evidence, not full LOAO robustness evidence.
- Keep Qwen zero-shot full LOAO as the open-weight robustness baseline.
- State the limitation explicitly: fine-tuned Qwen was not fully evaluated under LOAO taxonomy shift because the current SFT recipe failed the validation gate and a full 12-fold run was not justified.

## Next Immediate Order

1. Keep `docs/qwen_local_hybrid_direction.md` as the source of truth for the current modelling pivot.
2. Treat the local DistilBERT LOAO score/sentiment-confidence export as complete and reusable for future router diagnostics.
3. Treat the completed user-confirmed local-to-Qwen item 1 and item 2 as sufficient for the thesis-facing local-to-Qwen contribution: the global selected rule is `score_asym_rescue_le_0.05_confirm_le_0.30`, and the public Pareto CSV is `docs/thesis_figure_data/qwen_local_qwen_loao_pareto.csv`.
4. Keep lightweight defer routing as a non-default follow-up because it may overfit and may not beat the simple score-distance rule.
5. Keep per-aspect routing as an upper-bound diagnostic only; do not make it the main method because real new topics will not usually have their own validation fold.
6. Defer candidate-wise Qwen semantic judging and aspect-description/boundary-example prompting because they are a separate Qwen-inference route rather than a continuation of the completed score-distance router.
7. Use the existing full 12-fold Qwen LoRA command templates only after a revised recipe passes validation and GPU/storage conditions are confirmed.
