# Handoff Notes: 2026-06-16

These notes capture practical context that is useful for continuing the project but is not central enough for the main project overview.

## Communication And Code Style

- Communicate with the user in Chinese.
- Keep code, docstrings, comments, README content, and project documentation in English.
- Prefer concise, readable code because the repository will be reviewed by Aji.
- Avoid over-engineering; use small, explicit modules and scripts that are easy to inspect.

## Repository State

- The repository is still in setup stage and appears uncommitted locally.
- Current `git status --short` shows the project files as untracked.
- `outputs/` is ignored and contains local experiment results, split manifests, and per-label reports.
- Do not commit `outputs/`, model checkpoints, private/internal data, credentials, API keys, or confidential outputs.

## Current Best Local Results

The main documents record the headline results. The most useful operational summary is:

- Closed-topic DistilBERT remains the strongest full-split result so far.
- Traditional closed-topic and cross-organisation baselines appear close to a plateau after threshold tuning and small refined SVM sweeps.
- Held-out-organisation DistilBERT now clearly improves over the traditional cross-organisation baseline.
- The held-out-aspect lexical baseline is intentionally weak and should be treated as a lower bound, not as a serious final open-topic method.
- The first stronger held-out-aspect baseline is a candidate-aspect DistilBERT cross-encoder with a global TF-IDF Logistic Regression sentiment classifier. It improves over the lexical lower bound for both label-masked and example-filtered training.
- Qwen held-out-aspect zero-shot now works with an indexed candidate-label prompt. Plain canonical string prompts often produce near-miss aspect names; indexed prompts avoid that issue.
- Qwen has not yet been fully fine-tuned for the held-out-aspect protocol. The current full validation/test held-out-aspect Qwen numbers are zero-shot prompt results, while SFT/evaluation JSONL generation is prepared for stronger GPU access.
- Local Qwen QLoRA is feasible only for smoke tests on the laptop GPU; full Qwen experiments should wait for stronger GPU access.

Current best generalisation results:

```text
Held-out organisation, DistilBERT:
  test pair samples F1: 0.7575
  test pair micro F1:   0.7600
  test pair macro F1:   0.4035

Held-out aspect, candidate-aspect cross-encoder + global sentiment:
  label-masked test pair samples F1:     0.5595
  example-filtered test pair samples F1: 0.5816

Held-out aspect, Qwen3-4B indexed zero-shot:
  validation pair samples F1: 0.5753
  test pair samples F1:       0.5374
  valid JSON rate:            1.0000
```

## Recommended Continuation Order

1. Commit and push the clean setup if the user approves.
2. Use `docs/heldout_aspect_error_analysis.md` to guide the next Qwen fine-tuning design.
3. Run full Qwen candidate-label fine-tuning on a stronger GPU using the indexed SFT/evaluation JSONL files.
4. Add one more non-LLM label-aware held-out-aspect baseline, such as a bi-encoder or NLI-style model, if time allows.

## Useful Next Modelling Ideas

For held-out aspects, fixed-output classifiers are not enough because the evaluation labels are unseen during training. The candidate-aspect cross-encoder now consumes candidate labels at inference. Further useful baselines include:

- Bi-encoder similarity: encode feedback and candidate aspect text separately, then score candidate relevance.
- NLI-style formulation: convert each candidate aspect into a hypothesis and classify entailment/relevance.
- Qwen candidate-label prompting later, once GPU access is clearer.

For Qwen held-out-aspect work:

- Use indexed candidate labels and `aspect_id` outputs.
- Avoid free-form aspect strings in the prompt target.
- Prepared local SFT/evaluation files live under `outputs/qwen_heldout_aspect_sft_indexed/`.
- The output directory is ignored by Git; regenerate it with `scripts/prepare_qwen_heldout_aspect_sft_data.py` if needed.
- The method is grounded in instruction-tuned/generative ABSA and entailment-style zero-shot classification; see `docs/qwen_feasibility.md`.

For cross-organisation evaluation, the next useful step is error analysis rather than another small hyperparameter sweep. The current DistilBERT validation-selected run uses learning rate `6e-5`, square-root positive-class weighting, best epoch `8`, and threshold `0.44`.

## Important Local Paths

```text
Workspace:
C:\Msc_DSML\Msc_Project

Repository:
C:\Msc_DSML\Msc_Project\msc-project-chattermill-topic-classification

FABSA local export:
C:\Msc_DSML\Msc_Project\Project_Preparation\Public_Datasets\FABSA
```

## Verification Command

Before handing work back to the user, run:

```powershell
python -m unittest discover -s tests
```

The current expected result is:

```text
29 tests OK
```
