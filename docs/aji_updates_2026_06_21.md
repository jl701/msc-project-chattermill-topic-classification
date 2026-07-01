# Aji Updates: 2026-06-21

This note records the latest Slack exchange with Aji and the resulting project decisions. It is intended as a handoff note for future Codex/ChatGPT sessions.

## 1. User Progress Update Sent To Aji

The user told Aji that the initial clean repository setup had been pushed to GitHub, including reproducible loaders, split protocols, baseline scripts, tests, and documentation, while excluding data, outputs, checkpoints, and generated artifacts.

The update summarised the current evaluation setup:

- Closed-topic benchmark: use the provided FABSA train/validation/test split.
- Held-out organisation split: train excludes selected organisations, validation uses organisation `600`, and test uses organisations `369` and `727`.
- Held-out aspect split: hold out three aspects from training supervision and evaluate only those held-out aspect+sentiment labels. Candidate aspects are provided at inference.
- Held-out aspect training variants:
  - `label_masked`: keep training rows but remove held-out aspect labels.
  - `example_filtered`: remove any training row containing a held-out aspect.

The reported headline metric was sample-level F1 over aspect+sentiment pairs, with pair micro F1 and pair macro F1 also reported.

The reported headline results were:

| Setting | Result |
| --- | ---: |
| Closed-topic DistilBERT, test sample-level F1 | 0.7803 |
| Held-out organisation DistilBERT, test sample-level F1 | 0.7575 |
| Held-out organisation traditional SVM, test sample-level F1 | 0.7026 |
| Held-out aspect lexical lower bound, label-masked | 0.4698 |
| Held-out aspect lexical lower bound, example-filtered | 0.4626 |
| Held-out aspect candidate-aspect DistilBERT cross-encoder + global sentiment, label-masked | 0.5595 |
| Held-out aspect candidate-aspect DistilBERT cross-encoder + global sentiment, example-filtered | 0.5816 |
| Qwen3-4B indexed candidate-label zero-shot, validation | 0.5753 |
| Qwen3-4B indexed candidate-label zero-shot, test | 0.5374 |

The user also clarified that Qwen has not yet been fully fine-tuned for held-out-aspect evaluation. Completed Qwen work so far: indexed prompt design, zero-shot evaluation, error analysis, and GPU-ready SFT/evaluation JSONL preparation.

## 2. Aji's Evaluation Feedback

Aji said the work is excellent and moving quickly. He confirmed that the overall protocol is reasonable and does not need to be redesigned.

The key feedback was:

1. **Rotate held-out aspects before continuing.**
   - A single fixed choice of three held-out aspects makes the open-topic number fragile.
   - Scores may swing depending on whether the chosen aspects are common, rare, easy, or hard.
   - Aji suggested rotating held-out aspects, ideally leave-one-aspect-out if compute allows, and reporting the spread.
   - This should be done before heavy Qwen fine-tuning.

2. **Explain label-masked vs example-filtered carefully.**
   - Aji's hunch is that `example_filtered` may outperform `label_masked` because label masking injects false negatives.
   - In `label_masked`, the text may still contain a held-out aspect, but its supervision label is removed.
   - This should be treated as a meaningful ablation about incomplete-label noise, not just an arbitrary split variant.

3. **Check the global sentiment component.**
   - Aji asked whether the current "global sentiment" step predicts one polarity per document and applies it to all selected aspects.
   - This is correct for the current lexical and candidate-aspect cross-encoder baselines.
   - Since FABSA sentiment is per-aspect, mixed-sentiment documents may hurt pair-level held-out-aspect performance.

Aji also said to keep full Qwen fine-tuning parked until GPU resources are sorted.

## 3. Clarified Interpretation

### Leave-One-Aspect-Out

The next robustness step should be leave-one-aspect-out (LOAO) evaluation across all 12 FABSA aspects.

The current three-aspect held-out setup should be kept as the existing multi-candidate held-out-aspect baseline, but LOAO should be added to report robustness across aspects.

Important caveat for implementation:

- If LOAO evaluates only rows containing the held-out aspect and supplies only one candidate aspect, aspect selection becomes too easy.
- Use LOAO carefully as a robustness/per-aspect diagnostic.
- Consider whether evaluation should include negative candidate rows or use a candidate set that still makes false positives measurable.

### Label-Masked vs Example-Filtered

The two variants should be explained as follows:

- `label_masked` keeps more training rows but creates incomplete-label noise. A row may contain held-out-topic language while the held-out label is removed from supervision.
- `example_filtered` removes these potentially noisy rows, giving cleaner training supervision but less data.

This difference can be written up as an ablation on censored or incomplete held-out-topic supervision.

### Global Sentiment Limitation

The current held-out-aspect lexical baseline and candidate-aspect cross-encoder baseline use global sentiment:

1. Predict relevant candidate aspects.
2. Predict one sentiment from the whole review text.
3. Apply that same sentiment to every selected aspect in the row.

This was a quick first baseline to separate candidate aspect selection from sentiment prediction. It is not a final ABSA sentiment model.

Why it matters:

- FABSA sentiment is per-aspect, not per-document.
- A mixed-sentiment review can require different sentiments for different aspects.
- Global sentiment can therefore select the right aspect but assign the wrong pair label, lowering pair-level F1.

Useful next improvements:

- Aspect-conditioned sentiment classifier: `(review, candidate aspect) -> positive/negative/neutral`.
- Joint aspect+sentiment pair scorer: `(review, candidate aspect | sentiment) -> applicable/not applicable`.
- Qwen/Gemini indexed candidate-label outputs, which naturally allow different sentiments for different aspects.

## 4. GitHub Repository Fix

Aji noted that `https://github.com/jl701/msc-project-chattermill-topic-classification` appeared to contain only the initial minimal commit.

Cause:

- The full clean setup had been pushed to `master`.
- The repository default branch was `main`.
- Aji was seeing the stale default `main` branch.

Fix completed on 2026-06-21:

- Remote `main` was aligned with the full setup commit `868e0baa541839fe027473568b0304edb6ed9df4`.
- Local branch was renamed from `master` to `main`.
- Local `main` now tracks `origin/main`.
- GitHub default branch `main` now shows the full README, project overview, code, tests, and docs.

## 5. Local Environment Update

On the new machine, the project environment was configured on 2026-06-21. Keep both path roots in future handoffs, because the project may be opened from either machine:

- Current/new workspace: `D:\Msc_Project`
- Previous/alternate workspace: `C:\Msc_DSML\Msc_Project`
- Current/new repository: `D:\Msc_Project\msc-project-chattermill-topic-classification`
- Previous/alternate repository: `C:\Msc_DSML\Msc_Project\msc-project-chattermill-topic-classification`
- Current/new FABSA export: `D:\Msc_Project\Project_Preparation\Public_Datasets\FABSA`
- Previous/alternate FABSA export: `C:\Msc_DSML\Msc_Project\Project_Preparation\Public_Datasets\FABSA`
- Python: 3.11.9
- Git: 2.54.0
- Virtual environment: `.venv`
- PyTorch: `2.11.0+cu128`
- CUDA available: yes
- GPU: NVIDIA GeForce GTX 1660 Ti with Max-Q Design, 6 GB VRAM
- Unit tests: `29 tests OK`

The local GPU is useful for small DistilBERT and smoke-test work, but it is still not ideal for full Qwen fine-tuning.

## 6. Gemini / Vertex AI Access From Aji

Aji provided access to a private Gemini endpoint through an OpenAI-compatible API. The endpoint URL and actual API key must stay in local environment variables only and must not be committed or written into project files.

Local setup pattern:

```bash
export OPENAI_BASE_URL="<private-openai-compatible-endpoint>/v1"
export OPENAI_API_KEY="<set-locally-only>"
```

Recommended model IDs:

| Model ID | Notes |
| --- | --- |
| `vertex_ai/gemini-2.5-flash` | Fast and cheap; default first choice |
| `vertex_ai/gemini-2.5-pro` | Most capable |
| `vertex_ai/gemini-2.5-flash-lite` | Cheapest and lowest latency |

Operational notes:

- Keep endpoint URLs, API keys, budgets, and provider-specific access details out of tracked project files.
- Use the `vertex_ai/` model IDs with the private endpoint only when that access is configured locally.
- Gemini 2.5 spends tokens thinking first; too-low `max_tokens` can produce empty content, so use a few hundred tokens.

Suggested smoke test:

```bash
curl -s "$OPENAI_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "content-type: application/json" \
  -d '{"model":"vertex_ai/gemini-2.5-flash","max_tokens":200,"messages":[{"role":"user","content":"ping"}]}'
```

## 7. Updated Work Plan

The immediate plan should change from "full Qwen fine-tuning next" to:

1. Implement and run leave-one-aspect-out held-out-aspect evaluation across the 12 FABSA aspects.
2. Report spread across aspects: mean, standard deviation, min, max, and per-aspect table.
3. Preserve and explain `label_masked` vs `example_filtered` as an incomplete-label-noise ablation.
4. Add diagnostics separating aspect selection errors from sentiment-coupling errors.
5. Improve the global sentiment component with either:
   - aspect-conditioned sentiment, or
   - joint aspect+sentiment candidate scoring.
6. Add a Gemini OpenAI-compatible baseline using the same indexed candidate-label output format as Qwen.
7. Keep full Qwen fine-tuning parked until stronger GPU access is available.

Gemini should be treated as a hosted LLM baseline and practical trade-off measurement path, not as a replacement for the robustness and label-aware evaluation fixes above.
