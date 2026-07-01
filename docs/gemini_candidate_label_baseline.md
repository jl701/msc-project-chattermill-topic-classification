# Gemini Hosted Candidate-Label Baseline

Last updated: 2026-07-01

This note records the implementation status and reproduction plan for the hosted Gemini candidate-label baseline. The baseline is designed to match the existing Qwen indexed held-out-aspect protocol as closely as possible while adding the hosted-LLM diagnostics requested by Aji.

## Current Status

Implemented:

- `scripts/run_gemini_heldout_aspect.py`
- shared LLM candidate-label utilities in `src/msc_project/llm/candidate_label.py`
- parser/schema diagnostics tests in `tests/test_llm_candidate_label.py`
- dry-run validation of request construction
- real hosted Gemini smoke test
- small validation sweep over prompt/max-token settings
- full fixed held-out-aspect validation/test evaluation
- targeted fixed-split Gemini Flash error analysis
- 50-row Gemini Pro fixed-split subset comparison
- full fixed-split Gemini Pro validation/test evaluation
- full fixed-split Gemini Flash-Lite validation/test evaluation
- local-to-Gemini uncertainty cascade sweep over Flash-Lite, Flash, and Pro escalators

Completed hosted result:

```text
Model: vertex_ai/gemini-2.5-flash
Endpoint: Chattermill Vertex AI OpenAI-compatible endpoint
Prompt: indexed candidate labels
Response format: json_schema
Max tokens: 2048
Validation pair samples F1: 0.6146
Test pair samples F1: 0.6071
Test valid JSON rate: 1.0000
Test schema-valid rate: 1.0000
```

Completed hosted Pareto extensions:

```text
Model: vertex_ai/gemini-2.5-pro
Test pair samples F1: 0.7141
Test pair micro F1: 0.7425
Test pair macro F1: 0.6287
Test mean latency: 5.6510 seconds/example
Validation + test approximate cost: $2.9781

Model: vertex_ai/gemini-2.5-flash-lite
Test pair samples F1: 0.5516
Test pair micro F1: 0.5872
Test pair macro F1: 0.4876
Test mean latency: 0.3719 seconds/example
Validation + test approximate cost: $0.0171
```

Completed selective cascade:

| System | Test Pair Samples F1 | Test Pair Micro F1 | Test Pair Macro F1 | Test Aspect Samples F1 | Test Gemini Cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Local DistilBERT pipeline | 0.6071 | 0.5917 | 0.4890 | 0.6651 | n/a |
| Local -> Flash-Lite cascade | 0.6679 | 0.6579 | 0.5401 | 0.7259 | $0.0052 |
| Local -> Flash cascade | 0.7459 | 0.7348 | 0.6223 | 0.7993 | $0.2789 |
| Local -> Pro cascade | 0.8102 | 0.7955 | 0.6809 | 0.8493 | $1.5421 |

The cascade details are documented in `docs/local_gemini_cascade.md`.

Historical note: the first implementation pass was blocked because no compatible credentials were available in the process environment. The hosted run below was completed after the user supplied Aji's endpoint details and key. The key was used only as a process-local environment variable and was not written to the repository.

## Protocol

The runner uses the fixed held-out-aspect protocol:

- held-out aspects:
  - `Account management: Account access`
  - `Company brand: Competitor`
  - `Value: Discounts promotions`
- default training strategy metadata: `example_filtered`
- evaluation label scope: held-out labels only
- evaluation row scope: rows containing at least one held-out aspect
- default prompt variant: `indexed`

The default prompt supplies candidate IDs:

```text
Candidate aspects:
A1. Account management: Account access
A2. Company brand: Competitor
A3. Value: Discounts promotions
```

The model is asked to return candidate IDs rather than copied aspect names. This matches the Qwen indexed zero-shot prompt that avoided near-miss labels such as `Account management`.

## JSON Mode And Schema

The runner supports three response-format modes:

| Mode | Request setting | Prompt container | Intended use |
| --- | --- | --- | --- |
| `json_schema` | strict `response_format` JSON schema | `{"labels": [...]}` | preferred when the endpoint supports structured outputs |
| `json_object` | JSON-object response format | `{"labels": [...]}` | fallback JSON mode when strict schema is unavailable |
| `none` | no `response_format` | `[...]` by default | compatibility fallback matching the Qwen top-level array protocol |

The parser accepts both the Qwen-style top-level array and the JSON-mode object wrapper, but diagnostics distinguish valid JSON from schema-valid output.

Invalid handling:

- invalid JSON is scored as an empty prediction;
- top-level non-list/non-wrapper JSON is scored as an empty prediction;
- unknown candidate IDs or labels are dropped and counted;
- invalid sentiments are dropped and counted;
- duplicate valid pairs are deduplicated and counted;
- conflicting sentiments for the same aspect are counted as schema issues;
- exact aspect-name references can be mapped for scoring but make indexed schema validity fail.

## Diagnostics

For each completed run the runner reports:

- pair samples F1
- pair micro F1
- pair macro F1
- aspect samples F1
- sentiment accuracy when the gold aspect is predicted
- valid JSON rate
- schema-valid rate
- parse failure count
- invalid candidate label count
- invalid sentiment count
- duplicate prediction count
- conflicting sentiment count
- empty prediction count
- mean and median latency
- input tokens
- output/completion tokens
- reasoning/thinking tokens when reported
- total tokens
- estimated cost if token counts and cost rates are supplied

Token usage extraction supports both OpenAI-style fields and Gemini-style `usageMetadata` fields such as `thoughtsTokenCount`. For the Chattermill endpoint, `output_tokens` appears to include thinking tokens: `input_tokens + output_tokens = total_tokens`, while `reasoning_tokens` is a subset of `output_tokens`. Therefore cost estimates below bill output tokens once and report reasoning tokens separately.

## Hosted Results

### Smoke Test

Command:

```powershell
python .\scripts\run_gemini_heldout_aspect.py --split validation --limit 5 --response-format json_schema --response-format-fallback --output-dir .\outputs\llm\gemini_candidate_label_20260701_0110_smoke
```

Result:

| Metric | Value |
| --- | ---: |
| Examples | 5 |
| Pair samples F1 | 0.4000 |
| Pair micro F1 | 0.5455 |
| Pair macro F1 | 0.2222 |
| Aspect samples F1 | 0.4000 |
| Valid JSON rate | 1.0000 |
| Schema-valid rate | 1.0000 |
| Mean latency seconds | 2.5497 |
| Input tokens | 960 |
| Output tokens, including thinking | 1,832 |
| Reasoning tokens | 1,661 |

The smoke test confirmed that the endpoint accepts `response_format=json_schema` and returns reasoning-token diagnostics.

### Validation Sweep

All sweep rows used a sampled 50-row validation subset with seed 13.

| Configuration | Max Tokens | Pair Samples F1 | Pair Micro F1 | Pair Macro F1 | Valid JSON | Schema Valid | Reasoning Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `indexed` | 512 | 0.5867 | 0.7033 | 0.5353 | 0.8400 | 0.8400 | 15,803 |
| `indexed_conservative` | 512 | 0.4933 | 0.6207 | 0.4215 | 0.8200 | 0.8000 | 15,747 |
| `indexed_descriptive` | 512 | 0.5067 | 0.6429 | 0.4575 | 0.7800 | 0.7800 | 16,774 |
| `indexed` | 1024 | 0.6733 | 0.7475 | 0.5453 | 0.9800 | 0.9800 | 17,902 |
| `indexed` | 2048 | 0.6733 | 0.7327 | 0.5287 | 1.0000 | 1.0000 | 18,362 |

The first sweep exposed the exact failure mode Aji warned about: with `max_tokens=512`, Gemini often spent nearly all completion tokens on thinking and returned truncated JSON such as `{ "labels":`. Raising `max_tokens` solved the parse failures. A tiny `thinking_budget=0` smoke test was accepted by the endpoint but did not materially suppress reasoning tokens, so it was not used as the final configuration.

The selected configuration is:

```text
prompt_variant=indexed
response_format=json_schema
max_tokens=2048
temperature=0
```

This was selected because it achieved perfect schema validity on the sampled sweep while keeping predictive performance in the same band as the best 1024-token run.

### Full Fixed Held-Out-Aspect Evaluation

Command:

```powershell
python .\scripts\run_gemini_heldout_aspect.py --split both --limit 10000 --prompt-variant indexed --response-format json_schema --response-format-fallback --max-tokens 2048 --output-dir .\outputs\llm\gemini_candidate_label_20260701_0145_fixed_full
```

| Split | Examples | Pair Samples F1 | Pair Micro F1 | Pair Macro F1 | Aspect Samples F1 | Sentiment Accuracy When Gold Aspect Predicted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 212 | 0.6146 | 0.6446 | 0.4830 | 0.7129 | 0.8639 |
| test | 281 | 0.6071 | 0.6541 | 0.5547 | 0.6747 | 0.9052 |

Structured-output diagnostics:

| Split | Valid JSON | Schema Valid | Parse Failures | Invalid Candidates | Invalid Sentiments | Duplicates | Conflicts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 1.0000 | 0.9953 | 0 | 0 | 0 | 0 | 1 |
| test | 1.0000 | 1.0000 | 0 | 0 | 0 | 0 | 0 |

Latency and tokens:

| Split | Mean Latency | Median Latency | Input Tokens | Output Tokens, Including Thinking | Reasoning Tokens | Total Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 2.5074 s | 2.0986 s | 41,960 | 89,052 | 80,780 | 131,012 |
| test | 2.3721 s | 2.0598 s | 56,338 | 113,067 | 102,574 | 169,405 |

Approximate cost using public Gemini 2.5 Flash Standard rates from the [Gemini API pricing page](https://ai.google.dev/gemini-api/docs/pricing) as checked on 2026-07-01 (`$0.30 / 1M input tokens`, `$2.50 / 1M output tokens`, with output tokens including thinking):

| Split | Estimated Cost | Cost / 1k Reviews | Cost / 1M Reviews |
| --- | ---: | ---: | ---: |
| validation | $0.2352 | $1.1095 | $1,109.52 |
| test | $0.2996 | $1.0661 | $1,066.08 |
| validation + test | $0.5348 | $1.0848 | $1,084.76 |

This is an approximate public-rate calculation, not a Chattermill billing statement. It is useful for dissertation-scale comparison because it includes reasoning tokens through the billable output-token count.

## Interpretation

Gemini Flash is now a serious fixed held-out-aspect baseline. On the test split it matches the current strongest non-LLM fixed held-out-aspect headline score:

| Model | Test Pair Samples F1 | Test Pair Micro F1 | Test Pair Macro F1 | Test Aspect Samples F1 |
| --- | ---: | ---: | ---: | ---: |
| Candidate-aspect DistilBERT selector + DistilBERT aspect-conditioned sentiment | 0.6071 | 0.5917 | 0.4890 | 0.6651 |
| Qwen3-4B-Instruct indexed zero-shot | 0.5374 | 0.5300 | 0.4374 | 0.6340 |
| Gemini 2.5 Flash indexed JSON-schema | 0.6071 | 0.6541 | 0.5547 | 0.6747 |

Gemini is stronger than Qwen zero-shot and has better pair micro/macro F1 than the local non-LLM fixed result, while matching the local headline pair samples F1. However, the comparison is still only for the fixed three-aspect held-out protocol. It should not be treated as LOAO robustness evidence, and it should be discussed alongside latency, hosted-API governance, and cost.

The fixed-split error analysis in `docs/gemini_error_analysis.md` explains why the headline samples F1 ties the local DistilBERT result while micro/macro F1 improves. Gemini predicts fewer labels per row than the local candidate-aspect DistilBERT pipeline (`1.0498` versus `1.2811`) and has much higher pair precision (`0.6475` versus `0.5333`), but it also returns 42 empty predictions and misses many `Company brand: Competitor` labels. The local pipeline has slightly more exact rows, while Gemini has fewer aspect over-prediction rows.

A deterministic 50-row test subset was then used for a small Gemini Pro comparison. On the same rows, Pro improved over Flash from `0.6360` to `0.6933` pair samples F1, from `0.6731` to `0.7379` pair micro F1, and from `0.4627` to `0.5250` pair macro F1. The trade-off was practical: Pro was about `2.25x` slower and `5.53x` more expensive than Flash on this subset. This subset result justified full fixed-split Pro validation/test, but not full Pro LOAO.

The full fixed-split Pro run confirmed the subset direction:

| Model | Test Pair Samples F1 | Test Pair Micro F1 | Test Pair Macro F1 | Test Aspect Samples F1 | Mean Latency | Validation + Test Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemini 2.5 Flash-Lite | 0.5516 | 0.5872 | 0.4876 | 0.6062 | 0.3719 s | $0.0171 |
| Gemini 2.5 Flash | 0.6071 | 0.6541 | 0.5547 | 0.6747 | 2.3721 s | $0.5348 |
| Gemini 2.5 Pro | 0.7141 | 0.7425 | 0.6287 | 0.7746 | 5.6510 s | $2.9781 |

This is now a useful hosted Pareto comparison. Flash-Lite is extremely cheap and fast but below the strongest local/Flash results. Flash is the balanced hosted baseline and matches the strongest local fixed-split headline result. Pro is clearly strongest on the fixed split, but with substantially higher latency and cost.

The local-to-Gemini cascade is the strongest fixed-split system result so far. It selects escalation policies using validation-only local reliability features, then evaluates those policies on test. Validation-selected selective escalation reaches `0.6679` test pair samples F1 with Flash-Lite, `0.7459` with Flash, and `0.8102` with Pro. This is more dissertation-relevant than full Gemini LOAO because it tests a realistic selective-deployment pattern: local model first, hosted LLM only for locally uncertain rows.

## Reproduction Commands

Dry-run request construction without calling the API:

```powershell
python .\scripts\run_gemini_heldout_aspect.py --dry-run --split validation --limit 2 --response-format json_schema --output-dir .\outputs\llm\gemini_candidate_label_dry_run_check
```

Tiny hosted smoke test once credentials are available:

```powershell
$env:OPENAI_BASE_URL="https://llm-api.datascience.chattermill.xyz/v1"
# Set OPENAI_API_KEY locally before running; do not write it into the repository.
python .\scripts\run_gemini_heldout_aspect.py --split validation --limit 5 --response-format json_schema --response-format-fallback --output-dir .\outputs\llm\gemini_candidate_label_YYYYMMDD_HHMMSS
```

Small validation sweep:

```powershell
python .\scripts\run_gemini_heldout_aspect.py --split validation --limit 50 --sample --prompt-variant indexed --response-format json_schema --response-format-fallback --output-dir .\outputs\llm\gemini_candidate_label_sweep_json_schema

python .\scripts\run_gemini_heldout_aspect.py --split validation --limit 50 --sample --prompt-variant indexed_conservative --response-format json_schema --response-format-fallback --output-dir .\outputs\llm\gemini_candidate_label_sweep_conservative

python .\scripts\run_gemini_heldout_aspect.py --split validation --limit 50 --sample --prompt-variant indexed_descriptive --response-format json_schema --response-format-fallback --output-dir .\outputs\llm\gemini_candidate_label_sweep_descriptive
```

Full fixed held-out-aspect validation/test run after selecting the best configuration:

```powershell
python .\scripts\run_gemini_heldout_aspect.py --split both --limit 10000 --prompt-variant indexed --response-format json_schema --response-format-fallback --max-tokens 2048 --output-dir .\outputs\llm\gemini_candidate_label_YYYYMMDD_HHMMSS
```

Cost rates are intentionally explicit rather than hard-coded:

```powershell
python .\scripts\run_gemini_heldout_aspect.py --split validation --limit 50 --input-cost-per-1m 0.0 --output-cost-per-1m 0.0 --reasoning-cost-per-1m 0.0
```

Replace the cost-rate values with the current endpoint-specific rates before using cost estimates in the dissertation.

## Output Files

Each run writes to an ignored `outputs/llm/gemini_candidate_label_*` directory:

- `summary.json`
- `requests_<split>_<variant>.jsonl`
- `predictions_<split>_<variant>.jsonl` for real API runs

Generated outputs may contain review text and must not be committed.

## Current Validation

Completed on 2026-07-01:

```powershell
python -m unittest tests.test_llm_candidate_label
python -m unittest discover -s tests
python -m compileall -q src scripts tests
python .\scripts\run_gemini_heldout_aspect.py --dry-run --split validation --limit 2 --response-format json_schema --output-dir .\outputs\llm\gemini_candidate_label_dry_run_check
```

Results:

| Check | Result |
| --- | --- |
| Parser/schema diagnostics tests | 11 tests OK |
| Full unit test suite | 63 tests OK |
| Compile check | passed |
| Dry-run request construction | passed |
| Hosted API smoke test | passed |
| 50-row validation sweep | completed |
| Full validation/test evaluation | completed |
| Local-to-Gemini cascade sweep | completed |

## Remaining Follow-Up

Do not run full Gemini LOAO by default. Full Pro validation/test, Flash-Lite validation/test, and the local-to-Gemini cascade are now complete. A full Pro LOAO remains unjustified without a separate dissertation-value argument.

The remaining Gemini/local experiments should be framed around dissertation value rather than raw leaderboard chasing:

1. Test Gemini-generated candidate-aspect descriptions as label-representation support, generated without validation/test leakage.
2. Use Gemini Pro as a qualitative error-taxonomy aid with manual review, not as an automatic evaluator.
3. If time and budget permit, export local DistilBERT selector scores in a future rerun and repeat the cascade with calibrated score/margin uncertainty features.

This sequence supports a stronger industrial MSc story: local models, cheap hosted models, and stronger hosted models can be compared not only on F1, but also on cost, latency, governance, and selective deployment strategy.
