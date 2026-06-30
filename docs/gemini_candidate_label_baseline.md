# Gemini Hosted Candidate-Label Baseline

Last updated: 2026-07-01

This note records the implementation status and reproduction plan for the hosted Gemini candidate-label baseline. The baseline is designed to match the existing Qwen indexed held-out-aspect protocol as closely as possible while adding the hosted-LLM diagnostics requested by Aji.

## Current Status

Implemented:

- `scripts/run_gemini_heldout_aspect.py`
- shared LLM candidate-label utilities in `src/msc_project/llm/candidate_label.py`
- parser/schema diagnostics tests in `tests/test_llm_candidate_label.py`
- dry-run validation of request construction on the fixed held-out-aspect validation split

Not yet run:

- real Gemini API smoke test
- validation prompt/decoding sweep
- full fixed held-out-aspect validation/test evaluation

Blocked reason:

- no Gemini/OpenAI-compatible API key is currently available in the process environment;
- `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY`, and `GOOGLE_APPLICATION_CREDENTIALS` were all missing during the 2026-07-01 implementation session.

No Gemini F1 result should be reported until a real hosted run has completed.

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
- visible output tokens
- reasoning/thinking tokens when reported
- total tokens
- estimated cost if token counts and cost rates are supplied

Token usage extraction supports both OpenAI-style fields and Gemini-style `usageMetadata` fields such as `thoughtsTokenCount`.

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
python .\scripts\run_gemini_heldout_aspect.py --split both --limit 10000 --prompt-variant indexed --response-format json_schema --response-format-fallback --output-dir .\outputs\llm\gemini_candidate_label_YYYYMMDD_HHMMSS
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
| Full unit test suite | 59 tests OK |
| Compile check | passed |
| Dry-run request construction | passed |
| Real API call | blocked by missing credentials |

## Next Step

Run the 5-row hosted smoke test once a key is available, then compare at least `indexed`, `indexed_conservative`, and `indexed_descriptive` on a sampled validation subset before running full validation/test.
