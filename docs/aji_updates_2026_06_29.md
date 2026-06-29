# Aji Updates: 2026-06-29

This note records the latest Aji guidance shared by the user from a Slack screenshot. It should be used as a future reminder when resuming Gemini/LLM experiments and when framing the open-topic claim in the dissertation.

## Source Context

The user had replied to Aji that:

- the Gemini/Vertex AI key had been received and would be kept local only, without being committed anywhere;
- the endpoint would be useful for adding a hosted LLM baseline alongside local Qwen and DistilBERT work;
- the first Gemini test should be a small smoke test with `vertex_ai/gemini-2.5-flash`;
- the Gemini prompt should reuse the same indexed candidate-label JSON format as the Qwen held-out-aspect prompt;
- the smoke test should record valid JSON rate, latency, token usage, and F1 on a small validation sample before scaling;
- leave-one-aspect-out should still come before Gemini scaling, because it makes the open-topic evaluation more robust;
- Gemini Flash should be treated as the default cheap baseline, with Pro reserved for a smaller subset if needed;
- the Gemini `max_tokens`, reasoning-token, and `europe-west4` model-ID details were noted.

## Aji's Reply

Aji confirmed that the plan was right.

Key guidance:

1. **Leave-one-aspect-out matters most for the open-topic claim.**
   - Aji specifically said the LOAO setup is the part that matters most for the open-topic claim.
   - This supports treating LOAO as the robustness evidence around unseen-aspect generalisation.

2. **Use JSON mode / response format if the endpoint supports it.**
   - If the Gemini endpoint honours `response_format` or JSON mode, use it instead of only measuring valid-JSON rate.
   - Expected benefit: fewer parse failures and cleaner F1.

3. **Count Gemini reasoning tokens in cost comparisons.**
   - Cost comparisons must include Gemini reasoning/thinking tokens, not only visible output tokens.
   - Flash only remains a cheap baseline once thinking tokens are included in the total.
   - Otherwise Flash may look cheaper than it really is.

## Project Implications

For future Gemini experiments, record at least:

- model ID;
- endpoint region and route;
- prompt variant;
- whether `response_format` / JSON mode was used;
- validation/test split and sample size;
- pair samples F1, pair micro F1, pair macro F1;
- aspect-only F1 diagnostics;
- valid JSON / parse-failure rate;
- input tokens;
- visible output tokens;
- reasoning/thinking tokens if reported;
- total billable tokens;
- latency per example;
- estimated cost per 1,000 and 1,000,000 reviews.

Do not compare hosted Gemini, Qwen, and local DistilBERT only on F1. The dissertation should also discuss cost, latency, operational complexity, privacy/data-governance constraints, and output reliability.

## Status After This Guidance

Since this Slack exchange, the LOAO lexical robustness diagnostics have been implemented and documented, and the strongest local non-LLM fixed held-out-aspect baseline has been completed:

```text
Candidate-aspect DistilBERT selector
+ DistilBERT aspect-conditioned sentiment
```

Best fixed held-out-aspect result:

| Metric | Value |
| --- | ---: |
| Pair samples F1 | 0.6071 |
| Pair micro F1 | 0.5917 |
| Pair macro F1 | 0.4890 |

The next major modelling phase remains Gemini or Qwen candidate-label evaluation, but it is intentionally paused while the dissertation literature review and framework are developed.

