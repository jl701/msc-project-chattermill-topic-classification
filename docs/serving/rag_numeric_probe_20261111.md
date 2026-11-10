# Registered raw-logit saturation diagnostic

The completed development matrix contains many presence probabilities stored as
exactly `1.0`. This diagnostic asks how much ranking information is hidden by
float32 softmax rounding. It does not change model weights, prompts, score
precision, threshold selection or the core inference implementation.

`scripts/probe_rag_logit_margins_gpu.py` uses the original frozen serving
artifacts, cleaned retrieval index and the input-only development replay. It
selects twenty evenly spaced character-length ranks, with request ID breaking
ties and Python's `round` defining rank positions. Selection does not use labels
or predictions. Both fixed and E5 demonstration methods use those same inputs;
prefix caching and retrieval fallback are disabled. The encoder is the existing
local pinned E5 snapshot, with query encoding on CPU.

Before collecting margins, three development length ranks pass the legacy
full-grid versus sentiment-only correctness gate. This requires identical
encoded Stage2 inputs and truncation diagnostics, reports all 12-by-3 score
differences and checks decoded per-aspect labels at the frozen runner threshold.

A forward hook on the outer model receives the original call arguments. It
uses the same last-nonpadding positions and the same retained-position mapping
for `logits_to_keep` as the serving engine. It never returns replacement outputs
or modifies a model/input tensor. For every aspect pair it records the original
Y and N logits, their difference after conversion to float64, the current
float32 softmax probability, and a diagnostic float64 softmax probability.
Captured float32 probabilities must exactly reproduce the engine response;
otherwise the probe stops instead of interpreting misaligned logits.

Private results include per-pair values and summaries for all, seen-candidate
and held-out-candidate pairs. These scopes refer to candidate identities, not
review gold labels. Summaries count probabilities exactly zero or one at each
precision, give raw margin ranges, and count distinct margins and float64
probabilities inside the float32-one bucket. Float64 may itself saturate; raw
margins therefore remain part of the evidence.

The hook introduces extra arithmetic and CPU transfers, so elapsed times are not
latency benchmarks. Only after this diagnostic is inspected may a separately
registered common stable-score branch be considered. This script does not
promote such a branch or modify the historical evaluation scores.
