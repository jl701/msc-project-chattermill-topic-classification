# Boundary-only comparator routing registration

Registered 2026-11-11 before final classifier freeze or evaluation-label analysis.
This extends the approved manual-review study; classification and its seen-only
selection are unchanged. No GPU experiment is started by this registration.

For every candidate manual-review margin threshold:

1. Mandatory failures, empty outputs, severe query truncation and selected
   sentiment-input truncation go to review. Margins below the candidate threshold
   also go to review, without a comparator call.
2. Among the remaining development reviews, rank the observable risk margin,
   breaking ties by SHA256 of `review-routing-boundary:20261111:<request_id>`.
   The risk margin uses all candidate presence decisions and selected sentiments.
3. Take the lowest `ceil(0.20 * remaining_count)` ranks and freeze the largest
   margin/hash pair. This gate uses no gold labels. Each candidate threshold has
   its own gate, so a confidence-only review-all result does not preclude a
   different disagreement policy.
4. Only eligible reviews within that frozen band request the existing QLoRA
   comparator. A disagreement or an unavailable/failed comparator goes to human
   review. Other eligible reviews retain the primary classifier's output.
5. Apply the existing 5%/minimum10 random audit after targeted review. Select the
   qualified calibration policy by most automatic reviews, then fewer comparator
   calls, then higher manual margin. Evaluation never recomputes the boundary
   quantile and never changes the classifier or review thresholds.

Development eligibility and final qualification are distinct under the separate
`review_routing_20261111.json` registration; see
`rag_routing_selection_20261111.md`. The boundary construction, candidate grid,
tie order and audit remain unchanged. The original CP-based development choice
is retained only as a descriptive sensitivity, never an evaluation-time option.

Report additional full-review calls, candidate score requests, observed sentiment
prompt counts, missing/failed comparator responses and the stable call manifest.
An all-review fallback incurs zero comparator calls. A stored historical HTTP
latency is a retrospective cost proxy; offline scorer time is a different proxy.
Neither is reported as measured combined-path service latency or its P95. Actual
routed serving latency needs a separate combined-runtime measurement.

Raw comparator records, private request IDs and manifests stay in ignored local
outputs. Only aggregate findings and implementation belong in tracked reports.
