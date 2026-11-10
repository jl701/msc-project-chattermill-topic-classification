# Shared raw-margin development branch

This is the first of the two permitted directed development improvements. It is
registered after the seven-method historical development matrix and before any
historical evaluation or new-source gold is analysed. The original probability
matrix, thresholds and results remain immutable. Configuration:
`configs/experiments/serving_rag_margin_20261111.json`.

The original seen-only development comparison selected fixed examples: pair F1
64.20 versus63.32 for the best dynamic method,E5 (0–100 scale). Dynamic retrieval
reduced false negatives but increased false positives. All57 fixed false positives
and84 of88 E5 false positives were stored in a probability32 bucket equal to1.
This prevents threshold selection from distinguishing them from true positives
in the same bucket. It does not prove that numerical saturation caused semantic
errors or that raw scores will separate them successfully.

A separate input-only probe used20 length-stratified development reviews. The31
saturated fixed seen-candidate scores contained31 distinct raw Y−N margins,
from18.125 to36.15625. E5's45 saturated scores contained44 distinct margins,
from16.875 to37.484375. Float64 softmax still saturated once for E5, so merely
casting probabilities to float64 is not the chosen solution. Hook-captured
probability32 exactly matched the unchanged engine outputs. No gold label or
held-out-candidate outcome informed this diagnostic.

All seven methods now preserve the legacy probability in `score` and additionally
record `presence_logit_margin`. The new decision policy declares
`presence_score_space=logit_margin` and a separate `presence_margin_threshold`.
The model, examples, prompt budgets, quantization and sentiment stage stay fixed.
Classification and review-routing calculations must use the declared space;
presence risk distance is the absolute raw-margin distance to the frozen cutoff.
The old probability is not described as calibrated correctness probability.

Calibrate every method, including the fixed comparator, from seen labels only.
The margin grid contains99 percentile cutoffs, finite float64 endpoints just
outside the observed range, and the original probability threshold's logit only
when that threshold is strictly between0 and1. Deduplicate to at most102 points.
Keep the existing objective and tie rules. A legacy threshold of1 must not be
converted to infinity: its historical meaning was acceptance of rounded-to-one
probabilities. Retain same-forward probability32 calibration as a separate
diagnostic, so numerical postprocessing and retrieval effects can be separated.

Rerun Stage1 for all100 development reviews and all seven methods. The common
sentiment grid may be reused only after validating its source artifacts, input
identities, inference settings and immutable file digest. A real-GPU three-input
gate already verified the new sentiment-only path against the legacy full-grid
path:36 aspect vectors, identical encoded inputs and truncation audits, maximum
score difference0, no decoded-label differences. This removes discarded Stage1
work when creating sentiment grids for new populations.

After the shared-margin comparison, at most one further example-policy iteration
remains. Duplicate-group quarantine is not selected at this point: conflicting
training examples were not enriched in the observed seen-theme retrieval errors.
Final method/threshold/routing selection remains separate from evaluation.
# Classification and business routing scope

Strict unseen-theme classification continues to select methods, detection
thresholds and directed improvements from seen labels only. As authorised in
the approved plan, the separate business review-routing experiment may use
complete calibration labels, including the target theme, only after classifier
selection is frozen and hashed. Routing results cannot feed back into classifier
tuning. The combined classification-plus-routing experiment is therefore not an
end-to-end strict-unseen result. Historical evaluation labels are opened only
after both classifier and routing choices are frozen.

Before any service execution, read-only review found that a fixed60-second
normal-load window can offer fewer than10 requests at this model's capacity,
making the pre-existing recovery criterion unevaluable even with no failures.
Each of the three burst cycles therefore uses a normal-load duration of
`max(requested_baseline_seconds, ceil(20 / baseline_rate_rps) + 1)`, derived
only from the capacity pilots before observing burst outcomes. This ensures
at least20 scheduled baseline requests; fewer than10 successful timings still
produces an explicitly unevaluable recovery result. No historical-service
timing is silently substituted for a missing baseline.

