# Final directed development iteration: topic-aligned E5 negatives

This second and final directed iteration is registered after the complete
shared-margin100-review matrix and before business-routing outcomes or957-review
evaluation outcomes. Only E5 receives this new policy; BM25 and RRF are not
additional trials. Configuration: `serving_rag_negative_20261111.json`.

The shared-margin seen comparison retained fixed examples: pair F1=64.2045,
best dynamic BM25=63.7838, E5=63.1313, RRF=63.0728, all on the0–100 scale.
E5 reduced presence false negatives from48 to36 relative to fixed but increased
presence false positives from57 to87. Preserving raw margins alone did not make
a dynamic method pass the registered promotion gate.

The final hypothesis concerns negative demonstration relevance. E5 has207 of
1100 seen-candidate prompts without two same-aspect negative examples; these
contain30 presence false positives and2 false negatives. BM25 has only50 such
prompts, with3 false positives. E5's affected cohort also includes175 currently
correct decisions, so this is a testable priority, not an estimate of recoverable
errors. The association is confounded by theme and query difficulty.

Use method ID `e5_negative_seen`, underlying retriever `e5`, and policy
`same_aspect_negative_seen`. Keep the two original globally selected positive
examples unchanged. For a seen candidate, filter the existing E5 negative
ranking to that candidate aspect and choose two using the original within-class
review-ID deduplication. Preserve the original cross-class rule; do not add a
separate diversity or duplicate-removal intervention. For the held-out aspect,
retain the original global branch and demonstration identities/content.

This narrower intervention avoids forcing two same-aspect positives into the
426 E5 prompts that currently have one same-aspect positive and two same-aspect
negatives; that cohort has14 false positives among411 negative decisions. These
cohort numbers motivate the choice but are not causal estimates.

Keep YYNN order, model, quantisation, candidate definitions,1024-token few-shot
budget, demo-first trimming, batch4, and the frozen384-token sentiment stage.
Changing negative example lengths may reallocate the common trimming budget,
including retained positive text; record that effect. Identical held-out prompt
content also does not promise bitwise identical scores when neighbouring prompt
lengths change the padded microbatch shape.

Run the same100 reviews using the verified existing sentiment anchor, then
recalibrate from seen labels only with the unchanged margin grid and tie rules.
Compare with both original E5 and the matched fixed baseline. The existing
promotion requirement of at least0.5 pair-F1 points above fixed remains in force;
retain all outcomes if the variant loses. No additional directed example-policy
trial will follow this result. Final classifier selection, API correctness and
fault checks precede957-review evaluation and final service measurements.
