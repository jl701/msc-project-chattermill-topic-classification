# Two-Stage Global Router Validation v1: Results

Date completed: 2026-08-23

## Decision

The registered global Router did **not** meet its success rule. It should be
retained as a validation-only negative result and should not replace Frozen
Qwen few-shot as the primary L2-D system.

The all-validation fit produced the numerically highest D held-out pair F1
(`0.509080` versus `0.504883` for Frozen Qwen few-shot), but the improvement
was small, its paired bootstrap interval crossed zero, and the leave-one-aspect
cross-fitted estimate fell to `0.444618`. The cross-fitted estimate is the
principal estimate because it tests whether Router selection transfers to an
aspect not used to select the Router.

Official test remained sealed. Every input row had `split=validation`,
`include_official_test=false`, and `test_contract_count=0`.

## Frozen design

- Routing unit: one review--candidate-aspect instance.
- Eligible components: DistilBERT, Frozen Qwen few-shot and QLoRA Qwen.
- Search: all six ordered base/expert pairs and 81 fixed
  rescue/confirmation-band combinations per pair; 486 global policies total.
- Input feature: the base model's threshold-aligned distance only. No gold,
  aspect identity, condition identity or outcome feature enters routing.
- Decision: either retain the base's complete aspect/sentiment prediction or
  substitute the expert's complete prediction.
- Decoder: frozen capped-two sentiment output; no third sentiment is possible.
- Selection: D-validation only, ranked by held-out pair F1, overall pair F1,
  lower route rate and lexical policy ID.
- Transfer: the D-selected policy is applied unchanged to N.

## Selected final Router

- Base: Frozen Qwen few-shot.
- Expert: DistilBERT two-stage cross-encoder.
- Rescue band: `0.200`.
- Confirmation band: `0.300`.
- Mean routing rate: `2.647%` on D and `2.688%` on N.

This policy was selected on all 12 D-validation folds for a future frozen
application. Its all-validation metrics are descriptive fitted diagnostics,
not an unbiased estimate of new-aspect performance.

The exact compact deployment record is
`configs/experiments/taxonomy_two_stage_global_router_selected_v1.json`.

## Matched metrics

| Estimate | Condition | Held-out pair F1 | Overall pair F1 | Pooled held-out F1 | Held-out presence F1 | Presence AP | Oracle sentiment-set F1 | Route rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cross-fitted Router | D | 0.444618 | 0.553527 | 0.492537 | 0.500826 | 0.484272 | 0.775264 | 0.176318 |
| Frozen final Router fit | D | 0.509080 | 0.555618 | 0.565239 | 0.589150 | 0.507602 | 0.779469 | 0.026470 |
| Cross-fitted Router | N | 0.444016 | 0.551404 | 0.462228 | 0.488923 | 0.551383 | 0.830690 | 0.176469 |
| Frozen final Router fit | N | 0.496181 | 0.554925 | 0.558693 | 0.554675 | 0.578832 | 0.837805 | 0.026884 |

For the final fitted D Router, pooled held-out precision was `0.516000` and
recall was `0.624865`. For N, they were `0.575257` and `0.543057`.
For the cross-fitted D Router, pooled held-out precision was `0.471893` and
recall was `0.515070`; for N, they were `0.519115` and `0.416577`.

The fitted Router was slightly higher than Frozen Qwen few-shot on the primary
D metric but did not exceed QLoRA on D overall pair F1 (`0.555618` versus
`0.586771`). The cross-fitted Router ranked below Frozen Qwen zero-shot, QLoRA
and Frozen Qwen few-shot on D held-out pair F1.

## Uncertainty and generalisation

Twenty thousand synchronised review-cluster bootstrap draws reused the exact
scientific-freeze inference protocol.

| Router evidence | D held-out pair F1 | 95% review-bootstrap interval | Difference from Frozen Qwen few-shot | 95% interval for difference |
|---|---:|---:|---:|---:|
| Cross-fitted | 0.444618 | [0.415568, 0.470809] | -0.060265 | [-0.077306, -0.043434] |
| All-validation fitted diagnostic | 0.509080 | [0.474179, 0.539671] | +0.004197 | [-0.001575, 0.011416] |

The fitted gain is therefore compatible with no real improvement. The
cross-fitted loss is both materially larger and consistently negative under
review resampling.

## Why the Router did not generalise

The components are complementary, but the global threshold-distance rule does
not identify that complementarity reliably across aspects. On D, when Frozen
Qwen few-shot was the base, DistilBERT was exactly correct while the base was
wrong for a mean of 76 held-out aspect instances per fold. The reverse occurred
for a mean of 127.25 instances per fold. A global uncertainty band cannot tell
these two cases apart well enough.

Cross-fitting selected six distinct policies. The final Frozen-Qwen-to-
DistilBERT policy appeared in seven of 12 omitted-fold selections, while four
omitted folds selected a different model direction. This instability caused
route rates from `0.166%` to `88.166%` and produced large losses on folds a04,
a06 and a09. Relative to Frozen Qwen few-shot, the cross-fitted Router improved
four folds, tied one and worsened seven. The final fitted Router improved seven,
tied one and worsened four, which explains why the same-data fit looks better
than the transfer estimate.

## Integrity audit

- Verified formal campaign states: three workers complete, zero failures and
  zero test contracts.
- Verified published backup files: 4,007 unique files across 1,789 receipts.
- Router-critical verified inputs: 1,261 files with recomputed SHA-256 hashes.
- Policy/fold records: 11,664, covering 486 policies x 12 folds x 2 conditions.
- Cross-fitted records: 24; full Router metric records: 48.
- Non-finite numeric values: 0.
- Duplicate policy/fold/condition identities: 0.
- zero-positive prediction collapses: 0.
- maximum predicted sentiments for one selected aspect: 2.
- official-test reads and test contracts: 0.

## Thesis and execution recommendation

Report the Router as a controlled negative ablation: selective model division
has genuine oracle complementarity, but one globally shared score-distance
gate overfits the 12 validation aspects and does not beat the best single model
under cross-fitted estimation. Keep Frozen Qwen few-shot as the primary D
deployment choice. If the Router is later included in an authorised official
test run, use only the already frozen final policy above and label it as a
secondary comparison; do not redesign or retune it after this result.

Machine-readable tables are in
`docs/thesis_figure_data/taxonomy_two_stage_global_router_validation_v1/`.
The complete audit and row-level sufficient statistics are under
`outputs/experimental/taxonomy_two_stage_global_router_validation_v1/`.
