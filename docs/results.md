# Results: what can be concluded?

## 1. Separate detection from conditional sentiment

A low end-to-end score can have two different causes: missing the aspect or
predicting the wrong sentiment after detecting it. The central validation
diagnosis distinguishes these. Frozen Qwen few-shot has the stronger
thresholded unseen-aspect detector at the selected operating point. QLoRA has
the stronger presence ranking average precision and oracle-gated sentiment
score. An oracle gate supplies the true aspect presence only for diagnosis.
It is not a deployable system or an end-to-end result.

This crossed profile motivates a fixed composition: few-shot detection followed
by QLoRA sentiment. Each stage keeps its original threshold. No learned router
is fitted and no threshold is jointly retuned for the composition.

## 2. Original locked official evaluation

The primary endpoint is an equal-weight average over twelve aspects. For each
aspect's hold-out fold, compute micro-F1 from its aspect–sentiment pair counts,
then average the twelve values. Overall pair F1 instead includes seen and
held-out aspects. Pooled held-out F1 aggregates confusion counts before taking
F1. These answer different questions and are not interchangeable.

| System | Primary F1 |
|---|---:|
| Fixed stage composition | 0.5075 |
| Frozen Qwen few-shot | 0.4954 |
| QLoRA | 0.4864 |

The composition-minus-few-shot contrast is +0.0121 with a paired 95% interval
[0.0014, 0.0224]. The gated composition-minus-QLoRA contrast is +0.0211
[0.0002, 0.0411]. The first comparison is tested first. Only if it passes does
the second have confirmatory status. Both passed on the registered endpoint.

The intervals use 20,000 paired review-cluster bootstrap draws, seed 13. Each
draw uses the same resampled review identities across systems and folds. F1 is
recomputed from the resulting counts. This quantifies review-sampling
uncertainty for these trained systems and this taxonomy, not training-seed or
taxonomy-population uncertainty.

The finding is a modest benefit from complementary stages, not a universal
ranking. Omitting App/website reverses the QLoRA contrast to approximately
−0.0069. This is a post-hoc influence check, not a replacement primary endpoint.
The observed gain alone does not justify the cost of operating two components
in production. In addition, none of the three systems recovers both sentiments
in any of the 21 gold two-sentiment cases.

### Inspect the saved evidence

- [Exact contrasts and decision fields](thesis_figure_data/taxonomy_final_test_v1/official_confirmatory_results.csv)
- [All methods and N/D conditions](thesis_figure_data/taxonomy_final_test_v1/official_l2_model_condition_summary.csv)
- [Original authority and historical-test disclosure](taxonomy_current_authority_20260824.md)

`taxonomy-results` formats these stored point estimates and intervals. It does
not compute new confidence intervals from an aggregate table.

## 3. Later training-policy robustness

Review removal excludes any training review mentioning the held-out aspect.
Label masking retains the review and its other-aspect supervision but excludes
the held-out aspect from training targets entirely. It must not turn that
aspect into a negative example.

The [completed policy comparison](results/training_policy_completed/README.md)
reports twelve folds for both conditions. Each composition uses its own
policy's few-shot detector and QLoRA sentiment component. It is not a mixture
of a new detector and an old-policy sentiment model.

Under names plus definitions, retained-review scores are lower for the three
central systems, while the retained-review composition remains above its two
complete components on validation. These are nominal conditional validation
comparisons, developed after the original official test. They are not a second
confirmatory test. The policy contrast also changes the few-shot demonstration
pool, so it is not an identical-prompt comparison.

## 4. Supporting findings

Matched TF–IDF and E5 controls support true two-stage factorisation over their
one-stage pair-classification counterparts. Minimal descriptions have
model-dependent effects. Three parent-group hold-outs provide descriptive
structured-shift evidence, not a population estimate from three exchangeable
groups. Rich descriptions, description-to-weight transfer, learned routing and
other bounded extensions also provide useful negative findings.

The [history guide](history.md) locates these studies without presenting their
validation search as part of the confirmatory hypothesis family.
