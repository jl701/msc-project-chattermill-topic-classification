# Rich-description interface study v1

Date: 20 August 2026  
Status: **pre-registered before execution**

## Thesis boundary

The thesis mainline will contain exactly one rich-description condition, `R`,
inside the primary Level 2 leave-one-aspect-out experiment. The design search
is a bounded supporting study and will be reported in the appendix. It is not a
new taxonomy level and it does not change the main two-stage research question.

The existing frozen rich resource is not overwritten. Its direct-concatenation
rendering becomes the `R1_concat` development baseline. The development study
asks whether the same label-side content is better consumed through a shorter
positive-only string, separate positive prototypes, or a separate exclusion
penalty.

## Literature-grounded design principles

- Label-description training and matching can improve zero-shot text
  classification, but the model must have an interface that can interpret the
  descriptions rather than merely receiving more text (Gao et al., EMNLP
  2023; De Silva et al., EMNLP 2023).
- Multiple label descriptions can be selected or aggregated instead of being
  concatenated indiscriminately (Basile et al., 2022).
- Useful label keywords should balance task relevance, within-label diversity
  and across-label exclusivity (Yano et al., *SEM 2024).
- Label wording is a material source of zero-shot variance, so the authoring
  and selection boundary must be recorded (Chu et al., Findings ACL 2021).

Primary references:

- <https://aclanthology.org/2023.emnlp-main.853/>
- <https://aclanthology.org/2023.emnlp-main.475/>
- <https://arxiv.org/abs/2204.09481>
- <https://aclanthology.org/2024.starsem-1.9/>
- <https://aclanthology.org/2021.findings-acl.365/>

These sources motivate the design dimensions. They do not prescribe a single
universal rich-description generator, so the project pre-registers a small,
explicit comparison rather than claiming an external standard.

## Leakage-safe development protocol

Only the official training split is loaded. Its rows are assigned to three
deterministic partitions from `sha256(seed, row_uid) modulo 3`. For every one
of the twelve pseudo-unseen aspects and every row partition:

1. use two row partitions as the fit set;
2. remove every fit row containing the pseudo-unseen aspect;
3. fit any TF-IDF vocabulary using review and descriptor text for the eleven
   seen aspects only;
4. select an aspect threshold using only seen-aspect candidates in the held-out
   row partition; and
5. use the pseudo-unseen labels in that partition for evaluation only.

The official validation and test files are not loaded during development.

## Candidate interfaces

| ID | Interface |
|---|---|
| `D` | canonical name plus exact minimal definition |
| `R1_concat` | existing definition, aliases, inclusion and contrastive boundary concatenation |
| `R2_positive_concat` | definition, aliases and inclusion only; no exclusion text in the positive representation |
| `R3_prototype_max` | score the base definition, every alias and inclusion separately; take the maximum |
| `R4_prototype_top2` | score fields separately; average the two strongest positive similarities |
| `R5_contrastive_top2` | use the R4 positive score and subtract a separately scored exclusion penalty |

`R5` selects one global penalty from `0.1`, `0.25` and `0.5` inside the
training-only development results. The penalty is
`lambda * max(0, exclusion_similarity - positive_top2_mean)`, so an exclusion
field only reduces a score when the review matches it more strongly than the
positive representation. This is the only numeric interface grid.

### Pre-full-run smoke amendment

The first E5 path smoke showed that subtracting the absolute exclusion cosine
shifted the whole E5 score scale below the shared `D` threshold and therefore
created a zero-prediction operating point. Before any complete pseudo-unseen
run, the registered formula was amended to the scale-preserving excess-
similarity penalty above. No candidate was selected, and no official
validation or test data was opened when making this mechanical correction.

## Selection rule

The primary statistic is pseudo-unseen aspect-presence average precision. Rich
candidates are ranked by their unweighted mean AP delta from `D` across both
methods, twelve aspects and three row partitions. F1, false positives, score
separation and per-aspect stability are diagnostics. A difference of at most
`0.005` mean AP is a near tie and resolves in favour of the simpler registered
interface.

Exactly one rich interface is selected globally. Selecting different rich
interfaces by model, held-out aspect or formal-validation outcome is
prohibited.

## Stage-aware interpretation

The rich study targets Stage 1 aspect presence. In the eventual selected `R`
condition, Stage 2 retains the canonical name plus minimal definition. This
isolates the contribution of richer aspect semantics from sentiment-interface
changes and respects the genuine two-stage architecture.

## Formal boundary

After the selected interface, implementation contract and hashes are frozen,
TF-IDF and E5 receive one official-validation confirmation. That confirmation
cannot revise the interface. Qwen, DistilBERT or QLoRA expansion is allowed
only after the local gate is interpreted and the cloud manifest is regenerated.
The official test remains sealed.

## Outputs

- ignored row-level artifacts under
  `outputs/experimental/taxonomy_rich_description_interface_study_v1/`;
- a tracked aggregate JSON summary without review text;
- a tracked appendix-ready CSV with aggregate method/interface/fold metrics;
- a result note recording the selected interface, limitations and cloud-plan
  consequence.
