# Taxonomy-Generalisation Literature and Thesis Route Alignment

Date: 24 July 2026
Status: pre-execution literature alignment complete; no new official model run
was performed

## 1. Purpose

The thesis prose still reflected the earlier fixed held-out-aspect,
Gemini/Qwen, and router-centred route after the experiment mainline had moved to
an increasing-difficulty taxonomy-generalisation study. This review checked
whether the new route has direct literature support, identified where the
project is synthesising ideas rather than reproducing an established protocol,
and updated the local thesis before formal local execution.

The review used original papers and official proceedings or model reports.
Search themes were:

1. candidate-conditioned and zero/few-shot ABSA;
2. label-partially-unseen and generalised zero-shot text classification;
3. zero-shot multi-label learning with structured label spaces;
4. label names, descriptions, hierarchy, and leakage-safe description
   selection;
5. frozen semantic representations, contextual cross-encoders, Qwen, LoRA, and
   QLoRA; and
6. paired statistical evaluation in NLP.

## 2. Main alignment conclusion

The revised route is well supported as a synthesis of established research
questions:

- ABSA literature supports conditioning sentiment on a supplied aspect and
  constructing sentence-pair or prompt-based inputs.
- Zero-shot text-classification literature explicitly distinguishes training
  on some labels and testing jointly on all labels.
- Multi-label research supports separating seen and unseen performance and
  exploiting label descriptions and parent-child structure.
- Label-description research supports name-only, minimal definitions, richer
  keyword descriptions, and label-free refinement as different information
  regimes.
- Parameter-efficient adaptation literature supports a matched Frozen-Qwen
  versus QLoRA comparison.
- Recent multi-label evidence warns that supervised fine-tuning can improve
  overall performance while reducing zero-shot capability, making the
  seen/unseen and frozen/adapted comparisons scientifically necessary.

No reviewed paper supplies the exact FABSA sequence
`L1-D -> L2-D -> L3-DD -> L4-D`, the dual-unseen `NN/DN/ND/DD` crossover, or the
same five-method candidate-pair comparison. These should be presented as a
controlled project design motivated by adjacent literature, not as a previously
standard benchmark and not as an unsupported novelty claim.

## 3. Route-to-literature map

| Project component | Literature anchor | Alignment decision |
| --- | --- | --- |
| Aspect-conditioned sentiment | Sun et al. (2019); Seoh et al. (2021) | Keep one review-candidate claim and allow different sentiments per aspect. |
| Multi-label aspect-sentiment set | Huang et al. (2020); Kamila et al. (2022) | Keep empty, singleton, and multi-pair outputs; do not collapse to global sentiment. |
| Level 1 singleton unseen candidate | Candidate-conditioned ABSA and unseen-label literature | Call it an entry diagnostic, not label-fully-unseen classification. |
| Level 2 one unseen plus eleven seen | Yin et al. (2019) label-partially-unseen; Liu et al. (2021) GZS-MTC | This is the closest standard generalised zero-shot analogue. Report seen and unseen separately. |
| Level 3 two unseen candidates | Structured zero-shot multi-label literature | Treat as a harder generalised setting. Both `DN` and `ND` are required to remove aspect-identity confounding. |
| Level 3 descriptions | Gao et al. (2023); Basile et al. (2022); Chu et al. (2021); Yano et al. (2024) | Use a frozen minimal definition as the core treatment; keep corpus refinement and keyword generation outside the strict regime. |
| Level 4 parent-group holdout | Rios and Kavuluru (2018); Zhang et al. (2019); Liu et al. (2021) | Use the hierarchy to define the stress test, but do not claim hierarchical inference unless implemented. |
| Seen/unseen harmonic mean | Generalised zero-shot evaluation; Xian et al. (2017) | Retain overall, seen, unseen, and harmonic-mean reporting. |
| TF-IDF -> E5 -> DistilBERT | Classical lexical, frozen semantic, supervised contextual capacity | Keep the compact roster; do not add an unrestricted encoder sweep. |
| Frozen Qwen -> QLoRA | Qwen3 report; LoRA; QLoRA; Chen et al. (2025) | Interpret as adaptation versus preserved unseen-label access, not merely small versus large model. |
| Paired confidence intervals | Dror et al. (2018); Peyrard et al. (2021) | Preserve review pairing, recompute nonlinear metrics within replicates, and report paired differences. |

## 4. Terminology corrections

The thesis now uses the following terminology:

- **supplied-candidate single-unseen LOAO** for Level 1;
- **label-partially-unseen / generalised single-unseen** for Level 2;
- **dual-unseen generalised evaluation** for Level 3;
- **parent-group taxonomy holdout** for Level 4;
- **description-assisted zero-shot generalisation** when a definition is
  supplied; and
- **strict zero-label calibration** when held-out-aspect validation labels
  cannot select thresholds, prompts, checkpoints, or descriptions.

Level 1 is not called label-fully-unseen because task-specific training on the
other aspects remains available. "No description" means canonical hierarchical
name only; it never means an opaque ID in a core comparison.

## 5. Description construction and leakage boundary

The literature supports several description sources: label names, fixed
templates, dictionary or Wikipedia definitions, manually written paraphrases,
keywords, and unsupervised refinement. These sources carry different amounts of
supervision.

The strict core treatment remains:

```text
canonical hierarchical name + one frozen neutral taxonomy gloss
```

Permitted sources are the twelve names, their parent-child structure, and
general language knowledge. Review text, corpus terms or frequencies, gold
labels, model predictions, confusion matrices, and error analysis remain
forbidden. The description effect is measured after freezing by comparing
matched name-only and definition conditions. A result may show that a
description helps or harms a particular method; it must not trigger
result-informed rewriting.

Richer keywords, multiple descriptions, and unlabelled-corpus refinement remain
defensible future variants but constitute different information regimes and
must not silently enter the core `D` condition.

## 6. Updated thesis claim

The thesis is no longer organised around:

```text
closed taxonomy -> held-out organisation -> one fixed held-out split
-> Gemini/Qwen -> routing
```

The active spine is:

```text
L1-D supplied singleton unseen
  -> L2-D one unseen competing with eleven seen
  -> L3-DD two unseen competing under complete descriptions
  -> L4-D an unseen parent group
```

Two matched interventions sit across that spine:

1. `NN/DN/ND/DD` at Level 3 isolates incomplete label documentation.
2. Frozen Qwen versus QLoRA isolates task adaptation under the same
   candidate-pair interface.

Earlier closed-taxonomy, organisation-shift, Gemini, and router results remain
historical or secondary evidence. They do not define the experiments to run
next and should receive little or no main-text space unless they help explain
the research development.

## 7. Files updated

- `thesis/chapters/01_introduction.tex`
- `thesis/chapters/02_literature_review.tex`
- `thesis/main.tex`
- `thesis/references.bib`
- `docs/dissertation_loao_mainline_lock_2026_07_23.md`
- `report_notes.md`

## 8. Core sources added

- [Yin et al. (2019), zero-shot text-classification evaluation](https://aclanthology.org/D19-1404/)
- [Sun et al. (2019), auxiliary-sentence ABSA](https://aclanthology.org/N19-1035/)
- [Seoh et al. (2021), open aspect sentiment prompts](https://aclanthology.org/2021.emnlp-main.509/)
- [Rios and Kavuluru (2018), structured few/zero-shot multi-label learning](https://aclanthology.org/D18-1352/)
- [Liu et al. (2021), zero-shot multi-label hierarchy reasoning](https://aclanthology.org/2021.naacl-main.83/)
- [Zhang et al. (2019), class descriptions and hierarchy](https://aclanthology.org/N19-1108/)
- [Gao et al. (2023), label-description training](https://aclanthology.org/2023.emnlp-main.853/)
- [Chu et al. (2021), unsupervised label refinement](https://aclanthology.org/2021.findings-acl.365/)
- [Basile et al. (2022), label-free description ranking and aggregation](https://arxiv.org/abs/2204.09481)
- [Yano et al. (2024), relevant/diverse/exclusive keywords](https://aclanthology.org/2024.starsem-1.9/)
- [Chen et al. (2025), preserving zero-shot capability under fine-tuning](https://aclanthology.org/2025.findings-naacl.315/)
- [E5](https://arxiv.org/abs/2212.03533)
- [Qwen3](https://arxiv.org/abs/2505.09388)
- [LoRA](https://openreview.net/forum?id=nZeVKeeFYf9)
- [QLoRA](https://arxiv.org/abs/2305.14314)
- [Dror et al. (2018), statistical testing in NLP](https://aclanthology.org/P18-1128/)
- [Peyrard et al. (2021), paired NLP evaluation](https://aclanthology.org/2021.acl-long.179/)

## 9. Pre-execution conclusion

The literature review does not require a change to the approved Level 1--4
experiment design or five-method roster. It does require precise terminology,
separate seen/unseen analysis, a strict description-information boundary, and
careful interpretation of task adaptation. Those corrections are now reflected
in the local thesis.

Validation after the edits:

- all 37 unique citation keys resolve against 52 unique bibliography entries;
- XeLaTeX/BibTeX/XeLaTeX/XeLaTeX produced a 30-page `thesis/main.pdf` with no
  undefined citations, references, or LaTeX errors;
- `git diff --check` reported no whitespace errors; and
- the full repository test suite passed: 292 tests.

Formal local model execution remains intentionally unstarted in this stage.
