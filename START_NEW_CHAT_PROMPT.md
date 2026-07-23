# Prompt For Continuing In A New Chat

Use the text below when opening a fresh Codex/ChatGPT conversation for this project.

```text
I am continuing my UCL MSc project with Chattermill.

Please communicate with me in Chinese, but keep LaTeX, code, comments, docstrings, README content, and project documentation in British English. The code will be reviewed by Aji, so keep it concise, readable, and reproducible.

Workspace roots:
- preferred: D:\Msc_Project
- fallback: C:\Msc_DSML\Msc_Project

Repository:
- msc-project-chattermill-topic-classification
- private GitHub repository: https://github.com/jl701/msc-project-chattermill-topic-classification

FABSA data:
- preferred: D:\Msc_Project\Project_Preparation\Public_Datasets\FABSA
- fallback: C:\Msc_DSML\Msc_Project\Project_Preparation\Public_Datasets\FABSA

Before proposing or running work, read:
1. docs/dissertation_loao_mainline_lock_2026_07_23.md
2. docs/evaluation_protocol.md
3. docs/thesis_result_tables.md
4. docs/experiments/loao_bow_sentence_embedding_v1.md
5. docs/experiments/loao_similarity_aspect_conditioned_sentiment_rescore_v1.md
6. docs/experiments/loao_unified_candidate_pair_experimental_v1.md
7. docs/experiments/loao_strict_tfidf_aspect_qwen_router_v1.md
8. docs/experiment_reproducibility_register.md

The first file is the sole authority for method selection, active result numbers, the remaining-work checklist, and stopping decisions. Older roadmaps, meeting notes, result tables, router documents, and experiment logs are historical provenance only wherever they conflict with it.

Current task formulation:
- supplied-candidate aspect-sentiment classification under taxonomy shift;
- primary benchmark: twelve-fold all-row LOAO;
- one canonical candidate aspect per fold;
- all official evaluation rows;
- empty predictions allowed;
- pair-set output;
- primary metric: unweighted mean fold-level test pair micro-F1;
- target-calibrated and strict zero-label regimes must remain explicitly separated.

Aji's fixed modelling decision:
- every active method must predict sentiment conditional on the supplied candidate aspect;
- no global document sentiment may be used as an active baseline;
- Count, strict TF-IDF, MiniLM, and E5 share the same frozen DistilBERT aspect-conditioned sentiment component;
- each complete representation system uses its own validation-selected presence threshold;
- DistilBERT retains its candidate-specific sentiment component;
- frozen Qwen and QLoRA are candidate-conditioned by construction.

Active target-calibrated all-row LOAO results:
- Count BoW: 0.3144 mean pair micro-F1;
- strict train-only TF-IDF: 0.3856;
- MiniLM: 0.3889, appendix/repository only;
- E5-base-v2: 0.4013;
- DistilBERT candidate cross-encoder: 0.3158;
- frozen candidate-pair Qwen: 0.337804;
- candidate-pair QLoRA: 0.483158.

Retired from the active project:
- every global-document-sentiment baseline;
- shallow TF-IDF aspect-conditioned sentiment and its negative result;
- legacy TF-IDF 0.3780;
- legacy TF-IDF-to-Qwen router 0.4549;
- refined legacy router 0.4560;
- all derivative legacy-router tables, figures, boundary searches, and uncertainty analyses.

Do not quote, regenerate, tune from, or reintroduce retired results. They may remain only in historical experiment logs for provenance.

Continue from the ordered checklist in:
docs/dissertation_loao_mainline_lock_2026_07_23.md

Completed on 23 July:
- the aspect-conditioned similarity rescore files were integrated;
- the authoritative seven-row active baseline table was generated and guarded against retired rows;
- the strict TF-IDF to frozen candidate-pair Qwen router was rebuilt under validation-only selection;
- validation selected the zero-call local-only policy, test gain was exactly zero, and the router was dropped from the dissertation mainline.

The next unresolved items are:
1. complete strict zero-label Qwen/QLoRA analysis;
2. run registered additional QLoRA seed replication;
3. produce the separate matched-effect table;
4. freeze the final evidence set and finish the dissertation.
```
