# Thesis LaTeX Workflow

This note freezes the dissertation-writing workflow as of 2026-06-30.

## Source Of Truth

Dissertation prose should now be edited directly in LaTeX:

- Main entry point: `thesis/main.tex`
- Literature review draft: `thesis/chapters/02_literature_review.tex`
- Bibliography: `thesis/references.bib`
- Working notes not intended as final prose: `thesis/notes/`

Markdown files under `docs/` remain useful for experiment logs, planning notes, protocols, and handoff summaries, but they should not become parallel dissertation drafts. If a future task asks for dissertation wording, chapter edits, citation insertion, or literature-review revision, edit the LaTeX source first.

## Template

The initial LaTeX source was created from:

```text
C:\Msc_DSML\Msc_Project\UCL_Msc_Thesis.zip
```

The template was extracted into the repository as `thesis/`, with the original template entry file renamed to `thesis/main.tex`.

## Current Literature Review State

The Pro model feedback from 2026-06-30 was analysed and converted into:

- a working literature-review chapter in `thesis/chapters/02_literature_review.tex`;
- a BibTeX bibliography in `thesis/references.bib`;
- a revision checklist in `thesis/notes/pro_literature_review_feedback_2026_06_30.tex`.

The current working dissertation spine is:

```text
Structured candidate-label aspect-sentiment classification under taxonomy shift in customer feedback.
```

The literature review follows an hourglass shape:

1. Customer feedback mining and opinion mining.
2. Aspect-Based Sentiment Analysis.
3. FABSA as the project benchmark.
4. Multi-label classification and evaluation.
5. Domain and organisation shift.
6. Taxonomy shift and candidate-label classification.
7. Structured-output LLMs as candidate-label classifiers.

## Citation Status

The current BibTeX file contains the Tier 1 foundation references already identified in the project literature matrices. Several entries use `and others` for long author lists and should be tightened before final submission.

Must-fill before the final dissertation:

- exact multi-label metric definitions for samples, micro, and macro F1;
- final verification of the dynamic-label/taxonomy-shift citations now added to the LaTeX draft: Godbole et al. (2005), `Text Classification with Evolving Label-Sets`, and Wohlwend et al. (2019), `Metric Learning for Dynamic Text Classification`;
- precise FABSA details for dataset size, aspect hierarchy, sentiment labels, and the original candidate-aspect setup;
- methods-chapter explanation of all-row LOAO versus positive-row LOAO;
- exact model/API documentation only if Gemini or Qwen results become central.

## Build Guidance

Use XeLaTeX/BibTeX from the `thesis/` directory:

```powershell
cd C:\Msc_DSML\Msc_Project\msc-project-chattermill-topic-classification\thesis
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

Do not commit generated PDFs, logs, auxiliary LaTeX files, data, credentials, outputs, model checkpoints, or confidential material unless explicitly requested.
