# Experiment Log

This document is the running experiment log for the MSc project. It should be updated after every meaningful experiment, implementation change, or evaluation run before handing work back to the user.

## Logging Rule

Every completed experiment must add an entry to this file. The entry should be detailed enough that another Codex/ChatGPT session, the user, or Aji can reconstruct:

- what changed in the code or protocol
- why the experiment was run
- exactly which data split, labels, and model variant were used
- which command or script was run
- where local outputs were written
- which headline and supporting metrics were obtained
- what the result means
- what should be done next

Do not commit local `outputs/`, checkpoints, credentials, API keys, or private data. Summarise results in this document and keep generated artifacts local unless explicitly approved.

## Required Entry Template

````markdown
## YYYY-MM-DD: Short Experiment Name

### Purpose
- Why this experiment was needed.

### Code Or Protocol Changes
- Files changed.
- Behaviour changed.
- Any compatibility or reproducibility notes.

### Setup
- Dataset and split.
- Labels evaluated.
- Model or baseline.
- Important hyperparameters.
- Hardware or API endpoint, if relevant.

### Commands
```powershell
# exact command(s)
```

### Outputs
- Local output directory or files.
- Any files intentionally committed.
- Any generated files intentionally left uncommitted.

### Results
| Metric | Value |
| --- | ---: |
| Headline metric | 0.0000 |

### Interpretation
- What the result shows.
- Known limitations or failure modes.

### Next Step
- The next concrete action.
````

## 2026-06-21: Project Handoff And Aji Update Consolidation

### Purpose

- Preserve the latest Aji feedback and Gemini/Vertex AI access context for future project sessions.
- Fix the new-machine handoff so both possible workspace roots are recorded.
- Add this experiment logging rule before starting the leave-one-aspect-out work.

### Code Or Protocol Changes

- Added `docs/aji_updates_2026_06_21.md`.
- Updated `START_NEW_CHAT_PROMPT.md` with:
  - dual workspace path roots
  - latest Aji feedback
  - Gemini endpoint notes without the real API key
  - next-step ordering
- Added `docs/experiment_log.md`.

### Setup

- Current/new workspace: `D:\Msc_Project`
- Previous/alternate workspace: `C:\Msc_DSML\Msc_Project`
- Repository: `jl701/msc-project-chattermill-topic-classification`
- Branch: `main`

### Commands

```powershell
git status -sb
rg -n "sk-[A-Za-z0-9_-]+|OPENAI_API_KEY|llm-api\.datascience" . --glob '!outputs/**' --glob '!.venv/**' --glob '!data/**' --glob '!checkpoints/**'
```

### Outputs

- Documentation-only changes.
- No experiment outputs, checkpoints, data files, or credentials were added.
- The real Gemini API key is not stored in the repository.

### Results

| Check | Result |
| --- | --- |
| GitHub branch state | local `main` tracks `origin/main` |
| OneDrive sync for handoff docs | SHA256 hashes matched |
| Real API key committed | No |

### Interpretation

The project now has a durable handoff path for future sessions and a fixed rule for recording experiment details after each run.

### Next Step

Implement and run leave-one-aspect-out held-out-aspect evaluation across all 12 FABSA aspects, preserving both `label_masked` and `example_filtered`.
