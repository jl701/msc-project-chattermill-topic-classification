# GitHub Upload Scope

This note records what is safe to commit and push to the project GitHub repository at the current stage.

## Safe To Commit

The following files are project code, tests, configuration, or non-sensitive documentation and are safe to upload:

- `.gitignore`
- `README.md`
- `PROJECT_OVERVIEW.md`
- `START_NEW_CHAT_PROMPT.md`
- `requirements.txt`
- `requirements-llm.txt`
- `src/`
- `scripts/`
- `tests/`
- `docs/`

The documentation includes aggregated metrics and protocol notes only. It does not include raw private data, credentials, model checkpoints, or confidential customer text.

## Do Not Commit

Do not commit:

- `outputs/`
- `data/raw/`
- `data/interim/`
- `data/processed/`
- `data/internal/`
- `artifacts/`
- `models/`
- `checkpoints/`
- `runs/`
- `wandb/`
- `.env` or `.env.*`
- Any local API keys, credentials, private Chattermill material, raw model outputs containing confidential text, or large downloaded model files.

These paths are covered by `.gitignore` where possible. Before pushing, run:

```powershell
git status --short --ignored
```

and confirm that only source, tests, and documentation are staged.

## Current Recommended Commit Scope

The current recommended GitHub upload is a single initial commit containing:

- Reproducible FABSA loaders, split builders, metrics, and baseline scripts.
- Closed-topic, held-out-organisation, and held-out-aspect evaluation documentation.
- DistilBERT, classical, label-aware, and Qwen smoke-test tooling.
- Unit tests for data loading, splits, metrics, label-aware helpers, Qwen formatting, and error analysis.
- Project handoff notes and Git hygiene notes.

Experiment outputs and generated Qwen SFT/evaluation JSONL files should remain local and regenerated when needed.
