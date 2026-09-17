# Contributing and checking changes

This is a research repository with frozen scientific results. A refactor should
make the implementation easier to use without silently changing what was
measured. New scientific methods belong in explicitly separate studies.

## Install a development environment

Use Python 3.11 in a fresh virtual environment. From the repository root:

```bash
python -m pip install -r requirements-dev.lock
python -m pip install --no-deps -e .
python scripts/check_quality.py
python scripts/check_quality.py --test
```

The second command installs the project itself. The lock file pins the CPU
development dependencies. It is not the historical CUDA execution environment.
The lightweight tests need neither model downloads nor credentials.

For the complete suite, install the optional neural dependencies:

```bash
python -m pip install -e ".[dev,neural]"
python -m pytest -q
```

Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to verify that tests do not
depend on model downloads. CI sets both. On a CPU-only machine, install PyTorch
from the appropriate CPU wheel index before installing the neural extra. QLoRA
execution additionally needs the `qlora` extra and a compatible GPU environment.
Do not install GPU training dependencies just to read results or run the demo.

## What a good change includes

1. A clear purpose and small diff. Keep data loading, inference, decoding and
   reporting separate where possible.
2. Explicit input/output contracts. Validate identities, shapes and required
   fields before producing a scientific table.
3. Tests for success and failure cases. Fix seeds in synthetic fixtures and
   never use unpublished raw reviews as test fixtures.
4. Behaviour-preserving evidence for refactors. Compare counts, predictions
   and deterministic resampling, not just whether a command exits successfully.
5. Updated commands and documentation. Show what a reader can actually run.

Avoid hidden network downloads during imports, machine-specific absolute paths
in new reusable modules, bare `except` blocks, silent fallback to another
dataset, and broad rewrites of dated experiment runners for style alone.

## Checks and review

`scripts/check_quality.py` runs correctness lint across `src`, `scripts` and
`tests`. It also applies import, unused-name, bug-pattern and formatting checks
to maintained public interfaces. The stricter boundary is listed in that script,
not implied to cover every historical file. `--test` adds the CPU smoke suite.
The CI workflow also runs the full suite with neural dependencies on Linux.

Run all relevant tests before submitting a change. The Linux CI status can only
be known after the branch is pushed. A local Windows pass is not a claimed
remote CI pass.

## Data, credentials and publication

Never commit API tokens, SSH keys, raw customer reviews, private annotations,
cloud receipts containing secrets or model weights without permission.
Do not weaken the official-test release/reveal boundary to make a test pass.
Aggregate reporting and synthetic fixtures are the default public materials.

Check staged files explicitly with `git diff --cached` before committing. This
project can have unrelated thesis edits in progress. Stage a bounded change
set rather than `git add .`. A project-wide licence requires the author's
decision and does not override separate dataset/model terms.

The repository is public. Follow the
[public release boundary](docs/github_upload_scope.md): supervision messages,
working dissertation files, machine-specific handoff notes and live cloud
connection details remain author-only even when they are useful locally.

For a suspected secret exposure, contact the maintainer privately and do not
paste the credential into an issue. Rotate the credential through its provider.
