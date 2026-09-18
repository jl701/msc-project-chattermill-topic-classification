# Repository engineering review — 16 September 2026

## Outcome

The repository now has a current research README, installable Python package,
CPU-only walkthrough, aggregate-results command, contributor/reproduction/code
guides, regression tests and a CI definition. The work improves how a reader
understands, runs and changes the project. It does not modify scientific choices
or produce new model results.

## Changes and reasons

| Change | Reason | Verification |
|---|---|---|
| Replace the obsolete setup-stage README | Lead with the problem, findings and usable entry points | Tables cross-checked against saved aggregate CSVs |
| Separate original official evidence from completed policy validation | Prevent mixing distinct evidential roles or citing a partial composition | Scope-checked reporting command and negative tests |
| Add methods, architecture, history and reproduction guides | Let an unfamiliar ML reader find the relevant operation without reading dated logs | Local link checks and CLI help checks |
| Package metadata and CPU/neural/QLoRA extras | Avoid installing a GPU stack for simple data/decoder work | Fresh CPU installation and standalone wheel test |
| Extract CPU TF–IDF and defer neural imports | Remove accidental PyTorch coupling without changing model features | Same-environment bitwise feature/probability comparison and regression fixture |
| Extract shared policy bootstrap | Remove executable-script coupling from statistical reuse | Bitwise comparison at 1, 199, 200, 201, 401 and 20,000 draws |
| Validate public result scopes and protect output files | Fail clearly on malformed/mixed evidence and accidental overwrite | Invalid scope, non-finite, duplicate, malformed-row and overwrite tests |
| Add global correctness checks, stricter public-interface checks and CI | Make quality repeatable rather than a one-off edit | Local quality gate and complete test suite |
| Inventory source structure and common credential patterns | Identify actual remaining maintenance risks | Metadata-only audit, no credential values printed |

## Verification record

- Pre-edit baseline: 597 tests passed after installing the previously omitted
  `psutil` dependency. An initial collection failure was recorded, not hidden.
- Final full local suite: **644 passed**, with one CUDA compatibility warning
  from the existing local PyTorch environment detecting the laptop GPU. These
  tests do not establish that this environment can train on that GPU. No test
  suppression was introduced to achieve the pass.
- Fresh Python 3.11 CPU-only environment: **83 tests passed**. Neither PyTorch
  nor Transformers is installed or imported by the walkthrough.
- Repository-wide Ruff checks: `E9`, `F63`, `F7`, `F82` passed. Stricter
  `E,F,I,B` checks (excluding line length) and formatting passed for the ten
  explicitly maintained files. This is not a claim of strict lint compliance
  for every historical runner.
- The package built as a source archive and wheel. A distribution audit rejects
  author-only paths, credential-like text and author-machine paths. Both built
  archives passed. The demo and reporting command were also exercised outside
  the repository using a separate wheel-only environment. Reporting requires
  the checkout's aggregate CSVs via `--repo-root`.
- Bootstrap and TF–IDF comparisons use synthetic fixtures. They establish
  equivalence for those inputs, not a fresh end-to-end rerun of all models.
- New reference tables are byte-preserving copies of the completed author-held
  analysis. Existing thesis, configuration and result files remain protected.

## Review findings retained as explicit limitations

Several historical orchestration/analysis files exceed 1,000 lines, and some
load older runners dynamically. They remain candidates for future decomposition.
Rewriting every completed experiment before submission would add regression
risk without a new scientific requirement. Their purpose and boundaries are
now documented rather than concealed behind a claim of production readiness.

The lightweight dependency environment is locked. The original cloud recipe
is preserved separately. GPU/CUDA installation is not validated by a CPU unit
test, and no GPU job was launched during this cleanup.

CI is configured for Windows/Linux smoke checks and the Linux full suite; its
current remote status is shown by the README badge. The release scan covered
502 tracked or unignored text files and found no credential-pattern,
author-machine-path or live-SSH-endpoint candidates. A separate Git-history
search for common private-key and provider-token formats also found no match.
These checks are not a security certification. No project-wide licence was
selected on the author's behalf.

## Publication handling

The starting commit was `27cf9cd`, with pre-existing thesis and policy-study
changes. Work was prepared on branch `agent/repository-quality-20260916` and
preserves the historical execution commits. The current public tree excludes
author-only thesis and supervision material while retaining aggregate evidence.
