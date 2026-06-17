# FABSA Evaluation Protocol

This note records the current FABSA evaluation setup after Aji's feedback on the two main generalisation axes.

## 1. Closed-Topic Benchmark

The provided FABSA split is kept as the closed-topic benchmark for comparability:

| Split | Rows | Role |
| --- | ---: | --- |
| train | 7,930 | Train on the full fixed taxonomy |
| validation | 1,057 | Tune thresholds / select runs |
| test | 1,587 | Final closed-topic comparison |

All three provided splits contain the same 12 aspects. This makes the split useful for standard fixed-taxonomy modelling, but not for unseen-topic evaluation.

## 2. Held-Out Organisation Split

This split evaluates cross-company/domain-shift generalisation.

Current selected split:

| Split | Organisations | Rows | Supervision aspects | Pair labels |
| --- | --- | ---: | ---: | ---: |
| train | all except validation/test orgs | 7,020 | 12 | 36 |
| validation | `600` | 1,533 | 12 | 32 |
| test | `369`, `727` | 2,021 | 12 | 32 |

The selected organisations are large enough to preserve broad aspect coverage while keeping train/validation/test organisation sets disjoint.

Leakage checks:

- Row overlap between train/validation/test: `0`.
- Organisation overlap between train/validation/test: `0`.
- Label overlap is expected in this protocol because the taxonomy is fixed; the shift is in organisation/domain, not in labels.

## 3. Held-Out Aspect Split

This split evaluates open-topic/new-aspect generalisation under a candidate-label setup.

Current held-out aspects:

- `Account management: Account access`
- `Company brand: Competitor`
- `Value: Discounts promotions`

These aspects were selected because they cover different business areas and have enough validation/test examples for an initial benchmark.

Evaluation uses candidate labels at inference. The model should select from canonical labels rather than generate free-form topic names.

### Strategy A: Label-Masked Training

Rows from the official training split are kept if they still contain at least one seen aspect after held-out labels are removed.

| Split | Rows | Supervision aspects | Pair labels |
| --- | ---: | ---: | ---: |
| train | 7,632 | 9 | 27 |
| validation | 212 | 3 | 8 |
| test | 281 | 3 | 8 |

This setting tests whether a model can learn from texts that may contain held-out-topic language, while never receiving held-out-topic supervision.

### Strategy B: Example-Filtered Training

Any official training row containing a held-out aspect is removed entirely.

| Split | Rows | Supervision aspects | Pair labels |
| --- | ---: | ---: | ---: |
| train | 6,495 | 9 | 26 |
| validation | 212 | 3 | 8 |
| test | 281 | 3 | 8 |

This is stricter because the training texts should not contain held-out-topic examples.

Leakage checks for both strategies:

- Row overlap between train/validation/test: `0`.
- Train-vs-validation held-out supervision aspect overlap: `0`.
- Train-vs-test held-out supervision aspect overlap: `0`.
- Organisation overlap is expected because this protocol isolates the label axis, not the company axis.

## Reproduction

Analyse split candidates:

```powershell
python .\scripts\analyse_split_candidates.py
```

Build split manifests:

```powershell
python .\scripts\build_fabsa_splits.py
```

Generated manifests are written under:

```text
outputs/splits/
```

The output directory is ignored by Git. The committed code and this note document how to reproduce the split decisions.
