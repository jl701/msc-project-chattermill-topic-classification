# Rich-description interface study v1: development result

Date: 20 August 2026  
Status: **training-only development complete; one interface frozen and validation-confirmed**

## Outcome

The registered 72-pseudo-fold development grid completed without failure,
non-finite values, resume conflicts or test contracts. It used the official
training split only: two TF-IDF/E5 methods, twelve pseudo-unseen aspects and
three deterministic row partitions.

The global selection rule chose `R2_positive_concat`. The thesis-facing `R`
condition is now frozen as:

```text
Stage 1:
  canonical name
  + exact minimal definition
  + aliases
  + inclusion boundary

Stage 2:
  canonical name
  + exact minimal definition
  + candidate sentiment
```

The contrastive boundary is not concatenated into the positive Stage 1
representation. It remains appendix evidence about why naive rich text can
increase false positives.

## Selected comparison

| Development method | D AP | Selected R AP | R - D | Improved aspects | D FP/100 | R FP/100 |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF field similarity | 0.453021 | 0.464426 | +0.011405 | 8/12 | 19.666 | 28.648 |
| Frozen E5 field similarity | 0.338495 | 0.331019 | -0.007476 | 6/12 | 10.617 | 12.577 |
| Unweighted combined selection statistic | — | — | **+0.001965** | — | — | — |

This is weak and heterogeneous evidence. The selected interface was the only
rich candidate within the registered 0.005 near-tie band of the best combined
AP result, but it does not establish that rich text is better than a concise
definition. In particular, it helps the lexical development scorer and mildly
hurts the frozen semantic scorer.

## Mechanism evidence

The existing `R1_concat` condition was inferior to the selected interface:

- TF-IDF: AP 0.448781 and 36.767 FP/100;
- E5: AP 0.317561 and 14.004 FP/100.

Eleven of twelve contrastive boundaries explicitly name at least one competing
aspect. Concatenating those exclusions into a positive candidate therefore
introduces cross-label lexical content. The resource audit found a mean 55.42
positive-field tokens and 23.33 exclusion-field tokens per aspect.

Separate maximum/top-two prototype aggregation did not improve the global
selection statistic, and the scale-preserving exclusion penalties also did not
recover a stable gain. These alternatives remain appendix-only.

## Scientific interpretation

The development result supports a limited claim:

> Removing negative boundary text from the positive rich representation is
> preferable to naive concatenation, but richer positive label information is
> not reliably superior to the concise definition across representation
> models.

The selected `R` is retained because the thesis needs one controlled rich
condition. It must be described as the selected eligible interface, not as a
universally optimal description design.

## Frozen boundary

- Selection summary SHA-256:
  `c1801fb962a2cd868a0b1183343892749bb71c3e993c5730f6815623ffd23d6f`.
- Resource content SHA-256:
  `289ba3238cb6772f9bfda98eb73ac8a1108ba4b2bb3d23eecf72826881f415b9`.
- No validation or official-test data was loaded during development.
- The formal validation confirmation cannot revise the interface.
- Alternative-interface results belong in the appendix; the main text will
  show only one `R` condition.

## Validation gate outcome

The registered twelve-fold confirmation completed for strict TF-IDF and frozen
E5 with the selected Stage 1 `R` and unchanged Stage 2 `D`. Both methods lost
held-out pair F1, presence AP and presence F1 while increasing false positives.
Consequently, the gate is closed: `R` remains one controlled local ablation,
but it will not be propagated to expensive cloud methods. Full interface
exploration belongs in the appendix.

The confirmation evidence and exact-control audit are recorded in
`taxonomy_rich_description_validation_confirmation_v1_results.md`.
