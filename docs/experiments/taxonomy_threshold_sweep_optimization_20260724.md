# Exact Threshold-Sweep Optimisation

Date: 2026-07-24

## Reason

The first official-validation TF-IDF scope exposed an implementation
bottleneck. The registered search contains the 99 regular thresholds plus
every midpoint between adjacent unique validation scores. The previous
implementation rebuilt complete multilabel matrices for every threshold,
making the sweep approximately quadratic in the number of candidate pairs.

This was a computational problem, not a reason to reduce the preregistered
threshold set.

## Exact replacement

The replacement sorts candidate pairs once by descending score and maintains
the sufficient statistics needed by the unchanged registered ranking:

- pair-level true positives, false positives, and false negatives;
- per-review true-positive and predicted-pair counts for samples F1;
- pair micro precision; and
- presence false-positive rows per 100.

Each candidate pair enters the predicted set once. The candidate thresholds,
`score >= threshold` rule, metric definitions, ranking order, tie-breaks, and
selected-threshold artifact are unchanged.

## Verification

- A unit test compares every sweep row with the previous brute-force metric
  construction, including tied scores.
- Full repository suite after the change: `303 passed`.
- On the first 34,881-pair official-validation scope, the exact sweep evaluated
  32,965 registered thresholds in approximately 0.30 seconds.
- The interrupted quadratic run wrote no threshold-selection summary. Its
  completed train and validation score artifacts remain valid and resumable.

The change affects runtime only and does not alter the frozen scientific
protocol SHA-256
`d7ccded514ac1cbccf337e496e039ac418698566be0c0ca0c21e18608cc40f85`.
