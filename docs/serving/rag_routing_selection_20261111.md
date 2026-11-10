# Separate routing selection from final qualification

Registered before business-routing outcomes: the shared-margin100-review
development matrix is still running, no current routing policy result has been
analysed, and the957-review evaluation gold has not been opened. This is a
separate routing registration; the active classification campaign configuration,
threshold grid, classifier decisions and GPU execution are unchanged.

The original implementation reused the final one-sided95% Clopper–Pearson
lower-bound target during development selection. Even with a nominal minimum
of one automatic review, that gate needs at least59 entirely correct automatic
reviews to reach a95% lower bound. With only100 development reviews and at
least10 audit reviews, it can exclude smaller perfect candidate groups before
evaluation. Searching many development thresholds also means their individual
pointwise intervals do not certify the selected policy's coverage guarantee.

The primary development selector now requires, after the unchanged random audit:

- At least30 automatic reviews.
- Empirical complete label-set accuracy of at least95%.
- Oracle-corrected overall pair F1 of at least95%.

Among eligible candidates, maximise automatic review count, then minimise extra
comparator calls, then prefer the higher minimum margin, exactly as before.
Continue reporting every candidate's exact confidence lower bound descriptively.
Preserve the previous CP-based development choice as a separately labelled
sensitivity result; it must not become an alternative selected using evaluation.
If no primary candidate qualifies for selection, retain the all-review fallback.

Final qualification is unchanged: after audit there must be at least100 automatic
reviews, their one-sided95% Clopper–Pearson lower bound must be at least95%, and
oracle-corrected overall pair F1 must be at least95%. A failed final gate yields
zero qualified review-item reduction, with the observed reduction still shown.
The development empirical95% condition does not predict or guarantee final
qualification, particularly for candidates near that boundary. No evaluation
result can change the selected margin, comparator boundary, classifier, or the
choice between the primary and sensitivity rule.

Before any business-routing outcome or final evaluation access, the same
registration additionally fixes one primary routing policy. Only the already
selected classifier's confidence and confidence-plus-disagreement policies are
eligible for this primary role. Prefer a calibration-eligible policy, then more
automatic reviews after audit, fewer extra comparator calls, and a higher
minimum margin; an exact remaining tie selects confidence. If both fall back to
full review, confidence/full-review is the primary. The classifier is not
reselected through routing performance. Every other classifier/policy combination
is a secondary comparison, even if its final evaluation looks better.

The primary name and the complete calibration-based choice record are written
into `review_routing_selection.json` before evaluation labels are read. Final
routing output references that frozen name and labels every result primary or
secondary; it never derives a new winner from957-review outcomes. The old
CP-based sensitivity remains developmental only and cannot become the primary.

`configs/experiments/review_routing_20261111.json` owns these rules. The analysis
CLI accepts `--routing-config`, embeds the exact configuration and its SHA256 in
`review_routing_selection.json`, and refuses a later invocation under changed
configuration before reading calibration or evaluation labels. A
`--classification-only` run never accesses routing configuration. A subsequent
`--selection-only` run can freeze routing in the same unchanged classification
output or a separate analysis directory. Historical results remain historical
replay; new-source final claims still require the reviewed human-label release.
