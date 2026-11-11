# Purpose-bound adaptation use (registered before CPU validation)

## Scope and decision

This change is prepared in an isolated worktree based on `3633f87`. Do not
integrate it into the running campaign until the frozen queue has finished.
The objective is to prevent a registered adaptation record whose nonempty
retrieval word-normalised form matches protected input from reaching future
manual error-driven tuning, model/prompt improvement, fitting or feedback.
Word-normalised equality does not establish equal sentiment or meaning.

Keep all original inputs, assignments, human annotation, blind review and
immutable release records. Calibration and locked evaluation retain complete
populations. A purpose-specific view is an eligibility filter, not a new split.
Distinct nonempty originals with empty word-normalised forms remain distinct.
Other feedback filters still apply; the policy does not establish a legal
feedback-example count.

## Registration and validation plan

No GPU, model inference, training, threshold selection, real human labels or
model predictions are used. All execution is CPU-only synthetic unit testing.
No model/prompt/hyperparameter/seed/budget changes are proposed. Tests use
temporary synthetic datasets, reviewed releases, policies and feedback files.
Keep private source rows, private policy IDs/text, labels and outputs out of Git.

Require an explicit policy path, expected policy SHA256 and source root on
every adaptation-use export and feedback append. Validate the pinned campaign
source-to-policy registration, every policy source fingerprint, split and
population counts, excluded IDs/reasons/purposes and nonempty protected match
counts before opening reviewed gold. Keep the complete release loader for
annotation integrity and evaluation; expose a distinct guarded improvement view.
Re-export candidates with the same policy at append time, retaining existing
held-out-theme and duplicate protections. Store only counts and provenance
hashes in the public-safe policy audit.

Meaningful tests will cover policy/source hash mismatch, unknown/wrong IDs,
split and purpose mismatch, missing mandatory policy, full release population,
excluded improvement eligibility, feedback tampering despite refreshed candidate
hashes, and distinct normalised-empty inputs. Existing annotation, campaign
analysis and feedback suites remain regression checks. Python is
`C:/Msc_DSML/NLP/hdc-latency-20261108/.venv/Scripts/python.exe`; Ruff is
`uvx --offline --from ruff==0.16.7 ruff`.

This is a data-use safeguard, not another classifier-tuning round. Do not use
the historical `analyse_rag_seen_badcases.py` on renamed adaptation labels.
External notebooks must consume the guarded adaptation view; generic file I/O
cannot be made safe merely by changing an export filename.

## Implemented interface and migration

`AdaptationPolicy(path, source_root, expected_sha256)` is mandatory for
`load_adaptation_use`, `export_adaptation_use`, `export_feedback`,
`verified_feedback` and `append_feedback_index`. The source root resolves every
relative path in the private policy. The caller supplies the previously
registered policy hash. The current campaign's input hash is additionally pinned
to its policy hash in `configs/experiments/adaptation_use_20261111.json`.
An absent or changed policy/source fails closed before reviewed gold is opened.
Every registered source file and the active retrieval code must match byte for
byte. Git newline conversion can change a code-file hash in a separate checkout;
do not weaken the check or rewrite the registered policy to hide a mismatch.
Integration into the original checkout leaves its retrieval file untouched.

The purpose view contains eligible adaptation rows, annotations and labels only.
Its audit records original population counts, reviewed count, excluded count,
eligible count and source/policy fingerprints, without private IDs or text.
The unchanged `load_reviewed_split` remains the complete-population release
integrity/evaluation loader; it does not grant permission for improvement use.

The error-analysis/model-improvement export performs no classifier tuning. Use
the named purpose that describes the intended later operation:

```text
python scripts/review_annotation_workflow.py export-adaptation-use
  --data <frozen-annotation-dataset>
  --release <complete-reviewed-adaptation-release>
  --output <new-private-eligible-view>
  --purpose manual_error_driven_tuning
  --adaptation-use-policy <registered-private-policy.json>
  --adaptation-use-policy-sha256 <registered-policy-sha256>
  --policy-source-root <repository-containing-bound-source-files>
```

Other purposes are `model_or_prompt_improvement`, `training_or_fitting`, and
`demonstration_feedback_or_example_selection`. Optional `--predictions` accepts
a private JSONL file with exactly one record per original adaptation ID and no
other split. It writes only eligible prediction records, aligned by sorted ID
with `replay.jsonl` and `labels.jsonl`, and binds the source prediction hash.
The exported manifest also fingerprints every output file. The export requires
complete adaptation review; feedback export retains the existing partial-scope
support but applies the same exclusion to the reviewed scope.

Both `export-feedback` and `prepare_rag_feedback_index.py append` now require the
same three policy arguments. Old candidate directories are not retrofitted;
make a new immutable export and index version. Append independently regenerates
and compares pool, provenance and manifest, then retains all original protected
duplicate and strict-unseen checks. Its final manifest carries the policy audit.
Annotation, blind review and immutable release commands require no policy and
continue retaining every original member.

## Validation record

Independent read-only review found an ordering risk for a valid release whose
labels were serialized in a different order. The eligible export now sorts all
three aligned files by ID; a reverse-order synthetic release covers this case.
The review also checked fail-closed entry points and confirmed release/evaluation
code remains unchanged. Existing command templates were updated with mandatory
policy flags. No real human annotations, model predictions or models were read
or executed, and no private source data were copied into tracked files.

Validation command (CPU synthetic fixtures only):

```text
C:/Msc_DSML/NLP/hdc-latency-20261108/.venv/Scripts/python.exe -m pytest -q tests/test_adaptation_use.py tests/test_annotation_workflow.py tests/test_feedback_index.py tests/test_analyse_rag_campaign.py
uvx --offline --from ruff==0.16.7 ruff check src/msc_project/serving/adaptation_use.py src/msc_project/serving/annotation_workflow.py src/msc_project/serving/feedback_index.py scripts/review_annotation_workflow.py scripts/prepare_rag_feedback_index.py tests/test_adaptation_use.py tests/adaptation_policy_fixture.py tests/test_annotation_workflow.py tests/test_feedback_index.py
git diff --check
```

Result: **77 passed in 25.20 seconds**. Ruff and whitespace checks passed.
A private-ID scan of all 13 changed/new files found no registered private
exclusion ID. The scan read only the input-only policy, never labels or model
predictions, and reported no private values.

Do not integrate, push, run real feedback export or change current selections
while the original 13-job frozen campaign is active. After that queue completes,
the coordinator may integrate this isolated commit and use it for future
human-reviewed work. No feedback-quality or model-improvement result is claimed.

### Integrated after campaign completion

The original serial campaign completed all 13 stages with exit code zero on
2026-11-11. The 46-run service evidence and 16-run stability source bindings
were audited before integration, and an aggregate snapshot was extracted.
Commit `6c1b3779a01d218d2caafaf5390f071aed677950` was then fast-forwarded into
the campaign branch. The same focused suite passed again: **77 tests in
24.64 seconds**; Ruff and whitespace checks passed.

A real-input-only preflight called `validate_policy` for all four purposes,
without loading a human release. All source bindings and the pinned policy
hash matched, each purpose excluded one adaptation record, and the original
`retrieval.py` byte hash remained
`324a82d7715c687d85e6efd2a441b611c484a9f02035c0e0a5c4aca46e14ea45`.
The annotation and protected evaluation populations were unchanged. Actual
reviewed feedback export and any resulting quality gain still await labels.
