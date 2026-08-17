# Dataset-faithful capped-two sentiment validation v2: results

Date: 17 August 2026

Status: complete and audited

## Structural finding

The official train and validation data support many aspects per review but at
most two sentiments for one review-aspect instance. The output contract should
therefore be multi-label across aspects, with no top-k aspect cap, while each
selected aspect may receive one or two sentiments but never three.

| Split | Reviews | Reviews with at least 3 aspects | Rate | Maximum aspects | Two-sentiment review-aspect instances | Maximum sentiments per aspect |
|---|---:|---:|---:|---:|---:|---:|
| Train | 7,930 | 1,397 | 17.62% | 7 | 73 | 2 |
| Validation | 1,057 | 183 | 17.31% | 5 | 12 | 2 |

No official test data or labels were opened for this audit or experiment.

## Matched decoder comparison

Both decoders use the exact parent Study C aspect threshold and therefore
select the same aspect instances. Argmax emits one sentiment. Capped-two always
emits the top sentiment and adds the runner-up only when its score reaches a
threshold selected on seen-aspect validation. Values below are unweighted means
over twelve held-out-aspect validation folds.

| Method | Decoder | Pair P | Pair R | Pair F1 | Delta | Wins / ties / losses | Mean sentiments / selected aspect | Second-sentiment rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF | Argmax | 0.2744 | 0.3706 | **0.2780** | — | — | 1.0000 | 0.00% |
| TF-IDF | Capped-two | 0.2721 | 0.3727 | 0.2773 | **−0.0008** | 2 / 6 / 4 | 1.0162 | 1.62% |
| E5-base-v2 | Argmax | 0.2273 | 0.3702 | **0.2246** | — | — | 1.0000 | 0.00% |
| E5-base-v2 | Capped-two | 0.2192 | 0.3780 | 0.2222 | **−0.0024** | 1 / 1 / 10 | 1.0514 | 5.14% |

Freezing the aspect threshold removes the main confound found in v1. In
particular, TF-IDF `l2-a08` now exactly matches argmax at 0.3299 instead of
falling from 0.3299 to 0.2275 under the earlier joint threshold transfer.

## Two-sentiment recovery

The held-out folds contain 24 gold sentiment labels belonging to 12
two-sentiment review-aspect instances.

| Method | Argmax | Capped-two |
|---|---:|---:|
| TF-IDF | 9 / 24 (37.5%) | 9 / 24 (37.5%) |
| E5-base-v2 | 6 / 24 (25.0%) | 8 / 24 (33.3%) |

TF-IDF gains no additional true second labels and its tiny F1 reduction is due
to added false positives. E5 recovers two additional gold labels, increasing
recall, but its larger number of runner-up predictions lowers precision enough
to reduce total F1 slightly.

## Interpretation and decision

The capped-two decoder does not outperform argmax on mean validation F1, so it
must not be reported as a performance improvement. It does, however, represent
the annotation structure correctly and has only a small F1 cost: 0.0008 for
TF-IDF and 0.0024 for E5.

The accepted modelling contract is therefore:

1. permit zero, one, or multiple aspect predictions with no top-k aspect cap;
2. permit one or two sentiments per selected aspect;
3. add the second sentiment only through the frozen seen-validation threshold;
4. never emit a third sentiment; and
5. retain argmax as a constrained ablation and the empirical F1 winner.

This structural choice prioritises fidelity to the dataset rather than claiming
a score gain. Because v2 was designed after inspecting v1 validation evidence,
it is sequential development evidence, not independent confirmation. No rule
will be changed further using these held-out results. Any later official-test
evaluation must use the now-frozen choice once the wider method suite is ready.

## Safety and audit

- Methods: 2; completed folds: 24 / 24; failed folds: 0.
- Decoder rows: 48 / 48.
- Parent aspect threshold reused exactly in every fold.
- Parent Study C argmax precision, recall, and F1 reproduced to `1e-12`.
- Maximum predicted sentiments per review-aspect: 2.
- Official-test artifacts/accesses: 0.
- Non-finite values: 0.
- Audited artifacts with SHA-256 records: 28.
- Machine-readable audit:
  `outputs/experimental/taxonomy_capped_two_sentiment_validation_v2/audit.json`.

Raw fold outputs remain under the ignored experimental output root. The frozen
configuration, implementation, tests, audit script, and this aggregate report
are safe to track.
