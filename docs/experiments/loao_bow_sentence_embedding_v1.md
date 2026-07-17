# Protocol-Matched BoW And Sentence-Embedding LOAO Baselines

## Status

Pre-registered on 17 July 2026. No validation or test result had been inspected when this specification was written.

## Objective

Close the missing representation rungs in the primary all-row leave-one-aspect-out benchmark. The experiment adds one count-based Bag-of-Words baseline and two frozen sentence encoders. It also reruns the character TF-IDF reference under the strict train-only feature-fitting rule required by the frozen design.

## Common Contract

- Twelve held-out-aspect folds.
- `example_filtered` training only.
- Every official validation or test row retained.
- Gold pair labels projected to the current held-out aspect.
- One raw canonical candidate-aspect string supplied per fold.
- Empty predictions allowed.
- Presence threshold selected on validation pair micro F1, separately per method and aspect.
- One shared global word+character TF-IDF sentiment classifier, trained only on the permitted fold, so the experiment isolates the candidate-presence representation.
- Pair-set scoring remains authoritative because some review-aspect instances carry more than one gold sentiment.

## Registered Methods

| Method | Representation | FABSA fitting | Candidate/review formatting |
| --- | --- | --- | --- |
| `bow_count_1_2_train_vocab` | word count uni/bi-grams, cosine | vocabulary fitted on permitted train reviews only | raw/raw |
| `tfidf_char_3_5_train_vocab` | character `char_wb` 3–5 TF-IDF, cosine | vocabulary and IDF fitted on permitted train reviews only | raw/raw |
| `minilm_l6_v2` | frozen `sentence-transformers/all-MiniLM-L6-v2`, mean pooling, L2 | none | raw/raw |
| `e5_base_v2` | frozen `intfloat/e5-base-v2`, mean pooling, L2 | none | `query: ` candidate / `passage: ` review |

Model revisions, vectorizer parameters, truncation, thresholds, tie-breakers, output paths, and stopping rules are frozen in `configs/experiments/loao_bow_sentence_embedding_v1.json`.

The strict TF-IDF row is necessary because the historical lexical implementation fitted its vectorizer on permitted train reviews plus the candidate label text. That historical result remains reproducible evidence, but it is not used as the strict train-only reference in this comparison.

## Isolation And Execution Order

1. Run focused unit tests and a one-aspect validation smoke test.
2. Run all four methods on validation without loading the official test file.
3. Verify that every method has exactly twelve selected thresholds and freeze the selection manifest.
4. Run the test stage once, loading thresholds from that manifest and without reselecting any model or threshold.
5. Export only aggregate and per-aspect numeric evidence to tracked CSVs. Review-level predictions remain under ignored `outputs/`.

## Planned Metrics

Presence precision, recall, positive-class F1, average precision, prevalence, and false-positive/false-negative rows per 100 are reported separately from end-to-end aspect-sentiment pair metrics. The main comparison is the unweighted mean of the twelve test fold pair micro F1 values. Mean, median, population standard deviation, minimum, and maximum are retained, together with per-aspect rows and conditional/oracle-presence sentiment diagnostics.

## Known Limitation

The shared sentiment component emits one sentiment for a detected candidate, whereas the authoritative gold representation can contain multiple sentiments for the same review-aspect instance. This is a common pipeline limitation, not a change to the evaluation target.

## Results

Pending execution.
