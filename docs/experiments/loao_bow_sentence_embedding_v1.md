# Protocol-Matched BoW And Sentence-Embedding LOAO Baselines

## Status

Pre-registered on 17 July 2026. Full validation is complete and the 48 method-aspect thresholds are frozen. The official test file had not been loaded when the validation evidence below was recorded.

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

### Validation Gate

Command:

```powershell
python .\scripts\run_similarity_loao_baselines.py --stage validation --device cpu --local-files-only --output-dir .\outputs\baselines\loao_bow_sentence_embedding_v1 --public-output-dir .\docs\thesis_figure_data
```

The clean run used commit `63c3e724b336137c49425e1e7b59e257c037a5e4`, completed in `313.8` seconds on CPU, and loaded only the official train and validation CSVs. The output audit confirmed 4 methods, 12 aspects, 48 finite thresholds, 48 per-fold result rows, and 50,736 review-level validation predictions with no review text in the JSONL payloads.

| Method | Pair micro F1 | Presence F1 | Presence average precision |
| --- | ---: | ---: | ---: |
| Count BoW | 0.3269 | 0.3763 | 0.2820 |
| Strict train-only TF-IDF | 0.3943 | **0.4624** | 0.4027 |
| MiniLM-L6-v2 | 0.3603 | 0.4286 | 0.4029 |
| E5-base-v2 | **0.3944** | 0.4564 | **0.4327** |

Observed fact: E5 and strict TF-IDF are effectively tied on mean validation pair micro F1 at the displayed precision, while E5 has higher ranking-quality average precision and strict TF-IDF has slightly higher thresholded presence F1. MiniLM improves over BoW but is below both E5 and strict TF-IDF on this validation aggregate. These observations do not alter the pre-registered test plan: all four methods retain their frozen per-aspect thresholds and will be evaluated once.

### Test

Pending the frozen test-stage run.
