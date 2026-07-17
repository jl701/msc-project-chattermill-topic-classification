# Protocol-Matched BoW And Sentence-Embedding LOAO Baselines

## Status

Pre-registered and completed on 17 July 2026. Full validation froze all 48 method-aspect thresholds before the test stage loaded the official test split. The four registered methods were then evaluated once on test with no threshold or encoder reselection.

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

Command:

```powershell
python .\scripts\run_similarity_loao_baselines.py --stage test --device cpu --local-files-only --output-dir .\outputs\baselines\loao_bow_sentence_embedding_v1 --public-output-dir .\docs\thesis_figure_data
```

The clean test run used commit `c4a063757fd2260b28e526d362e1316f1d166d13` and completed in `321.2` seconds on CPU. Before loading `test.csv`, the runner required the validation manifest to declare a protocol-complete 4-method by 12-aspect selection with exact registered method/aspect keys, a matching configuration fingerprint, and finite thresholds inside the registered grid. The audit found 48 test result rows and 76,176 review-level predictions. The maximum absolute difference between the validation and test threshold mappings was `0.0`; no test threshold sweep was produced. Review text is absent from the prediction payloads.

| Method | Pair micro F1 | Pair samples F1 | Pair macro F1 | Presence F1 | Presence average precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| Count BoW | 0.3225 | 0.0855 | 0.2309 | 0.3631 | 0.2562 |
| Strict train-only TF-IDF | 0.3667 | 0.0902 | 0.2654 | 0.4161 | 0.3476 |
| MiniLM-L6-v2 | 0.3699 | 0.0854 | 0.2690 | 0.4138 | 0.3882 |
| E5-base-v2 | **0.3791** | **0.0956** | **0.2756** | **0.4272** | **0.4177** |

### Paired Aspect-Level Evidence

The following post-run exploratory stability analysis reads only the frozen numeric per-aspect CSVs. It uses the twelve held-out aspects as paired units, reports all six method pairs, applies a two-sided exact sign-flip test with Holm correction within each split, and uses a seeded 100,000-resample paired bootstrap percentile interval for the mean pair-micro-F1 difference. Because the contrasts were not pre-registered and the folds share reviews and substantially overlapping training data, these quantities describe stability across the twelve named taxonomy aspects rather than confirmatory inference to a population of independent aspects.

```powershell
python .\scripts\analyse_similarity_loao_results.py
```

| Test comparison (challenger − reference) | Mean difference | 95% paired-bootstrap interval | Exact p | Holm p | Wins / ties / losses |
| --- | ---: | ---: | ---: | ---: | ---: |
| E5 − strict TF-IDF | +0.0123 | [−0.0788, 0.0940] | 0.8022 | 1.0000 | 8 / 0 / 4 |
| MiniLM − strict TF-IDF | +0.0032 | [−0.0897, 0.0781] | 0.9561 | 1.0000 | 8 / 0 / 4 |
| E5 − MiniLM | +0.0091 | [−0.0222, 0.0376] | 0.5840 | 1.0000 | 7 / 0 / 5 |
| Strict TF-IDF − Count BoW | +0.0442 | [0.0202, 0.0710] | 0.0039 | 0.0234 | 9 / 2 / 1 |
| E5 − Count BoW | +0.0566 | [−0.0321, 0.1367] | 0.2290 | 1.0000 | 9 / 0 / 3 |
| MiniLM − Count BoW | +0.0474 | [−0.0385, 0.1192] | 0.2847 | 1.0000 | 9 / 0 / 3 |

### Interpretation

- E5 is the strongest registered method on the test aggregate, including pair micro F1, thresholded presence F1, and ranking-quality presence average precision.
- The numerical gaps among E5, MiniLM, and strict TF-IDF are small relative to their across-aspect variation. Their paired intervals include zero, so this experiment does not establish that one of those three is reliably superior across aspects.
- Strict train-only TF-IDF clearly improves on unweighted Count BoW in this twelve-aspect sample. This supports the value of lexical weighting and character-level matching over raw word counts.
- Frozen semantic encoders close or slightly reverse the aggregate gap to strict TF-IDF without any FABSA fitting. Their strongest defensible claim here is competitive unseen-candidate transfer, not a statistically resolved win over the lexical reference.
- The historical TF-IDF result of `0.3780` remains a legacy row because candidate text participated in vectorizer fitting. Its similar headline score must not be used as the strict same-protocol comparator; the strict reference is `0.3667`.

### Limitations

- There are only twelve paired aspect units, and aspect prevalence/difficulty varies substantially. The exact test and bootstrap interval therefore have limited power and should accompany, not replace, the per-aspect table.
- Thresholds are target-calibrated on each held-out aspect's validation labels. This is the frozen primary cold-start protocol, not the stricter zero-label threshold-transfer condition.
- The sentence encoders use only canonical aspect names, not definitions or supervised contrastive tuning.
- The shared sentiment classifier predicts one sentiment for a detected candidate, while a small number of gold review-aspect instances may contain conflicting sentiment pairs.

### Tracked Evidence

- `docs/thesis_figure_data/loao_bow_sentence_embedding_v1_selected_thresholds.csv`
- `docs/thesis_figure_data/loao_bow_sentence_embedding_v1_validation_aggregate.csv`
- `docs/thesis_figure_data/loao_bow_sentence_embedding_v1_validation_per_aspect.csv`
- `docs/thesis_figure_data/loao_bow_sentence_embedding_v1_test_aggregate.csv`
- `docs/thesis_figure_data/loao_bow_sentence_embedding_v1_test_per_aspect.csv`
- `docs/thesis_figure_data/loao_bow_sentence_embedding_v1_paired_comparisons.csv`

The complete local manifests, fold diagnostics, and review-level no-text prediction records remain under ignored `outputs/baselines/loao_bow_sentence_embedding_v1/`.
