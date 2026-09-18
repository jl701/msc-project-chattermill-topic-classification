# Closed-Topic FABSA Baselines

This note records the first closed-topic baseline results for FABSA.

## Task

The task is multi-label classification over the fixed FABSA aspect+sentiment taxonomy.

The prediction unit is the full pair:

```text
Aspect | sentiment
```

For example, `Online experience: App website | negative` and `Online experience: App website | positive` are different labels. Pair-level F1 requires both the aspect and sentiment to be correct.

## Data

The local FABSA export contains:

| Split | Rows |
| --- | ---: |
| train | 7,930 |
| validation | 1,057 |
| test | 1,587 |

The training label space contains 36 aspect+sentiment labels from 12 aspects.

## Models Tried

The current traditional baselines include:

- Bag-of-Words + One-vs-Rest Logistic Regression.
- Binary Bag-of-Words + One-vs-Rest Logistic Regression.
- TF-IDF + One-vs-Rest Logistic Regression.
- Word+character TF-IDF + One-vs-Rest Logistic Regression.
- Word+character TF-IDF + One-vs-Rest Linear SVM.
- LSA dense vectors from TF-IDF + One-vs-Rest Logistic Regression.

Thresholds are tuned on the validation split only. The test split is used after selecting the best validation configuration.

## Best Validation Configuration

The best validation result is:

```text
model: Linear SVM
features: word+character TF-IDF
word ngrams: 1-3
max features: 100,000 total
C: 0.2
class weight: none
decision threshold: -0.38
```

| Split | Pair Samples F1 | Pair Micro F1 | Pair Macro F1 | Aspect Samples F1 | Aspect Micro F1 | Aspect Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 0.7234 | 0.7190 | 0.4621 | 0.7921 | 0.7827 | 0.7284 |
| test | 0.7090 | 0.7042 | 0.4207 | 0.7736 | 0.7666 | 0.6752 |

## Interpretation

Threshold tuning gives a substantial improvement over the first TF-IDF Logistic Regression baseline.

The strongest traditional models use combined word and character TF-IDF features. Linear SVM gives the best validation result. A small refined search improved validation F1, but did not improve test F1 beyond the simpler SVM word+char configuration, which suggests that the traditional baseline is close to a plateau.

The gap between pair-level F1 and aspect-only F1 shows that sentiment and aspect+sentiment pairing remain important error sources. Pair macro F1 remains much lower than pair micro F1, which indicates weaker performance on rare labels.

## Reproduction

Run all traditional baselines:

```powershell
python .\scripts\run_classical_baselines.py
```

Outputs are written to:

```text
outputs/baselines/classical/
```

The output directory is ignored by Git.

## BERT-Style Encoder Baseline

A BERT-style closed-topic baseline was added after the traditional sweep. The model is trained with a multi-label classification head and BCE loss over the 36 aspect+sentiment labels.

Because FABSA is label-imbalanced, the strongest encoder runs use square-root positive-class weighting:

```text
pos_weight = sqrt((negative examples) / (positive examples))
```

This improves macro F1 substantially compared with unweighted BCE.

### Best Validation-Selected Encoder

The best validation-selected encoder run is:

```text
model: distilbert-base-uncased
max length: 256
batch size: 16
learning rate: 5e-5
epochs run: 10
best epoch: 9
positive-class weighting: sqrt
threshold: 0.53
```

| Split | Pair Samples F1 | Pair Micro F1 | Pair Macro F1 | Aspect Samples F1 | Aspect Micro F1 | Aspect Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 0.7819 | 0.7785 | 0.5454 | 0.8213 | 0.8138 | 0.8004 |
| test | 0.7744 | 0.7663 | 0.5377 | 0.8135 | 0.8058 | 0.7717 |

### Best Observed Test Encoder

A very similar DistilBERT run had slightly lower validation micro F1 but higher test micro F1:

```text
model: distilbert-base-uncased
learning rate: 4e-5
epochs run: 10
best epoch: 8
threshold: 0.64
```

| Split | Pair Samples F1 | Pair Micro F1 | Pair Macro F1 | Aspect Samples F1 | Aspect Micro F1 | Aspect Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 0.7830 | 0.7784 | 0.5356 | 0.8264 | 0.8151 | 0.7989 |
| test | 0.7803 | 0.7738 | 0.5377 | 0.8185 | 0.8108 | 0.7742 |

The validation difference between the `4e-5` and `5e-5` runs is very small, so the safer interpretation is that DistilBERT has reached roughly:

```text
test pair micro F1: 0.766-0.774
test pair macro F1: about 0.538
```

### Comparison With Traditional Baselines

| Model | Test Pair Samples F1 | Test Pair Micro F1 | Test Pair Macro F1 | Test Aspect Samples F1 |
| --- | ---: | ---: | ---: | ---: |
| Word+char TF-IDF + Linear SVM | 0.7090 | 0.7042 | 0.4207 | 0.7736 |
| DistilBERT, validation-selected | 0.7744 | 0.7663 | 0.5377 | 0.8135 |
| DistilBERT, best observed test | 0.7803 | 0.7738 | 0.5377 | 0.8185 |

The encoder baseline gives a clear improvement over the traditional baseline, especially on macro F1.

### BERT-Base Check

One `bert-base-uncased` run was tested:

```text
learning rate: 2e-5
batch size: 8
gradient accumulation: 2
epochs: 6
positive-class weighting: sqrt
```

It reached:

```text
test pair micro F1: 0.7312
test pair macro F1: 0.5094
```

This did not beat the tuned DistilBERT runs in the current setup.
