# Methods in practical terms

## Input and output

For a review and every supplied candidate aspect, predict whether the aspect
is present. For a detected aspect, predict one or at most two sentiments from
negative, neutral and positive. An absent aspect emits no pair. A candidate
list is always supplied, so this is not open-world discovery of new names.

## Training construction

Each leave-one-aspect-out fold withholds all task-specific supervision for
one aspect. Evaluation still uses the complete twelve-aspect candidate grid.

| Policy | Mixed review mentioning held-out A and seen B | Training targets |
|---|---|---|
| Review filtering | Remove the entire review | Neither A nor B from that review |
| Label masking | Retain the review text | B and other eligible seen aspects, never A |

Masking applies to positive **and negative** targets for A. It also removes A's
annotation channels from the training representation. The held-out text may
remain visible as context. That is the intended difference between the policies.
Only training rows are transformed. Validation and test are not filtered in
this way.

## What each model learns

| Method | Fitted component | Candidate information |
|---|---|---|
| TF–IDF | Logistic regression on text/candidate features | Lexical overlap and sentiment cues |
| Frozen E5 | No encoder adaptation | Embedding similarity to candidate text |
| Description-to-classifier weight transfer | Map descriptions to seen-aspect classifier weights | Embedding of the unseen aspect's description |
| DistilBERT | Task-specific neural classifiers | Review and candidate text |
| Qwen zero-shot | No task-specific weights | Instructions and candidate text |
| Qwen few-shot | No task-specific weights | Instructions plus selected training examples |
| QLoRA | Low-rank adapter on the quantised base model | Separate presence and sentiment prompt formats |

See the [architecture guide](architecture.md) for implementation locations.
Descriptions and frozen resources are in `configs/`. Names-only (`N`) and
minimal-definition (`D`) conditions use registered interfaces. Their contrast
should not be described as a universal causal effect of richer semantics.

## QLoRA: two tasks, one adapter

The model answers two prompt types, not one unrestricted aspect-plus-sentiment
generation. Presence prompts have Y/N answers. Conditional sentiment prompts
use A/B/C for negative/neutral/positive. One adapter per fold model learns both
tasks and is queried separately for each stage at inference.

The registered sample budget is 4,096 training prompts: 2,048 presence and 2,048
sentiment prompts. Sampling balances answer classes within each task, without
replacement. It does not allocate prompts proportionally to aspect frequency.
Presence targets equal Y/N quotas. For sentiment, the small remainder when dividing
2,048 by three is allocated deterministically. If a class cannot fill its quota,
the remaining capacity is allocated to other classes without duplicating examples.
An insufficient total pool raises an error. Eligibility, deduplication and
the deterministic seed are applied before training.

The frozen recipe uses one epoch, rank 4, alpha 8, dropout 0.05, maximum length
384, batch size 1 with eight gradient-accumulation steps, and q/k/v/o projection
adapters. Learning-rate candidates are 2e-6, 5e-6 and 1e-5. Selection uses the
registered seen-only validation criterion, not held-out-aspect outcomes.
Training manifests record the actual prompts used. The configuration and
implementation remain the authority for execution details.

## Few-shot examples

Demonstrations are selected from eligible training examples and then remain
fixed for the corresponding fold/stage contract. The target review changes
between inference calls. In the completed policy study, applying the same
selection rule to different training pools can select different demonstrations.
The two policies therefore do not necessarily share identical prompts.

If a frozen model instead received exactly the same demonstrations, target
reviews, candidate text and inference settings in both conditions, there would
be no training-policy change in its inputs to explain a systematic difference.
That fixed-demonstration control is not claimed to have been run here.

## Thresholds, decoder and composition

Thresholds are selected for each model and outer fold using seen-aspect
validation labels only. A single threshold applies across candidates within
that fold, rather than a fitted threshold for each unseen aspect. Stage 1 is
selected first. The Stage-2 runner-up threshold is then selected with Stage 1
fixed. Candidate scope, tie rules and selection objectives are registered.

For a present aspect, the decoder always emits its highest-scoring sentiment.
It emits the runner-up only if its score reaches the frozen threshold. It never
emits a third sentiment. Ties use the canonical sentiment order.

The composition reuses the few-shot presence scores/threshold and QLoRA
sentiment scores/runner-up threshold. It has no new trainable gate. The
completed policy study performs this construction separately within each policy.

For sampling code, see
[`taxonomy_two_stage_training.py`](../src/msc_project/experiments/taxonomy_two_stage_training.py).
For exact prompts, see
[`qwen_two_stage_classifier.py`](../src/msc_project/llm/qwen_two_stage_classifier.py).
For decoding and selection, see
[`taxonomy_two_stage.py`](../src/msc_project/experiments/taxonomy_two_stage.py).
