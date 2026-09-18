# Level 3 Rich-Guidance Pre-flight Evidence

Date: 2026-07-24

Status: passed; official test remains sealed.

## Frozen identities

| Item | SHA-256 |
| --- | --- |
| Minimal definitions (`D`) | `fc93cf27efdb64ad335f39f4a0dbdbd3dad13b1aae5de0010280931867af7d4c` |
| Rich taxonomy guidance (`R`) | `289ba3238cb6772f9bfda98eb73ac8a1108ba4b2bb3d23eecf72826881f415b9` |
| Bound description bundle | `fcf546d227ad2ac52ccfa9682fc3685d2e396069ed911016f0bc12bf205f1367` |
| Scientific protocol | `d7ccded514ac1cbccf337e496e039ac418698566be0c0ca0c21e18608cc40f85` |

## Static verification

- Focused taxonomy resource, protocol, pipeline, analysis, audit, runner, and
  plan suite: `52 passed`.
- Full repository suite: `297 passed in 11.01s`.
- All edited JSON resources parsed successfully.
- Dissertation build: 30 pages, no undefined references or citations and no
  LaTeX errors. Three non-blocking layout warnings remain.
- `git diff --check`: passed apart from Git's Windows line-ending notices.

## Real-model synthetic smoke

All four local methods executed the same Level 3 synthetic fold under
`NN`, `DN`, `ND`, `DD`, and `RR`. Every run:

- built 47 synthetic training pairs;
- evaluated 108 logical pairs per condition;
- reduced 540 logical condition-pairs to 144 unique rendered inputs;
- wrote and validated both resumable score shards for every condition; and
- recorded `official_data_read: false`.

| Method | Device | Fit (s) | Score (s) | Smoke summary SHA-256 |
| --- | --- | ---: | ---: | --- |
| Strict train-only TF-IDF | CUDA host environment; CPU estimator | 0.035 | 2.998 | `203a5939d5b4dda5a5ff444411c2d698598dadc325bb44d4a7d599a827d6472e` |
| E5-base-v2 | CUDA | 0.997 | 3.648 | `0bee7d6eb4589ce26a146d695adca2183456656d0756f13a6f7c94d9b5a526e0` |
| DistilBERT cross-encoder | CUDA | 1.854 | 3.069 | `cb330481f1ce68f992f5b209c2840216c0c1acb791892fcaa632e56420cafb2e` |
| Frozen candidate-pair Qwen | CUDA | 10.447 | 17.568 | `d50aa1523dd000c90275f9988ddcc952d735b31e0353758d3d44b5daea1018f2` |

Synthetic F1 values are deliberately not treated as performance evidence.
These runs test resource rendering, model compatibility, raw-score caching,
artifact contracts, sharding, and resume validation only.

## Regenerated execution plan

Plan:
`outputs/experimental/taxonomy_hybrid_execution_plan_v2_20260724.json`

Plan SHA-256:
`e2a06d17ed35db8ce0afd3c67c22122dd04ba299019d11c34c6f1504f84e9685`

The plan contains 2,611 dependency-ordered jobs:

| Placement | Jobs |
| --- | ---: |
| Local CPU | 1,733 |
| Local GPU | 354 |
| Cloud GPU | 294 |
| Cloud control/analysis | 230 |

The plan opens local official validation only after the implementation commit.
It does not authorise official test inspection before every admitted method,
threshold, seed, and analysis rule has been frozen.
