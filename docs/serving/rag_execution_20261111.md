# Dynamic demonstration retrieval and review routing

The user approved local execution on 11 November 2026. This campaign compares fixed, random, BM25, E5 and fused demonstration selection for the frozen l2-a01 two-stage Qwen service. The tracked configuration is `configs/experiments/serving_rag_20261111.json`.

The first execution track uses the existing 100 development and 957 historical validation replay reviews. These samples were exposed in historical selection; this track supplies regression and engineering evidence, not new blind generalisation results. A separate 800-review acquisition and human annotation workflow will provide independent development, calibration and evaluation records. New-source gold metrics cannot be produced until actual human annotations are available.

The preflight began from clean commit `6da0148392cb254292a58eccbc2de661069b6ef2`, fetched the remote and created branch `codex/chattermill-rag-20261111`. Existing experiment outputs are preserved. GPU jobs run serially. Raw text, scores, labels and paths remain under ignored `outputs/rag_20261111`.

For offline selection, all twelve sentiment vectors are computed with the same frozen adapter once per review and reused across detection variants. This gives identical Stage 2 evidence for rethresholding and avoids unnecessary repeated computation. Offline matrix elapsed time is not API latency. The final service comparison executes actual retrieval and conditional sentiment scoring and includes queueing, failures and fallbacks.

Strict held-out-theme methods exclude Account access examples and reviews containing that theme. Threshold selection and directed improvements use seen-theme labels only. A later optional support-example onboarding branch is separate from strict unseen evaluation. The official test split is outside this campaign.

All substantive configuration changes and their timing relative to outcomes will be recorded here. No resume is edited as part of execution.
