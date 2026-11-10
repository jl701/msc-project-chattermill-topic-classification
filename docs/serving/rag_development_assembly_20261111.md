# Immutable final development matrix assembly

This is a CPU-only evidence organisation step, registered before assembly. It
does not recompute scores, inspect gold labels, choose a classifier, change
thresholds or analyse heldout outcomes. Classification selection can use the
original seven-method directory plus the additional method file directly;
byte-preserving assembly must not change those score hashes or selection results.

`scripts/assemble_rag_development.py` requires both source runs to be complete.
It checks the same100 review IDs in every registered method, unique record IDs,
successful response identities/counts and each completed score-file SHA256.
Original artifact, replay, index-file hashes, runtime-artifact bytes, and Stage2
reuse receipts must agree. Both copied anchor files and the referenced original
anchor/identity/runtime/state provenance hashes are checked.

The source inference-code versions and experiment configurations intentionally
differ. They are preserved per source and per method; no common inference-code
hash or synthetic common configuration is invented. The passed
`negative_policy_native_gate/summary.json` must bind the original E5 score file,
common inputs and the newer core/retrieval-policy source hashes. It supplies
sampled, input-only native seen-score control-parity evidence, not a claim that
all methods or all future inputs were retested under one code version.

```powershell
& $env:RAG_PYTHON scripts/assemble_rag_development.py `
  --source outputs/rag_20261111/development_margin `
  --source outputs/rag_20261111/development_negative `
  --parity-gate outputs/rag_20261111/negative_policy_native_gate/summary.json `
  --output outputs/rag_20261111/development_final --reviews 100
```

Only after all validation passes is a fresh output directory created. Method
JSONL files are copied byte-for-byte, source metadata and the parity receipt are
archived, and `identity.json` records every method's origin. `summary.methods`
retains the original source summaries and hashes. Original component timings are
not presented as assembly time or HTTP latency. A complete state is written last,
with an assembly receipt hash. Existing or partial combined outputs are never
overwritten. All raw score files, IDs and absolute local provenance paths remain
under the ignored `outputs` directory.

Execution completed for the registered source pair: eight methods, each with100
matching review IDs. All method bytes/hashes and common Stage2 provenance passed
validation. The combined directory is `outputs/rag_20261111/development_final`;
its receipt SHA256 is
`fe94931c3ed10f71228c27c4c44bb3d8e5d4cb6c7ad9ac18b9c94bf748879a4c`.
Eleven synthetic CPU tests passed, covering immutable copying, separate versions,
incomplete runs, duplicate IDs, tampered scores, incompatible inputs, altered
anchor provenance and failed/mismatched parity evidence. No quality analysis or
GPU execution was part of this assembly.
