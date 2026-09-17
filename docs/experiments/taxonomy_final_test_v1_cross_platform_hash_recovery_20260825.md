# Taxonomy final test v1: pre-scoring cross-platform hash recovery

Date: 2026-08-25  
Status: operational recovery before the registered one-time result reveal

## Scope and outcome boundary

The user authorised the three-GPU final-test launch on 2026-08-24.  The GPU
workers received only the registered label-free review file with columns
`row_uid,text`; the official label vault remained local and was not copied to
RunPod.  No aggregate metric, fold metric, official label or comparison was
computed or inspected during the attempts described here.

The first launch failed before scoring because the migrated containers resolved
the portable `python` token to the base image interpreter, which lacked the
audited scientific packages.  All three failure records and logs were retained
under the local final-test backup's `_failure_evidence` directory and under a
separate remote failure-evidence root.

An exact retry bound each worker to its existing formal-v2 virtual environment.
Those environments retained the frozen package versions and CUDA support.  The
retry then exposed two operational preflight issues:

- worker B first needed the already-pinned E5 revision restored to its
  persistent Hugging Face cache; and
- workers A and B rejected QLoRA checkpoint trees even though every file size
  and every per-file SHA-256 matched the frozen local backup.

Worker D passed preflight and wrote one label-free score CSV before the common
release was stopped.  That score bundle was not verified or published as a
completed sync unit, no labels were joined to it, and it is excluded from the
replacement run.  The complete partial worker directory was archived without
opening the score contents.

## Root cause

`_tree_sha256` sorted `Path` objects directly before serialising per-file hash
records.  `WindowsPath` comparisons are case-insensitive, while Linux path
comparisons are case-sensitive.  A checkpoint containing both
`adapter/README.md` and lower-case adapter files therefore produced a different
tree digest on Linux despite identical file contents.

For the first worker-A checkpoint, the frozen digest was
`175b5e7a1e249ec66283c7c92f4788191a26b2bc2b8fc2f78dce05a84075269e`.
The uncorrected Linux aggregate differed, but all 12 relative paths, byte counts
and per-file SHA-256 values matched the local backup exactly.  Recomputing the
tree with a case-folded relative POSIX-path order reproduced the frozen digest.
The same check reproduced all eight worker-A QLoRA checkpoint tree digests.

## Repair boundary

The repair sorts checkpoint files by
`(relative_posix_path.casefold(), relative_posix_path)` before applying the
unchanged canonical JSON and SHA-256 calculation.  It adds a regression test
covering an upper-case `README.md` beside lower-case adapter files.

The repair changes no model weights, checkpoint file, selected threshold,
prompt, demonstration, candidate description, decoder, fold, condition,
metric, hypothesis, test row or label boundary.  A replacement execution
release must bind the repair commit, pass the focused and full test suites,
re-verify all 93 assigned artifact sets on Linux, and restart all three workers
from clean output roots.  The stopped attempts cannot contribute scores to the
replacement result graph.
