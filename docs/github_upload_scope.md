# Public repository boundary

This repository is public and linked from the author's CV. Its current tree is
therefore curated as a research-software portfolio, not as a mirror of the
author's complete local workspace.

## Included

- reusable source code and experiment entry points;
- synthetic test fixtures and offline checks;
- registered configurations and aggregate result tables;
- methods, limitations, evidence roles and reproducibility guidance;
- historical scientific records needed to interpret reported results.

Aggregate evidence supports inspection of the reported claims. It does not make
the repository a self-contained reproduction bundle for every trained model.

## Excluded

- raw review text and private or unpublished annotations;
- row-level predictions, score shards and official-label vaults;
- model weights, adapters, checkpoints, caches and cloud backups;
- API tokens, SSH keys, credentials, live endpoints and billing receipts;
- supervisor messages, meeting/email drafts and agent handoff prompts;
- working dissertation sources, draft PDFs and private editorial reviews;
- absolute author-machine paths when they are not essential historical facts.

These materials remain outside Git or are ignored locally. Removing a file from
the current tree does not erase an earlier Git object. If a real credential or
confidential dataset is ever committed, rotate or contain it first, then use a
reviewed history-rewrite procedure rather than relying on a normal deletion.

Some immutable historical manifests name non-public inputs that were present
when an audit ran. A recorded path or hash is provenance, not evidence that the
underlying private artifact is distributed in the current tree.

## Before every public push

1. Review `git status --short` and `git diff --cached` file by file.
2. Run `python scripts/check_quality.py --test`.
3. Run `python scripts/audit_repository.py --output NEW_JSON_PATH` and inspect
   every credential-pattern candidate without printing the candidate value.
4. Confirm no raw text, private correspondence, machine identity or active
   infrastructure address was added.
5. Confirm result tables are aggregate evidence and their evidential role is
   stated accurately.

GitHub secret scanning and push protection complement this review. They do not
replace it and cannot detect every form of confidential research material.
