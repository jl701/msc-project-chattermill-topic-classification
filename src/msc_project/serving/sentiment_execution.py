"""Shared lightweight identity for full-grid-compatible sentiment execution."""
import hashlib
import json

SENTIMENT_EXECUTIONS = ("selected_only", "canonical_full_grid_groups")


def build_sentiment_group_contract(candidates, *, execution, pair_batch_size, max_batch_tokens, logits_mode):
    if execution not in SENTIMENT_EXECUTIONS or logits_mode not in ("full", "last_token"):
        raise ValueError("Unknown sentiment execution or logits policy")
    if pair_batch_size < 1 or max_batch_tokens < 1:
        raise ValueError("Batch limits must be positive")
    contract = {"execution":execution,
        "candidate_order_sha256":hashlib.sha256(json.dumps(candidates,sort_keys=True,
            ensure_ascii=False).encode("utf-8")).hexdigest(),
        "pair_batch_size":pair_batch_size,"max_batch_tokens":max_batch_tokens,
        "logits_mode":logits_mode,
        "grouping":"original_full_grid_pair_order_preserving_padded_token_budget",
        "scope":"same engine request batch; grouping may depend on other requests outside the registered 12x4 protocol"}
    return {**contract,"contract_sha256":hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()}
