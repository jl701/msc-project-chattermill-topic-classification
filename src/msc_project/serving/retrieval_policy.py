"""Pure, shared identities for registered retrieval comparisons."""
from __future__ import annotations

GLOBAL = "global"
SAME_ASPECT_NEGATIVE_SEEN = "same_aspect_negative_seen"
SELECTION_POLICIES = (GLOBAL, SAME_ASPECT_NEGATIVE_SEEN)


def validate_selection_policy(method: str, selection_policy: str) -> str:
    if selection_policy not in SELECTION_POLICIES:
        raise ValueError(f"Unknown retrieval selection policy {selection_policy}")
    if selection_policy != GLOBAL and method != "e5":
        raise ValueError("Same-aspect negative policy is registered for E5 only")
    return selection_policy


def resolve_retrieval_method(method_id: str) -> dict:
    """Return a fresh constructor mapping; reject unregistered method IDs."""
    if method_id in ("fixed", "bm25", "e5", "rrf"):
        return {"method": method_id, "seed": 13, "selection_policy": GLOBAL}
    if method_id in ("random_13", "random_37", "random_73"):
        return {"method": "random", "seed": int(method_id.rsplit("_", 1)[1]),
                "selection_policy": GLOBAL}
    if method_id == "e5_negative_seen":
        return {"method": "e5", "seed": 13,
                "selection_policy": SAME_ASPECT_NEGATIVE_SEEN}
    raise ValueError(f"Unknown registered retrieval method {method_id}")


def validate_method_definition(definition: dict) -> dict:
    expected = resolve_retrieval_method(definition["id"])
    supplied = {"method": definition["method"], "seed": definition["seed"],
                "selection_policy": definition.get("selection_policy", GLOBAL)}
    if supplied != expected:
        raise ValueError(f"Registered retrieval definition conflicts with method ID {definition['id']}")
    return expected
