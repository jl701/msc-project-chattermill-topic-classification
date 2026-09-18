from __future__ import annotations

from scripts.audit_taxonomy_post_supervisor_reuse import _all_named_values


def test_recursive_contract_counter_finds_nested_values() -> None:
    value = {"test_contract_count": 0, "nested": [{"test_contract_count": 0}]}
    assert list(_all_named_values(value, "test_contract_count")) == [0, 0]
