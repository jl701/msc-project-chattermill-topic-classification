import pytest

from msc_project.serving.retrieval_policy import (
    resolve_retrieval_method, validate_method_definition, validate_selection_policy,
)


def test_registered_mapping_is_explicit_and_independent():
    expected = {"method":"e5", "seed":13, "selection_policy":"same_aspect_negative_seen"}
    assert resolve_retrieval_method("e5_negative_seen") == expected
    assert validate_method_definition({"id":"e5_negative_seen", **expected}) == expected
    assert resolve_retrieval_method("random_37")["seed"] == 37
    assert validate_method_definition({"id":"e5","method":"e5","seed":13})["selection_policy"] == "global"
    changed = resolve_retrieval_method("e5")
    changed["selection_policy"] = "bad"
    assert resolve_retrieval_method("e5")["selection_policy"] == "global"


@pytest.mark.parametrize("value", ["e5_same_seen", "random_99", "random", "unknown"])
def test_unregistered_method_rejected(value):
    with pytest.raises(ValueError, match="Unknown"):
        resolve_retrieval_method(value)


def test_definition_or_policy_cannot_silently_change_comparator():
    with pytest.raises(ValueError, match="conflicts"):
        validate_method_definition({"id":"e5_negative_seen","method":"e5","seed":13})
    with pytest.raises(ValueError, match="E5 only"):
        validate_selection_policy("bm25", "same_aspect_negative_seen")
    with pytest.raises(ValueError, match="Unknown"):
        validate_selection_policy("e5", "all_same")
