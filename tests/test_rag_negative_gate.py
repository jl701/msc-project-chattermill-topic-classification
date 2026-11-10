from dataclasses import replace

import pytest

from msc_project.llm.qwen_two_stage_classifier import TwoStageDemonstration
from check_rag_negative_policy import compare_selection, item_sha


def demos():
    return [TwoStageDemonstration(str(i),aspect,str(i),aspect,label)
            for i,(aspect,label) in enumerate(zip(["A","B","A","B"],"YYNN"))]


def test_gate_distinguishes_allowed_negative_change_and_bundled_changes():
    old=demos()
    new=old[:3]+[replace(old[3],candidate_aspect="A",aspect_candidate="A")]
    assert compare_selection(old,new,aspect="A",heldout="H")["selection_changed"]
    assert not compare_selection(old,old,aspect="H",heldout="H")["selection_changed"]
    with pytest.raises(AssertionError,match="positive"):
        compare_selection(old,[replace(new[0],review_text="other"),*new[1:]],aspect="A",heldout="H")
    with pytest.raises(AssertionError,match="held-out"):
        compare_selection(old,new,aspect="H",heldout="H")
    with pytest.raises(AssertionError,match="aligned"):
        compare_selection(new,old,aspect="A",heldout="H")
    with pytest.raises(AssertionError,match="repeats"):
        compare_selection(old,new[:2]+[new[2],new[2]],aspect="A",heldout="H")
    assert item_sha({"input_ids":[1,2]}) != item_sha({"input_ids":[1,3]})
