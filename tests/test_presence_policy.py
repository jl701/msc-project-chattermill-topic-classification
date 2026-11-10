import math

import pytest

from msc_project.serving.presence_policy import (
    active_presence_threshold,finite_margin,presence_selected,presence_space,validated_presence_policy,
)


def test_probability_is_preserved_and_margin_threshold_is_separate_binary64():
    value = 30.0
    policy = {"aspect_threshold":1.0,"presence_score_space":"logit_margin",
              "presence_margin_threshold":math.nextafter(value,math.inf)}
    assert active_presence_threshold(policy) > value
    assert not presence_selected({"score":1.,"presence_logit_margin":value},policy)
    assert presence_selected({"score":1.},{"aspect_threshold":1.})
    assert validated_presence_policy(policy)["aspect_threshold"] == 1.


def test_missing_unknown_and_nonfinite_policies_are_rejected():
    with pytest.raises(ValueError):
        presence_space({"presence_score_space":"logits"})
    with pytest.raises(ValueError,match="requires"):
        active_presence_threshold({"presence_score_space":"logit_margin","aspect_threshold":.5})
    for value in [float("inf"),float("nan"),True,None]:
        with pytest.raises(ValueError):
            finite_margin(value)
