import copy
import json
import math

import numpy as np
import pytest

import analyse_rag_campaign as analysis
import analyse_rag_seen_badcases as badcases
import run_rag_service_study as service
from msc_project.serving.rag_evaluation import (
    active_presence_threshold, decode_records, margin_threshold_candidates,
    score_maps, select_threshold, validate_record_spaces,
)
from msc_project.serving.review_routing import assign_review, risk_features
from test_analyse_rag_campaign import fixture, jsonl
from test_rag_evaluation import ASPECTS, record
from test_rag_service_study import api_response, make_args


def margin_record(key, margin=25., heldout_margin=-10.):
    row = record(key, seen=1.)
    row["response"]["presence_score_space"] = "logit_margin"
    for item in row["response"]["aspect_scores"]:
        item["presence_logit_margin"] = margin if item["aspect"] == "seen" else heldout_margin
    return row


def test_saturated_probabilities_remain_unchanged_but_margin_can_separate_them():
    rows = {"false": margin_record("false", 18.), "true": margin_record("true", 25.)}
    gold = {"false": set(), "true": {("seen", "negative")}}
    legacy = select_threshold(rows, gold, ASPECTS, "heldout", .5, .5)
    selected = select_threshold(rows, gold, ASPECTS, "heldout", .5, .5, presence_score_space="logit_margin")
    assert legacy["seen_pair_f1"] == pytest.approx(2/3)
    assert selected["seen_pair_f1"] == 1
    assert selected["aspect_threshold"] == .5
    assert 18 < active_presence_threshold(selected) <= 25
    assert rows["false"]["response"]["aspect_scores"][0]["score"] == 1.
    changed = copy.deepcopy(rows)
    for row in changed.values():
        row["response"]["aspect_scores"][1]["presence_logit_margin"] = 999999.
    changed_gold = {key: pairs | {("heldout", "positive")} for key, pairs in gold.items()}
    assert selected == select_threshold(changed, changed_gold, ASPECTS, "heldout", .5, .5,
                                        presence_score_space="logit_margin")


@pytest.mark.parametrize("old", [-.1, 0., .5, 1., 1.1])
def test_margin_grid_budget_finite_endpoints_and_old_probability_boundaries(old):
    values = np.asarray([-32., 0., 33.], dtype=np.float32)
    grid = margin_threshold_candidates(values, old)
    assert len(grid) <= 102 and all(math.isfinite(value) for value in grid)
    assert grid[0] < -32 and grid[-1] > 33
    rows = {"a": margin_record("a", 33., -32.)}
    assert not decode_records(rows, ["a"], ASPECTS, grid[-1], .5,
                              presence_score_space="logit_margin")["a"]
    assert len(decode_records(rows, ["a"], ASPECTS, grid[0], .5,
                              presence_score_space="logit_margin")["a"]) == 2
    assert len(margin_threshold_candidates([30., 30.], old)) <= 4
    json.dumps(grid, allow_nan=False)


def test_margin_input_is_explicit_and_never_silently_falls_back_to_probability():
    row = record("a")
    with pytest.raises(ValueError, match="presence_logit_margin"):
        score_maps(row, ASPECTS, presence_score_space="logit_margin")
    bad = margin_record("a", float("nan"))
    with pytest.raises(ValueError, match="finite"):
        score_maps(bad, ASPECTS, presence_score_space="logit_margin")
    with pytest.raises(ValueError, match="Mixed"):
        validate_record_spaces({"a": row, "b": margin_record("b")}, "logit_margin")
    with pytest.raises(ValueError, match="presence_margin_threshold"):
        active_presence_threshold({"presence_score_space": "logit_margin", "aspect_threshold": 1.})


def test_routing_margin_distance_is_not_probability_clipped_and_policy_space_is_bound():
    rows = {"a": margin_record("a", 30.25)}
    features = risk_features(rows, ["a"], ASPECTS, 30.125, .5, presence_score_space="logit_margin")
    assert features["a"]["margin"] == pytest.approx(.125)
    assert not features["a"]["mandatory"]
    with pytest.raises(ValueError, match="different presence"):
        assign_review(features, {"min_margin": 0, "presence_score_space": "probability"})


def margin_analysis_fixture(tmp_path):
    args = fixture(tmp_path)
    args.presence_score_space = "logit_margin"
    for directory, labels, prefix, size in ((args.development_root, args.development_labels, "dev", 100),
                                           (args.evaluation_root, args.evaluation_labels, "eval", 200)):
        for method in ("fixed", "bm25"):
            jsonl(directory / f"{method}.jsonl", [margin_record(f"{prefix}{i}", 30 if i%2 else 18) for i in range(size)])
        jsonl(labels, [{"request_id": f"{prefix}{i}", "pairs": [{"aspect": "seen", "sentiment": "negative"}]
                      if i%2 else []} for i in range(size)])
    return args


def test_analysis_keeps_matched_fixed_and_same_forward_probability_recalibration(tmp_path):
    args = margin_analysis_fixture(tmp_path)
    result = analysis.run(args)
    selection = json.loads((args.output / "selection.json").read_text())
    assert result["presence_score_space"] == "logit_margin"
    assert result["methods"]["fixed"]["scopes"]["overall"]["pair"]["micro_f1"] == 1
    assert result["contrasts"]["bm25_vs_fixed"]["scopes"]["overall"]["pair_f1_difference_points"] == 0
    legacy = result["legacy_probability_recalibration"]
    assert legacy["methods"]["fixed"]["scopes"]["overall"]["pair"]["micro_f1"] == pytest.approx(2/3)
    assert legacy["postprocessing_margin_vs_probability"]["fixed"]["scopes"]["overall"]["pair_f1_difference_points"] > 30
    assert selection["legacy_probability_recalibration"]["thresholds"]["fixed"]["aspect_threshold"] <= 1
    assert selection["thresholds"]["fixed"]["presence_margin_threshold"] > 18


def test_mixed_analysis_space_is_rejected_before_selection_is_written(tmp_path):
    args = margin_analysis_fixture(tmp_path)
    args.classification_only = True
    jsonl(args.development_root / "bm25.jsonl", [record(f"dev{i}") for i in range(100)])
    with pytest.raises(ValueError, match="Mixed"):
        analysis.run(args)
    assert not (args.output / "selection.json").exists()


def test_seen_badcase_selector_matches_margin_selection_after_heldout_strip():
    rows = {"a": margin_record("a", 18.), "b": margin_record("b", 25.)}
    gold = {"a": set(), "b": {("seen", "negative")}}
    expected = select_threshold(rows, gold, ASPECTS, "heldout", .5, .5, presence_score_space="logit_margin")
    rows["a"]["response"]["aspect_scores"][1]["presence_logit_margin"] = float("nan")
    stripped = {key: badcases.strip_nonseen(row, {"seen"}) for key, row in rows.items()}
    actual = badcases.calibrate_seen(stripped, gold, ["seen"], .5, .5, presence_score_space="logit_margin")
    assert active_presence_threshold(actual) == active_presence_threshold(expected)
    assert actual["seen_pair_f1"] == expected["seen_pair_f1"]


def test_service_margin_clone_keeps_probability_and_fixed_fallback_thresholds(tmp_path):
    args = make_args(tmp_path)
    selection = service.read_json(args.selection)
    selection["presence_score_space"] = "logit_margin"
    for method, threshold in (("fixed", 24.), ("bm25", 26.)):
        selection["thresholds"][method].update(presence_score_space="logit_margin", presence_margin_threshold=threshold)
        jsonl(args.development_score_root / f"{method}.jsonl", [margin_record(f"dev{i}", 30.) for i in range(30)])
    args.selection.write_text(json.dumps(selection))
    plan = service.prepare_plan(args)
    assert plan["presence_score_space"] == "logit_margin"
    clone = service.read_json(args.output / "inputs/artifacts_bm25.json")
    assert clone["variants"]["hybrid"]["aspect_threshold"] == .9
    assert clone["variants"]["hybrid"]["presence_margin_threshold"] == 26
    assert clone["retrieval_fallback_policy"]["presence_margin_threshold"] == 24
    assert clone["retrieval_fallback_policy"]["presence_score_space"] == "logit_margin"


def test_api_correctness_uses_margin_despite_identical_probabilities():
    reference = margin_record("a", 30.)
    response = api_response(reference, presence=1.)
    response["presence_score_space"] = "logit_margin"
    response["aspect_scores"][0]["presence_logit_margin"] = 30.
    response["aspect_scores"][1]["presence_logit_margin"] = -10.
    thresholds = {"aspect_threshold": .5, "runner_up_threshold": .5,
                  "presence_score_space": "logit_margin", "presence_margin_threshold": 25.}
    match = service.compare_api_response(reference, response, ASPECTS, thresholds)
    assert match["exact_label_set_match"] and match["presence_logit_margin_max_abs_delta"] == 0
    response["aspect_scores"][0]["presence_logit_margin"] = 24.
    response["predictions"] = []
    mismatch = service.compare_api_response(reference, response, ASPECTS, thresholds)
    assert not mismatch["exact_label_set_match"]
    assert mismatch["presence_probability_max_abs_delta"] == 0
    assert mismatch["presence_logit_margin_max_abs_delta"] == 6
