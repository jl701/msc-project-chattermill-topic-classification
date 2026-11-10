import copy
import json

import pytest

import analyse_rag_seen_badcases as badcases
from msc_project.serving.rag_evaluation import select_threshold
from test_rag_evaluation import ASPECTS, record


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def jsonl(path, rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding="utf-8")


def args_fixture(tmp_path):
    artifacts = tmp_path / "artifacts.json"
    dump(artifacts, {"taxonomy": {"candidates": [{"aspect": a} for a in ASPECTS],
        "heldout_aspects": ["heldout"]}, "variants": {"hybrid": {"aspect_threshold": .5, "runner_up_threshold": .5}}})
    labels = tmp_path / "development_seen_labels.jsonl"
    jsonl(labels, [{"request_id": str(i), "pairs": [] if i % 3 == 0 else
                   [{"aspect": "seen", "sentiment": "negative"}]} for i in range(20)])
    replay = tmp_path / "replay.jsonl"
    jsonl(replay, [{"request_id": str(i), "text": "Synthetic review " + str(i)} for i in range(20)])
    root = tmp_path / "development"
    root.mkdir()
    jsonl(root / "fixed.jsonl", [record(str(i), seen=.8 + i*.001) for i in range(20)])
    dump(root / "identity.json", {"methods": ["fixed", "bm25"], "request_ids": [str(i) for i in range(20)],
        "replay": badcases.sha256(replay), "artifacts": badcases.sha256(artifacts)})
    dump(root / "summary.json", {"methods": {"fixed": {"file_sha256": badcases.sha256(root / "fixed.jsonl")}}})
    dump(root / "state.json", {"status": "running"})
    args = badcases.parser().parse_args(["--artifacts", str(artifacts), "--seen-labels", str(labels),
        "--replay", str(replay), "--development-root", str(root), "--method", "fixed",
        "--output", str(tmp_path / "private")])
    args.synthetic_test = True
    return args


def test_partial_fixed_profile_never_selects_an_improvement(tmp_path):
    args = args_fixture(tmp_path)
    result = badcases.run(args)
    assert not result["matrix_complete"] and not result["improvement_selected"]
    assert result["profiled_methods"] == ["fixed"]
    assert not result["heldout_predictions_and_labels_used"]
    assert "request_id" not in (args.output / "summary.json").read_text()
    assert (args.output / "cases_fixed.md").is_file()
    badcases.run(args)


def test_full_matrix_waits_for_registered_completion(tmp_path):
    args = args_fixture(tmp_path)
    args.method = None
    with pytest.raises(RuntimeError, match="Wait for all"):
        badcases.run(args)
    (args.development_root / "bm25.jsonl").write_bytes((args.development_root / "fixed.jsonl").read_bytes())
    dump(args.development_root / "summary.json", {"methods": {method: {
        "file_sha256": badcases.sha256(args.development_root / f"{method}.jsonl")} for method in ("fixed", "bm25")}})
    dump(args.development_root / "state.json", {"status": "complete"})
    result = badcases.run(args)
    assert result["matrix_complete"]
    assert result["methods"]["bm25"]["versus_fixed"]["seen_pair_f1_difference_points"] == 0


def test_unsanitized_gold_path_is_rejected_before_any_label_read(tmp_path, monkeypatch):
    args = args_fixture(tmp_path)
    args.seen_labels = tmp_path / "labels_development.jsonl"
    monkeypatch.setattr(badcases, "read_jsonl", lambda *a: pytest.fail("No labels should be read"))
    with pytest.raises(ValueError, match="sanitized"):
        badcases.run(args)


def test_heldout_gold_rejected_even_in_misnamed_sanitized_view(tmp_path):
    args = args_fixture(tmp_path)
    jsonl(args.seen_labels, [{"request_id": "0", "pairs": [{"aspect": "heldout", "sentiment": "negative"}]}])
    with pytest.raises(ValueError, match="Unknown aspect"):
        badcases.run(args)


def test_heldout_predictions_scores_and_truncation_are_dropped_before_profile():
    row = record("a")
    poisoned = copy.deepcopy(row)
    poisoned["response"]["aspect_scores"][1]["score"] = float("nan")
    poisoned["response"]["full_grid_scores"][1]["sentiment_scores"] = {"invalid": "poison"}
    poisoned["response"]["diagnostics"] = {"severe_truncation": True,
        "batch_aspect": {"retrieval": [{"candidate_aspect": "heldout", "truncation": {"severe_query_truncation": True}}]}}
    poisoned["response"]["sentiment_truncation_by_aspect"] = {"heldout": {"left_truncated": True}}
    assert badcases.strip_nonseen(row, {"seen"}) == badcases.strip_nonseen(poisoned, {"seen"})


def test_seen_only_calibrator_matches_registered_selection_without_heldout_scores():
    rows = {str(i): record(str(i), seen=.1 + i*.08) for i in range(10)}
    gold = {str(i): {("seen", "negative")} if i > 4 else set() for i in range(10)}
    expected = select_threshold(rows, gold, ASPECTS, "heldout", .5, .5)
    stripped = {key: badcases.strip_nonseen(row, {"seen"}) for key, row in rows.items()}
    actual = badcases.calibrate_seen(stripped, gold, ["seen"], .5, .5)
    for key in ("aspect_threshold", "seen_pair_f1", "seen_presence_f1", "seen_aspect_calls", "candidate_count"):
        assert actual[key] == expected[key]


def test_demo_alignment_duplicate_identity_and_error_denominators():
    row = badcases.strip_nonseen(record("a"), {"seen"})
    row["response"]["retrieval"] = [{"candidate_aspect": "seen", "selected_answers": list("YYNN"),
        "selected_ids": ["one|seen", "two|seen", "one|seen", "three|seen"],
        "truncation": {"original_prompt_tokens": 1200, "final_prompt_tokens": 1024,
            "query_characters_removed": 0, "demo_characters_removed": [2, 0, 0, 0]}}]
    result = badcases.demo_profile({"a": row}, {"a": {("seen", "negative")}}, {"a": set()})
    assert result["counts"]["prompts_with_repeated_review"] == 1
    assert result["counts"]["aligned_positive"] == 2
    assert result["counts"]["demo_trimmed_prompts"] == 1
    assert result["alignment_error_buckets"]["same_theme_positive2_negative2"]["false_positive_rate"] == 1


def test_in_progress_or_modified_score_file_cannot_be_profiled(tmp_path):
    args = args_fixture(tmp_path)
    with (args.development_root / "fixed.jsonl").open("a") as stream:
        stream.write('{}\n')
    with pytest.raises(ValueError, match="registered completed"):
        badcases.run(args)


def test_private_output_guard_runs_before_label_loading(tmp_path, monkeypatch):
    args = args_fixture(tmp_path)
    args.synthetic_test = False
    monkeypatch.setattr(badcases, "read_jsonl", lambda *a: pytest.fail("No labels should be read"))
    with pytest.raises(ValueError, match="Private diagnostics"):
        badcases.run(args)


def test_conflict_exposure_distinguishes_ambiguous_demo_labels_from_other_group_conflicts():
    rows = {}
    for key, uid in (("a", "ambiguous-uid"), ("b", "other-conflict-uid"), ("c", "clean-uid")):
        row = badcases.strip_nonseen(record(key), {"seen"})
        row["response"]["retrieval"] = [{"candidate_aspect": "seen", "selected_ids": [f"{uid}|seen"]}]
        rows[key] = row
    groups = [{"retained_rows": ["ambiguous-uid"], "seen_aspect_sets": [[], ["seen"]]},
              {"retained_rows": ["other-conflict-uid"], "seen_aspect_sets": [["seen"], ["seen", "other"]]}]
    predictions = {key: {("seen", "negative")} for key in rows}
    gold = {key: set() if key == "a" else pairs for key, pairs in predictions.items()}
    summary, private = badcases.training_conflict_profile(rows, predictions, gold, groups, ["seen", "other"])
    assert summary["unique_conflict_reviews_selected"] == 2
    assert summary["prompt_cohorts"]["ambiguous_demo_label"]["presence_errors"] == 1
    assert summary["prompt_cohorts"]["conflict_group_only"]["presence_errors"] == 0
    assert summary["prompt_cohorts"]["unexposed"]["prompts"] == 1
    assert private["retained_review_exposures"]["other-conflict-uid"]["ambiguous_demo_label_prompts"] == 0
    assert "ambiguous-uid" not in json.dumps(summary)
    groups[0]["seen_aspect_sets"].append(["heldout"])
    with pytest.raises(ValueError, match="nonseen"):
        badcases.training_conflict_profile(rows, predictions, gold, groups, ["seen", "other"])
