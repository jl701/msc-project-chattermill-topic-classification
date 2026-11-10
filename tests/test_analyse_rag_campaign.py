import json
from pathlib import Path

import pytest

import analyse_rag_campaign as analysis
from test_rag_evaluation import ASPECTS, record
from msc_project.serving.annotation_workflow import (
    compare_blind, load_reviewed_split, prepare_blind, read_json, release_labels, sha, write_json,
)
from msc_project.serving.review_data import write_jsonl
from serve_review_adjudication import AdjudicationStore


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_registered_method_list_excludes_telemetry_journals(tmp_path):
    dump(tmp_path / "identity.json", {"methods": ["fixed", "e5"]})
    for name in ("fixed", "e5", "telemetry", "_sentiment_anchor"):
        (tmp_path / f"{name}.jsonl").write_text("{}\n", encoding="utf-8")
    assert analysis.method_paths([], tmp_path) == {
        "fixed": tmp_path / "fixed.jsonl", "e5": tmp_path / "e5.jsonl"}
    dump(tmp_path / "identity.json", {"methods": ["../unregistered"]})
    with pytest.raises(ValueError, match="registered method"):
        analysis.method_paths([], tmp_path)


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def fixture(tmp_path):
    artifacts = tmp_path / "artifacts.json"
    dump(artifacts, {"taxonomy": {"candidates": [{"aspect": a} for a in ASPECTS],
        "heldout_aspects": ["heldout"]}, "variants": {"hybrid": {
            "aspect_threshold": .5, "runner_up_threshold": .5}}})
    dev = tmp_path / "dev"
    evaluation = tmp_path / "evaluation"
    dev.mkdir()
    evaluation.mkdir()
    for directory, prefix, size in ((dev, "dev", 100), (evaluation, "eval", 200)):
        for method in ("fixed", "bm25"):
            jsonl(directory / f"{method}.jsonl", [record(f"{prefix}{i}") for i in range(size)])
        jsonl(tmp_path / f"{prefix}_labels.jsonl", [{"request_id": f"{prefix}{i}",
            "pairs": [{"aspect": "seen", "sentiment": "negative"}]} for i in range(size)])
    args = analysis.parser().parse_args([
        "--artifacts", str(artifacts), "--development-root", str(dev),
        "--evaluation-root", str(evaluation),
        "--development-labels", str(tmp_path / "dev_labels.jsonl"),
        "--evaluation-labels", str(tmp_path / "eval_labels.jsonl"),
        "--output", str(tmp_path / "out"), "--bootstrap-draws", "20"])
    return args


def test_end_to_end_selection_quality_routing_and_immutable_outputs(tmp_path):
    args = fixture(tmp_path)
    result = analysis.run(args)
    assert result["selected_method"] == "fixed"
    assert result["methods"]["fixed"]["scopes"]["overall"]["pair"]["micro_f1"] == 1
    routing = json.loads((args.output / "review_routing.json").read_text())
    assert routing["evaluation"]["fixed__confidence"]["qualified_review_item_reduction_percent"] == 95
    assert "request_id" not in (args.output / "quality.json").read_text()
    analysis.run(args)  # an identical rerun does not overwrite different evidence
    with pytest.raises(FileExistsError):
        analysis.write_json(args.output / "quality.json", {"different": True})


def test_selection_only_does_not_open_final_labels(tmp_path):
    args = fixture(tmp_path)
    args.selection_only = True
    args.evaluation_labels.unlink()
    result = analysis.run(args)
    assert result["selected_method"] == "fixed"
    assert (args.output / "review_routing_selection.json").is_file()
    assert not (args.output / "quality.json").exists()


def test_population_overlap_rejected(tmp_path):
    args = fixture(tmp_path)
    args.evaluation_labels = args.development_labels
    with pytest.raises(ValueError, match="overlap"):
        analysis.run(args)


def test_classification_only_never_accesses_routing_or_final_population(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    args.classification_only = True
    args.evaluation_root = None
    args.evaluation_labels.unlink()
    for name in ("risk_features", "select_policy", "comparator_predictions", "comparator_records"):
        monkeypatch.setattr(analysis, name, lambda *a, **kw: pytest.fail("Routing was accessed"))
    result = analysis.run(args)
    assert result["selection_outcomes"] == "seen_aspects_only"
    assert {path.name for path in args.output.iterdir()} == {"selection.json"}


def test_auxiliary_anchor_is_not_a_method(tmp_path):
    args = fixture(tmp_path)
    jsonl(args.development_root / "_sentiment_anchor.jsonl", [{"not": "a method"}])
    assert set(analysis.method_paths([], args.development_root)) == {"fixed", "bm25"}


def test_comparator_uses_http_sample_identity_and_first_success(tmp_path):
    path = tmp_path / "http.jsonl"
    base = {"sample_id": "original-review", "request_id": "http-op", "success": True,
            "response": {"predictions": [{"aspect": "seen", "sentiments": ["negative"]}]}}
    later = dict(base, started_s=2, response={"predictions": []})
    jsonl(path, [later, dict(base, started_s=1)])
    assert analysis.comparator_predictions(path) == {"original-review": {("seen", "negative")}}


def test_comparator_does_not_hide_first_attempt_failure_with_a_successful_retry(tmp_path):
    path = tmp_path / "http.jsonl"
    failed = {"sample_id": "a", "request_id": "operation1", "success": False,
              "response": {}, "started_s": 1, "latency_ms": 120000}
    retried = {"sample_id": "a", "request_id": "operation2", "success": True,
               "response": {"predictions": []}, "started_s": 2, "latency_ms": 1000}
    jsonl(path, [retried, failed])
    assert analysis.comparator_predictions(path) == {}
    assert analysis.comparator_records(path)["a"]["latency_ms"] == 120000


def test_end_to_end_boundary_manifest_and_cost_are_separate_from_quality(tmp_path):
    args = fixture(tmp_path)
    for attribute, prefix, size in (("comparator_development", "dev", 100),
                                     ("comparator_evaluation", "eval", 200)):
        path = tmp_path / f"{prefix}_comparator.jsonl"
        jsonl(path, [{"request_id": f"{prefix}{i}", "success": True, "latency_ms": 10,
            "response": {"predictions": [{"aspect": "seen", "sentiments": ["negative"]}],
                         "diagnostics": {"sentiment_prompt_count": 1}}} for i in range(size)])
        setattr(args, attribute, path)
    analysis.run(args)
    routing = json.loads((args.output / "review_routing.json").read_text())
    result = routing["evaluation"]["fixed__confidence_disagreement"]
    manifest = json.loads((args.output / "review_routing_call_manifest_evaluation.json").read_text())
    calls = sum(row["comparator_call_required"] for row in manifest["policies"]["fixed__confidence_disagreement"])
    assert 0 < calls < 200
    assert result["comparator_cost"]["additional_comparator_review_requests"] == calls
    assert not result["comparator_cost"]["combined_service_latency_measured"]
    assert "request_id" not in (args.output / "quality.json").read_text()


def reviewed_fixture(tmp_path, *, disagreement=True):
    """A small registered synthetic snapshot, never a real annotation store."""
    args = fixture(tmp_path)
    args.scope = "new_source_locked"
    data, queue, comparison, release = (tmp_path / name for name in
                                        ("annotation", "blind", "comparison", "release"))
    data.mkdir()
    taxonomy = [{"aspect": aspect, "definition": f"Definition {aspect}",
                 "zh_name": aspect, "zh_description": aspect} for aspect in ASPECTS]
    artifacts = read_json(args.artifacts)
    artifacts["taxonomy"]["candidates"] = [{"aspect": row["aspect"], "text": row["definition"]}
                                            for row in taxonomy]
    write_json(args.artifacts, artifacts)
    rows = []
    for split, prefix, size, directory in (("adaptation", "adapt", 4, None),
            ("calibration", "dev", 12, args.development_root),
            ("locked", "eval", 24, args.evaluation_root)):
        rows.extend({"request_id": f"{prefix}{i}", "text": f"Synthetic English {prefix} review {i}",
                     "app": "Synthetic", "split": split, "pilot": split == "adaptation"}
                    for i in range(size))
        if directory is not None:
            for method in ("fixed", "bm25"):
                jsonl(directory / f"{method}.jsonl", [record(f"{prefix}{i}") for i in range(size)])
    write_jsonl(data / "annotation_reviews.jsonl", rows)
    write_json(data / "annotation_taxonomy.json", taxonomy)
    write_json(data / "annotation_manifest.json", {
        "annotation_reviews_sha256": sha(data / "annotation_reviews.jsonl"),
        "assignment": {"counts": {"adaptation": 4, "calibration": 12, "locked": 24}},
        "labels_present": False})
    first = {row["request_id"]: {"request_id": row["request_id"], "split": row["split"],
        "source": "human_annotation", "revision": 1, "saved_at": "2026-11-11T00:00:00+00:00",
        "reviewer": "Synthetic First Human", "status": "confirmed", "labels": [["seen", "negative"]],
        "english_confirmed": True, "no_defined_theme": False, "ambiguous": False,
        "out_of_scope": False, "incomplete_text": False, "note": "Synthetic annotation"} for row in rows}
    (data / "human_labels").mkdir()
    write_json(data / "human_labels/annotations.json", first)
    write_jsonl(data / "human_labels/annotation_history.jsonl", [
        {"previous": None, "current": annotation} for annotation in first.values()])
    prepare_blind(data, queue)
    second = {identity: dict(value, reviewer="Synthetic Second Human",
                            saved_at="2026-11-12T01:00:00+00:00")
              for identity, value in read_json(queue / "first_round_private.json").items()}
    if disagreement:
        for value in second.values():
            value["labels"] = [["seen", "positive"]]
    write_json(queue / "human_labels/annotations.json", second)
    compare_blind(queue, comparison)
    store = AdjudicationStore(comparison / "adjudication")
    for identity in store.order:
        store.save({"request_id": identity, "expected_revision": 0,
            "reviewer": "Synthetic Adjudicator", "status": "confirmed", "labels": [["seen", "negative"]],
            "english_confirmed": True, "no_defined_theme": False, "ambiguous": False,
            "out_of_scope": False, "incomplete_text": False, "note": "Synthetic human final reason"})
    release_labels(data, comparison, store.path if store.order else comparison / "decisions_template.json", release)
    args.annotation_data, args.reviewed_release = data, release
    args.development_labels = release / "labels_calibration.jsonl"
    args.evaluation_labels = release / "labels_locked.jsonl"
    return args


def update_split_hash(release, split, name):
    path = release / "release_manifest.json"
    manifest = read_json(path)
    manifest["split_artifacts"][split][name] = sha(release / name)
    write_json(path, manifest)


def test_new_source_requires_reviewed_provenance_not_only_scope_flag(tmp_path):
    args = fixture(tmp_path)
    args.scope = "new_source_locked"
    with pytest.raises(ValueError, match="requires --annotation-data"):
        analysis.run(args)
    assert not (args.output / "selection.json").exists()


def test_reviewed_new_source_full_population_and_missing_failed_predictions(tmp_path):
    args = reviewed_fixture(tmp_path)
    path = args.evaluation_root / "fixed.jsonl"
    values = analysis.read_jsonl(path)
    values[0] = {"request_id": values[0]["request_id"], "status": "error", "elapsed_ms": 2}
    jsonl(path, values[:-1])
    result = analysis.run(args)
    fixed = result["methods"]["fixed"]
    assert fixed["review_count"] == 24
    assert fixed["operational"]["successful_reviews"] == 22
    assert fixed["operational"]["explicit_failed_reviews"] == 1
    assert fixed["operational"]["missing_record_reviews"] == 1
    assert fixed["scopes"]["seen"]["pair"]["fn"] == 2
    assert result["reviewed_evaluation_provenance"]["registered_count"] == 24
    assert result["interpretation"] == "reviewed_frozen_new_source_locked_population"
    assert "request_id" not in (args.output / "quality.json").read_text()


@pytest.mark.parametrize("mode", ["classification_only", "selection_only"])
def test_new_source_selection_does_not_open_locked_gold_or_history(tmp_path, monkeypatch, mode):
    args = reviewed_fixture(tmp_path)
    setattr(args, mode, True)
    original_text, original_bytes = Path.read_text, Path.read_bytes

    def check(path):
        if path.parent == args.reviewed_release and "_locked." in path.name:
            pytest.fail(f"Locked gold/history was accessed before freeze: {path.name}")

    def text(path, *a, **kw):
        check(path)
        return original_text(path, *a, **kw)

    def byte_data(path, *a, **kw):
        check(path)
        return original_bytes(path, *a, **kw)

    monkeypatch.setattr(Path, "read_text", text)
    monkeypatch.setattr(Path, "read_bytes", byte_data)
    result = analysis.run(args)
    assert result["reviewed_calibration_provenance"]["registered_count"] == 12


def test_gold_and_prediction_dropped_together_cannot_shrink_new_source_population(tmp_path):
    args = reviewed_fixture(tmp_path)
    labels = analysis.read_jsonl(args.evaluation_labels)
    removed = labels[0]["request_id"]
    jsonl(args.evaluation_labels, labels[1:])
    # Even recomputing this one file hash does not bypass the original ID population.
    update_split_hash(args.reviewed_release, "locked", "labels_locked.jsonl")
    for path in args.evaluation_root.glob("*.jsonl"):
        jsonl(path, [row for row in analysis.read_jsonl(path) if row["request_id"] != removed])
    with pytest.raises(ValueError, match="Gold/replay coverage"):
        analysis.run(args)
    assert not (args.output / "quality.json").exists()


@pytest.mark.parametrize("part", ["taxonomy", "snapshot", "review_metadata", "history", "correction", "legacy"])
def test_new_source_rejects_changed_source_or_review_history(tmp_path, part):
    args = reviewed_fixture(tmp_path)
    if part == "taxonomy":
        artifacts = read_json(args.artifacts)
        artifacts["taxonomy"]["candidates"][0]["text"] = "Different rubric"
        write_json(args.artifacts, artifacts)
    elif part == "snapshot":
        path = args.annotation_data / "annotation_manifest.json"
        value = read_json(path)
        value["new_version"] = True
        write_json(path, value)
    elif part == "review_metadata":
        path = args.reviewed_release / "review_comparison.json"
        value = read_json(path)
        value["agreement_count"] += 1
        write_json(path, value)
    elif part == "history":
        path = args.reviewed_release / "primary_history_calibration.jsonl"
        values = analysis.read_jsonl(path)
        values[0]["previous"] = {"revision": 0}
        jsonl(path, values)
        update_split_hash(args.reviewed_release, "calibration", path.name)
    elif part == "correction":
        path = args.reviewed_release / "corrections_calibration.jsonl"
        values = analysis.read_jsonl(path)
        assert values
        values[0]["resolved"]["labels"] = [["seen", "positive"]]
        jsonl(path, values)
        update_split_hash(args.reviewed_release, "calibration", path.name)
    else:
        path = args.reviewed_release / "release_manifest.json"
        value = read_json(path)
        value.pop("integrity_schema")
        write_json(path, value)
    with pytest.raises(ValueError):
        analysis.run(args)
    assert not (args.output / "quality.json").exists()


def test_new_source_uses_frozen_history_not_mutable_live_annotations(tmp_path):
    args = reviewed_fixture(tmp_path)
    for name in ("annotations.json", "annotation_history.jsonl"):
        (args.annotation_data / "human_labels" / name).write_text("not valid JSON", encoding="utf-8")
    result = analysis.run(args)
    assert result["methods"]["fixed"]["review_count"] == 24


def test_new_source_wrong_gold_split_rejected(tmp_path):
    args = reviewed_fixture(tmp_path)
    args.development_labels = args.evaluation_labels
    with pytest.raises(ValueError, match="Provided calibration gold file differs"):
        analysis.run(args)


def test_reusable_loader_limits_adaptation_release_to_reviewed_scope(tmp_path):
    args = reviewed_fixture(tmp_path, disagreement=False)
    verified = load_reviewed_split(args.annotation_data, args.reviewed_release, "adaptation")
    assert verified["ids"] == {f"adapt{i}" for i in range(4)}
    assert verified["registered_count"] == verified["reviewed_count"] == 4
