import json

import pytest

import analyse_rag_campaign as analysis
from test_rag_evaluation import ASPECTS, record


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


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
    for name in ("risk_features", "select_policy", "comparator_predictions"):
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
