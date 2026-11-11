"""Synthetic eligibility, source integrity and immutable-release regressions."""
from dataclasses import replace
import json

import pytest

from adaptation_policy_fixture import synthetic_policy
from test_annotation_workflow import dataset as source_fixture, make_release
from msc_project.serving import adaptation_use
from msc_project.serving.adaptation_use import (
    PURPOSES, export_adaptation_use, load_adaptation_use, validate_policy,
)
from msc_project.serving.annotation_workflow import (
    export_feedback, load_reviewed_split, read_json, read_lines, sha, write_json,
)
from msc_project.serving.feedback_index import verified_feedback
from msc_project.serving.review_data import write_jsonl


@pytest.fixture
def population(tmp_path):
    data = source_fixture.__wrapped__(tmp_path)
    rows = read_lines(data / "annotation_reviews.jsonl")
    texts = ["Cannot log in!", "!!!", "???", "…", "An independent message",
             "Cannot log in?", "☆", "♫", "One protected input", "Another protected input"]
    for row, text in zip(rows, texts):
        row["text"] = text
    write_jsonl(data / "annotation_reviews.jsonl", rows)
    manifest = read_json(data / "annotation_manifest.json")
    manifest["annotation_reviews_sha256"] = sha(data / "annotation_reviews.jsonl")
    write_json(data / "annotation_manifest.json", manifest)
    policy = synthetic_policy(data)
    _, _, release = make_release(data, tmp_path)
    return data, release, policy


@pytest.mark.parametrize("purpose", PURPOSES)
def test_policy_filters_use_not_annotation_release(population, purpose):
    data, release, policy = population
    before = {p: sha(p) for folder in (data, release) for p in folder.rglob("*") if p.is_file()}
    view = load_adaptation_use(data, release, purpose=purpose, policy=policy)
    assert view["ids"] == {"r1", "r2", "r3", "r4"}
    assert set(view["rows"]) == set(view["annotations"]) == view["ids"]
    assert {r["request_id"] for r in view["labels"]} == view["ids"]
    assert view["use_audit"]["excluded_adaptation_count"] == 1
    assert view["use_audit"]["reviewed_adaptation_count"] == 5
    assert view["use_audit"]["eligible_count"] == 4
    assert "r0" not in json.dumps(view["use_audit"])
    for split, expected in (("adaptation", 5), ("calibration", 2), ("locked", 3)):
        assert load_reviewed_split(data, release, split)["reviewed_count"] == expected
    assert {p: sha(p) for p in before} == before
    # All five distinct punctuation/symbol inputs remain in the full release.
    retained = {r["text"] for r in read_lines(data / "annotation_reviews.jsonl")
                if not adaptation_use.retrieval.normalise_text(r["text"])}
    assert retained == {"!!!", "???", "…", "☆", "♫"}


def rewrite_policy(policy, mutate):
    value = read_json(policy.path)
    mutate(value)
    write_json(policy.path, value)
    return replace(policy, expected_sha256=sha(policy.path))


@pytest.mark.parametrize("mutation,reason", [
    (lambda p: p["excluded_adaptation_records"][0].update(request_id="unknown"), "ID or split"),
    (lambda p: p["excluded_adaptation_records"][0].update(request_id="r5"), "ID or split"),
    (lambda p: p["excluded_adaptation_records"][0].update(original_split="calibration"), "ID or split"),
    (lambda p: p["excluded_adaptation_records"][0].update(request_id="r1"), "protected match"),
    (lambda p: p["excluded_adaptation_records"][0].update(blocked_uses=[PURPOSES[0]]), "blocked purposes"),
    (lambda p: p["excluded_adaptation_records"][0].update(protected_split_match_counts={"calibration": 2, "locked": 0}), "protected match"),
    (lambda p: p.update(excluded_adaptation_records=[], excluded_record_count=0), "cover every"),
    (lambda p: p["empty_normal_forms"].update(merge_on_empty_form=True), "empty-normal"),
    (lambda p: p["registered_population_counts"].update(adaptation=4), "population"),
])
def test_wrong_id_split_purpose_or_incomplete_exclusion_rejected_before_gold(population, monkeypatch, mutation, reason):
    data, release, policy = population
    policy = rewrite_policy(policy, mutation)
    def no_gold(*args, **kwargs):
        pytest.fail("Policy failure must precede reviewed-gold reads")
    monkeypatch.setattr(adaptation_use, "load_reviewed_split", no_gold)
    with pytest.raises(ValueError, match=reason):
        load_adaptation_use(data, release, purpose=PURPOSES[0], policy=policy)


def test_unknown_purpose_and_missing_policy_are_fail_closed(population, monkeypatch):
    data, release, policy = population
    monkeypatch.setattr(adaptation_use, "load_reviewed_split", lambda *a, **k: pytest.fail("Gold opened"))
    with pytest.raises(ValueError, match="Unknown adaptation-use purpose"):
        load_adaptation_use(data, release, purpose="evaluation", policy=policy)
    with pytest.raises(ValueError, match="Explicit"):
        load_adaptation_use(data, release, purpose=PURPOSES[0], policy=None)
    with pytest.raises(TypeError, match="policy"):
        export_feedback(data, release, data.parent / "feedback")


@pytest.mark.parametrize("name", ["annotation_reviews.jsonl", "annotation_manifest.json",
                                  "replay_new_inputs.jsonl", "replay_new_inputs_manifest.json"])
def test_source_and_assignment_hash_changes_rejected(population, name):
    data, _, policy = population
    path = data / name
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source hash mismatch"):
        validate_policy(data, policy, purpose=PURPOSES[0])


def test_policy_hash_and_registered_policy_substitution_rejected(population, tmp_path, monkeypatch):
    data, _, policy = population
    with pytest.raises(ValueError, match="policy hash mismatch"):
        validate_policy(data, replace(policy, expected_sha256="0" * 64), purpose=PURPOSES[0])
    registration = tmp_path / "registration.json"
    write_json(registration, {"policies_by_source_sha256": {sha(data / "annotation_reviews.jsonl"): policy.expected_sha256}})
    monkeypatch.setattr(adaptation_use, "REGISTRATION", registration)
    modified = rewrite_policy(policy, lambda p: p.update(untrusted_extra=True))
    with pytest.raises(ValueError, match="registered campaign"):
        validate_policy(data, modified, purpose=PURPOSES[0])


def test_bound_normalisation_code_cannot_be_substituted(population):
    data, _, policy = population
    code = policy.source_root / "retrieval.py"
    code.write_bytes(code.read_bytes() + b"# synthetic change\n")
    modified = rewrite_policy(policy, lambda p: p["source_sha256"].update({"retrieval.py": sha(code)}))
    with pytest.raises(ValueError, match="Active retrieval normalisation"):
        validate_policy(data, modified, purpose=PURPOSES[0])


@pytest.mark.parametrize("entry", ["export-feedback", "append"])
def test_cli_requires_explicit_policy_flags(entry, monkeypatch, capsys):
    if entry == "export-feedback":
        from review_annotation_workflow import main
        monkeypatch.setattr("sys.argv", ["review_annotation_workflow.py", entry,
            "--data", "unused", "--release", "unused", "--output", "unused"])
        run = main
    else:
        from prepare_rag_feedback_index import main
        args = [entry]
        for name in ("parent", "data", "release", "feedback", "artifacts", "output"):
            args.extend(["--" + name, "unused"])
        run = lambda: main(args)
    with pytest.raises(SystemExit) as error:
        run()
    assert error.value.code == 2
    assert "--adaptation-use-policy" in capsys.readouterr().err


def test_eligible_analysis_export_aligns_predictions_without_changing_full_population(population, tmp_path):
    data, release, policy = population
    # A valid release may serialise labels in any order; the view must align files.
    label_path = release / "labels_adaptation.jsonl"
    write_jsonl(label_path, list(reversed(read_lines(label_path))))
    release_manifest = read_json(release / "release_manifest.json")
    release_manifest["split_artifacts"]["adaptation"][label_path.name] = sha(label_path)
    write_json(release / "release_manifest.json", release_manifest)
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(predictions, [{"request_id": f"r{i}", "synthetic_prediction": True} for i in range(5)])
    output = tmp_path / "use"
    manifest = export_adaptation_use(data, release, output, purpose=PURPOSES[0], policy=policy, predictions=predictions)
    assert manifest["eligible_count"] == 4
    for name in ("replay.jsonl", "labels.jsonl", "predictions.jsonl"):
        assert [r["request_id"] for r in read_lines(output / name)] == ["r1", "r2", "r3", "r4"]
    assert "r0" not in (output / "adaptation_use_manifest.json").read_text()
    assert manifest["prediction_source_sha256"] == sha(predictions)
    with pytest.raises(ValueError, match="Output exists"):
        export_adaptation_use(data, release, output, purpose=PURPOSES[0], policy=policy)
    write_jsonl(predictions, [{"request_id": "r5"}])
    with pytest.raises(ValueError, match="complete adaptation"):
        export_adaptation_use(data, release, tmp_path / "wrong", purpose=PURPOSES[0], policy=policy, predictions=predictions)
    assert not (tmp_path / "wrong").exists()


def test_tampered_feedback_cannot_reintroduce_excluded_candidate(population, tmp_path):
    data, release, policy = population
    feedback = tmp_path / "feedback"
    export_feedback(data, release, feedback, policy=policy)
    pool = read_lines(feedback / "feedback_pool.jsonl")
    assert "r0" not in {r["row_uid"] for r in pool}
    forged = dict(pool[0], row_uid="r0", review_text="Cannot log in!", answer="Y")
    write_jsonl(feedback / "feedback_pool.jsonl", pool + [forged])
    manifest = read_json(feedback / "feedback_manifest.json")
    manifest["feedback_pool_sha256"] = sha(feedback / "feedback_pool.jsonl")
    write_json(feedback / "feedback_manifest.json", manifest)
    with pytest.raises(ValueError, match="verified reviewed-source"):
        verified_feedback(data, release, feedback, policy=policy)
