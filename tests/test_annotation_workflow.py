from __future__ import annotations

import json
from pathlib import Path

import pytest

from msc_project.serving.annotation import AnnotationStore
from msc_project.serving.annotation_workflow import (
    HELDOUT, compare_blind, export_feedback, prepare_blind, read_json,
    release_labels, sha, write_json,
)
from msc_project.serving.review_data import write_jsonl
from serve_review_adjudication import AdjudicationStore, EXTRA_SCRIPT
from adaptation_policy_fixture import synthetic_policy


@pytest.fixture
def dataset(tmp_path):
    data = tmp_path / "source"
    data.mkdir()
    rows = [{"request_id": f"r{i}", "text": f"Synthetic unique example number {i}",
             "app": "Test", "split": "adaptation" if i < 5 else "calibration" if i < 7 else "locked",
             "pilot": i < 5} for i in range(10)]
    # A new-source duplicate with a locked input must never enter feedback.
    rows[4]["text"] = rows[8]["text"]
    taxonomy = [{"aspect": name, "definition": name, "zh_name": name, "zh_description": name}
                for name in (HELDOUT, "Other")]
    write_jsonl(data / "annotation_reviews.jsonl", rows)
    write_json(data / "annotation_taxonomy.json", taxonomy)
    write_json(data / "annotation_manifest.json", {"annotation_reviews_sha256": sha(data / "annotation_reviews.jsonl"),
        "assignment": {"counts": {"adaptation": 5, "calibration": 2, "locked": 3}}})
    (data / "human_labels").mkdir()
    labels = {r["request_id"]: {"request_id": r["request_id"], "split": r["split"],
        "source": "human_annotation", "revision": 1, "saved_at": "2026-11-11T00:00:00+00:00",
        "reviewer": "Human One", "status": "confirmed", "labels": [[HELDOUT if i == 3 else "Other", "positive"]],
        "english_confirmed": True, "no_defined_theme": False, "ambiguous": False,
        "out_of_scope": False, "incomplete_text": False, "note": "synthetic test"} for i, r in enumerate(rows)}
    write_json(data / "human_labels/annotations.json", labels)
    write_jsonl(data / "human_labels/annotation_history.jsonl", [
        {"previous": None, "current": value} for value in labels.values()])
    return data


def second_round(queue: Path, *, reviewer="Human Two", disagreement=False, saved_at="2026-11-12T01:00:00+00:00"):
    first = read_json(queue / "first_round_private.json")
    second = {i: dict(a, reviewer=reviewer, saved_at=saved_at) for i, a in first.items()}
    if disagreement:
        identity = next(iter(second))
        second[identity]["labels"] = [["Other", "negative"]]
    write_json(queue / "human_labels/annotations.json", second)


def make_release(dataset, tmp_path, *, disagreement=False):
    queue, comparison, release = tmp_path / "blind", tmp_path / "comparison", tmp_path / "release"
    prepare_blind(dataset, queue)
    second_round(queue, disagreement=disagreement)
    compare_blind(queue, comparison)
    decision_file = adjudicate(comparison)
    release_labels(dataset, comparison, decision_file, release)
    return queue, comparison, release


def decision(identity, **updates):
    return dict({"request_id": identity, "expected_revision": 0,
                 "labels": [["Other", "neutral"]], "reviewer": "Human Adjudicator",
                 "note": "Synthetic adjudication reason", "status": "confirmed",
                 "english_confirmed": True, "no_defined_theme": False, "ambiguous": False,
                 "out_of_scope": False, "incomplete_text": False}, **updates)


def adjudicate(comparison):
    store = AdjudicationStore(comparison / "adjudication")
    for identity in store.order:
        store.save(decision(identity))
    return store.path if store.order else comparison / "decisions_template.json"


def test_twenty_percent_deterministic_blind_queue_no_first_labels(dataset, tmp_path):
    first, second = tmp_path / "queue1", tmp_path / "queue2"
    summary = prepare_blind(dataset, first)
    assert summary["scope_count"] == 10 and summary["sample_count"] == 2
    prepare_blind(dataset, second)
    assert read_json(first / "blind_review_manifest.json")["sample_ids"] == read_json(second / "blind_review_manifest.json")["sample_ids"]
    store = AnnotationStore(first / "annotation_reviews.jsonl", first / "annotation_taxonomy.json", first / "human_labels")
    assert not store.annotations
    for identity in store.order:
        assert store.item(identity)["annotation"] is None
        assert "note" not in store.item(identity)
    with pytest.raises(ValueError, match="Output exists"):
        prepare_blind(dataset, first)


def test_incomplete_and_nonhuman_first_round_block_audit(dataset, tmp_path):
    path = dataset / "human_labels/annotations.json"
    labels = read_json(path)
    labels["r0"]["status"] = "needs_review"
    write_json(path, labels)
    with pytest.raises(ValueError, match="not confirmed"):
        prepare_blind(dataset, tmp_path / "blind")
    labels["r0"]["status"] = "confirmed"
    labels["r0"]["source"] = "model_prediction"
    write_json(path, labels)
    with pytest.raises(ValueError, match="Only human"):
        prepare_blind(dataset, tmp_path / "blind")


def test_independent_reviewer_identity_and_delayed_rereview(dataset, tmp_path):
    queue = tmp_path / "independent"
    prepare_blind(dataset, queue)
    second_round(queue, reviewer="Human One")
    with pytest.raises(ValueError, match="same reviewer"):
        compare_blind(queue, tmp_path / "comparison")
    delayed = tmp_path / "delayed"
    prepare_blind(dataset, delayed, mode="delayed_same_reviewer", delay_hours=24)
    second_round(delayed, reviewer="Human One", saved_at="2026-11-11T10:00:00+00:00")
    with pytest.raises(ValueError, match="interval"):
        compare_blind(delayed, tmp_path / "delayed_comparison")
    second_round(delayed, reviewer="Human One")
    summary = compare_blind(delayed, tmp_path / "delayed_comparison")
    assert summary["review_mode"] == "delayed_same_reviewer"
    assert summary["agreement_fraction"] == 1


def test_disagreement_requires_human_adjudication_preserves_primary(dataset, tmp_path):
    primary_sha = sha(dataset / "human_labels/annotations.json")
    queue = tmp_path / "blind"
    compare = tmp_path / "compare"
    prepare_blind(dataset, queue)
    second_round(queue, disagreement=True)
    summary = compare_blind(queue, compare)
    assert summary["agreement_count"] == 1
    with pytest.raises(ValueError, match="Missing adjudication decision bindings"):
        release_labels(dataset, compare, compare / "decisions_template.json", tmp_path / "release")
    path = adjudicate(compare)
    identity = next(iter(read_json(path)))
    released = release_labels(dataset, compare, path, tmp_path / "release")
    assert released["adjudication_count"] == 1
    assert sha(dataset / "human_labels/annotations.json") == primary_sha
    resolved = read_json(tmp_path / "release/resolved_annotations.json")
    assert resolved[identity]["revision"] == 2
    assert resolved[identity]["labels"] == [["Other", "neutral"]]
    assert resolved[identity]["adjudication"]["adjudication_manifest_sha256"] == sha(compare / "adjudication/adjudication_manifest.json")
    assert len((tmp_path / "release/human_corrections.jsonl").read_text().splitlines()) == 1


def test_changed_primary_or_dataset_rejected(dataset, tmp_path):
    queue, compare, _ = make_release(dataset, tmp_path)
    labels = read_json(dataset / "human_labels/annotations.json")
    labels["r0"]["revision"] += 1
    write_json(dataset / "human_labels/annotations.json", labels)
    with pytest.raises(ValueError, match="Primary annotation changed"):
        release_labels(dataset, compare, tmp_path / "human_decisions.json", tmp_path / "release2")


def test_feedback_adaptation_only_heldout_and_duplicate_exclusion(dataset, tmp_path):
    policy = synthetic_policy(dataset)
    _, _, release = make_release(dataset, tmp_path)
    output = tmp_path / "feedback"
    summary = export_feedback(dataset, release, output, policy=policy)
    assert summary["review_count"] == 3  # r0,r1,r2; r3 heldout, r4 duplicates locked r8.
    assert summary["skipped"] == {"heldout_review": 1, "non_adaptation": 5, "adaptation_use_policy": 1}
    assert summary["live_index_modified"] is False
    pool = [json.loads(line) for line in (output / "feedback_pool.jsonl").read_text().splitlines()]
    assert {r["row_uid"] for r in pool} == {"r0", "r1", "r2"}
    assert all(r["candidate_aspect"] != HELDOUT for r in pool)
    provenance = read_json(output / "provenance.json")
    assert all(r["original_split"] == "adaptation" for r in provenance.values())
    with pytest.raises(ValueError, match="Output exists"):
        export_feedback(dataset, release, output, policy=policy)


def test_feedback_rejects_forged_split_or_modified_labels(dataset, tmp_path):
    policy = synthetic_policy(dataset)
    _, _, release = make_release(dataset, tmp_path)
    path = release / "resolved_adaptation.json"
    labels = read_json(path)
    labels["r0"]["split"] = "locked"
    write_json(path, labels)
    with pytest.raises(ValueError, match="artifact changed"):
        export_feedback(dataset, release, tmp_path / "feedback", policy=policy)
    manifest = read_json(release / "release_manifest.json")
    manifest["split_artifacts"]["adaptation"][path.name] = sha(path)
    write_json(release / "release_manifest.json", manifest)
    with pytest.raises(ValueError, match="ID/split"):
        export_feedback(dataset, release, tmp_path / "feedback", policy=policy)


@pytest.mark.parametrize("name", ["annotation_reviews.jsonl", "annotation_taxonomy.json", "references.json"])
@pytest.mark.parametrize("stage", ["startup", "save", "release"])
def test_adjudication_inputs_are_bound_at_every_stage(dataset, tmp_path, name, stage):
    _, comparison, _ = make_release(dataset, tmp_path, disagreement=True)
    data = comparison / "adjudication"
    store = AdjudicationStore(data)
    annotation_sha = sha(store.path)
    primary_sha = sha(dataset / "human_labels/annotations.json")
    path = data / name
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen adjudication input changed"):
        if stage == "startup":
            AdjudicationStore(data)
        elif stage == "save":
            store.save(decision(store.order[0], expected_revision=1))
        else:
            release_labels(dataset, comparison, store.path, tmp_path / "release2")
    assert sha(store.path) == annotation_sha
    assert sha(dataset / "human_labels/annotations.json") == primary_sha


def test_changed_adjudication_manifest_blocks_save_and_release(dataset, tmp_path):
    _, comparison, _ = make_release(dataset, tmp_path, disagreement=True)
    data = comparison / "adjudication"
    store = AdjudicationStore(data)
    path = data / "adjudication_manifest.json"
    manifest = read_json(path)
    manifest["second_annotations_sha256"] = "0" * 64
    write_json(path, manifest)
    with pytest.raises(ValueError, match="Adjudication manifest changed"):
        AdjudicationStore(data)
    with pytest.raises(ValueError, match="Adjudication manifest changed"):
        store.save(decision(store.order[0], expected_revision=1))
    with pytest.raises(ValueError, match="Adjudication manifest changed"):
        release_labels(dataset, comparison, store.path, tmp_path / "release2")


def test_adjudication_edited_decision_and_wrong_version_bindings_rejected(dataset, tmp_path):
    _, comparison, _ = make_release(dataset, tmp_path, disagreement=True)
    store = AdjudicationStore(comparison / "adjudication")
    identity = store.order[0]
    original = read_json(store.path)
    modified = read_json(store.path)
    modified[identity]["labels"] = [["Other", "positive"]]
    write_json(store.path, modified)
    with pytest.raises(ValueError, match="decision binding mismatch"):
        release_labels(dataset, comparison, store.path, tmp_path / "release2")
    with pytest.raises(ValueError, match="changed in another process"):
        store.save(decision(identity, expected_revision=1))
    write_json(store.path, original)
    path = store.storage / "adjudication_bindings.json"
    binding = read_json(path)
    binding["adjudication_manifest_sha256"] = "0" * 64
    write_json(path, binding)
    with pytest.raises(ValueError, match="decision binding mismatch"):
        AdjudicationStore(comparison / "adjudication")
    with pytest.raises(ValueError, match="decision binding mismatch"):
        release_labels(dataset, comparison, store.path, tmp_path / "release2")


def test_bound_adjudication_revisions_and_language_pending_remain_non_gold(dataset, tmp_path):
    queue, comparison = tmp_path / "blind", tmp_path / "comparison"
    prepare_blind(dataset, queue)
    second_round(queue, disagreement=True)
    compare_blind(queue, comparison)
    store = AdjudicationStore(comparison / "adjudication")
    identity = store.order[0]
    primary_sha = sha(dataset / "human_labels/annotations.json")
    with pytest.raises(ValueError, match="非英文"):
        store.save(decision(identity, english_confirmed=False))
    saved = store.save(decision(identity, status="needs_review", english_confirmed=False,
                               note="Synthetic language review remains pending"))
    assert saved["revision"] == 1
    with pytest.raises(ValueError, match="Unresolved adjudication"):
        release_labels(dataset, comparison, store.path, tmp_path / "release")
    # Resolve uncertainty on this English synthetic input; no translation rule is implied.
    store.save(decision(identity, expected_revision=1))
    release_labels(dataset, comparison, store.path, tmp_path / "release")
    assert len((store.storage / "annotation_history.jsonl").read_text().splitlines()) == 2
    assert sha(dataset / "human_labels/annotations.json") == primary_sha


def test_adjudication_page_displays_every_disagreement_flag():
    for field in ("status", "no_defined_theme", "ambiguous", "out_of_scope", "incomplete_text", "english_confirmed"):
        assert f"r.{field}" in EXTRA_SCRIPT
