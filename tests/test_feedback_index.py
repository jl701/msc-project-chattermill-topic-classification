"""Reviewed-feedback provenance, split isolation and immutable re-encoding."""
from copy import deepcopy
import json

import numpy as np
import pytest

from msc_project.serving.annotation_workflow import export_feedback, read_json, write_json
from msc_project.serving.feedback_index import append_feedback_index, encode_immutable_index, sha_file
from msc_project.serving.retrieval import DemonstrationRetriever, digest, E5_REVISION
from msc_project.serving.review_data import write_jsonl
from adaptation_policy_fixture import synthetic_policy


@pytest.fixture
def campaign(tmp_path):
    from test_annotation_workflow import dataset as source_fixture, make_release

    data = source_fixture.__wrapped__(tmp_path)
    rows = [json.loads(s) for s in (data / "annotation_reviews.jsonl").read_text().splitlines()]
    distinct = ["Payment confirmation arrived after several days of waiting",
        "Helpful assistant resolved the address problem immediately",
        "Application layout is confusing and icons are difficult to locate",
        "Unable to access my account after changing password", "duplicate placeholder",
        "The courier left my order next to the blue gate", "Pricing seems higher than the alternative service",
        "Discount coupon worked during checkout", "The screen became blank immediately after updating",
        "I prefer the previous design of this product"]
    for row,text in zip(rows,distinct):
        row["text"] = text
    rows[4]["text"] = rows[8]["text"]
    write_jsonl(data / "annotation_reviews.jsonl", rows)
    frozen_manifest = read_json(data / "annotation_manifest.json")
    frozen_manifest["annotation_reviews_sha256"] = sha_file(data/"annotation_reviews.jsonl")
    write_json(data / "annotation_manifest.json", frozen_manifest)
    _, _, release = make_release(data, tmp_path)
    feedback = tmp_path / "feedback"
    policy = synthetic_policy(data)
    export_feedback(data, release, feedback, policy=policy)
    rows = [json.loads(s) for s in (data / "annotation_reviews.jsonl").read_text().splitlines()]
    full = tmp_path / "all_new_inputs.jsonl"
    historical = tmp_path / "historical_inputs.jsonl"
    write_jsonl(full, [{"request_id": r["request_id"], "text": r["text"]} for r in rows])
    write_jsonl(historical, [{"request_id": "historical", "text": "Historical protected evaluation customer input"}])
    taxonomy = read_json(data / "annotation_taxonomy.json")
    texts = ["Parcel arrived in excellent condition", "Agent called me after lunch",
             "The support email contained a broken link", "Refund confirmation was late yesterday"]
    pool = [{"row_uid": f"train-{i}", "candidate_aspect": "Other", "review_text": text,
             "aspect_candidate": "Other", "answer": "Y" if i < 2 else "N"} for i,text in enumerate(texts)]
    parent = tmp_path / "parent"
    parent.mkdir()
    write_jsonl(parent / "pool.jsonl", pool)
    np.save(parent / "embeddings.npy", np.eye(4, dtype=np.float32))
    write_json(parent / "manifest.json", {"schema_version": "chattermill_demo_index_v1",
        "pool_sha256": digest(pool), "pool_pairs": 4, "legal_reviews": 4,
        "heldout_aspect": taxonomy[0]["aspect"], "train_csv_sha256": "synthetic-source-hash",
        "exclusion_source_hashes": {str(full): sha_file(full), str(historical): sha_file(historical)},
        "duplicate_rule": "registered_test_rule", "encoded": True,
        "embeddings_sha256": sha_file(parent / "embeddings.npy"), "e5_revision": E5_REVISION})
    artifacts = tmp_path / "artifacts.json"
    write_json(artifacts, {"taxonomy": {"candidates": [{"aspect": c["aspect"], "text": c["definition"]} for c in taxonomy]},
                           "few_shot": {"demonstrations": {"aspect": pool}}})
    return dict(parent=parent, data=data, release=release, feedback=feedback, artifacts=artifacts,
                output=tmp_path / "combined", parent_replays=[full,historical], policy=policy)


def test_reviewed_adaptation_append_preserves_origin_and_reissues_exclusion_certificate(campaign):
    original = {p.name: sha_file(p) for p in campaign["parent"].iterdir()}
    result = append_feedback_index(**campaign)
    output = campaign["output"]
    assert result["reviews_by_origin"] == {"official_train":4, "human_reviewed_adaptation":3}
    assert result["feedback"]["added_reviews"] == 3
    assert result["feedback"]["adaptation_use_audit"]["policy_sha256"] == campaign["policy"].expected_sha256
    assert result["feedback"]["adaptation_use_audit"]["excluded_reviewed_count"] == 1
    assert not result["encoded"] and not (output / "embeddings.npy").exists()
    assert not result["adaptation_evaluation_certified_disjoint"]
    origins = read_json(output / "origin_by_review.json")
    assert origins["r0"]["original_split"] == "adaptation"
    assert origins["r0"]["label_source"] == "human_annotation"
    assert {p.name: sha_file(p) for p in campaign["parent"].iterdir()} == original
    full_hash, historical_hash = map(sha_file, campaign["parent_replays"])
    assert full_hash not in result["exclusion_source_hashes"].values()
    assert historical_hash in result["exclusion_source_hashes"].values()
    assert sha_file(output/"replay_calibration.jsonl") in result["exclusion_source_hashes"].values()
    assert sha_file(output/"replay_locked.jsonl") in result["exclusion_source_hashes"].values()
    assert any(row["adaptation_rows_not_certified_disjoint"] == 5 for row in result["parent_exclusion_partition_audit"])
    # Strict held-out and protected duplicate examples remain absent.
    assert not {"r3", "r4", "r5", "r8"} & set(origins)
    retriever = DemonstrationRetriever(output, method="bm25")
    assert retriever.fingerprint == result["pool_sha256"]
    with pytest.raises(FileExistsError):
        append_feedback_index(**campaign)


def test_feedback_manifest_provenance_and_original_split_tampering_rejected(campaign):
    provenance_path = campaign["feedback"] / "provenance.json"
    provenance = read_json(provenance_path)
    provenance["r0"]["original_split"] = "train"
    write_json(provenance_path, provenance)
    with pytest.raises(ValueError, match="verified reviewed-source"):
        append_feedback_index(**campaign)
    assert not campaign["output"].exists()


def test_all_parent_exclusion_inputs_must_be_accounted_for(campaign):
    campaign["parent_replays"] = campaign["parent_replays"][:1]
    with pytest.raises(ValueError, match="exactly all parent"):
        append_feedback_index(**campaign)


@pytest.mark.parametrize("suffix,reason", [("!", "evaluation_exact"), (" x", "evaluation_near")])
def test_extra_protected_duplicate_filters_feedback_without_dropping_evaluation(campaign, tmp_path, suffix, reason):
    extra = tmp_path / "extra.jsonl"
    write_jsonl(extra, [{"request_id":"extra", "text":"Payment confirmation arrived after several days of waiting"+suffix}])
    before = sha_file(extra)
    result = append_feedback_index(**campaign, extra_protected_replays=[extra])
    assert result["feedback"]["added_reviews"] == 2
    assert result["feedback"]["protected_duplicate_audit"][reason] == 1
    assert "r0" not in read_json(campaign["output"] / "origin_by_review.json")
    assert sha_file(extra) == before


def test_parent_matches_to_new_protected_inputs_require_rebuild(campaign):
    protected = campaign["parent_replays"][1]
    pool = [json.loads(s) for s in (campaign["parent"] / "pool.jsonl").read_text().splitlines()]
    write_jsonl(protected, [{"request_id":"historical", "text":pool[0]["review_text"]}])
    path = campaign["parent"] / "manifest.json"
    manifest = read_json(path)
    manifest["exclusion_source_hashes"][str(protected)] = sha_file(protected)
    write_json(path, manifest)
    with pytest.raises(ValueError, match="Parent contains protected-input"):
        append_feedback_index(**campaign)


def test_immutable_encoding_rebuilds_all_rows_and_preserves_original(campaign, tmp_path, monkeypatch):
    from msc_project.serving import retrieval
    append_feedback_index(**campaign)
    source = campaign["output"]
    before = {p.name: sha_file(p) for p in source.iterdir()}
    calls = []
    class Encoder:
        def __init__(self, path, device):
            assert device == "cpu"
        def encode(self, values):
            calls.extend(values)
            return np.tile(np.array([[.6,.8]], dtype=np.float32), (len(values),1))
    monkeypatch.setattr(retrieval, "E5Encoder", Encoder)
    output = tmp_path / "encoded"
    result = encode_immutable_index(source=source, output=output,
                                    encoder_path=tmp_path/E5_REVISION, device="cpu")
    assert result["encoded"] and len(np.load(output/"embeddings.npy")) == 7
    assert len(calls) == 8  # Seven unique reviews and one candidate card.
    assert {p.name: sha_file(p) for p in source.iterdir()} == before
    assert result["pool_sha256"] == read_json(source/"manifest.json")["pool_sha256"]
    with pytest.raises(FileExistsError):
        encode_immutable_index(source=source, output=output, encoder_path=tmp_path/E5_REVISION)
    with pytest.raises(ValueError, match="pinned"):
        encode_immutable_index(source=source, output=tmp_path/"wrong", encoder_path=tmp_path/"unverified")


def test_failed_encoding_preserves_failure_receipt_without_complete_index(campaign, tmp_path, monkeypatch):
    from msc_project.serving import retrieval
    append_feedback_index(**campaign)
    class Encoder:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Synthetic encoding failure")
    monkeypatch.setattr(retrieval, "E5Encoder", Encoder)
    output = tmp_path / "failed_encoding"
    with pytest.raises(RuntimeError, match="Synthetic"):
        encode_immutable_index(source=campaign["output"], output=output, encoder_path=tmp_path/E5_REVISION)
    assert read_json(output/"encoding_state.json")["status"] == "failed"
    assert not (output/"manifest.json").exists()
