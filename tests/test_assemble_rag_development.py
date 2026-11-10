"""Synthetic, CPU-only checks of immutable score-matrix assembly."""
import json
from pathlib import Path

import pytest

import assemble_rag_development as assembly


def put(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(tmp_path):
    ids = [f"synthetic-{i}" for i in range(3)]
    anchor = tmp_path / "common_anchor"
    anchor.mkdir()
    (anchor / "_sentiment_anchor.jsonl").write_text('synthetic byte-preserved anchor\n', encoding="utf-8")
    for name in ("identity.json", "runtime.json"):
        put(anchor / name, {"synthetic": True})
    put(anchor / "state.json", {"status": "complete"})
    receipt = {"source": str(anchor / "_sentiment_anchor.jsonl"),
        "anchor_sha256": assembly.digest(anchor / "_sentiment_anchor.jsonl"),
        "identity_sha256": assembly.digest(anchor / "identity.json"),
        "runtime_sha256": assembly.digest(anchor / "runtime.json"),
        "state_sha256": assembly.digest(anchor / "state.json"), "source_reviews": 3, "reused_reviews": 3}
    sources = []
    for name, methods in (("old", ["fixed", "e5"]), ("new", ["e5_negative_seen"])):
        path = tmp_path / name
        path.mkdir()
        sources.append(path)
        (path / "_inference_artifacts.json").write_bytes(b'{"frozen":true}\r\n')
        (path / "_sentiment_anchor.jsonl").write_bytes((anchor / "_sentiment_anchor.jsonl").read_bytes())
        put(path / "sentiment_anchor_reuse.json", receipt)
        put(path / "runtime.json", {"source": name})
        identity = {"artifacts": "same-artifacts", "replay": "same-replay", "config": f"config-{name}",
            "methods": methods, "request_ids": ids, "index_files": {"manifest.json": "same-index"},
            "runtime_artifacts": assembly.digest(path / "_inference_artifacts.json"),
            "sentiment_anchor_reuse": receipt,
            "inference_code": {f"src/msc_project/serving/{file}.py": f"{name}-{file}"
                               for file in ("inference", "retrieval", "retrieval_policy")}}
        put(path / "identity.json", identity)
        results = {}
        for method in methods:
            rows = [{"request_id": identity, "status": "ok", "response": {"request_id": identity,
                     "opaque_scores": "Never interpreted by assembly"}} for identity in ids]
            (path / f"{method}.jsonl").write_bytes(('\r\n'.join(json.dumps(r) for r in rows)+'\r\n').encode())
            results[method] = {"successful": 3, "file_sha256": assembly.digest(path / f"{method}.jsonl"),
                               "mean_offline_presence_ms": 1.0}
        put(path / "summary.json", {"methods": results, "reviews": 3, "no_labels_read": True, "elapsed_s": 1})
        put(path / "state.json", {"status": "complete", "reviews": 3, "methods": methods})
    gate = tmp_path / "gate.json"
    put(gate, {"status": "complete", "gates_pass": True, "no_gold_read": True,
        "legacy_seen_probability_and_margin_exact": True, "scope": "native_seen_numeric_parity_only",
        "reviews": 3, "encoded_pairs": 72,
        "input_sha256": {"artifacts": "same-artifacts", "development_replay": "same-replay",
            "index_manifest": "same-index", "reference_e5": assembly.digest(sources[0] / "e5.jsonl")},
        "source_sha256": assembly.read_json(sources[1] / "identity.json")["inference_code"]})
    return sources, tmp_path / "combined", gate


def test_copies_exact_bytes_and_preserves_separate_code_versions_and_parent_receipts(tmp_path):
    sources, output, gate = fixture(tmp_path)
    result = assembly.assemble(sources, output, gate, reviews=3)
    assert result["status"] == "complete"
    identity = assembly.read_json(output / "identity.json")
    assert identity["methods"] == ["fixed", "e5", "e5_negative_seen"]
    assert "inference_code" not in identity and "config" not in identity
    assert identity["source_runs"]["source_01"]["inference_code"] != identity["source_runs"]["source_02"]["inference_code"]
    for method, source_index in (("fixed", 0), ("e5", 0), ("e5_negative_seen", 1)):
        assert (output / f"{method}.jsonl").read_bytes() == (sources[source_index] / f"{method}.jsonl").read_bytes()
        assert identity["method_sources"][method]["file_sha256"] == assembly.digest(output / f"{method}.jsonl")
    for i, source in enumerate(sources, 1):
        for name in ("identity.json", "summary.json", "state.json", "runtime.json"):
            assert (output / "provenance" / f"source_{i:02d}" / name).read_bytes() == (source / name).read_bytes()
    assert (output / "provenance/control_parity_gate.json").read_bytes() == gate.read_bytes()
    assert assembly.read_json(output / "state.json")["assembly_receipt_sha256"] == assembly.digest(output / "assembly_receipt.json")
    with pytest.raises(FileExistsError, match="never overwritten"):
        assembly.assemble(sources, output, gate, reviews=3)


def test_running_source_fails_before_creating_output(tmp_path):
    sources, output, gate = fixture(tmp_path)
    put(sources[1] / "state.json", {"status": "running"})
    with pytest.raises(RuntimeError, match="not complete"):
        assembly.assemble(sources, output, gate, reviews=3)
    assert not output.exists()


def test_changed_method_bytes_are_rejected(tmp_path):
    sources, output, gate = fixture(tmp_path)
    with (sources[0] / "fixed.jsonl").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="score SHA"):
        assembly.assemble(sources, output, gate, reviews=3)
    assert not output.exists()


def test_duplicate_score_ids_fail_even_with_an_updated_file_hash(tmp_path):
    sources, output, gate = fixture(tmp_path)
    path = sources[1] / "e5_negative_seen.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1] = rows[0]
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    summary = assembly.read_json(sources[1] / "summary.json")
    summary["methods"]["e5_negative_seen"]["file_sha256"] = assembly.digest(path)
    put(sources[1] / "summary.json", summary)
    with pytest.raises(ValueError, match="duplicate"):
        assembly.assemble(sources, output, gate, reviews=3)


@pytest.mark.parametrize("field", ["artifacts", "replay", "index_files"])
def test_incompatible_common_inputs_are_rejected(tmp_path, field):
    sources, output, gate = fixture(tmp_path)
    identity = assembly.read_json(sources[1] / "identity.json")
    identity[field] = {"manifest.json": "different"} if field == "index_files" else "different"
    put(sources[1] / "identity.json", identity)
    with pytest.raises(ValueError, match="provenance differ"):
        assembly.assemble(sources, output, gate, reviews=3)
    assert not output.exists()


def test_changed_original_stage2_anchor_provenance_is_rejected(tmp_path):
    sources, output, gate = fixture(tmp_path)
    receipt = assembly.read_json(sources[0] / "sentiment_anchor_reuse.json")
    Path(receipt["source"]).write_text("changed common anchor")
    with pytest.raises(ValueError, match="Original Stage2"):
        assembly.assemble(sources, output, gate, reviews=3)


@pytest.mark.parametrize("mutation", ["failed_gate", "wrong_reference", "wrong_code"])
def test_cross_version_gate_must_pass_and_bind_the_actual_sources(tmp_path, mutation):
    sources, output, gate = fixture(tmp_path)
    value = assembly.read_json(gate)
    if mutation == "failed_gate":
        value["gates_pass"] = False
    elif mutation == "wrong_reference":
        value["input_sha256"]["reference_e5"] = "wrong"
    else:
        value["source_sha256"]["src/msc_project/serving/inference.py"] = "wrong"
    put(gate, value)
    with pytest.raises(ValueError, match="parity gate"):
        assembly.assemble(sources, output, gate, reviews=3)
    assert not output.exists()
