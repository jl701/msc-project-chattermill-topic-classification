"""Byte-preserving assembly of completed, provenance-compatible score matrices.

No model, labels, classification thresholds or quality metrics are accessed.
Source code versions remain separate, linked by an explicit passed control gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def copy_exact(source: Path, target: Path, expected: str) -> None:
    hashed = hashlib.sha256()
    with source.open("rb") as reader, target.open("xb") as writer:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            writer.write(block)
            hashed.update(block)
    if hashed.hexdigest() != expected or digest(source) != expected:
        raise ValueError("Source changed while copying immutable evidence")


def inspect_scores(path: Path, expected_ids: set[str], expected_sha: str) -> int:
    if digest(path) != expected_sha:
        raise ValueError("Method score SHA differs from its completed source summary")
    identities, successful = set(), 0
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            identity = row.get("request_id")
            if identity not in expected_ids or identity in identities:
                raise ValueError("Method scores have duplicate or unregistered review IDs")
            identities.add(identity)
            if row.get("status") not in ("ok", "error"):
                raise ValueError("Method score has unsupported execution status")
            if row["status"] == "ok":
                if row.get("response", {}).get("request_id") != identity:
                    raise ValueError("Successful response identity differs from request")
                successful += 1
    if identities != expected_ids or digest(path) != expected_sha:
        raise ValueError("Method score population is incomplete or changed during validation")
    return successful


def inspect_source(path: Path, reviews: int) -> dict:
    path = path.resolve()
    state = read_json(path / "state.json")
    if state.get("status") != "complete":
        raise RuntimeError(f"Source is not complete: {path.name}; retry after scoring finishes")
    identity, summary = read_json(path / "identity.json"), read_json(path / "summary.json")
    ids, methods = identity.get("request_ids", []), identity.get("methods", [])
    if len(ids) != reviews or len(set(ids)) != reviews or any(not isinstance(v, str) or not v for v in ids):
        raise ValueError("Source identity has wrong review count or duplicate IDs")
    if (not methods or len(set(methods)) != len(methods) or any(not isinstance(m, str) or
            not m or not all(c.isascii() and (c.isalnum() or c == "_") for c in m) for m in methods)):
        raise ValueError("Invalid registered source method names")
    if state.get("reviews") != reviews or summary.get("reviews") != reviews or \
            state.get("methods") != methods or set(summary.get("methods", {})) != set(methods):
        raise ValueError("Completed source identity, state and summary disagree")
    required = ("artifacts", "replay", "runtime_artifacts", "index_files", "inference_code", "config", "sentiment_anchor_reuse")
    if any(not identity.get(key) for key in required):
        raise ValueError("Source lacks required input or code provenance")
    metadata = ("identity.json", "summary.json", "state.json", "runtime.json", "sentiment_anchor_reuse.json")
    hashes = {name: digest(path / name) for name in metadata}
    if digest(path / "_inference_artifacts.json") != identity["runtime_artifacts"]:
        raise ValueError("Runtime artifact SHA mismatch")
    anchor = identity["sentiment_anchor_reuse"]
    if read_json(path / "sentiment_anchor_reuse.json") != anchor or \
            digest(path / "_sentiment_anchor.jsonl") != anchor["anchor_sha256"] or anchor["reused_reviews"] != reviews:
        raise ValueError("Stage2 anchor file or reuse receipt differs")
    anchor_path = Path(anchor["source"])
    original_anchor_files = {anchor_path: anchor["anchor_sha256"],
        anchor_path.parent / "identity.json": anchor["identity_sha256"],
        anchor_path.parent / "runtime.json": anchor["runtime_sha256"],
        anchor_path.parent / "state.json": anchor["state_sha256"]}
    if any(digest(file) != expected for file, expected in original_anchor_files.items()):
        raise ValueError("Original Stage2 anchor provenance changed")
    if read_json(anchor_path.parent / "state.json").get("status") != "complete":
        raise ValueError("Original Stage2 anchor run is not complete")
    for method in methods:
        successful = inspect_scores(path / f"{method}.jsonl", set(ids), summary["methods"][method]["file_sha256"])
        if successful != summary["methods"][method]["successful"]:
            raise ValueError("Method successful count differs from completed summary")
    if any(digest(path / name) != value for name, value in hashes.items()):
        raise ValueError("Source metadata changed during validation")
    return {"path": path, "identity": identity, "summary": summary, "hashes": hashes}


def assemble(sources: list[Path], output: Path, parity_gate: Path, *, reviews: int = 100,
             reference_method: str = "e5", new_method: str = "e5_negative_seen") -> dict:
    started = time.perf_counter()
    output = output.resolve()
    if output.exists():
        raise FileExistsError("Combined output already exists; immutable evidence is never overwritten")
    if reviews < 1 or len(sources) < 2 or len({p.resolve() for p in sources}) != len(sources):
        raise ValueError("Use at least two distinct completed sources and a positive review count")
    inspected = [inspect_source(path, reviews) for path in sources]
    first = inspected[0]["identity"]
    common_fields = ("artifacts", "replay", "runtime_artifacts", "index_files", "sentiment_anchor_reuse")
    owners, methods = {}, []
    for i, source in enumerate(inspected):
        identity = source["identity"]
        if any(identity[k] != first[k] for k in common_fields) or set(identity["request_ids"]) != set(first["request_ids"]):
            raise ValueError("Source populations, artifacts, replay, index or Stage2 provenance differ")
        for method in identity["methods"]:
            if method in owners:
                raise ValueError("Duplicate method across source runs")
            owners[method] = i
            methods.append(method)
    if reference_method not in owners or new_method not in owners or owners[reference_method] == owners[new_method]:
        raise ValueError("Reference and new method must identify separate source runs")
    gate_sha = digest(parity_gate)
    gate = read_json(parity_gate)
    reference = inspected[owners[reference_method]]
    newer = inspected[owners[new_method]]
    expected_gate_inputs = {"artifacts": first["artifacts"], "development_replay": first["replay"],
        "index_manifest": first["index_files"]["manifest.json"],
        "reference_e5": reference["summary"]["methods"][reference_method]["file_sha256"]}
    if (gate.get("status") != "complete" or gate.get("gates_pass") is not True or
            gate.get("legacy_seen_probability_and_margin_exact") is not True or
            gate.get("no_gold_read") is not True or any(
                gate.get("input_sha256", {}).get(k) != value for k, value in expected_gate_inputs.items())):
        raise ValueError("Control parity gate did not pass or does not bind the source inputs/reference")
    required_gate_code = ("src/msc_project/serving/inference.py", "src/msc_project/serving/retrieval.py",
                          "src/msc_project/serving/retrieval_policy.py")
    if any(not gate.get("source_sha256", {}).get(p) or
           gate["source_sha256"][p] != newer["identity"]["inference_code"].get(p) for p in required_gate_code):
        raise ValueError("New source inference code differs from the passed parity gate")
    source_records, per_method = {}, {}
    for i, source in enumerate(inspected):
        name = f"source_{i + 1:02d}"
        source_records[name] = {"path": str(source["path"]), "metadata_sha256": source["hashes"],
            "config_sha256": source["identity"]["config"],
            "inference_code": source["identity"]["inference_code"],
            "methods": source["identity"]["methods"]}
        for method in source["identity"]["methods"]:
            per_method[method] = {"source_run": name, "source_path": str(source["path"] / f"{method}.jsonl"),
                "file_sha256": source["summary"]["methods"][method]["file_sha256"]}
    control = {"source": str(parity_gate.resolve()), "sha256": gate_sha,
        "copied_path": "provenance/control_parity_gate.json", "scope": gate["scope"],
        "reviews": gate["reviews"], "encoded_pairs": gate["encoded_pairs"],
        "legacy_seen_probability_and_margin_exact": True,
        "interpretation": "input-only sampled E5 control parity; source-specific code versions are preserved, not unified"}
    identity = {"schema_version": "assembled_rag_development_v1", **{k: first[k] for k in common_fields},
        "methods": methods, "request_ids": first["request_ids"], "source_runs": source_records,
        "method_sources": per_method, "cross_version_control_parity": control,
        "code_provenance_scope": "source_specific_inference_code_and_config_no_fabricated_common_version"}
    receipt = {"registered_utc": datetime.now(timezone.utc).isoformat(), "no_labels_read": True,
        "no_scores_recomputed": True, "source_method_files_copied_byte_for_byte": True,
        "assembler_sha256": digest(Path(__file__)), "reviews_per_method": reviews,
        "methods": methods, "source_runs": source_records, "control_parity": control}
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    write_json(output / "assembly_registration.json", receipt)
    for i, source in enumerate(inspected):
        target = provenance / f"source_{i + 1:02d}"
        target.mkdir()
        for name, value in source["hashes"].items():
            copy_exact(source["path"] / name, target / name, value)
        for method in source["identity"]["methods"]:
            copy_exact(source["path"] / f"{method}.jsonl", output / f"{method}.jsonl", per_method[method]["file_sha256"])
    for name, value in (("_inference_artifacts.json", first["runtime_artifacts"]),
                        ("_sentiment_anchor.jsonl", first["sentiment_anchor_reuse"]["anchor_sha256"]),
                        ("sentiment_anchor_reuse.json", inspected[0]["hashes"]["sentiment_anchor_reuse.json"])):
        copy_exact(inspected[0]["path"] / name, output / name, value)
    copy_exact(parity_gate, provenance / "control_parity_gate.json", gate_sha)
    write_json(output / "identity.json", identity)
    summary = {"methods": {m: inspected[owners[m]]["summary"]["methods"][m] for m in methods},
        "reviews": reviews, "scope": "assembled_historical_development_scores_no_new_inference",
        "no_labels_read": True, "source_method_timing_scope": "original_offline_component_times_not_assembly_or_HTTP_latency",
        "source_run_elapsed_s": {f"source_{i + 1:02d}": s["summary"].get("elapsed_s") for i, s in enumerate(inspected)}}
    write_json(output / "summary.json", summary)
    receipt.update(completed_utc=datetime.now(timezone.utc).isoformat(), assembly_elapsed_s=time.perf_counter() - started,
        combined_identity_sha256=digest(output / "identity.json"), combined_summary_sha256=digest(output / "summary.json"))
    write_json(output / "assembly_receipt.json", receipt)
    write_json(output / "state.json", {"status": "complete", "reviews": reviews, "methods": methods,
        "assembly_receipt_sha256": digest(output / "assembly_receipt.json")})
    return {"status": "complete", "methods": methods, "reviews": reviews, "output": str(output),
            "receipt_sha256": digest(output / "assembly_receipt.json")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parity-gate", type=Path, required=True)
    parser.add_argument("--reviews", type=int, default=100)
    args = parser.parse_args(argv)
    print(json.dumps(assemble(args.source, args.output, args.parity_gate, reviews=args.reviews)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
