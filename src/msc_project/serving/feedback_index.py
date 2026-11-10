"""Immutable, provenance-preserving promotion of reviewed adaptation examples.

The official-train index builder remains unchanged. This separate operation
never relabels adaptation records as official train or certifies adaptation
inputs as disjoint evaluation inputs after their examples have been added.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

from msc_project.serving.annotation_workflow import export_feedback, read_dataset
from msc_project.serving.retrieval import digest, exclude_duplicates


def sha_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def write_jsonl(path: Path, values: Sequence[dict]) -> None:
    path.write_text("".join(json.dumps(v, ensure_ascii=False, allow_nan=False)+"\n" for v in values),
                    encoding="utf-8")


def grouped_pool(pool: list[dict], candidates: dict[str, str]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    fields = {"row_uid", "candidate_aspect", "review_text", "aspect_candidate", "answer"}
    for row in pool:
        if (set(row) != fields or row["answer"] not in {"Y", "N"} or
                row["candidate_aspect"] not in candidates or
                row["aspect_candidate"] != candidates[row["candidate_aspect"]]):
            raise ValueError("Pool row does not match the frozen demonstration schema/taxonomy")
        groups[row["row_uid"]].append(row)
    for rows in groups.values():
        if (len(rows) != len(candidates) or {r["candidate_aspect"] for r in rows} != set(candidates)
                or len({r["review_text"] for r in rows}) != 1):
            raise ValueError("Each review requires one complete, text-consistent seen-aspect grid")
    return dict(groups)


def verified_feedback(data: Path, release: Path, feedback: Path):
    """Re-export from the reviewed source to validate all current release gates."""
    manifest = read_json(feedback / "feedback_manifest.json")
    if manifest.get("feedback_pool_sha256") != sha_file(feedback / "feedback_pool.jsonl"):
        raise ValueError("Feedback pool hash differs from its export manifest")
    # The export routine owns human-review/adjudication validation. Reusing it
    # avoids silently bypassing future strengthening of those release gates.
    with tempfile.TemporaryDirectory(prefix="rag-feedback-verify-") as temporary:
        owned = Path(temporary).resolve()
        if owned.parent != Path(tempfile.gettempdir()).resolve():
            raise RuntimeError("Unexpected verification temporary-directory boundary")
        expected = owned / "expected"
        export_feedback(data, release, expected)
        for name in ("feedback_pool.jsonl", "provenance.json", "feedback_manifest.json"):
            loader = read_jsonl if name.endswith("jsonl") else read_json
            if loader(feedback / name) != loader(expected / name):
                raise ValueError(f"Feedback differs from verified reviewed-source export: {name}")
    return read_jsonl(feedback / "feedback_pool.jsonl"), read_json(feedback / "provenance.json"), manifest


def append_feedback_index(*, parent: Path, data: Path, release: Path, feedback: Path,
                          artifacts: Path, output: Path, parent_replays: Sequence[Path],
                          extra_protected_replays: Sequence[Path] = ()) -> dict:
    if output.exists():
        raise FileExistsError("Output index already exists; use a new version directory")
    parent_manifest = read_json(parent / "manifest.json")
    parent_pool = read_jsonl(parent / "pool.jsonl")
    if digest(parent_pool) != parent_manifest.get("pool_sha256"):
        raise ValueError("Parent pool fingerprint mismatch")
    feedback_pool, feedback_provenance, feedback_manifest = verified_feedback(data, release, feedback)
    dataset, taxonomy = read_dataset(data)
    config = read_json(artifacts)
    heldout = parent_manifest["heldout_aspect"]
    if (feedback_manifest.get("heldout_aspect") != heldout or not feedback_manifest.get("strict_unseen")
            or feedback_manifest.get("calibration_and_locked_included") is not False):
        raise ValueError("Feedback does not preserve strict held-out-theme scope")
    all_candidates = {str(c["aspect"]): str(c["text"]) for c in config["taxonomy"]["candidates"]}
    if {r["aspect"]: r["definition"] for r in taxonomy} != all_candidates:
        raise ValueError("Human annotation taxonomy differs from the frozen model taxonomy")
    candidates = {a: c for a,c in all_candidates.items() if a != heldout}
    base_groups = grouped_pool(parent_pool, candidates)
    additions = grouped_pool(feedback_pool, candidates)
    if set(additions) != set(feedback_provenance):
        raise ValueError("Feedback provenance coverage differs from its reviews")

    # Preserve the origin of every existing review, including prior feedback
    # versions. No adaptation record is ever recast as official training data.
    if parent_manifest.get("origin_by_review_sha256"):
        if sha_file(parent / "origin_by_review.json") != parent_manifest["origin_by_review_sha256"]:
            raise ValueError("Parent review-origin record changed")
        origins = read_json(parent / "origin_by_review.json")
        if set(origins) != set(base_groups):
            raise ValueError("Parent review-origin coverage mismatch")
    elif (parent_manifest.get("schema_version") == "chattermill_demo_index_v1"
          and parent_manifest.get("train_csv_sha256")):
        origins = {i: {"origin": "official_train", "train_csv_sha256": parent_manifest["train_csv_sha256"]}
                   for i in base_groups}
    else:
        raise ValueError("Parent index lacks verifiable per-review origin")
    if any(v.get("origin") not in {"official_train", "human_reviewed_adaptation"} for v in origins.values()):
        raise ValueError("Unknown parent review origin")

    protected = {i: {"request_id": i, "text": r["text"]} for i,r in dataset.items()
                 if r["split"] in {"calibration", "locked"}}
    parent_registered = set(parent_manifest.get("exclusion_source_hashes", {}).values())
    provided = {sha_file(path): path for path in parent_replays}
    if set(provided) != parent_registered:
        raise ValueError("Provide exactly all parent exclusion replays with their registered hashes")
    full_exclusion_hashes, partition_audit = {}, []

    def protect(row):
        if set(row) != {"request_id", "text"} or not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError("Protected replay must contain input-only records")
        previous = protected.get(row["request_id"])
        if previous is not None and previous["text"] != row["text"]:
            raise ValueError("Conflicting protected review identity")
        protected[row["request_id"]] = dict(row)

    for value, path in sorted(provided.items()):
        moved = 0
        rows = read_jsonl(path)
        for row in rows:
            original = dataset.get(row.get("request_id"))
            if original is not None and original["split"] == "adaptation":
                if set(row) != {"request_id", "text"} or row["text"] != original["text"]:
                    raise ValueError("Adaptation partition differs from its frozen input snapshot")
                moved += 1
            else:
                protect(row)
        partition_audit.append({"source_sha256": value, "source_path": str(path.resolve()),
                                "source_rows": len(rows), "adaptation_rows_not_certified_disjoint": moved,
                                "protected_rows": len(rows)-moved})
        if not moved:
            full_exclusion_hashes[str(path.resolve())] = value
    for path in extra_protected_replays:
        for row in read_jsonl(path):
            protect(row)
        full_exclusion_hashes[str(path.resolve())] = sha_file(path)
    protected_rows = [protected[i] for i in sorted(protected)]
    excluded_texts = [r["text"] for r in protected_rows]
    base_reviews = [{"row_uid": i, "text": rows[0]["review_text"]} for i,rows in base_groups.items()]
    safe_base, base_audit = exclude_duplicates(base_reviews, excluded_texts)
    if len(safe_base) != len(base_reviews):
        raise ValueError("Parent contains protected-input matches; rebuild it with these exclusions first")

    already_present = []
    for identity in set(additions) & set(base_groups):
        if sorted(additions[identity], key=lambda r:r["candidate_aspect"]) != sorted(
                base_groups[identity], key=lambda r:r["candidate_aspect"]):
            raise ValueError("Existing review has changed labels/text; explicit replacement workflow required")
        already_present.append(identity)
    new_reviews = [{"row_uid": i, "text": rows[0]["review_text"]} for i,rows in additions.items()
                   if i not in already_present]
    safe_feedback, protected_audit = exclude_duplicates(new_reviews, excluded_texts)
    safe_feedback, base_overlap_audit = exclude_duplicates(
        safe_feedback, [row["text"] for row in base_reviews])
    accepted_ids = {r["row_uid"] for r in safe_feedback}
    accepted_pairs = [r for r in feedback_pool if r["row_uid"] in accepted_ids]
    combined = parent_pool + accepted_pairs
    legal = {digest(r) for r in combined}
    invalid_fixed = [d["row_uid"]+"|"+d["candidate_aspect"]
                     for d in config["few_shot"]["demonstrations"]["aspect"] if digest(d) not in legal]
    if invalid_fixed:
        raise ValueError("Combined index would invalidate frozen fixed demonstrations")
    for identity in sorted(accepted_ids):
        if dataset[identity]["split"] != "adaptation":
            raise ValueError("Only the frozen adaptation partition may enter feedback")
        origins[identity] = {"origin": "human_reviewed_adaptation", **feedback_provenance[identity]}
    source_counts = Counter(v["origin"] for v in origins.values())
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "pool.jsonl", combined)
    write_json(output / "origin_by_review.json", origins)
    write_jsonl(output / "protected_replay.jsonl", protected_rows)
    full_exclusion_hashes[str((output / "protected_replay.jsonl").resolve())] = sha_file(output / "protected_replay.jsonl")
    for split in ("calibration", "locked"):
        split_path = output / f"replay_{split}.jsonl"
        write_jsonl(split_path, [{"request_id": i, "text": dataset[i]["text"]} for i in sorted(dataset)
                                if dataset[i]["split"] == split])
        full_exclusion_hashes[str(split_path.resolve())] = sha_file(split_path)
    manifest = {"schema_version": "chattermill_demo_index_with_feedback_v1",
        "origin_scope": "official_train_plus_human_reviewed_adaptation",
        "heldout_aspect": heldout, "pool_sha256": digest(combined), "pool_pairs": len(combined),
        "legal_reviews": len(origins), "reviews_by_origin": dict(source_counts),
        "origin_by_review_sha256": sha_file(output / "origin_by_review.json"),
        "exclusion_source_hashes": full_exclusion_hashes,
        "parent_exclusion_partition_audit": partition_audit,
        "parent_index_manifest_sha256": sha_file(parent / "manifest.json"),
        "parent_pool_sha256": parent_manifest["pool_sha256"],
        "base_index_audit": parent_manifest,
        "feedback": {"reviewed_release_manifest_sha256": sha_file(release / "release_manifest.json"),
            "feedback_manifest_sha256": sha_file(feedback / "feedback_manifest.json"),
            "feedback_pool_sha256": sha_file(feedback / "feedback_pool.jsonl"),
            "provenance_sha256": sha_file(feedback / "provenance.json"),
            "source_dataset_sha256": sha_file(data / "annotation_reviews.jsonl"),
            "added_reviews": len(accepted_ids), "added_pairs": len(accepted_pairs),
            "already_present_reviews": len(already_present), "protected_duplicate_audit": protected_audit,
            "base_overlap_audit": base_overlap_audit},
        "duplicate_rule": parent_manifest.get("duplicate_rule"),
        "frozen_fixed_ineligible_ids": [], "encoded": False,
        "embedding_state": "requires_full_rebuild_for_combined_pool",
        "encoder_path": parent_manifest.get("encoder_path"), "e5_revision": parent_manifest.get("e5_revision"),
        "index_encoding_device": None, "index_encoding_batch_size": None,
        "calibration_and_locked_examples_included": False,
        "adaptation_evaluation_certified_disjoint": False,
        "parent_index_modified": False}
    write_json(output / "manifest.json", manifest)
    return manifest


def encode_immutable_index(*, source: Path, output: Path, encoder_path: Path,
                           device: str = "cpu") -> dict:
    """Rebuild E5 vectors in another version; never reuse stale parent vectors."""
    import numpy as np
    from msc_project.serving.retrieval import E5Encoder, E5_REVISION, encode_pool

    if output.exists():
        raise FileExistsError("Encoded output already exists; use a new version directory")
    manifest, pool = read_json(source / "manifest.json"), read_jsonl(source / "pool.jsonl")
    if manifest.get("e5_revision") != E5_REVISION or encoder_path.name != E5_REVISION:
        raise ValueError("Encoding requires the pinned E5 snapshot revision")
    if digest(pool) != manifest.get("pool_sha256"):
        raise ValueError("Index pool changed before encoding")
    if manifest.get("origin_by_review_sha256") and sha_file(source / "origin_by_review.json") != manifest["origin_by_review_sha256"]:
        raise ValueError("Review-origin record changed before encoding")
    output.mkdir(parents=True, exist_ok=False)
    for path in source.iterdir():
        if path.is_file() and path.name not in {"embeddings.npy", "manifest.json", "encoding_state.json"}:
            shutil.copy2(path, output / path.name)
    write_json(output / "encoding_state.json", {"status": "running", "source_manifest_sha256": sha_file(source / "manifest.json")})
    try:
        embeddings = encode_pool(pool, E5Encoder(encoder_path, device=device))
        if embeddings.ndim != 2 or len(embeddings) != len(pool) or not np.isfinite(embeddings).all():
            raise ValueError("E5 embedding rows must match the complete pool and be finite")
        np.save(output / "embeddings.npy", embeddings)
        manifest = deepcopy(manifest)
        manifest.update(encoded=True, embedding_state="complete", encoder_path=str(encoder_path.resolve()),
            embedding_dimension=int(embeddings.shape[1]), embeddings_sha256=sha_file(output / "embeddings.npy"),
            index_encoding_device=device, index_encoding_batch_size=32,
            source_unencoded_manifest_sha256=sha_file(source / "manifest.json"))
        write_json(output / "manifest.json", manifest)
        write_json(output / "encoding_state.json", {"status": "completed", "pool_sha256": manifest["pool_sha256"]})
        return manifest
    except BaseException as exc:
        write_json(output / "encoding_state.json", {"status": "failed", "error_type": type(exc).__name__})
        raise
