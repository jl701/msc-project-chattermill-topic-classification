"""Blind re-review, human adjudication and adaptation-only example feedback.

All outputs are new immutable directories. The live first-round annotation
store is read-only here; no machine-generated label can become human gold.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from msc_project.serving.review_data import deduplicate, write_jsonl

HELDOUT = "Account management: Account access"
FLAGS = ("no_defined_theme", "ambiguous", "out_of_scope", "incomplete_text", "english_confirmed")
ADJUDICATION_FILES = ("annotation_reviews.jsonl", "annotation_taxonomy.json", "references.json")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def new_directory(path: Path) -> None:
    if path.exists():
        raise ValueError("Output exists; choose a new version directory")
    path.mkdir(parents=True)


def read_dataset(data: Path) -> tuple[dict[str, dict], list[dict]]:
    source = data / "annotation_reviews.jsonl"
    rows = [json.loads(s) for s in source.read_text(encoding="utf-8").splitlines() if s.strip()]
    by_id = {r["request_id"]: r for r in rows}
    if len(rows) != len(by_id):
        raise ValueError("Duplicate dataset ID")
    if any(r.get("split") not in {"adaptation", "calibration", "locked"} for r in rows):
        raise ValueError("Unknown frozen split")
    manifest = data / "annotation_manifest.json"
    if manifest.exists() and read_json(manifest).get("annotation_reviews_sha256") != sha(source):
        raise ValueError("Frozen annotation dataset hash changed")
    taxonomy = read_json(data / "annotation_taxonomy.json")
    return by_id, taxonomy


def scope_ids(rows: dict, group: str) -> list[str]:
    if group not in {"pilot", "adaptation", "calibration", "locked", "all"}:
        raise ValueError("Unknown annotation group")
    return sorted(i for i, r in rows.items() if group == "all" or
                  (group == "pilot" and r.get("pilot")) or r["split"] == group)


def validate_label(value: dict, row: dict, taxonomy: list[dict], *, confirmed: bool = True) -> None:
    if value.get("source") != "human_annotation":
        raise ValueError("Only human_annotation provenance is accepted")
    if value.get("request_id") != row["request_id"] or value.get("split") != row["split"]:
        raise ValueError("Label ID/split does not match frozen dataset")
    if not isinstance(value.get("reviewer"), str) or not value["reviewer"].strip():
        raise ValueError("Human reviewer required")
    if not isinstance(value.get("revision"), int) or value["revision"] < 1:
        raise ValueError("Positive annotation revision required")
    timestamp(value.get("saved_at", ""))
    if value.get("status") not in {"confirmed", "needs_review"}:
        raise ValueError("Invalid annotation status")
    if any(not isinstance(value.get(f), bool) for f in FLAGS):
        raise ValueError("Missing annotation flags")
    allowed = {t["aspect"] for t in taxonomy}
    pairs = value.get("labels")
    if not isinstance(pairs, list):
        raise ValueError("Labels must be a list")
    seen = set()
    for pair in pairs:
        if not isinstance(pair, list) or len(pair) != 2 or pair[0] not in allowed or pair[1] not in {"negative", "neutral", "positive"}:
            raise ValueError("Invalid aspect/sentiment")
        if tuple(pair) in seen:
            raise ValueError("Duplicate label pair")
        seen.add(tuple(pair))
    if pairs and value["no_defined_theme"]:
        raise ValueError("Labels conflict with no_defined_theme")
    if confirmed and (value["status"] != "confirmed" or value["ambiguous"] or not value["english_confirmed"]
                      or (not pairs and not value["no_defined_theme"])):
        raise ValueError("Annotation is not confirmed English human gold")


def timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("Timezone-aware timestamp required")
    return parsed


def signature(value: dict) -> dict:
    return {"labels": sorted(tuple(x) for x in value["labels"]),
            **{f: value[f] for f in FLAGS}, "status": value["status"]}


def prepare_blind(data: Path, output: Path, *, group: str = "all", seed: int = 20261111,
                  mode: str = "independent", delay_hours: float = 24.0) -> dict:
    if mode not in {"independent", "delayed_same_reviewer"} or delay_hours < 0:
        raise ValueError("Invalid review mode or delay")
    rows, taxonomy = read_dataset(data)
    first = read_json(data / "human_labels/annotations.json")
    ids = scope_ids(rows, group)
    if not ids:
        raise ValueError("Empty scope")
    for identity in ids:
        if identity not in first:
            raise ValueError("Scope has unlabelled reviews; complete the first round first")
        validate_label(first[identity], rows[identity], taxonomy)
    count = math.ceil(.2 * len(ids))
    selected = sorted(ids, key=lambda i: canonical([seed, group, i]))[:count]
    new_directory(output)
    # Identical row identity/split, but no labels, ratings, or first-round notes.
    queue = [{"request_id": i, "text": rows[i]["text"], "app": rows[i]["app"],
              "split": rows[i]["split"], "pilot": True} for i in selected]
    write_jsonl(output / "annotation_reviews.jsonl", queue)
    write_json(output / "annotation_taxonomy.json", taxonomy)
    write_json(output / "first_round_private.json", {i: first[i] for i in selected})
    manifest = {"schema": "blind_review_queue_v1", "group": group, "scope_ids": ids,
                "scope_count": len(ids), "sample_count": count, "sample_fraction": .2,
                "sample_ids": selected, "seed": seed,
                "sampling": "smallest canonical SHA256(seed,group,id); independent of labels/errors",
                "mode": mode, "minimum_delay_hours": delay_hours if mode == "delayed_same_reviewer" else 0,
                "source_dataset_sha256": sha(data / "annotation_reviews.jsonl"),
                "source_taxonomy_sha256": sha(data / "annotation_taxonomy.json"),
                "primary_scope_hashes": {i: canonical(first[i]) for i in ids},
                "annotation_reviews_sha256": sha(output / "annotation_reviews.jsonl"),
                "annotation_taxonomy_sha256": sha(output / "annotation_taxonomy.json"),
                "private_snapshot_sha256": sha(output / "first_round_private.json"),
                "blinding": "Separate empty human_labels store. UI exposes no primary labels/notes or ratings. Private snapshot is not served.",
                "created_at": datetime.now(timezone.utc).isoformat()}
    write_json(output / "blind_review_manifest.json", manifest)
    (output / "human_labels").mkdir()
    return {k: manifest[k] for k in ("group", "scope_count", "sample_count", "mode")}


def load_queue(queue: Path) -> tuple[dict, dict, dict, list[dict]]:
    manifest = read_json(queue / "blind_review_manifest.json")
    if (sha(queue / "annotation_reviews.jsonl") != manifest["annotation_reviews_sha256"]
            or sha(queue / "first_round_private.json") != manifest["private_snapshot_sha256"]
            or sha(queue / "annotation_taxonomy.json") != manifest["annotation_taxonomy_sha256"]):
        raise ValueError("Blind queue or primary snapshot changed")
    rows, taxonomy = read_dataset(queue)
    if set(rows) != set(manifest["sample_ids"]):
        raise ValueError("Blind sample mismatch")
    return manifest, rows, read_json(queue / "first_round_private.json"), taxonomy


def load_adjudication(data: Path, *, expected_manifest_sha256: str | None = None) -> dict:
    """Check the exact frozen comment, rubric and human references shown by the UI."""
    manifest_path = data / "adjudication_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Missing frozen adjudication manifest; prepare a new comparison")
    if expected_manifest_sha256 is None:
        comparison_path = data.parent / "comparison.json"
        if not comparison_path.is_file():
            raise ValueError("Keep the adjudication queue with its frozen parent comparison")
        expected_manifest_sha256 = read_json(comparison_path).get("adjudication_manifest_sha256")
        if not isinstance(expected_manifest_sha256, str):
            raise ValueError("Parent comparison lacks an adjudication version binding")
    if sha(manifest_path) != expected_manifest_sha256:
        raise ValueError("Adjudication manifest changed")
    manifest = read_json(manifest_path)
    if manifest.get("schema") != "human_adjudication_queue_v1" or set(manifest.get("files", {})) != set(ADJUDICATION_FILES):
        raise ValueError("Invalid adjudication manifest")
    for name in ADJUDICATION_FILES:
        if not (data / name).is_file() or sha(data / name) != manifest["files"][name]:
            raise ValueError(f"Frozen adjudication input changed: {name}")
    rows, taxonomy = read_dataset(data)
    references = read_json(data / "references.json")
    if set(references) != set(rows) or set(rows) != set(manifest["request_ids"]):
        raise ValueError("Adjudication reference IDs do not match queue")
    for identity, reference in references.items():
        if set(reference) != {"first", "second"}:
            raise ValueError("Invalid adjudication references")
        validate_label(reference["first"], rows[identity], taxonomy)
        validate_label(reference["second"], rows[identity], taxonomy, confirmed=False)
    return manifest


def validate_decision_bindings(decisions_path: Path, decisions: dict, manifest_sha256: str) -> None:
    """Reject copied/edited decisions not bound to the UI's frozen input version."""
    path = decisions_path.parent / "adjudication_bindings.json"
    if not path.is_file():
        raise ValueError("Missing adjudication decision bindings; save decisions in the adjudication UI")
    bindings = read_json(path)
    if (bindings.get("schema") != "human_adjudication_bindings_v1"
            or bindings.get("adjudication_manifest_sha256") != manifest_sha256
            or bindings.get("decisions") != {i: canonical(value) for i, value in decisions.items()}):
        raise ValueError("Adjudication decision binding mismatch")


def bind_decisions(decisions_path: Path, decisions: dict, manifest_sha256: str) -> None:
    """Publish a fail-closed sidecar without changing the main annotation store API.

    The annotation store has already saved its append-only revision history. If
    this separate atomic write fails, release rejects the unmatched decision.
    """
    path = decisions_path.parent / "adjudication_bindings.json"
    temporary = path.with_suffix(".tmp")
    write_json(temporary, {"schema": "human_adjudication_bindings_v1",
                          "adjudication_manifest_sha256": manifest_sha256,
                          "decisions": {i: canonical(value) for i, value in decisions.items()}})
    os.replace(temporary, path)


def read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_primary_history(history: list[dict], annotations: dict, rows: dict,
                             taxonomy: list[dict]) -> None:
    """Check complete per-item revision chains against a frozen primary snapshot."""
    latest = {}
    for event in history:
        current = event["current"]
        identity = current.get("request_id")
        if identity not in annotations:
            raise ValueError("Primary history contains an ID outside its frozen scope")
        validate_label(current, rows[identity], taxonomy, confirmed=False)
        previous = latest.get(identity)
        if event.get("previous") != previous or current["revision"] != (previous["revision"] + 1 if previous else 1):
            raise ValueError("Primary annotation history has a broken revision chain")
        latest[identity] = current
    if latest != annotations:
        raise ValueError("Primary annotation history does not reproduce the frozen labels")


def split_artifact_names(split: str) -> tuple[str, ...]:
    if split not in {"adaptation", "calibration", "locked"}:
        raise ValueError("Unknown reviewed split")
    return (f"labels_{split}.jsonl", f"replay_{split}.jsonl", f"primary_{split}.json",
            f"primary_history_{split}.jsonl", f"resolved_{split}.json",
            f"corrections_{split}.jsonl", f"decisions_{split}.json")


def compare_blind(queue: Path, output: Path) -> dict:
    manifest, rows, first, taxonomy = load_queue(queue)
    second = read_json(queue / "human_labels/annotations.json")
    missing = set(rows) - set(second)
    if missing or set(second) - set(rows):
        raise ValueError("Second-round annotations are incomplete or contain unknown IDs")
    disagreements, agreements = [], []
    for identity in manifest["sample_ids"]:
        old, new = first[identity], second[identity]
        validate_label(new, rows[identity], taxonomy, confirmed=False)
        same_reviewer = old["reviewer"].strip().casefold() == new["reviewer"].strip().casefold()
        if manifest["mode"] == "independent" and same_reviewer:
            raise ValueError("Independent review cannot reuse the same reviewer identity")
        if manifest["mode"] == "delayed_same_reviewer":
            if not same_reviewer:
                raise ValueError("Review mode says same reviewer, but identities differ")
            hours = (timestamp(new["saved_at"]) - timestamp(old["saved_at"])).total_seconds() / 3600
            if hours < manifest["minimum_delay_hours"]:
                raise ValueError("Required delayed-review interval not met")
        if signature(old) == signature(new) and new["status"] == "confirmed":
            agreements.append(identity)
        else:
            disagreements.append({"request_id": identity, "text": rows[identity]["text"],
                                  "split": rows[identity]["split"], "first": old, "second": new})
    new_directory(output)
    write_jsonl(output / "adjudication_tasks.jsonl", disagreements)
    # Deliberately blank decisions, never copied from either annotator.
    template = {d["request_id"]: {"request_id": d["request_id"], "labels": [],
                "reviewer": "", "reason": "", "status": "pending", "no_defined_theme": False,
                "ambiguous": False, "out_of_scope": False, "incomplete_text": False,
                "english_confirmed": False} for d in disagreements}
    write_json(output / "decisions_template.json", template)
    adjudication = output / "adjudication"
    adjudication.mkdir()
    write_jsonl(adjudication / "annotation_reviews.jsonl", [
        {"request_id": d["request_id"], "text": d["text"], "app": rows[d["request_id"]]["app"],
         "split": d["split"], "pilot": True} for d in disagreements])
    write_json(adjudication / "annotation_taxonomy.json", taxonomy)
    write_json(adjudication / "references.json", {d["request_id"]: {"first": d["first"], "second": d["second"]}
                                                 for d in disagreements})
    (adjudication / "human_labels").mkdir()
    adjudication_manifest = {"schema": "human_adjudication_queue_v1",
        "request_ids": [d["request_id"] for d in disagreements],
        "files": {name: sha(adjudication / name) for name in ADJUDICATION_FILES},
        "tasks_sha256": sha(output / "adjudication_tasks.jsonl"),
        "queue_manifest_sha256": sha(queue / "blind_review_manifest.json"),
        "second_annotations_sha256": sha(queue / "human_labels/annotations.json")}
    write_json(adjudication / "adjudication_manifest.json", adjudication_manifest)
    comparison = {"schema": "blind_review_comparison_v1", "queue_manifest": manifest,
                  "queue_manifest_sha256": sha(queue / "blind_review_manifest.json"),
                  "second_annotations_sha256": sha(queue / "human_labels/annotations.json"),
                  "agreement_ids": agreements, "disagreement_ids": [x["request_id"] for x in disagreements],
                  "sample_count": len(rows), "agreement_count": len(agreements),
                  "agreement_fraction": len(agreements) / len(rows),
                  "review_mode": manifest["mode"], "tasks_sha256": sha(output / "adjudication_tasks.jsonl"),
                  "adjudication_manifest_sha256": sha(adjudication / "adjudication_manifest.json")}
    write_json(output / "comparison.json", comparison)
    return {k: comparison[k] for k in ("sample_count", "agreement_count", "agreement_fraction", "review_mode")}


def release_labels(data: Path, comparison_dir: Path, decisions_path: Path, output: Path) -> dict:
    rows, taxonomy = read_dataset(data)
    comparison = read_json(comparison_dir / "comparison.json")
    manifest = comparison["queue_manifest"]
    sampled = set(manifest["sample_ids"])
    agree, disagree = set(comparison["agreement_ids"]), set(comparison["disagreement_ids"])
    if (agree & disagree or agree | disagree != sampled or not sampled.issubset(manifest["scope_ids"])
            or comparison["sample_count"] != len(sampled)
            or comparison["agreement_count"] != len(agree)
            or set(manifest["primary_scope_hashes"]) != set(manifest["scope_ids"])):
        raise ValueError("Comparison sample/scope accounting is inconsistent")
    if sha(data / "annotation_reviews.jsonl") != manifest["source_dataset_sha256"] or sha(data / "annotation_taxonomy.json") != manifest["source_taxonomy_sha256"]:
        raise ValueError("Source dataset/taxonomy changed after blind queue construction")
    if sha(comparison_dir / "adjudication_tasks.jsonl") != comparison["tasks_sha256"]:
        raise ValueError("Adjudication tasks changed")
    adjudication_sha = comparison.get("adjudication_manifest_sha256")
    if not isinstance(adjudication_sha, str):
        raise ValueError("Comparison lacks frozen adjudication binding; prepare a new comparison")
    adjudication_manifest = load_adjudication(comparison_dir / "adjudication",
                                             expected_manifest_sha256=adjudication_sha)
    if (set(adjudication_manifest["request_ids"]) != disagree
            or any(adjudication_manifest[key] != comparison[key] for key in
                   ("tasks_sha256", "queue_manifest_sha256", "second_annotations_sha256"))):
        raise ValueError("Adjudication inputs do not belong to this comparison")
    first = read_json(data / "human_labels/annotations.json")
    for identity in manifest["scope_ids"]:
        if identity not in first or canonical(first[identity]) != manifest["primary_scope_hashes"][identity]:
            raise ValueError("Primary annotation changed; rebuild review for this version")
        validate_label(first[identity], rows[identity], taxonomy)
    history_path = data / "human_labels/annotation_history.jsonl"
    if not history_path.is_file():
        raise ValueError("Primary annotation revision history is required for a reviewed release")
    primary = {i: first[i] for i in manifest["scope_ids"]}
    history = [event for event in read_lines(history_path)
               if event["current"]["request_id"] in primary]
    validate_primary_history(history, primary, rows, taxonomy)
    decisions = read_json(decisions_path)
    if set(decisions) != set(comparison["disagreement_ids"]):
        raise ValueError("One human adjudication decision is required for every disagreement")
    if decisions:
        validate_decision_bindings(decisions_path, decisions, adjudication_sha)
    resolved = {i: dict(first[i]) for i in manifest["scope_ids"]}
    changes = []
    for identity, decision in decisions.items():
        if decision.get("source", "human_annotation") != "human_annotation":
            raise ValueError("Adjudication must be human, not model output")
        reason = decision.get("reason", decision.get("note", ""))
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Human adjudication reason required")
        if decision.get("status") != "confirmed":
            raise ValueError("Unresolved adjudication cannot enter a release")
        value = dict(decision, reason=reason, source="human_annotation", split=rows[identity]["split"],
                     revision=first[identity]["revision"] + 1, saved_at=datetime.now(timezone.utc).isoformat())
        validate_label(value, rows[identity], taxonomy)
        value["adjudication"] = {"decision_file_sha256": sha(decisions_path), "reason": reason,
                                  "previous_revision": first[identity]["revision"],
                                  "adjudication_manifest_sha256": adjudication_sha,
                                  "decision_sha256": canonical(decision)}
        resolved[identity] = value
        changes.append({"request_id": identity, "split": rows[identity]["split"],
                        "previous": first[identity], "resolved": value,
                        "labels_changed": signature(first[identity]) != signature(value)})
    new_directory(output)
    write_json(output / "resolved_annotations.json", resolved)
    write_jsonl(output / "human_corrections.jsonl", changes)
    # Immutable, split-specific private evidence permits calibration validation
    # without opening the locked split's labels or review history.
    (output / "review_comparison.json").write_bytes((comparison_dir / "comparison.json").read_bytes())
    (output / "review_adjudication_manifest.json").write_bytes(
        (comparison_dir / "adjudication/adjudication_manifest.json").read_bytes())
    bindings = read_json(decisions_path.parent / "adjudication_bindings.json") if decisions else {
        "schema": "human_adjudication_bindings_v1", "adjudication_manifest_sha256": adjudication_sha,
        "decisions": {}}
    write_json(output / "review_decision_bindings.json", bindings)
    split_artifacts = {}
    for split in ("adaptation", "calibration", "locked"):
        identities = [i for i in manifest["scope_ids"] if rows[i]["split"] == split]
        write_jsonl(output / f"labels_{split}.jsonl", [
            {"request_id": i, "pairs": [{"aspect": a, "sentiment": s} for a, s in resolved[i]["labels"]],
             "aspects": sorted({a for a, _ in resolved[i]["labels"]}), "source": "human_annotation",
             "revision": resolved[i]["revision"]} for i in identities])
        write_jsonl(output / f"replay_{split}.jsonl", [{"request_id": i, "text": rows[i]["text"]} for i in identities])
        write_json(output / f"primary_{split}.json", {i: primary[i] for i in identities})
        write_jsonl(output / f"primary_history_{split}.jsonl", [event for event in history
                    if event["current"]["request_id"] in identities])
        write_json(output / f"resolved_{split}.json", {i: resolved[i] for i in identities})
        write_jsonl(output / f"corrections_{split}.jsonl", [item for item in changes if item["request_id"] in identities])
        write_json(output / f"decisions_{split}.json", {i: decisions[i] for i in identities if i in decisions})
        split_artifacts[split] = {name: sha(output / name) for name in split_artifact_names(split)}
    release = {"schema": "reviewed_human_label_release_v1", "source_dataset_sha256": manifest["source_dataset_sha256"],
               "integrity_schema": "frozen_population_and_review_history_v1",
               "source_snapshot_manifest_sha256": sha(data / "annotation_manifest.json"),
               "source_taxonomy_sha256": manifest["source_taxonomy_sha256"],
               "comparison_sha256": sha(comparison_dir / "comparison.json"),
               "adjudication_manifest_sha256": adjudication_sha,
               "decision_bindings_sha256": sha(decisions_path.parent / "adjudication_bindings.json") if decisions else None,
               "decisions_sha256": sha(decisions_path), "resolved_sha256": sha(output / "resolved_annotations.json"),
               "human_corrections_sha256": sha(output / "human_corrections.jsonl"),
               "split_artifacts": split_artifacts,
               "review_evidence": {name: sha(output / name) for name in (
                   "review_comparison.json", "review_adjudication_manifest.json", "review_decision_bindings.json")},
               "scope_ids": manifest["scope_ids"], "scope_group": manifest["group"],
               "scope_counts": dict(Counter(rows[i]["split"] for i in manifest["scope_ids"])),
               "blind_sample_count": comparison["sample_count"], "agreement_count": comparison["agreement_count"],
               "adjudication_count": len(decisions), "review_mode": manifest["mode"],
               "primary_store_modified": False}
    write_json(output / "release_manifest.json", release)
    return {k: release[k] for k in ("scope_counts", "blind_sample_count", "agreement_count", "adjudication_count", "review_mode")}


def load_reviewed_split(data: Path, release_dir: Path, split: str, *,
                        require_complete: bool = True) -> dict:
    """Verify one reviewed population without opening another split's gold.

    Reads only immutable release snapshots and the original label-free dataset,
    never the live first-round store. The returned rows/annotations are private.
    Consumers should publish only counts and the provenance hashes.
    """
    names = split_artifact_names(split)
    rows, taxonomy = read_dataset(data)
    snapshot_path = data / "annotation_manifest.json"
    snapshot = read_json(snapshot_path)
    actual_counts = dict(Counter(row["split"] for row in rows.values()))
    if snapshot.get("assignment", {}).get("counts") != actual_counts:
        raise ValueError("Frozen snapshot assignment counts do not match its registered population")
    manifest = read_json(release_dir / "release_manifest.json")
    if (manifest.get("schema") != "reviewed_human_label_release_v1"
            or manifest.get("integrity_schema") != "frozen_population_and_review_history_v1"):
        raise ValueError("Reviewed release lacks population/history integrity evidence; create a new version")
    for name, key in (("annotation_manifest.json", "source_snapshot_manifest_sha256"),
                      ("annotation_reviews.jsonl", "source_dataset_sha256"),
                      ("annotation_taxonomy.json", "source_taxonomy_sha256")):
        if sha(data / name) != manifest.get(key):
            raise ValueError(f"Frozen source version mismatch: {name}")
    evidence_names = {"review_comparison.json", "review_adjudication_manifest.json", "review_decision_bindings.json"}
    evidence = manifest.get("review_evidence", {})
    if set(evidence) != evidence_names:
        raise ValueError("Incomplete frozen review evidence")
    for name in evidence_names:
        if sha(release_dir / name) != evidence[name]:
            raise ValueError(f"Frozen review evidence changed: {name}")
    comparison = read_json(release_dir / "review_comparison.json")
    adjudication = read_json(release_dir / "review_adjudication_manifest.json")
    bindings = read_json(release_dir / "review_decision_bindings.json")
    queue = comparison["queue_manifest"]
    scope = set(manifest["scope_ids"])
    if (len(scope) != len(manifest["scope_ids"]) or scope != set(queue["scope_ids"])
            or scope != set(scope_ids(rows, queue["group"]))
            or not scope.issubset(rows)
            or manifest["scope_counts"] != dict(Counter(rows[i]["split"] for i in scope))
            or queue["source_dataset_sha256"] != manifest["source_dataset_sha256"]
            or queue["source_taxonomy_sha256"] != manifest["source_taxonomy_sha256"]
            or sha(release_dir / "review_comparison.json") != manifest["comparison_sha256"]):
        raise ValueError("Reviewed scope does not match the original frozen snapshot")
    sampled = set(queue["sample_ids"])
    expected_sample = set(sorted(scope, key=lambda i: canonical([queue["seed"], queue["group"], i]))[:math.ceil(.2 * len(scope))])
    agree, disagree = set(comparison["agreement_ids"]), set(comparison["disagreement_ids"])
    if (sampled != expected_sample or agree & disagree or agree | disagree != sampled
            or comparison["sample_count"] != len(sampled) or comparison["agreement_count"] != len(agree)
            or manifest["blind_sample_count"] != len(sampled) or manifest["adjudication_count"] != len(disagree)
            or manifest["agreement_count"] != len(agree)
            or manifest["review_mode"] != comparison["review_mode"]
            or manifest["review_mode"] != queue["mode"]
            or queue["mode"] not in {"independent", "delayed_same_reviewer"}
            or set(adjudication["request_ids"]) != disagree
            or set(bindings.get("decisions", {})) != disagree
            or bindings.get("schema") != "human_adjudication_bindings_v1"
            or bindings.get("adjudication_manifest_sha256") != manifest["adjudication_manifest_sha256"]
            or sha(release_dir / "review_adjudication_manifest.json") != manifest["adjudication_manifest_sha256"]
            or comparison["adjudication_manifest_sha256"] != manifest["adjudication_manifest_sha256"]
            or any(adjudication[key] != comparison[key] for key in
                   ("tasks_sha256", "queue_manifest_sha256", "second_annotations_sha256"))):
        raise ValueError("Reviewed release has inconsistent blind/adjudication history")
    population = {i for i, row in rows.items() if row["split"] == split}
    identities = population & scope
    if require_complete and identities != population:
        raise ValueError(f"Reviewed release does not cover the complete registered {split} population")
    files = manifest.get("split_artifacts", {}).get(split, {})
    if set(files) != set(names):
        raise ValueError(f"Missing frozen {split} label/replay/history artifacts")
    for name in names:
        if sha(release_dir / name) != files[name]:
            raise ValueError(f"Frozen reviewed split artifact changed: {name}")
    primary = read_json(release_dir / f"primary_{split}.json")
    resolved = read_json(release_dir / f"resolved_{split}.json")
    decisions = read_json(release_dir / f"decisions_{split}.json")
    corrections_list = read_lines(release_dir / f"corrections_{split}.jsonl")
    corrections = {item["request_id"]: item for item in corrections_list}
    if (set(primary) != identities or set(resolved) != identities
            or set(decisions) != disagree & identities or set(corrections) != set(decisions)
            or len(corrections) != len(corrections_list)):
        raise ValueError("Reviewed split/history identity coverage differs from the frozen population")
    validate_primary_history(read_lines(release_dir / f"primary_history_{split}.jsonl"), primary, rows, taxonomy)
    for identity in identities:
        validate_label(primary[identity], rows[identity], taxonomy)
        validate_label(resolved[identity], rows[identity], taxonomy)
        if canonical(primary[identity]) != queue["primary_scope_hashes"][identity]:
            raise ValueError("Primary snapshot differs from the blind-review version")
        if identity not in decisions:
            if resolved[identity] != primary[identity]:
                raise ValueError("Unadjudicated label changed after blind review")
            continue
        decision, correction, final = decisions[identity], corrections[identity], resolved[identity]
        validate_label(decision, rows[identity], taxonomy)
        reason = decision.get("reason", decision.get("note", ""))
        if (not isinstance(reason, str) or not reason.strip()
                or canonical(decision) != bindings["decisions"][identity]
                or correction["previous"] != primary[identity] or correction["resolved"] != final
                or final["revision"] != primary[identity]["revision"] + 1
                or final.get("reason") != reason
                or final.get("adjudication", {}).get("decision_sha256") != canonical(decision)
                or final["adjudication"].get("adjudication_manifest_sha256") != manifest["adjudication_manifest_sha256"]
                or any(final[field] != decision[field] for field in (*FLAGS, "labels", "status", "reviewer"))):
            raise ValueError("Resolved labels do not reproduce the bound human adjudication")
    labels = read_lines(release_dir / f"labels_{split}.jsonl")
    replay = read_lines(release_dir / f"replay_{split}.jsonl")
    label_map = {row["request_id"]: row for row in labels}
    replay_map = {row["request_id"]: row for row in replay}
    if (set(label_map) != identities or len(label_map) != len(labels)
            or set(replay_map) != identities or len(replay_map) != len(replay)):
        raise ValueError("Gold/replay coverage must equal the full registered reviewed split")
    for identity in identities:
        expected = {"request_id": identity,
                    "pairs": [{"aspect": a, "sentiment": s} for a, s in resolved[identity]["labels"]],
                    "aspects": sorted({a for a, _ in resolved[identity]["labels"]}),
                    "source": "human_annotation", "revision": resolved[identity]["revision"]}
        if label_map[identity] != expected or replay_map[identity] != {"request_id": identity, "text": rows[identity]["text"]}:
            raise ValueError("Gold/replay contents differ from the frozen reviewed source")
    return {"rows": rows, "taxonomy": taxonomy, "annotations": resolved, "labels": labels,
            "ids": identities, "registered_count": len(population), "reviewed_count": len(identities),
            "provenance": {"release_manifest_sha256": sha(release_dir / "release_manifest.json"),
                "source_snapshot_manifest_sha256": sha(snapshot_path),
                "source_dataset_sha256": manifest["source_dataset_sha256"],
                "source_taxonomy_sha256": manifest["source_taxonomy_sha256"],
                "comparison_sha256": manifest["comparison_sha256"], "split": split,
                "registered_count": len(population), "reviewed_count": len(identities),
                "review_mode": manifest["review_mode"], "blind_sample_count_in_split": len(sampled & identities),
                "agreement_count_in_split": len(agree & identities),
                "adjudication_count_in_split": len(disagree & identities), "files": files}}


def export_feedback(data: Path, release_dir: Path, output: Path) -> dict:
    """Export adaptation-only candidates; explicitly retain their actual origin.

    Returned candidates match the five-field serving demonstration schema, but
    are a separate proposed feedback pool. This does not modify the live index.
    """
    verified = load_reviewed_split(data, release_dir, "adaptation", require_complete=False)
    rows, taxonomy, annotations = verified["rows"], verified["taxonomy"], verified["annotations"]
    manifest = read_json(release_dir / "release_manifest.json")
    candidates, skipped, provenance = [], Counter(), {}
    non_adaptation = sum(rows[identity]["split"] != "adaptation" for identity in manifest["scope_ids"])
    if non_adaptation:
        skipped["non_adaptation"] = non_adaptation
    for identity, annotation in annotations.items():
        if identity not in rows:
            raise ValueError("Unknown review ID")
        validate_label(annotation, rows[identity], taxonomy)
        # Never trust a split supplied in an annotation file to grant eligibility.
        if rows[identity]["split"] != "adaptation":
            raise ValueError("Non-adaptation ID entered the verified feedback population")
        if HELDOUT in {a for a, _ in annotation["labels"]}:
            skipped["heldout_review"] += 1
            continue
        if annotation["out_of_scope"] or annotation["incomplete_text"]:
            skipped["scope_or_incomplete"] += 1
            continue
        candidates.append(dict(rows[identity]))
        provenance[identity] = {"original_split": "adaptation", "revision": annotation["revision"],
                                "label_source": "human_annotation", "release_sha256": sha(release_dir / "release_manifest.json")}
    excluded = [r["text"] for r in rows.values() if r["split"] != "adaptation"]
    accepted, duplicate_audit = deduplicate(candidates, excluded)
    pool = []
    for row in accepted:
        identity = row["request_id"]
        aspects = {a for a, _ in annotations[identity]["labels"]}
        for candidate in taxonomy:
            if candidate["aspect"] == HELDOUT:
                continue
            pool.append({"row_uid": identity, "candidate_aspect": candidate["aspect"],
                         "review_text": row["text"], "aspect_candidate": candidate["definition"],
                         "answer": "Y" if candidate["aspect"] in aspects else "N"})
    new_directory(output)
    write_jsonl(output / "feedback_pool.jsonl", pool)
    accepted_ids = {r["request_id"] for r in accepted}
    write_json(output / "provenance.json", {i: provenance[i] for i in sorted(accepted_ids)})
    result = {"schema": "human_adaptation_feedback_candidates_v1", "strict_unseen": True,
              "heldout_aspect": HELDOUT, "review_count": len(accepted), "pair_count": len(pool),
              "skipped": dict(skipped), "duplicate_audit": duplicate_audit,
              "source_dataset_sha256": manifest["source_dataset_sha256"],
              "release_manifest_sha256": sha(release_dir / "release_manifest.json"),
              "feedback_pool_sha256": sha(output / "feedback_pool.jsonl"),
              "actual_origin": "new-source human-labelled adaptation partition; not official train",
              "calibration_and_locked_included": False, "live_index_modified": False}
    write_json(output / "feedback_manifest.json", result)
    return {k: result[k] for k in ("review_count", "pair_count", "skipped", "strict_unseen", "live_index_modified")}
