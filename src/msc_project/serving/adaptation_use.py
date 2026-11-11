"""Source-bound eligibility for future adaptation use, never a label-release filter."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re

from msc_project.serving.annotation_workflow import (
    load_reviewed_split, new_directory, read_dataset, read_json, read_lines, sha, write_json,
)
from msc_project.serving import retrieval
from msc_project.serving.review_data import write_jsonl

PURPOSES = (
    "manual_error_driven_tuning", "model_or_prompt_improvement", "training_or_fitting",
    "demonstration_feedback_or_example_selection",
)
REASON = "nonempty_retrieval_word_normal_form_shared_with_protected_split"
REGISTRATION = Path(__file__).resolve().parents[3] / "configs/experiments/adaptation_use_20261111.json"
SOURCE_NAMES = {
    "annotation_reviews.jsonl", "annotation_manifest.json", "replay_new_inputs.jsonl",
    "replay_new_inputs_manifest.json", "retrieval.py", "input_population_audit_supplement.json",
}


@dataclass(frozen=True)
class AdaptationPolicy:
    path: Path
    source_root: Path
    expected_sha256: str


def validate_policy(data: Path, policy: AdaptationPolicy, *, purpose: str) -> tuple[set[str], dict]:
    """Use only label-free inputs; fail before any reviewed gold is opened."""
    if purpose not in PURPOSES:
        raise ValueError("Unknown adaptation-use purpose")
    if not isinstance(policy, AdaptationPolicy):
        raise ValueError("Explicit adaptation-use policy is required")
    if (not re.fullmatch(r"[0-9a-f]{64}", policy.expected_sha256)
            or sha(policy.path) != policy.expected_sha256):
        raise ValueError("Adaptation-use policy hash mismatch")
    value = read_json(policy.path)
    if value.get("schema") != "adaptation_purpose_exclusion_policy_v1":
        raise ValueError("Unknown adaptation-use policy schema")
    source_hash = sha(data / "annotation_reviews.jsonl")
    registration = read_json(REGISTRATION)
    pinned = registration["policies_by_source_sha256"].get(source_hash)
    if pinned is not None and pinned != policy.expected_sha256:
        raise ValueError("Policy differs from the registered campaign source policy")
    bindings = value.get("source_sha256", {})
    root = policy.source_root.resolve()
    paths = {}
    for name, expected in bindings.items():
        relative = Path(name)
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root) or path.name in paths:
            raise ValueError("Invalid policy source path")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Invalid policy source hash")
        if not path.is_file() or sha(path) != expected:
            raise ValueError("Policy source hash mismatch")
        paths[path.name] = path
    if set(paths) != SOURCE_NAMES:
        raise ValueError("Incomplete policy source bindings")
    for name in ("annotation_reviews.jsonl", "annotation_manifest.json", "replay_new_inputs.jsonl",
                 "replay_new_inputs_manifest.json"):
        if paths[name] != (data / name).resolve():
            raise ValueError("Policy is bound to a different annotation dataset")
    if sha(paths["retrieval.py"]) != sha(Path(retrieval.__file__)):
        raise ValueError("Active retrieval normalisation differs from the registered code")
    rows, _ = read_dataset(data)
    counts = dict(Counter(row["split"] for row in rows.values()))
    if (value.get("registered_population_counts") != counts
            or read_json(data / "annotation_manifest.json").get("assignment", {}).get("counts") != counts):
        raise ValueError("Policy population or assignment counts differ")
    # Empty forms are not a duplicate key: punctuation-only originals are retained.
    protected = Counter((row["split"], retrieval.normalise_text(row["text"]))
                        for row in rows.values() if row["split"] != "adaptation")
    matches = {}
    for identity, row in rows.items():
        normal = retrieval.normalise_text(row["text"])
        if row["split"] == "adaptation" and normal:
            overlap = {split: protected[split, normal] for split in ("calibration", "locked")}
            if any(overlap.values()):
                matches[identity] = overlap
    records = value.get("excluded_adaptation_records")
    if not isinstance(records, list) or value.get("excluded_record_count") != len(records):
        raise ValueError("Invalid policy exclusion accounting")
    excluded = set()
    for record in records:
        identity = record.get("request_id")
        if (identity not in rows or identity in excluded or rows[identity]["split"] != "adaptation"
                or record.get("original_split") != "adaptation"):
            raise ValueError("Invalid policy exclusion ID or split")
        if (record.get("reason_code") != REASON or identity not in matches
                or record.get("protected_split_match_counts") != matches[identity]
                or not isinstance(record.get("blocked_uses"), list)
                or set(record["blocked_uses"]) != set(PURPOSES)
                or record.get("annotation_and_blind_rereview_allowed") is not True):
            raise ValueError("Invalid policy reason, protected match or blocked purposes")
        excluded.add(identity)
    if excluded != set(matches):
        raise ValueError("Policy must cover every nonempty protected normal-form match")
    empty = value.get("empty_normal_forms", {})
    if empty.get("excluded_by_this_policy") is not False or empty.get("merge_on_empty_form") is not False:
        raise ValueError("Policy must preserve distinct empty-normal-form originals")
    return excluded, {
        "schema": "adaptation_use_audit_v1", "purpose": purpose,
        "policy_sha256": policy.expected_sha256, "source_dataset_sha256": source_hash,
        "source_sha256": bindings, "registered_population_counts": counts,
        "excluded_adaptation_count": len(excluded), "annotation_population_modified": False,
        "calibration_and_locked_population_modified": False,
        "eligibility_is_not_final_feedback_acceptance": True,
    }


def load_adaptation_use(data: Path, release: Path, *, purpose: str,
                        policy: AdaptationPolicy, require_complete: bool = True) -> dict:
    """Return only eligible private adaptation rows/labels for the named purpose.

    Full immutable release validation still checks excluded rows internally.
    The returned view exposes neither their labels nor protected-split inputs.
    """
    excluded, audit = validate_policy(data, policy, purpose=purpose)
    complete = load_reviewed_split(data, release, "adaptation", require_complete=require_complete)
    identities = complete["ids"] - excluded
    audit.update(reviewed_adaptation_count=complete["reviewed_count"],
                 excluded_reviewed_count=len(complete["ids"] & excluded), eligible_count=len(identities))
    return {"ids": identities, "rows": {i: complete["rows"][i] for i in sorted(identities)},
            "annotations": {i: complete["annotations"][i] for i in sorted(identities)},
            "labels": sorted((row for row in complete["labels"] if row["request_id"] in identities),
                             key=lambda row: row["request_id"]),
            "taxonomy": complete["taxonomy"], "provenance": complete["provenance"], "use_audit": audit}


def export_adaptation_use(data: Path, release: Path, output: Path, *, purpose: str,
                          policy: AdaptationPolicy, predictions: Path | None = None) -> dict:
    """Private eligible analysis/fitting inputs; no tuning or model execution."""
    view = load_adaptation_use(data, release, purpose=purpose, policy=policy)
    aligned = None
    if predictions is not None:
        records = read_lines(predictions)
        by_id = {row["request_id"]: row for row in records}
        rows, _ = read_dataset(data)
        population = {i for i, row in rows.items() if row["split"] == "adaptation"}
        if len(by_id) != len(records) or set(by_id) != population:
            raise ValueError("Predictions must cover exactly the complete adaptation population")
        aligned = [by_id[i] for i in sorted(view["ids"])]
    new_directory(output)
    write_jsonl(output / "replay.jsonl", [{"request_id": i, "text": row["text"]}
                                          for i, row in view["rows"].items()])
    write_jsonl(output / "labels.jsonl", view["labels"])
    if aligned is not None:
        write_jsonl(output / "predictions.jsonl", aligned)
    audit = {**view["use_audit"], "reviewed_release": view["provenance"],
             "prediction_source_sha256": sha(predictions) if predictions else None,
             "files": {p.name: sha(p) for p in output.iterdir() if p.is_file()}}
    write_json(output / "adaptation_use_manifest.json", audit)
    return audit


def add_policy_arguments(parser) -> None:
    parser.add_argument("--adaptation-use-policy", type=Path, required=True)
    parser.add_argument("--adaptation-use-policy-sha256", required=True)
    parser.add_argument("--policy-source-root", type=Path, required=True)


def policy_from_args(args) -> AdaptationPolicy:
    return AdaptationPolicy(args.adaptation_use_policy, args.policy_source_root,
                            args.adaptation_use_policy_sha256)
