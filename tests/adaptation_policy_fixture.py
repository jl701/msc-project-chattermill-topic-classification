"""Synthetic-only policy builder; never access campaign inputs or annotations."""
from collections import Counter
from pathlib import Path

from msc_project.serving.adaptation_use import AdaptationPolicy, PURPOSES, REASON
from msc_project.serving.annotation_workflow import read_dataset, sha, write_json
from msc_project.serving import retrieval
from msc_project.serving.review_data import write_jsonl


def synthetic_policy(data):
    root = data.parent
    rows, _ = read_dataset(data)
    write_jsonl(data / "replay_new_inputs.jsonl", [
        {"request_id": i, "text": row["text"]} for i, row in rows.items()])
    write_json(data / "replay_new_inputs_manifest.json", {
        "sha256": sha(data / "replay_new_inputs.jsonl")})
    write_json(root / "input_population_audit_supplement.json", {"synthetic": True})
    (root / "retrieval.py").write_bytes(Path(retrieval.__file__).read_bytes())
    records = []
    for identity, row in rows.items():
        normal = retrieval.normalise_text(row["text"])
        if row["split"] != "adaptation" or not normal:
            continue
        matches = {split: sum(r["split"] == split and retrieval.normalise_text(r["text"]) == normal
                              for r in rows.values()) for split in ("calibration", "locked")}
        if any(matches.values()):
            records.append({"request_id": identity, "original_split": "adaptation",
                "reason_code": REASON, "protected_split_match_counts": matches,
                "blocked_uses": list(PURPOSES), "annotation_and_blind_rereview_allowed": True})
    paths = [data / name for name in ("annotation_reviews.jsonl", "annotation_manifest.json",
        "replay_new_inputs.jsonl", "replay_new_inputs_manifest.json")]
    paths += [root / "retrieval.py", root / "input_population_audit_supplement.json"]
    path = root / "synthetic_policy.json"
    write_json(path, {"schema": "adaptation_purpose_exclusion_policy_v1",
        "source_sha256": {p.relative_to(root).as_posix(): sha(p) for p in paths},
        "registered_population_counts": dict(Counter(r["split"] for r in rows.values())),
        "excluded_adaptation_records": records, "excluded_record_count": len(records),
        "empty_normal_forms": {"excluded_by_this_policy": False, "merge_on_empty_form": False}})
    return AdaptationPolicy(path, root, sha(path))
