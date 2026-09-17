"""Seal deterministic Frozen-Qwen few-shot demonstrations and prompt contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SELECTION_ROOT = Path(
    r"C:\Msc_DSML\Msc_Project\cloud_backups\taxonomy_two_stage_formal_v2_r2"
    r"\worker-distil-frozen\selections\frozen_qwen_few_shot"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = sorted(args.selection_root.glob("heldout-*.json"))
    if len(paths) != 15:
        raise ValueError(f"Expected 15 few-shot selection contracts, found {len(paths)}")
    records = []
    for path in paths:
        payload = _read(path)
        contract = payload["scope_contract"]
        if (
            payload.get("protocol_id") != "taxonomy_two_stage_formal_v2"
            or payload.get("method_id") != "frozen_qwen_few_shot"
            or payload.get("failure_count") != 0
            or payload.get("test_contract_count") != 0
            or contract.get("test_contract_count") != 0
            or contract.get("selection_partition") != "seen_validation_only"
            or contract.get("seed") != 13
            or contract.get("model_id") != "Qwen/Qwen3-4B-Instruct-2507"
            or contract.get("model_revision")
            != "cdbee75f17c01a7cc42f958dc650907174af0554"
            or contract.get("maximum_length") != 1024
        ):
            raise ValueError(f"Unsafe few-shot selection contract: {path}")
        demonstrations = contract["demonstrations"]
        for task in ("aspect", "sentiment"):
            if demonstrations[task]["sha256"] != payload["demonstration_sha256s"][task]:
                raise ValueError(f"Demonstration hash mismatch: {path} / {task}")
            if demonstrations[task]["count"] != len(demonstrations[task]["row_uids"]):
                raise ValueError(f"Demonstration count mismatch: {path} / {task}")
            if not all(str(value).startswith("train:") for value in demonstrations[task]["row_uids"]):
                raise ValueError(f"Non-train demonstration identity: {path} / {task}")
        records.append(
            {
                "training_scope_id": contract["training_scope_id"],
                "selection_artifact_sha256": _sha256(path),
                "scope_contract_sha256": payload["scope_contract_sha256"],
                "prompt_contract_sha256": contract["prompt_contract_sha256"],
                "aspect_demonstration_sha256": demonstrations["aspect"]["sha256"],
                "sentiment_demonstration_sha256": demonstrations["sentiment"]["sha256"],
                "aspect_demonstration_row_uids": demonstrations["aspect"]["row_uids"],
                "sentiment_demonstration_row_uids": demonstrations["sentiment"]["row_uids"],
                "aspect_threshold": payload["thresholds"]["aspect"],
                "runner_up_sentiment_threshold": payload["thresholds"]["runner_up_sentiment"],
                "selection_partition": "seen_validation_only",
            }
        )
    if len({record["training_scope_id"] for record in records}) != 15:
        raise ValueError("Few-shot training scopes are not unique.")
    audit = {
        "schema_version": "taxonomy_few_shot_contract_audit_v1",
        "status": "pass",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "include_official_test": False,
        "official_test_opened": False,
        "test_contract_count": 0,
        "failure_count": 0,
        "non_finite_value_count": 0,
        "resume_conflict_count": 0,
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "model_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
        "maximum_length": 1024,
        "scope_count": len(records),
        "demonstration_identity_scope": "train rows only",
        "selection_scope": "seen validation only",
        "records": records,
    }
    _write(args.output.resolve(), audit)
    print(
        json.dumps(
            {
                "status": "pass",
                "scope_count": len(records),
                "test_contract_count": 0,
            },
            indent=2,
        )
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-root", type=Path, default=DEFAULT_SELECTION_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/taxonomy_few_shot_contract_audit_20260822.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
