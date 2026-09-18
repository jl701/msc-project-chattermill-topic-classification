"""Audit the rich-description development and validation evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msc_project.experiments.taxonomy_rich_description import (  # noqa: E402
    canonical_json_sha256,
)


METHODS = ("strict_train_only_tfidf", "e5_base_v2")
DEVELOPMENT_METHODS = (
    "tfidf_field_similarity",
    "e5_base_v2_field_similarity",
)
INTERFACES = {
    "D",
    "R1_concat",
    "R2_positive_concat",
    "R3_prototype_max",
    "R4_prototype_top2",
    "R5_contrastive_top2_l0.10",
    "R5_contrastive_top2_l0.25",
    "R5_contrastive_top2_l0.50",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _assert_finite(value: object, location: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"Non-finite value at {location}.")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{location}.{key}")


def _verify_summary_hash(summary: dict[str, Any], path: Path) -> None:
    observed = str(summary.get("summary_sha256", ""))
    payload = {
        key: value
        for key, value in summary.items()
        if key not in {"seconds", "summary_sha256"}
    }
    expected = canonical_json_sha256(payload)
    if observed != expected:
        raise ValueError(f"Summary hash mismatch: {path}")


def _audit_development(root: Path) -> dict[str, object]:
    summary_path = root / "summary.json"
    summary = _load(summary_path)
    _verify_summary_hash(summary, summary_path)
    if (
        summary.get("completed_pseudo_folds") != 72
        or summary.get("expected_pseudo_folds") != 72
        or summary.get("selection", {}).get("record_keys") != 72
        or summary.get("selection", {}).get("selected_interface")
        != "R2_positive_concat"
    ):
        raise ValueError("Development grid or selected interface is incomplete.")
    paths = sorted(
        path
        for path in root.rglob("*.json")
        if path.name not in {"summary.json", "resource_audit.json"}
    )
    if len(paths) != 72:
        raise ValueError(f"Expected 72 development fold artifacts, found {len(paths)}.")
    keys: set[tuple[str, str, int]] = set()
    for path in paths:
        value = _load(path)
        key = (
            str(value.get("method_id")),
            str(value.get("pseudo_unseen_aspect")),
            int(value.get("row_partition", -1)),
        )
        if key in keys:
            raise ValueError(f"Duplicate development fold identity: {key}")
        keys.add(key)
        if (
            key[0] not in DEVELOPMENT_METHODS
            or key[2] not in {0, 1, 2}
            or value.get("failure_count") != 0
            or value.get("non_finite_value_count") != 0
            or value.get("test_contract_count") != 0
            or {str(item.get("interface")) for item in value.get("records", [])}
            != INTERFACES
        ):
            raise ValueError(f"Invalid development artifact: {path}")
        _assert_finite(value, str(path))
    return {
        "pseudo_fold_count": len(paths),
        "record_count": sum(len(_load(path)["records"]) for path in paths),
        "selected_interface": "R2_positive_concat",
        "summary_sha256": summary["summary_sha256"],
    }


def _audit_confirmation(root: Path, baseline_root: Path) -> dict[str, object]:
    method_results: dict[str, object] = {}
    for method in METHODS:
        method_root = root / method
        summary_path = method_root / "summary.json"
        summary = _load(summary_path)
        _verify_summary_hash(summary, summary_path)
        paths = sorted((method_root / "folds").glob("l2-a*.json"))
        if (
            len(paths) != 12
            or summary.get("completed_folds") != 12
            or summary.get("expected_folds") != 12
            or summary.get("failure_count") != 0
            or summary.get("non_finite_value_count") != 0
            or summary.get("resume_conflict_count") != 0
            or summary.get("test_contract_count") != 0
        ):
            raise ValueError(f"Incomplete confirmation for {method}.")
        identities: set[str] = set()
        for path in paths:
            value = _load(path)
            fold_id = str(value.get("fold_id"))
            if fold_id in identities:
                raise ValueError(f"Duplicate confirmation fold: {fold_id}")
            identities.add(fold_id)
            if (
                set(value.get("conditions", {})) != {"D", "R"}
                or value.get("failure_count") != 0
                or value.get("non_finite_value_count") != 0
                or value.get("resume_conflict_count") != 0
                or value.get("test_contract_count") != 0
                or not value.get("seen_score_sha256")
                or not value.get("sentiment_score_sha256")
            ):
                raise ValueError(f"Invalid confirmation artifact: {path}")
            _assert_finite(value, str(path))

        prior = _load(baseline_root / method / "summary.json")
        prior_d = prior["aggregate"]["condition_macro_means"]["D"]
        observed_d = summary["aggregate"]["condition_macro_means"]["D"]
        exact_controls = {
            "presence_average_precision": "heldout_presence_ap",
            "presence_f1": "heldout_presence_f1",
            "heldout_pair_micro_f1": "heldout_pair_micro_f1",
            "overall_pair_micro_f1": "overall_pair_micro_f1",
        }
        for observed_key, prior_key in exact_controls.items():
            if observed_d[observed_key] != prior_d[prior_key]:
                raise ValueError(
                    f"D control mismatch for {method}: {observed_key}/{prior_key}."
                )
        method_results[method] = {
            "fold_count": 12,
            "D_exactly_matches_prior_control": True,
            "summary_sha256": summary["summary_sha256"],
            "R_minus_D": summary["aggregate"]["R_minus_D"],
        }
    return method_results


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--development-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experimental/taxonomy_rich_description_interface_study_v1",
    )
    parser.add_argument(
        "--confirmation-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experimental/taxonomy_rich_description_validation_confirmation_v1",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experimental/taxonomy_post_supervisor_local_v1/level2_local_ndr",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experimental/taxonomy_rich_description_validation_confirmation_v1/audit.json",
    )
    args = parser.parse_args()
    result: dict[str, object] = {
        "schema_version": "taxonomy_rich_description_evidence_audit_v1",
        "status": "pass",
        "development": _audit_development(args.development_root.resolve()),
        "confirmation": _audit_confirmation(
            args.confirmation_root.resolve(), args.baseline_root.resolve()
        ),
        "failure_count": 0,
        "non_finite_value_count": 0,
        "resume_conflict_count": 0,
        "test_contract_count": 0,
        "cloud_expansion_of_R": False,
    }
    result["audit_sha256"] = canonical_json_sha256(result)
    _write(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
