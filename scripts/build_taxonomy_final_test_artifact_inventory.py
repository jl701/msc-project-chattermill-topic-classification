"""Build a public, review-text-free inventory for final-test artifact reuse.

The script reads only validation-era selections, checkpoint files and compact
result records.  It never opens a FABSA split.  Paths are stored relative to
logical roots so the inventory remains portable and does not expose a local
user profile or cloud credential.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msc_project.experiments.taxonomy_final_test import (
    L2_SCOPES,
    L4_SCOPES,
    PROTOCOL_ID,
    canonical_sha256,
    file_sha256,
    load_json_object,
    preregistration_sha256,
    validate_artifact_inventory,
    validate_preregistration,
)

DEFAULT_PREREGISTRATION = (
    PROJECT_ROOT / "configs" / "experiments" / "taxonomy_final_test_v1.json"
)
DEFAULT_CLOUD_ROOT = (
    PROJECT_ROOT.parent / "cloud_backups" / "taxonomy_two_stage_formal_v2_r2"
)
DEFAULT_LOCAL_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "experimental"
    / "taxonomy_post_supervisor_local_v1"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "docs"
    / "experiments"
    / "taxonomy_final_test_v1_artifact_inventory.json"
)

ALL_SCOPES = (*L2_SCOPES, *L4_SCOPES)


def _sealed_payload_sha256(value: Mapping[str, object], field: str) -> str:
    payload = dict(value)
    observed = payload.pop(field, None)
    expected = canonical_sha256(payload)
    if observed != expected:
        raise ValueError(f"Sealed payload hash mismatch for {field}.")
    return expected


def _tree_record(root: Path) -> dict[str, object]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"Checkpoint tree is empty: {root}")
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in files
    ]
    return {
        "sha256": canonical_sha256(records),
        "file_count": len(records),
        "bytes": sum(int(item["bytes"]) for item in records),
    }


def _entry(
    *,
    root_alias: str,
    path: Path,
    root: Path,
    kind: str,
    method_id: str | None = None,
    scope_id: str | None = None,
    sha256: str | None = None,
    bytes_value: int | None = None,
    file_count: int = 1,
) -> dict[str, object]:
    if sha256 is None:
        sha256 = file_sha256(path)
    if bytes_value is None:
        bytes_value = path.stat().st_size
    value: dict[str, object] = {
        "root_alias": root_alias,
        "path": path.relative_to(root).as_posix(),
        "kind": kind,
        "sha256": sha256,
        "bytes": int(bytes_value),
        "file_count": int(file_count),
    }
    if method_id is not None:
        value["method_id"] = method_id
    if scope_id is not None:
        value["scope_id"] = scope_id
    return value


def _selection_entries(
    *,
    worker_root: Path,
    root_alias: str,
    method_id: str,
    scopes: Iterable[str],
    include_checkpoints: bool,
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for scope_id in scopes:
        selection_path = worker_root / "selections" / method_id / f"{scope_id}.json"
        selection = load_json_object(selection_path)
        _sealed_payload_sha256(selection, "selection_payload_sha256")
        if (
            selection.get("protocol_id") != "taxonomy_two_stage_formal_v2"
            or selection.get("method_id") != method_id
            or selection.get("training_scope_id") != scope_id
            or selection.get("failure_count") != 0
            or selection.get("test_contract_count") != 0
        ):
            raise ValueError(f"Selection identity or safety mismatch: {selection_path}")
        entries.append(
            _entry(
                root_alias=root_alias,
                path=selection_path,
                root=worker_root,
                kind="frozen_selection",
                method_id=method_id,
                scope_id=scope_id,
            )
        )
        if not include_checkpoints:
            continue
        relative = selection.get("selected_checkpoint_relative_path")
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"Selection lacks checkpoint path: {selection_path}")
        checkpoint = worker_root / Path(relative)
        tree = _tree_record(checkpoint)
        entries.append(
            _entry(
                root_alias=root_alias,
                path=checkpoint,
                root=worker_root,
                kind="selected_checkpoint_tree",
                method_id=method_id,
                scope_id=scope_id,
                sha256=str(tree["sha256"]),
                bytes_value=int(tree["bytes"]),
                file_count=int(tree["file_count"]),
            )
        )
        manifest = checkpoint / "checkpoint.manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        entries.append(
            _entry(
                root_alias=root_alias,
                path=manifest,
                root=worker_root,
                kind="checkpoint_manifest",
                method_id=method_id,
                scope_id=scope_id,
            )
        )
    return entries


def build_inventory(
    *,
    preregistration_path: Path,
    cloud_root: Path,
    local_root: Path,
) -> dict[str, object]:
    preregistration = load_json_object(preregistration_path)
    validate_preregistration(preregistration)
    entries: list[dict[str, object]] = []

    distil_root = cloud_root / "worker-distil-frozen"
    entries.extend(
        _selection_entries(
            worker_root=distil_root,
            root_alias="cloud_worker_distil_frozen",
            method_id="distilbert_review_candidate_cross_encoder",
            scopes=ALL_SCOPES,
            include_checkpoints=True,
        )
    )
    entries.extend(
        _selection_entries(
            worker_root=distil_root,
            root_alias="cloud_worker_distil_frozen",
            method_id="frozen_qwen_few_shot",
            scopes=ALL_SCOPES,
            include_checkpoints=False,
        )
    )

    qlora_workers = {
        "worker-qlora-a": cloud_root / "worker-qlora-a",
        "worker-qlora-b": cloud_root / "worker-qlora-b",
    }
    seen_qlora_scopes: set[str] = set()
    for worker_id, worker_root in qlora_workers.items():
        selection_dir = worker_root / "selections" / "qwen_candidate_pair_qlora"
        scopes = sorted(path.stem for path in selection_dir.glob("*.json"))
        duplicates = seen_qlora_scopes.intersection(scopes)
        if duplicates:
            raise ValueError(f"QLoRA scopes duplicated across workers: {sorted(duplicates)}")
        seen_qlora_scopes.update(scopes)
        entries.extend(
            _selection_entries(
                worker_root=worker_root,
                root_alias=f"cloud_{worker_id.replace('-', '_')}",
                method_id="qwen_candidate_pair_qlora",
                scopes=scopes,
                include_checkpoints=True,
            )
        )
    if seen_qlora_scopes != set(ALL_SCOPES):
        raise ValueError(
            "QLoRA worker union does not equal the 15 frozen scopes: "
            f"missing={sorted(set(ALL_SCOPES) - seen_qlora_scopes)}, "
            f"extra={sorted(seen_qlora_scopes - set(ALL_SCOPES))}."
        )

    local_method_dirs = {
        "strict_train_only_tfidf": local_root
        / "level2_local_ndr"
        / "strict_train_only_tfidf"
        / "folds",
        "e5_base_v2": local_root / "level2_local_ndr" / "e5_base_v2" / "folds",
        "frozen_qwen_candidate_pair": local_root
        / "level2_local_ndr"
        / "frozen_qwen_candidate_pair"
        / "folds",
        "description_to_classifier_weight_transfer": local_root / "dcwt" / "folds",
    }
    for method_id, fold_dir in local_method_dirs.items():
        for fold_id in (f"l2-a{index:02d}" for index in range(1, 13)):
            path = fold_dir / f"{fold_id}.json"
            value = load_json_object(path)
            if value.get("failure_count") != 0 or value.get("test_contract_count") != 0:
                raise ValueError(f"Local frozen fold record failed safety checks: {path}")
            entries.append(
                _entry(
                    root_alias="post_supervisor_local_v1",
                    path=path,
                    root=local_root,
                    kind="local_fold_selection_and_result",
                    method_id=method_id,
                    scope_id=fold_id,
                )
            )

    safe_repo_inputs = (
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "fabsa_aspect_descriptions_minimal_v2.json",
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "taxonomy_two_stage_three_gpu_parallel_v2.json",
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "taxonomy_two_stage_stage_hybrid_calibration_v1.json",
        PROJECT_ROOT / "src" / "msc_project" / "experiments" / "taxonomy_methods.py",
        PROJECT_ROOT
        / "src"
        / "msc_project"
        / "experiments"
        / "taxonomy_two_stage_runtime.py",
        PROJECT_ROOT
        / "src"
        / "msc_project"
        / "experiments"
        / "taxonomy_few_shot_cache.py",
    )
    for path in safe_repo_inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
        entries.append(
            _entry(
                root_alias="repository",
                path=path,
                root=PROJECT_ROOT,
                kind="locked_code_or_resource",
            )
        )

    inventory: dict[str, object] = {
        "schema_version": "taxonomy_final_test_artifact_inventory_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "pass",
        "created_at": "2026-08-24",
        "scope": "validation-era reusable artifacts only; no FABSA split was opened",
        "preregistration_sha256": preregistration_sha256(preregistration),
        "distilbert_selected_scopes": 15,
        "qlora_selected_scopes": 15,
        "few_shot_selected_scopes": 15,
        "local_L2_folds_per_method": 12,
        "local_L2_method_count": 4,
        "entries": entries,
        "entry_manifest_sha256": canonical_sha256(entries),
        "failure_count": 0,
        "test_contract_count": 0,
    }
    validate_artifact_inventory(inventory, preregistration)
    return inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--cloud-root", type=Path, default=DEFAULT_CLOUD_ROOT)
    parser.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--write", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inventory = build_inventory(
        preregistration_path=args.preregistration.resolve(),
        cloud_root=args.cloud_root.resolve(),
        local_root=args.local_root.resolve(),
    )
    args.write.parent.mkdir(parents=True, exist_ok=True)
    args.write.write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "status": inventory["status"],
                "entries": len(inventory["entries"]),
                "entry_manifest_sha256": inventory["entry_manifest_sha256"],
                "test_contract_count": inventory["test_contract_count"],
                "output": str(args.write),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
