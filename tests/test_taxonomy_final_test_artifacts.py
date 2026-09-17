from __future__ import annotations

from pathlib import Path

from msc_project.experiments.taxonomy_final_test import canonical_sha256, file_sha256
from msc_project.experiments.taxonomy_final_test_artifacts import (
    ArtifactRoots,
    _tree_sha256,
    verify_inventory_entries,
)


def test_repository_inventory_alias_resolves_inside_project_root(tmp_path: Path) -> None:
    resource = tmp_path / "resource.json"
    resource.write_text("{}\n", encoding="utf-8")
    roots = ArtifactRoots(
        project_root=tmp_path,
        local_validation_root=tmp_path / "local",
        cloud_worker_distil_frozen=tmp_path / "distil",
        cloud_worker_qlora_a=tmp_path / "qlora-a",
        cloud_worker_qlora_b=tmp_path / "qlora-b",
    )
    inventory = {
        "entries": [
            {
                "root_alias": "repository",
                "path": "resource.json",
                "kind": "locked_code_or_resource",
                "sha256": file_sha256(resource),
                "bytes": resource.stat().st_size,
                "file_count": 1,
            }
        ]
    }
    audit = verify_inventory_entries(inventory, roots)
    assert audit["status"] == "pass"
    assert audit["entries"] == 1
    assert audit["hash_mismatch_count"] == 0


def test_tree_hash_uses_platform_independent_casefolded_posix_order(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoint"
    adapter = root / "adapter"
    adapter.mkdir(parents=True)
    lower = adapter / "adapter_config.json"
    upper = adapter / "README.md"
    lower.write_text("config\n", encoding="utf-8")
    upper.write_text("readme\n", encoding="utf-8")

    expected_records = [
        {
            "path": "adapter/adapter_config.json",
            "bytes": lower.stat().st_size,
            "sha256": file_sha256(lower),
        },
        {
            "path": "adapter/README.md",
            "bytes": upper.stat().st_size,
            "sha256": file_sha256(upper),
        },
    ]

    assert _tree_sha256(root) == canonical_sha256(expected_records)
