from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from msc_project.experiments.taxonomy_final_test import canonical_sha256, file_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_taxonomy_final_test.py"
SPEC = importlib.util.spec_from_file_location("run_taxonomy_final_test", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
_load_materialised = MODULE._load_materialised


def _synthetic_worker_root(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "output"
    private = root / "private_release_data"
    private.mkdir(parents=True)
    reviews = pd.DataFrame(
        {"row_uid": ["synthetic:1", "synthetic:2"], "text": ["alpha", "beta"]}
    )
    unlabelled = private / "unlabelled_reviews.csv"
    reviews.to_csv(unlabelled, index=False)
    release: dict[str, object] = {
        "authorisation_id": "synthetic-authorisation",
        "official_test_row_uid_sha256": canonical_sha256(reviews["row_uid"].tolist()),
    }
    manifest = {
        "schema_version": "taxonomy_final_test_materialised_data_v1",
        "protocol_id": "taxonomy_two_stage_final_test_v1",
        "authorisation_id": release["authorisation_id"],
        "unlabelled_reviews_sha256": file_sha256(unlabelled),
        "label_vault_sha256": "a" * 64,
    }
    (private / "data_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return root, release


def test_worker_loader_does_not_require_or_open_label_vault(tmp_path: Path) -> None:
    root, release = _synthetic_worker_root(tmp_path)
    reviews, uids, manifest = _load_materialised(root, release)
    assert uids == ["synthetic:1", "synthetic:2"]
    assert list(reviews.columns) == ["row_uid", "text"]
    assert manifest["label_vault_sha256"] == "a" * 64
    assert not (root / "private_release_data" / "label_vault.csv").exists()


def test_analysis_loader_requires_and_hash_checks_private_vault(tmp_path: Path) -> None:
    root, release = _synthetic_worker_root(tmp_path)
    with pytest.raises(FileNotFoundError, match="Private label vault"):
        _load_materialised(root, release, require_label_vault=True)

    vault = root / "private_release_data" / "label_vault.csv"
    vault.write_text("private synthetic labels\n", encoding="utf-8")
    with pytest.raises(ValueError, match="label-vault hash mismatch"):
        _load_materialised(root, release, require_label_vault=True)
