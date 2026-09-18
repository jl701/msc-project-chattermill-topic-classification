from __future__ import annotations

import json
from pathlib import Path

import pytest

from msc_project.experiments.taxonomy_reuse import file_sha256


def test_file_sha256_is_content_sensitive(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps({"value": 1}), encoding="utf-8")
    first = file_sha256(path)
    path.write_text(json.dumps({"value": 2}), encoding="utf-8")
    assert file_sha256(path) != first


def test_file_sha256_fails_closed_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        file_sha256(tmp_path / "missing.json")
