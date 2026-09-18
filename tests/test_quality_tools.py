"""Quality tools inspect source metadata without leaking candidate secrets."""

import importlib.util
import zipfile
from pathlib import Path


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_reports_location_not_secret(tmp_path):
    (tmp_path / "src").mkdir()
    synthetic = "ghp_" + "A" * 36
    (tmp_path / "src/example.py").write_text(f"TOKEN = {synthetic!r}\n", encoding="utf-8")
    report = load_script("audit_repository").scan(tmp_path)
    assert report["python_files"] == 1
    assert report["credential_pattern_candidates"] == [
        {"file": "src/example.py", "line": 1, "pattern": "github_token"}
    ]
    assert synthetic not in str(report)


def test_audit_does_not_read_artifacts(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints/private.py").write_text("not valid Python!", encoding="utf-8")
    assert load_script("audit_repository").scan(tmp_path)["python_files"] == 0


def test_audit_flags_machine_paths_without_echoing_them(tmp_path):
    (tmp_path / "docs").mkdir()
    private_path = "C:" + "/Users/example/.ssh/id_ed25519"
    (tmp_path / "docs/notes.md").write_text(private_path, encoding="utf-8")
    report = load_script("audit_repository").scan(tmp_path)
    assert report["publication_pattern_candidates"] == [
        {"file": "docs/notes.md", "line": 1, "pattern": "author_machine_path"}
    ]
    assert private_path not in str(report)


def test_distribution_audit_rejects_author_only_file(tmp_path):
    archive = tmp_path / "example.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("project/docs/aji_private.md", "No credentials needed")
    assert load_script("check_distribution").audit_archive(archive) == [
        {
            "file": "project/docs/aji_private.md",
            "pattern": "author_only_path",
        }
    ]


def test_maintained_links_exist():
    assert load_script("check_quality").broken_links() == []
