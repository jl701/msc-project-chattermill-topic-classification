"""Local launcher checks use only synthetic folders and temporary loopback ports."""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from create_annotation_launcher import create_launcher


@pytest.fixture
def data(tmp_path):
    folder = tmp_path / "synthetic inputs with spaces"
    folder.mkdir()
    (folder / "annotation_reviews.jsonl").write_text("", encoding="utf-8")
    (folder / "annotation_taxonomy.json").write_text("[]", encoding="utf-8")
    (folder / "human_labels").mkdir()
    (folder / "human_labels/annotations.json").write_text("synthetic sentinel", encoding="utf-8")
    (folder / "开始标注.html").write_text(
        "<html>如果打不开，通知 Codex 重启本地标注服务即可，已保存标签不会丢失。</html>",
        encoding="utf-8",
    )
    return folder


def test_generate_idempotent_and_preserves_synthetic_labels(data):
    from pathlib import Path
    first = create_launcher(data, Path(sys.executable))
    assert first["entry_guide_updated"] and not first["service_started"]
    second = create_launcher(data, Path(sys.executable))
    assert not second["entry_guide_updated"]
    assert (data / "human_labels/annotations.json").read_text() == "synthetic sentinel"
    assert "启动标注.cmd" in (data / "开始标注.html").read_text(encoding="utf-8")
    assert "通知 Codex" in (data / "开始标注_添加启动入口前.html").read_text(encoding="utf-8")
    assert not (data / "launcher_logs").exists()
    (data / "启动标注.cmd").write_text("user change", encoding="utf-8")
    with pytest.raises(ValueError, match="Existing launcher differs"):
        create_launcher(data, Path(sys.executable))
    assert (data / "启动标注.cmd").read_text() == "user change"


def test_invalid_paths_and_port_fail_before_generation(data):
    from pathlib import Path
    with pytest.raises(FileNotFoundError):
        create_launcher(data, data / "missing_python.exe")
    with pytest.raises(ValueError, match="port"):
        create_launcher(data, Path(sys.executable), port=0)
    (data / "annotation_taxonomy.json").unlink()
    with pytest.raises(ValueError, match="Missing input"):
        create_launcher(data, Path(sys.executable))
    assert not (data / "start_annotation.ps1").exists()


POWERSHELL = shutil.which("powershell.exe")
windows_only = pytest.mark.skipif(os.name != "nt" or not POWERSHELL, reason="Windows PowerShell required")


def check_only(data):
    return subprocess.run(
        [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
         str(data / "start_annotation.ps1"), "-CheckOnly"],
        capture_output=True, timeout=15,
    )


@windows_only
def test_inactive_check_validates_paths_without_starting(data):
    from pathlib import Path
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    create_launcher(data, Path(sys.executable), port=port)
    result = check_only(data)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ready_to_start" and payload["started"] is False
    assert payload["url"] == f"http://127.0.0.1:{port}/"
    assert not (data / "launcher_logs").exists()
    # Removing a required synthetic input causes refusal, never recreation.
    (data / "annotation_reviews.jsonl").unlink()
    assert check_only(data).returncode != 0
    assert not (data / "annotation_reviews.jsonl").exists()


@windows_only
@pytest.mark.parametrize("kind,ready,allowed", [
    ("human_annotation", True, True),
    ("unrelated_application", True, False),
    ("human_annotation", False, False),
])
def test_existing_service_reuse_or_refusal(data, kind, ready, allowed):
    from pathlib import Path

    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"kind": kind, "ready": ready}).encode())

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        create_launcher(data, Path(sys.executable), port=server.server_port)
        result = check_only(data)
        assert (result.returncode == 0) is allowed, result.stderr
        if allowed:
            assert json.loads(result.stdout)["status"] == "reused_existing"
        assert not (data / "launcher_logs").exists()
        assert thread.is_alive()  # Launcher never terminates the occupying service.
        assert (data / "human_labels/annotations.json").read_text() == "synthetic sentinel"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
