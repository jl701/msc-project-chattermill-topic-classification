from __future__ import annotations
import json
import threading
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from msc_project.serving.annotation import AnnotationStore, make_server
from msc_project.serving.review_data import assign_splits, deduplicate, parse_feed, private_safe_text


def test_feed_omits_author_and_rating_is_not_label():
    data = {"feed": {"entry": [{"id": {"label": "11"}, "author": {"name": {"label": "private"}},
             "im:rating": {"label": "1"}, "content": {"label": "Email me at person@example.com"},
             "title": {"label": "Title"}}]}}
    row = parse_feed(data, app="Synthetic", app_id="1", country="gb", source_url="https://example.org")[0]
    assert "author" not in row and "labels" not in row
    assert row["text"] == "Email me at [EMAIL]"
    assert row["rating_metadata_not_gold"] == "1"


def test_redaction_preserves_polarity_money_and_noise():
    assert private_safe_text(" Not good! £1.50 😡 <3 ") == "Not good! £1.50 😡 <3"
    assert "[PHONE]" in private_safe_text("Call +44 7700 900123 please")


def test_duplicates_and_history_are_isolated():
    history = ["Same old review", "This is a very long unique original comment about my account which cannot login at all"]
    rows = [{"request_id": str(i), "text": text, "app": "A"} for i, text in enumerate([
        "  SAME old review  ", "This is a very long unique original comment about my account which cannot login at all!",
        "New review", "new REVIEW", "Other review"])]
    kept, audit = deduplicate(rows, history)
    assert {r["text"] for r in kept} == {"New review", "Other review"}
    assert audit["historical_overlap_removed"] == 2
    assert audit["new_duplicate_group_members_removed"] == 1


def test_split_counts_determinism_and_proportions():
    rows = [{"request_id": str(i), "duplicate_group_id": str(i), "app": "A" if i < 500 else "B"} for i in range(1000)]
    assigned, meta = assign_splits(rows)
    assert meta["counts"] == {"adaptation": 200, "calibration": 200, "locked": 400}
    assert len({r["duplicate_group_id"] for r in assigned}) == 800
    assert sum(r["pilot"] for r in assigned) == 50
    assert assigned == assign_splits(list(reversed(rows)))[0]
    assert meta["source_by_split"]["locked"] == {"A": 200, "B": 200}
    with pytest.raises(ValueError):
        assign_splits([rows[0], rows[0]])


@pytest.fixture
def store(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps({"request_id": "test", "text": "Synthetic annotation fixture",
        "app": "Test", "split": "locked", "pilot": False, "rating_metadata_not_gold": "1",
        "prediction": ["must not be exposed"]}) + "\n", encoding="utf-8")
    taxonomy = tmp_path / "taxonomy.json"
    taxonomy.write_text(json.dumps([{"aspect": "Theme", "zh_name": "主题", "definition": "Test", "zh_description": "测试"}]), encoding="utf-8")
    return AnnotationStore(dataset, taxonomy, tmp_path / "labels")


def sample_annotation():
    return {"request_id": "test", "expected_revision": 0, "reviewer": "test reviewer",
        "status": "confirmed", "labels": [["Theme", "positive"]], "no_defined_theme": False,
        "ambiguous": False, "out_of_scope": False, "incomplete_text": False,
        "english_confirmed": True, "note": "synthetic test"}


def test_store_persistence_revision_and_gold_validation(store):
    assert "rating_metadata_not_gold" not in store.item("test")
    assert "prediction" not in store.item("test")
    assert store.item("test")["annotation"] is None
    saved = store.save(sample_annotation())
    assert saved["revision"] == 1 and saved["source"] == "human_annotation"
    assert json.loads(store.path.read_text())["test"]["labels"] == [["Theme", "positive"]]
    assert store.bootstrap("locked")["confirmed"] == 1
    with pytest.raises(ValueError, match="另一窗口"):
        store.save(sample_annotation())
    value = dict(sample_annotation(), expected_revision=1, labels=[])
    with pytest.raises(ValueError, match="请标主题"):
        store.save(value)
    with pytest.raises(ValueError, match="Invalid aspect"):
        store.save(dict(value, labels=[["Unknown", "positive"]]))
    with pytest.raises(ValueError, match="歧义"):
        store.save(dict(sample_annotation(), expected_revision=1, ambiguous=True))
    store.save(dict(value, no_defined_theme=True, out_of_scope=True))
    assert len((store.storage / "annotation_history.jsonl").read_text().splitlines()) == 2


def test_http_local_page_and_save_origin(store):
    server = make_server(store, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(origin + "/") as response:
            assert "首批 50 条" in response.read().decode()
        body = json.dumps(sample_annotation()).encode()
        bad = urllib.request.Request(origin + "/api/save", data=body, headers={"Content-Type": "application/json", "Origin": "https://evil.example"})
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(bad)
        assert error.value.code == 403
        good = urllib.request.Request(origin + "/api/save", data=body, headers={"Content-Type": "application/json", "Origin": origin})
        with urllib.request.urlopen(good) as response:
            assert json.load(response)["revision"] == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_export_keeps_human_gold_schema_and_split(store, tmp_path):
    store.save(sample_annotation())
    (tmp_path / "annotation_reviews.jsonl").write_bytes((tmp_path / "dataset.jsonl").read_bytes())
    (tmp_path / "annotation_taxonomy.json").write_bytes((tmp_path / "taxonomy.json").read_bytes())
    (tmp_path / "human_labels").mkdir()
    (tmp_path / "human_labels/annotations.json").write_bytes(store.path.read_bytes())
    destination = tmp_path / "export"
    script = Path(__file__).resolve().parents[1] / "scripts/export_review_annotations.py"
    subprocess.run([sys.executable, str(script), "--data", str(tmp_path), "--output", str(destination)],
                   check=True, capture_output=True)
    exported = json.loads((destination / "labels_locked.jsonl").read_text())
    assert exported["pairs"] == [{"aspect": "Theme", "sentiment": "positive"}]
    assert exported["aspects"] == ["Theme"]
    assert not (destination / "labels_adaptation.jsonl").read_text()
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run([sys.executable, str(script), "--data", str(tmp_path), "--output", str(destination)],
                       check=True, capture_output=True)
