"""Tests for denominators, unique-review quality and unlabelled signals."""
import pytest

from msc_project.serving.monitoring import (
    evaluate_quality, js_divergence, make_review_queues, monitor_windows,
    prediction_pairs, summarise_load, unlabelled_summary,
)


def record(sample, pairs=(), success=True, index=0, length_bin="short", **kwargs):
    predictions = [{"aspect": a, "sentiments": [s]} for a, s in pairs]
    return {"request_id": f"attempt-{index}", "sample_id": sample, "success": success,
            "started_s": index, "scheduled_s": index, "finished_s": index + 1,
            "latency_ms": 1000, "scheduled_latency_ms": 1000, "schedule_lag_ms": 0,
            "input_chars": 100, "length_bin": length_bin, "status_code": 200 if success else 503,
            "error": None if success else "http_503",
            "response": {"variant": "hybrid", "model_version": "a01",
                         "predictions": predictions}, **kwargs}


def test_load_denominator_includes_timeout_rejection_and_client_drop():
    rows = [record("a"), record("b", success=False, index=1),
            {"request_id": "drop", "success": False, "started_s": None,
             "scheduled_s": 2, "finished_s": 2, "latency_ms": None,
             "scheduled_latency_ms": None, "error": "client_capacity_drop"}]
    got = summarise_load(rows, wall_seconds=4)
    assert got["attempts"] == 3
    assert got["sent"] == 2
    assert got["error_rate"] == pytest.approx(2 / 3)
    assert got["successful_reviews_per_second"] == 0.25
    assert got["successful_latency"]["count"] == 1
    assert got["response_budgets_seconds"]["2"]["fraction_of_all_attempts"] == pytest.approx(1 / 3)


def test_delayed_labels_deduplicate_but_report_repeat_disagreement_and_failures():
    rows = [record("a", [("held", "negative")]),
            record("a", [("held", "positive")], index=1),
            record("b", success=False, index=2)]
    labels = [{"request_id": "a", "pairs": [{"aspect": "held", "sentiment": "negative"}]},
              {"request_id": "b", "pairs": [{"aspect": "seen", "sentiment": "positive"}]}]
    got = evaluate_quality(rows, labels, ["held"], ["held", "seen"])
    assert got["unique_successful_reviews"] == 1
    assert got["repeated_success_prediction_disagreements"] == 1
    assert got["successful_unique_quality"]["heldout"]["pair"]["micro_f1"] == 1
    assert got["successful_unique_quality"]["heldout"]["gold_pair_support"] == 1
    assert got["successful_unique_quality"]["heldout"]["positive_review_count"] == 1
    assert got["delivered_quality_failure_as_empty"]["overall"]["pair"]["fn"] == 1
    assert got["delivered_quality_failure_as_empty"]["overall"]["pair"]["micro_f1"] == pytest.approx(2 / 3)


def test_presence_does_not_double_count_two_sentiments_and_pair_detects_error():
    rows = [record("a", [("x", "negative"), ("x", "positive")])]
    labels = [{"request_id": "a", "pairs": [{"aspect": "x", "sentiment": "negative"}]}]
    got = evaluate_quality(rows, labels)["successful_unique_quality"]["overall"]
    assert got["presence"]["micro_f1"] == 1
    assert got["pair"]["micro_f1"] == pytest.approx(2 / 3)


def test_unknown_labels_and_mixed_versions_fail_closed():
    with pytest.raises(ValueError, match="No independent labels"):
        evaluate_quality([record("unknown")], [])
    rows = [record("a"), record("b")]
    rows[1]["response"]["variant"] = "qlora"
    with pytest.raises(ValueError, match="grouped"):
        evaluate_quality(rows, [])
    rows[1]["response"]["variant"] = "hybrid"
    rows[0]["response"]["fold_id"] = "a01"
    rows[1]["response"]["fold_id"] = "a02"
    with pytest.raises(ValueError, match="grouped"):
        evaluate_quality(rows, [])


def test_js_and_window_shift_is_descriptive_only():
    assert js_divergence({"a": 10}, {"a": 10}) == 0
    assert js_divergence({"a": 10}, {"b": 10}) == 1
    assert js_divergence({}, {"a": 10}) is None
    reference = [record("a")]
    now = [record("b", length_bin="long")]
    output = monitor_windows(now, reference)
    assert output["windows"][0]["input_length_shift_signal"]
    assert "significance" in output["interpretation"]
    assert "micro_f1" not in str(output)
    assert unlabelled_summary(reference)["input_character_length"]["p50"] == 100


def test_review_queues_are_unique_disjoint_and_seeded():
    rows = [record(str(i), index=i) for i in range(10)] + [record("0", index=11)]
    first = make_review_queues(rows, random_count=3, targeted_count=4, seed=4)
    second = make_review_queues(rows, random_count=3, targeted_count=4, seed=4)
    assert first == second
    assert len(first["random_audit_sample_ids"]) == 3
    assert not set(first["random_audit_sample_ids"]) & set(first["targeted_inspection_sample_ids"])


def test_prediction_schema_supports_aspect_and_aspect_id():
    assert prediction_pairs({"predictions": [{"aspect_id": "a", "sentiments": ["negative"]}]}) == {("a", "negative")}


def test_unknown_predicted_aspect_is_not_silently_removed():
    rows = [record("a", [("not_in_taxonomy", "negative")])]
    labels = [{"request_id": "a", "pairs": []}]
    with pytest.raises(ValueError, match="outside"):
        evaluate_quality(rows, labels, all_aspects=["x"])
    got = evaluate_quality(rows, labels)["successful_unique_quality"]["overall"]
    assert got["pair"]["fp"] == 1


def test_http_driver_keeps_failures_and_open_model_capacity_drops(tmp_path):
    import json
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from pathlib import Path
    import subprocess
    import sys
    import threading
    import time

    observed_keys = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, value, status=200):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except BrokenPipeError:
                pass

        def do_GET(self):
            self.reply({"ready": True})

        def do_POST(self):
            value = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            observed_keys.append(set(value))
            time.sleep(0.25 if value["text"] == "phase" else 0.1)
            if value["text"] == "reject":
                self.reply({"error": "busy"}, 503)
            else:
                self.reply({"request_id": value["request_id"], "variant": "hybrid", "predictions": []})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    replay = tmp_path / "replay.jsonl"
    replay.write_text('\n'.join(json.dumps({"request_id": str(i), "text": text, "gold": "never_send"})
                                for i, text in enumerate(["ok", "reject", "ok"])), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts/benchmark_serving.py"
    common = [sys.executable, str(script), "--url", f"http://127.0.0.1:{server.server_port}",
              "--replay", str(replay), "--timeout", "2", "--expected-variant", "hybrid"]
    try:
        subprocess.run(common + ["--output", str(tmp_path / "closed"), "--requests", "3",
                                 "--mode", "closed", "--concurrency", "2"], check=True, capture_output=True, timeout=15)
        closed = json.loads((tmp_path / "closed/summary.json").read_text())
        assert closed["attempts"] == 3
        assert closed["successful"] == 2
        assert closed["http_status_counts"]["503"] == 1
        subprocess.run(common + ["--output", str(tmp_path / "open"), "--requests", "6",
                                 "--mode", "open", "--rate", "100", "--max-inflight", "1"],
                       check=True, capture_output=True, timeout=15)
        opened = json.loads((tmp_path / "open/summary.json").read_text())
        assert opened["attempts"] == 6
        assert opened["failure_counts"]["client_capacity_drop"] >= 1
        rows = [json.loads(line) for line in (tmp_path / "open/requests.jsonl").read_text().splitlines()]
        assert opened["wall_seconds_including_drain"] >= max(r["finished_s"] for r in rows)
        phase_replay = tmp_path / "phase_replay.jsonl"
        phase_replay.write_text(json.dumps({"request_id": "phase_sample", "text": "phase"}), encoding="utf-8")
        phase_plan = tmp_path / "phases.json"
        phase_plan.write_text(json.dumps({"phases": [
            {"name": name, "duration": 0.02, "rate": 25, "replay": str(phase_replay)}
            for name in ["normal", "burst", "recovery"]]}), encoding="utf-8")
        subprocess.run(common + ["--output", str(tmp_path / "phased"), "--phase-plan", str(phase_plan),
                                 "--max-inflight", "8"], check=True, capture_output=True, timeout=15)
        phased = json.loads((tmp_path / "phased/summary.json").read_text())
        phase_rows = {r["phase"]: r for r in [json.loads(line) for line in (tmp_path / "phased/requests.jsonl").read_text().splitlines()]}
        assert phased["attempts"] == phased["successful"] == 3
        assert phase_rows["recovery"]["started_s"] < phase_rows["normal"]["finished_s"]
        assert phased["phases"][0]["completed_after_phase_boundary"] == 1
        assert all(r["sample_id"] == "phase_sample" for r in phase_rows.values())
        phase_config = json.loads((tmp_path / "phased/config.json").read_text())
        assert phase_config["rate"] is None
        assert phase_config["arrival_schedule"] == "phased"
        subprocess.run(common + ["--output", str(tmp_path / "duration"), "--mode", "open",
                                 "--duration", "0.2", "--rate", "1"],
                       check=True, capture_output=True, timeout=15)
        duration = json.loads((tmp_path / "duration/summary.json").read_text())
        assert duration["wall_seconds_including_drain"] >= 0.2
        assert all(keys == {"request_id", "text"} for keys in observed_keys)
    finally:
        server.shutdown()
        server.server_close()
