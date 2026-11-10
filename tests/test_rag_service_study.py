import json
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import run_rag_service_study as study
from test_rag_evaluation import ASPECTS, record


def make_args(tmp_path):
    index = tmp_path / "index"
    index.mkdir()
    (index / "manifest.json").write_text('{}')
    (index / "pool.jsonl").write_text('{}\n')
    artifacts = tmp_path / "artifacts.json"
    artifacts.write_text(json.dumps({"taxonomy": {"candidates": [{"aspect": a} for a in ASPECTS]},
        "variants": {"hybrid": {
        "aspect_threshold": .9, "runner_up_threshold": .4}}, "unchanged": "reference"}))
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"baseline": "fixed", "best_dynamic_method": "bm25",
        "selected_method": "fixed", "thresholds": {
            "fixed": {"aspect_threshold": .8, "runner_up_threshold": .4},
            "bm25": {"aspect_threshold": .7, "runner_up_threshold": .4}}}))
    replay = tmp_path / "replay.jsonl"
    replay.write_text(json.dumps({"request_id": "synthetic", "text": "Synthetic test."}) + '\n')
    capacity = tmp_path / "capacity.jsonl"
    capacity.write_text(json.dumps({"request_id": "capacity", "text": "Capacity review."}) + '\n')
    development = tmp_path / "development.jsonl"
    development.write_text(''.join(json.dumps({"request_id": f"dev{i}", "text": "Synthetic." * (i+1)})+'\n'
                                   for i in range(30)))
    scores = tmp_path / "scores"
    scores.mkdir()
    for method in ("fixed", "bm25"):
        (scores / f"{method}.jsonl").write_text(''.join(json.dumps(record(f"dev{i}"))+'\n' for i in range(30)))
    args = study.parser().parse_args(["--artifacts", str(artifacts), "--index", str(index),
        "--replay", str(replay), "--selection", str(selection), "--output", str(tmp_path / "out"),
        "--development-replay", str(development), "--development-score-root", str(scores),
        "--capacity-replay", str(capacity)])
    return args


def completed_capacity(output, method, repeat, capacity, *, interrupted=False, successes=30):
    run = output / "runs" / f"capacity__{method}__c1__r{repeat}"
    summary = {"interrupted": interrupted, "attempts": 30, "successful": successes,
               "success_rate": successes / 30, "successful_reviews_per_second": capacity}
    study.save_json(run / "summary.json", summary)
    study.save_json(run / "complete.json", {"summary_sha256": study.digest(run / "summary.json")})


def test_plan_preserves_original_and_clones_thresholds_without_start(tmp_path, monkeypatch):
    args = make_args(tmp_path)
    original = args.artifacts.read_bytes()
    monkeypatch.setattr(study.subprocess, "Popen", lambda *a, **kw: pytest.fail("No process in plan"))
    plan = study.prepare_plan(args)
    assert args.artifacts.read_bytes() == original
    assert plan["methods"]["fixed"]["prefix_cache"] == "on"
    assert plan["methods"]["bm25"]["prefix_cache"] == "off"
    assert not plan["candidate_promoted_by_quality"]
    clone = study.read_json(args.output / "inputs/artifacts_bm25.json")
    assert clone["variants"]["hybrid"]["aspect_threshold"] == .7
    assert clone["unchanged"] == "reference"
    assert clone["retrieval_fallback_policy"]["aspect_threshold"] == .8
    baseline_path = args.output / "inputs/artifacts_fixed.json"
    assert clone["retrieval_fallback_policy"]["source_artifacts_sha256"] == study.digest(baseline_path)
    assert "retrieval_fallback_policy" not in study.read_json(baseline_path)
    assert study.prepare_plan(args) == plan
    args.closed_requests = 31
    with pytest.raises(FileExistsError):
        study.prepare_plan(args)


def test_label_in_performance_replay_rejected(tmp_path):
    args = make_args(tmp_path)
    args.replay.write_text(json.dumps({"request_id": "x", "text": "synthetic", "labels": []}))
    with pytest.raises(ValueError, match="only"):
        study.prepare_plan(args)


def test_new_e5_method_keeps_distinct_clone_identity_and_passes_policy_to_server(tmp_path):
    args = make_args(tmp_path)
    selection = study.read_json(args.selection)
    selection.update(best_dynamic_method="e5_negative_seen", selected_method="e5_negative_seen")
    selection["thresholds"]["e5_negative_seen"] = selection["thresholds"].pop("bm25")
    args.selection.write_text(json.dumps(selection))
    (args.development_score_root / "e5_negative_seen.jsonl").write_bytes(
        (args.development_score_root / "bm25.jsonl").read_bytes())
    plan = study.prepare_plan(args)
    config = plan["methods"]["e5_negative_seen"]
    assert config["retrieval_method"] == "e5"
    assert config["retrieval_seed"] == 13
    assert config["retrieval_selection_policy"] == "same_aspect_negative_seen"
    assert config["prefix_cache"] == "off"
    clone = study.read_json(args.output / "inputs/artifacts_e5_negative_seen.json")
    assert clone["service_calibration"]["method"] == "e5_negative_seen"
    assert clone["service_calibration"]["retrieval"] == {
        "method": "e5", "seed": 13, "selection_policy": "same_aspect_negative_seen"}
    fixed = study.read_json(args.output / "inputs/artifacts_fixed.json")
    assert fixed["service_calibration"]["retrieval"]["selection_policy"] == "global"
    assert clone["retrieval_fallback_policy"]["source_artifacts_sha256"] == study.digest(
        args.output / "inputs/artifacts_fixed.json")
    argv = study.server_command("e5_negative_seen", plan, args.output / "session")
    assert argv[argv.index("--retrieval-method") + 1] == "e5"
    assert argv[argv.index("--retrieval-selection-policy") + 1] == "same_aspect_negative_seen"
    assert "src/msc_project/serving/retrieval_policy.py" in plan["source_hashes"]


def test_closed_counts_and_matched_common_rates(tmp_path):
    args = make_args(tmp_path)
    plan = study.prepare_plan(args)
    assert len(study.stage_runs("closed", "fixed", plan, None)) == 12
    pilots = study.stage_runs("capacity", "fixed", plan, None)
    assert len(pilots) == 3
    assert {run["replay"] for run in pilots} == {plan["capacity_replay"]}
    assert {run["replay"] for run in study.stage_runs("closed", "fixed", plan, None)} == {plan["replay"]}
    for method, capacities in (("fixed", [2, 3, 4]), ("bm25", [1, 1.5, 2])):
        for repeat, capacity in enumerate(capacities, start=1):
            completed_capacity(args.output, method, repeat, capacity)
    rates = study.freeze_common_rates(plan, args.output)
    assert rates["minimum_capacity"] == 1.5
    fixed = study.stage_runs("open", "fixed", plan, rates)
    dynamic = study.stage_runs("open", "bm25", plan, rates)
    assert len(fixed) == 6
    assert [r["rate"] for r in fixed] == [r["rate"] for r in dynamic] == [.75] * 3 + [1.125] * 3
    assert [r["seed"] for r in fixed] == [r["seed"] for r in dynamic]
    assert study.stage_runs("soak", "bm25", plan, rates)[0]["duration"] == 3600


def test_burst_peak_is_candidate_capacity_and_no_interphase_drain(tmp_path):
    args = make_args(tmp_path)
    plan = study.prepare_plan(args)
    rates = {"minimum_capacity": 1, "burst_peak_rate": 6}
    runs = study.stage_runs("burst", "bm25", plan, rates)
    assert len(runs) == 3
    assert [p["duration"] for p in runs[0]["phases"]] == [60, 30, 180]
    assert [p["rate"] for p in runs[0]["phases"]] == [.5, 6, .5]
    directory = args.output / "test_run"
    resolved = study.resolved_run(runs[0], directory)
    argv = study.client_command(resolved, directory, plan)
    assert "--phase-plan" in argv and "--duration" not in argv
    assert "--expected-variant" in argv


@pytest.mark.parametrize("minimum_capacity,requested,expected_seconds", [
    (.3, 60, 135), (.2, 60, 201), (.01, 60, 4001), (1, 60, 60), (.3, 200, 200)])
def test_all_burst_cycles_have_capacity_derived_baseline_minimum(
        tmp_path, minimum_capacity, requested, expected_seconds):
    args = make_args(tmp_path)
    args.baseline_seconds = requested
    plan = study.prepare_plan(args)
    baseline = study.burst_baseline_schedule(plan, minimum_capacity)
    assert baseline["duration_seconds"] == expected_seconds
    assert baseline["planned_requests"] >= 20
    assert plan["burst_baseline_rule"]["historical_baseline_fallback"] is False
    rates = {"minimum_capacity": minimum_capacity, "burst_peak_rate": 2 * minimum_capacity,
             "burst_baseline": baseline}
    runs = study.stage_runs("burst", "bm25", plan, rates)
    assert len(runs) == 3
    for run in runs:
        normal, burst, recovery = run["phases"]
        assert normal["duration"] == expected_seconds
        assert study.scheduled_attempts(normal["duration"], normal["rate"]) >= 20
        assert [burst["duration"], recovery["duration"]] == [30, 180]
        assert normal["rate"] == recovery["rate"] == .5 * minimum_capacity
    rates["burst_baseline"] = {**baseline, "duration_seconds": expected_seconds + 1}
    with pytest.raises(ValueError, match="frozen"):
        study.stage_runs("burst", "bm25", plan, rates)


def test_capacity_freezes_burst_baseline_before_any_burst(tmp_path):
    args = make_args(tmp_path)
    plan = study.prepare_plan(args)
    for method in ("fixed", "bm25"):
        for repeat in range(1, 4):
            completed_capacity(args.output, method, repeat, .3)
    rates = study.freeze_common_rates(plan, args.output)
    assert rates["burst_baseline"]["duration_seconds"] == 135
    assert rates["burst_baseline"]["planned_requests"] == 21
    assert study.read_json(args.output / "common_rates.json")["burst_baseline"] == rates["burst_baseline"]


def test_paired_repeat_order_balances_methods_and_reloads_per_method_block(tmp_path):
    args = make_args(tmp_path)
    plan = study.prepare_plan(args)
    rates = {"minimum_capacity": 1, "common_rates": {"0.5": .5, "0.75": .75}, "burst_peak_rate": 2}
    for stage, expected, runs_per_block in (
            ("capacity", ["fixed", "bm25", "bm25", "fixed", "fixed", "bm25"], 1),
            ("closed", ["bm25", "fixed", "fixed", "bm25", "bm25", "fixed"], 4),
            ("open", ["fixed", "bm25", "bm25", "fixed", "fixed", "bm25"], 2)):
        blocks = study.execution_blocks(stage, plan, rates)
        assert [block["method"] for block in blocks] == expected
        assert all(len(block["runs"]) == runs_per_block for block in blocks)
        assert [set(run["seed"] for run in block["runs"]) for block in blocks] == [
            {plan["seed"]}, {plan["seed"]}, {plan["seed"] + 1}, {plan["seed"] + 1},
            {plan["seed"] + 2}, {plan["seed"] + 2}]
        all_names = [run["name"] for block in blocks for run in block["runs"]]
        assert len(set(all_names)) == len(all_names)
    assert len(study.execution_blocks("soak", plan, rates)) == 1


def test_failed_capacity_cannot_freeze_rates(tmp_path):
    args = make_args(tmp_path)
    plan = study.prepare_plan(args)
    for method in ("fixed", "bm25"):
        for repeat in range(1, 4):
            completed_capacity(args.output, method, repeat, 1, successes=29 if method == "bm25" else 30)
    with pytest.raises(ValueError, match="failures"):
        study.freeze_common_rates(plan, args.output)


def test_incomplete_result_preserved_on_resume(tmp_path):
    args = make_args(tmp_path)
    plan = study.prepare_plan(args)
    run = study.stage_runs("closed", "fixed", plan, None)[0]
    directory = args.output / "runs" / run["name"]
    directory.mkdir(parents=True)
    (directory / "requests.jsonl").write_text('partial evidence')
    with pytest.raises(RuntimeError, match="Incomplete"):
        study.completed_run(args.output, run)
    assert (directory / "requests.jsonl").read_text() == 'partial evidence'


def test_completed_resume_verifies_manifest_and_summary(tmp_path):
    args = make_args(tmp_path)
    plan = study.prepare_plan(args)
    run = study.stage_runs("capacity", "fixed", plan, None)[0]
    completed_capacity(args.output, "fixed", 1, 1)
    assert study.completed_run(args.output, run)
    directory = args.output / "runs" / run["name"]
    summary = study.read_json(directory / "summary.json")
    summary["attempts"] = 29
    study.save_json(directory / "summary.json", summary, mutable=True)
    with pytest.raises(RuntimeError, match="changed"):
        study.completed_run(args.output, run)


def test_occupied_port_is_not_contacted_or_stopped(monkeypatch):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        monkeypatch.setattr(study, "endpoint", lambda *a, **kw: pytest.fail("Do not contact foreign port"))
        with pytest.raises(OSError):
            study.ensure_port_free(sock.getsockname()[1])


def test_startup_failure_cleans_only_owned_children(tmp_path, monkeypatch):
    args = make_args(tmp_path)
    plan = study.prepare_plan(args)
    children = []
    class FakeProcess:
        def __init__(self, argv, **kwargs):
            self.pid = 100 + len(children)
            self.returncode = 1 if not children else None
            children.append(self)
        def poll(self):
            return self.returncode
        def wait(self, timeout=None):
            if self.returncode is None:
                raise subprocess.TimeoutExpired('synthetic', timeout)
            return self.returncode
    stopped = []
    def stop(process, timeout=20):
        stopped.append(process)
        process.returncode = 0
        study._OWNED_FAMILIES.pop(id(process), None)
    monkeypatch.setattr(study, "ensure_port_free", lambda port: None)
    monkeypatch.setattr(study.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(study, "stop_owned", stop)
    monkeypatch.setattr(study, "endpoint", lambda *a, **kw: pytest.fail("No foreign shutdown"))
    with pytest.raises(RuntimeError, match="before readiness"):
        with study.server_session("closed", "fixed", plan, args.output):
            pytest.fail("Not ready")
    assert stopped == children  # both roots are audited even when the server root already exited


def test_annotation_port_is_reserved(tmp_path):
    args = make_args(tmp_path)
    args.port = 8769
    with pytest.raises(ValueError, match="annotation"):
        study.prepare_plan(args)


def test_open_load_retains_offered_arrivals_when_client_inflight_cap_is_hit(tmp_path):
    class FakeAPI(BaseHTTPRequestHandler):
        def reply(self, payload):
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(data)
        def do_GET(self):
            self.reply({'ready': True, 'unfinished_tasks': 0})
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            time.sleep(.15)
            self.reply({'request_id': payload['request_id'], 'predictions': [], 'variant': 'hybrid'})
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), FakeAPI)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    replay = tmp_path / 'synthetic.jsonl'
    replay.write_text(json.dumps({'request_id': 's', 'text': 'Synthetic CPU test.'})+'\n')
    output = tmp_path / 'load'
    argv = [sys.executable, str(study.ROOT / 'scripts/benchmark_serving.py'),
        '--url', f'http://127.0.0.1:{server.server_port}', '--replay', str(replay),
        '--output', str(output), '--mode', 'open', '--duration', '.3', '--rate', '100',
        '--max-inflight', '1', '--timeout', '2', '--expected-variant', 'hybrid']
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=15,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        assert result.returncode == 0, result.stderr
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    summary = study.read_json(output / 'summary.json')
    assert summary['attempts'] == study.scheduled_attempts(.3, 100) == 30
    assert summary['failure_counts']['client_capacity_drop'] > 0
    assert summary['sent'] < summary['attempts']
    assert summary['successful'] + summary['failed'] == 30
    study.validate_run(summary, {'mode': 'open', 'duration': .3, 'rate': 100})


def test_open_validation_rejects_silent_arrival_reduction():
    summary = {'interrupted': False, 'wall_seconds_including_drain': 90, 'attempts': 30}
    with pytest.raises(RuntimeError, match='offered'):
        study.validate_run(summary, {'mode': 'open', 'duration': 90, 'rate': 1})


def api_response(reference, *, presence=.9, label="negative"):
    return {"request_id": reference["request_id"], "variant": "hybrid",
        "aspect_scores": [{"aspect": "seen", "score": presence}, {"aspect": "heldout", "score": .1}],
        "predictions": [{"aspect": "seen", "sentiments": [label],
                         "sentiment_scores": {"negative": .8, "neutral": .1, "positive": .1}}]}


def test_correctness_tolerates_probability_rounding_only_when_label_sets_match():
    reference = record("a")
    thresholds = {"aspect_threshold": .8, "runner_up_threshold": .4}
    compared = study.compare_api_response(reference, api_response(reference, presence=.900001), ASPECTS, thresholds)
    assert compared["exact_label_set_match"]
    assert compared["presence_probability_max_abs_delta"] == pytest.approx(.000001)
    assert compared["conditional_sentiment_probability_comparisons"] == 3
    changed = study.compare_api_response(reference, api_response(reference, label="positive"), ASPECTS, thresholds)
    assert not changed["exact_label_set_match"]
    assert changed["offline_only"] == [("seen", "negative")]
    assert changed["api_only"] == [("seen", "positive")]


def test_correctness_rejects_wrong_identity_and_missing_presence_grid():
    reference = record("a")
    response = api_response(reference)
    response["request_id"] = "other"
    with pytest.raises(ValueError, match="identity"):
        study.compare_api_response(reference, response, ASPECTS, {"aspect_threshold": .8, "runner_up_threshold": .4})
    response = api_response(reference)
    response["aspect_scores"].pop()
    with pytest.raises(ValueError, match="complete presence"):
        study.compare_api_response(reference, response, ASPECTS, {"aspect_threshold": .8, "runner_up_threshold": .4})


def test_correctness_detects_different_retrieval_even_if_labels_agree():
    reference = record("a")
    response = api_response(reference)
    contract = {"batch_aspect": {"retrieval": [{"candidate_aspect": "seen",
        "demonstration_sha256": "frozen-demo", "truncation": {"query_characters_removed": 0}}]}}
    reference["response"]["diagnostics"] = contract
    response["diagnostics"] = json.loads(json.dumps(contract))
    thresholds = {"aspect_threshold": .8, "runner_up_threshold": .4}
    assert study.compare_api_response(reference, response, ASPECTS, thresholds)["retrieval_prompt_contract_match"]
    response["diagnostics"]["batch_aspect"]["retrieval"][0]["demonstration_sha256"] = "different"
    check = study.compare_api_response(reference, response, ASPECTS, thresholds)
    assert check["exact_label_set_match"]
    assert check["retrieval_prompt_contract_match"] is False


def test_correctness_normalizes_legacy_global_metadata_but_rejects_policy_mismatch():
    reference = record("a")
    response = api_response(reference)
    contract = {"batch_aspect": {"retrieval": [{"candidate_aspect": "seen",
        "demonstration_sha256": "same"}]}}
    reference["response"]["diagnostics"] = contract
    response["diagnostics"] = json.loads(json.dumps(contract))
    row = response["diagnostics"]["batch_aspect"]["retrieval"][0]
    row.update(selection_policy="global", effective_selection_policy="global")
    thresholds = {"aspect_threshold": .8, "runner_up_threshold": .4}
    assert study.compare_api_response(reference, response, ASPECTS, thresholds)["retrieval_prompt_contract_match"]
    row.update(selection_policy="same_aspect_negative_seen", effective_selection_policy="same_aspect_negative_seen")
    result = study.compare_api_response(reference, response, ASPECTS, thresholds)
    assert result["exact_label_set_match"]
    assert result["retrieval_prompt_contract_match"] is False


def test_stratified_correctness_covers_length_range_without_labels():
    rows = [{"request_id": str(i), "text": "x"*(i+1)} for i in range(100)]
    sampled = study.length_stratified_rows(rows, 30, 17)
    assert len({row["request_id"] for row in sampled}) == 30
    assert sampled[0]["request_id"] == "0" and sampled[-1]["request_id"] == "99"
    assert sampled == study.length_stratified_rows(list(reversed(rows)), 30, 17)
    with pytest.raises(ValueError, match="at least30"):
        study.length_stratified_rows(rows, 29, 17)


def test_capacity_benchmark_and_correctness_inputs_cannot_overlap(tmp_path):
    args = make_args(tmp_path)
    args.capacity_replay = args.replay
    with pytest.raises(ValueError, match="disjoint"):
        study.prepare_plan(args)


def test_correctness_reference_is_bound_to_frozen_selection(tmp_path):
    args = make_args(tmp_path)
    selection = study.read_json(args.selection)
    selection["input_sha256"] = {"development_scores": {"fixed": "unexpectedhash"}}
    args.selection.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="frozen development"):
        study.prepare_plan(args)


@pytest.mark.parametrize("inject_failure", [False, True])
def test_correctness_preserves_all_requests_and_enforces_timing_gate(tmp_path, monkeypatch, inject_failure):
    args = make_args(tmp_path)
    plan = study.prepare_plan(args)
    seen_ids = []
    class Reply:
        status = 200
        def __init__(self, response):
            self.response = response
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            return json.dumps(self.response).encode()
    class Opener:
        def open(self, request, **kwargs):
            payload = json.loads(request.data)
            seen_ids.append(payload["request_id"])
            assert set(payload) == {"request_id", "text"}
            if inject_failure and payload["request_id"] == "dev3":
                raise TimeoutError("Synthetic timeout")
            return Reply(api_response(record(payload["request_id"]),
                         label="positive" if inject_failure and payload["request_id"] == "dev4" else "negative"))
    monkeypatch.setattr(study.urllib.request, "build_opener", lambda *a: Opener())
    monkeypatch.setattr(study, "wait_idle", lambda *a, **kw: {"unfinished_tasks": 0})
    for method in plan["methods"]:
        summary = study.run_correctness(method, plan, args.output, object())
        assert summary["attempts"] == 30 and summary["successful"] == 30 - inject_failure
        assert summary["exact_matches"] == 30 - 2*inject_failure
        assert summary["passed"] is not inject_failure
        assert study.correctness_completed(args.output, method)
        rows = study.read_jsonl(args.output / "correctness" / method / "requests.jsonl")
        assert len(rows) == 30
    assert seen_ids[:30] == plan["correctness"]["request_ids"] == seen_ids[30:]
    if inject_failure:
        with pytest.raises(RuntimeError, match="mismatch"):
            study.verify_correctness(args.output, plan)
    else:
        assert study.verify_correctness(args.output, plan)["passed"]
    assert study.read_json(args.output / "correctness/gate.json")["timing_promotion_blocked"] is inject_failure
    assert not (args.output / "runs").exists()


def test_missing_correctness_gate_stops_before_server_start_and_releases_lock(tmp_path, monkeypatch):
    from types import SimpleNamespace
    args = make_args(tmp_path)
    args.stage = "closed"
    monkeypatch.setattr(study, "parser", lambda: SimpleNamespace(parse_args=lambda argv: args))
    monkeypatch.setattr(study, "server_session", lambda *a: pytest.fail("Unverified timing server started"))
    with pytest.raises(RuntimeError, match="correctness"):
        study.main([])
    assert not (args.output / "running.lock").exists()


@pytest.mark.skipif(study.os.name != "nt", reason="Windows execution-state contract")
def test_keep_awake_never_requests_display_and_restores_on_error(monkeypatch):
    from types import SimpleNamespace
    calls = []
    def execution_state(flags):
        calls.append(flags)
        return 0x80000000
    monkeypatch.setattr(study.ctypes, "windll", SimpleNamespace(kernel32=SimpleNamespace(
        SetThreadExecutionState=execution_state)))
    with pytest.raises(ValueError, match="synthetic"):
        with study.keep_system_awake():
            raise ValueError("synthetic failure")
    assert calls == [0x80000001, 0x80000000]
    assert all(not value & 0x2 for value in calls)  # ES_DISPLAY_REQUIRED is never set


def test_cleanup_stops_pinned_orphan_but_not_recycled_pid_or_unrelated_service(monkeypatch):
    import psutil
    class ExitedParent:
        pid = 101
        def poll(self):
            return 1
        def terminate(self):
            pytest.fail("Exited parent must not be signalled")
    terminated = []
    class Child:
        def __init__(self, pid):
            assert pid in (102, 103), "Unrelated service was inspected"
            self.pid = pid
        def create_time(self):
            return 11.0 if self.pid == 102 else 999.0
        def is_running(self):
            return True
        def terminate(self):
            terminated.append(self.pid)
    parent = ExitedParent()
    study._OWNED_FAMILIES[id(parent)] = {101: 10.0, 102: 11.0, 103: 12.0}
    monkeypatch.setattr(psutil, "Process", Child)
    monkeypatch.setattr(psutil, "wait_procs", lambda children, timeout: (children, []))
    study.stop_owned(parent)
    assert terminated == [102]
    assert id(parent) not in study._OWNED_FAMILIES


def test_fallback_reporting_counts_reviews_once_and_retains_failed_requests():
    entries = [{"fallback": True, "error_type": "RetrievalTimeoutError"} for _ in range(12)]
    records = [{"success": True, "response": {"diagnostics": {"batch_aspect": {"retrieval": entries}}}},
               {"success": False, "response": None},
               {"success": True, "response": {"diagnostics": {"batch_aspect": {"retrieval": [{"fallback": False}]}}}}]
    summary = study.retrieval_observations(records)
    assert summary["fallback_review_count"] == 1
    assert summary["fallback_aspect_count"] == 12
    assert summary["fallback_error_review_counts"] == {"RetrievalTimeoutError": 1}
    assert summary["responses_with_retrieval_diagnostics"] == 2
