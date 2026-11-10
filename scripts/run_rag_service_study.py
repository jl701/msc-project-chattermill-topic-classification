"""Serial, resumable HTTP service study for fixed and selected dynamic examples.

Raw artifacts, process logs and responses belong in ignored local outputs.
Each stage warms a fresh owned server, preserves failures, and drains its queue.
Run --plan-only to register without loading a model or starting subprocesses.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_serving_study import validate_completed_run
from msc_project.serving.rag_evaluation import (
    SENTIMENTS, decode_scores, read_jsonl, records_by_id, score_maps, successful,
)

_OWNED_FAMILIES: dict[int, dict[int, float]] = {}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def keep_system_awake():
    """Prevent automatic Windows system sleep, without keeping the display on."""
    if os.name != "nt":
        yield
        return
    execution_state = ctypes.windll.kernel32.SetThreadExecutionState
    execution_state.argtypes = [ctypes.c_uint]
    execution_state.restype = ctypes.c_uint
    continuous, system_required = 0x80000000, 0x00000001
    previous = execution_state(continuous | system_required)
    if not previous:
        raise OSError("Windows rejected the process-thread keep-awake request")
    try:
        yield
    finally:
        execution_state(previous | continuous)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, value: object, *, mutable: bool = False) -> None:
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not mutable:
        if path.read_text(encoding="utf-8") != data:
            raise FileExistsError(f"Existing evidence differs: {path}")
        return
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(data, encoding="utf-8")
    temporary.replace(path)


def copy_immutable(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(source) != digest(target):
            raise FileExistsError(f"Frozen input differs: {target}")
    else:
        with target.open("xb") as stream:
            stream.write(source.read_bytes())


def choose_methods(selection: dict) -> tuple[str, str]:
    baseline = selection.get("baseline", "fixed")
    if baseline != "fixed":
        raise ValueError("This study requires the fixed demonstration baseline")
    candidate = selection.get("best_dynamic_method") or "bm25"
    if candidate not in {"bm25", "e5", "rrf"}:
        raise ValueError("Selected dynamic method must be bm25, e5 or rrf")
    if any(method not in selection["thresholds"] for method in (baseline, candidate)):
        raise ValueError("Both methods require frozen calibrated thresholds")
    return baseline, candidate


def replay_population(path: Path) -> list[dict]:
    rows = read_jsonl(path)
    if not rows or any(set(row) != {"request_id", "text"} or
                       not isinstance(row["request_id"], str) or not row["request_id"] or
                       not isinstance(row["text"], str) or not row["text"].strip()
                       for row in rows):
        raise ValueError("Performance replay must contain only nonempty request_id and text")
    if len({row["request_id"] for row in rows}) != len(rows):
        raise ValueError("Performance replay IDs must be unique")
    return rows


def length_stratified_rows(rows: list[dict], count: int, seed: int) -> list[dict]:
    """Evenly spaced length ranks, with deterministic ID-hash tie breaking."""
    if count < 30 or len(rows) < 30:
        raise ValueError("Correctness requires at least30 development reviews")
    ordered = sorted(rows, key=lambda row: (len(row["text"]),
        hashlib.sha256(f"{seed}:{row['request_id']}".encode()).hexdigest()))
    count = min(count, len(ordered))
    return [ordered[round(i * (len(ordered) - 1) / (count - 1))] for i in range(count)]


def prepare_plan(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    selection = read_json(args.selection)
    baseline, candidate = choose_methods(selection)
    artifacts = read_json(args.artifacts)
    index = args.index.resolve()
    index_files = {path.name: digest(path) for path in sorted(index.iterdir()) if path.is_file()}
    if "manifest.json" not in index_files or "pool.jsonl" not in index_files:
        raise ValueError("Retrieval index lacks its manifest or pool")
    replay_rows = replay_population(args.replay)
    capacity_rows = replay_population(args.capacity_replay)
    development_replay = args.development_replay or args.replay
    development_rows = replay_population(development_replay)
    populations = [replay_rows, capacity_rows, development_rows]
    for i, left in enumerate(populations):
        for right in populations[i + 1:]:
            if {row["request_id"] for row in left} & {row["request_id"] for row in right} or \
                    {" ".join(row["text"].casefold().split()) for row in left} & \
                    {" ".join(row["text"].casefold().split()) for row in right}:
                raise ValueError("Capacity, benchmark and development reviews must be disjoint")
    correctness_rows = length_stratified_rows(development_rows,
                                             args.correctness_reviews, args.seed)
    aspects = [candidate["aspect"] for candidate in artifacts["taxonomy"]["candidates"]]
    score_inputs = {}
    for method in (baseline, candidate):
        source = args.development_score_root / f"{method}.jsonl"
        source_hash = digest(source)
        registered_hash = selection.get("input_sha256", {}).get("development_scores", {}).get(method)
        if registered_hash is not None and source_hash != registered_hash:
            raise ValueError("Correctness scores differ from frozen development selection")
        records = records_by_id(read_jsonl(source))
        for row in correctness_rows:
            record = records.get(row["request_id"])
            if not successful(record):
                raise ValueError("Every correctness review requires a successful offline score reference")
            score_maps(record, aspects)
        score_inputs[method] = {"source": str(source.resolve()), "sha256": source_hash,
            "frozen_path": str(output / "inputs" / f"development_{method}.jsonl")}
    if args.port == 8769 or not 1024 <= args.port <= 65535:
        raise ValueError("Use an unprivileged service port distinct from annotation port8769")
    numeric = [args.repeats, args.closed_requests, args.open_seconds, args.soak_seconds,
               args.baseline_seconds, args.burst_seconds, args.recovery_seconds, args.startup_timeout,
               args.retrieval_timeout_s]
    if any(not math.isfinite(float(value)) or value <= 0 for value in numeric):
        raise ValueError("All run counts and durations must be positive")
    sources = ["scripts/run_rag_service_study.py", "scripts/benchmark_serving.py",
               "scripts/serve_chattermill.py", "src/msc_project/serving/server.py",
               "src/msc_project/serving/inference.py", "src/msc_project/serving/retrieval.py",
               "src/msc_project/serving/prefix_cache.py", "src/msc_project/serving/rag_evaluation.py",
               "src/msc_project/serving/timed_retrieval.py"]
    plan = {"schema_version": "rag_http_service_study_v1", "scope": "local_historical_replay",
        "input_hashes": {"artifacts": digest(args.artifacts), "selection": digest(args.selection),
                         "replay": digest(args.replay), "capacity_replay": digest(args.capacity_replay),
                         "index_files": index_files},
        "source_hashes": {name: digest(ROOT / name) for name in sources},
        "replay": str(output / "inputs" / "replay.jsonl"), "replay_unique_reviews": len(replay_rows),
        "capacity_replay": str(output / "inputs" / "capacity_replay.jsonl"),
        "capacity_unique_reviews": len(capacity_rows),
        "pool_separation": "capacity_benchmark_development_disjoint_IDs_and_normalized_exact_text",
        "correctness": {"development_replay_sha256": digest(development_replay),
            "reviews": len(correctness_rows), "requested_reviews": args.correctness_reviews,
            "request_ids": [row["request_id"] for row in correctness_rows],
            "sampling": "deterministic_evenly_spaced_character_length_ranks_no_gold_access",
            "replay": str(output / "inputs" / "development_replay.jsonl"),
            "offline_scores": score_inputs, "aspects": aspects,
            "gate": "all_successful_exact_review_label_sets_match_offline_frozen_thresholds",
            "probability_deltas": "recorded_descriptively_no_arbitrary_numerical_tolerance",
            "timing_promotion_requires_pass": True},
        "index": str(index), "encoder_path": str(args.encoder_path.resolve()) if args.encoder_path else None,
        "baseline": baseline, "dynamic_candidate": candidate,
        "candidate_promoted_by_quality": selection.get("selected_method") == candidate,
        "candidate_scope": "best_dynamic_compared_even_if_quality_did_not_promote_it",
        "methods": {}, "port": args.port, "seed": args.seed,
        "warmup_requests": 3, "repeats": args.repeats, "concurrencies": [1, 2, 4, 8],
        "closed_requests": args.closed_requests, "open_seconds": args.open_seconds,
        "open_rate_fractions": [.5, .75], "soak_seconds": args.soak_seconds,
        "soak_rate_fraction": .5, "burst_cycles": 3,
        "burst_phase_seconds": [args.baseline_seconds, args.burst_seconds, args.recovery_seconds],
        "burst_rate_fractions": [.5, 2.0, .5],
        "burst_peak_reference": "dynamic_candidate_C1_capacity",
        "capacity_definition": "independent_capacity_pool_minimum_of_two_method_C1_three_repeat_median_successful_QPS",
        "startup_timeout_seconds": args.startup_timeout,
        "server": {"variant": "hybrid", "pair_batch_size": 4, "max_batch_tokens": 4096,
                   "request_batch_size": 1, "queue_size": 8, "timeout_s": 120,
                   "logits_mode": "last_token", "retrieval_fallback": "fixed",
                   "retrieval_timeout_s": args.retrieval_timeout_s},
        "client": {"timeout": 130, "max_inflight": 64, "metrics_interval": 5},
        "environment": {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                        "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"},
        "warmup_scope": "fresh_session_warmup_excluded_from_all_headline_runs",
        "failure_scope": "all_failed_timed_out_rejected_and_unsent_attempts_retained"}
    plan["keep_awake"] = "Windows_system_only_thread_state_restored_on_exit_no_display_requirement"
    plan["paired_repeat_order"] = {stage: [
        {"repeat": repeat + 1, "method_order": [baseline, candidate][::
            -1 if ((repeat % 2 == 1) != (stage == "closed")) else 1]}
        for repeat in range(args.repeats)] for stage in ("capacity", "closed", "open")}
    plan["session_policy"] = "reload_per_paired_repeat_method_block_with_three_excluded_warmups"
    for method in (baseline, candidate):
        clone = json.loads(json.dumps(artifacts))
        selected = selection["thresholds"][method]
        for key in ("aspect_threshold", "runner_up_threshold"):
            value = float(selected[key])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Selected threshold outside [0,1]")
            clone["variants"]["hybrid"][key] = value
        clone["service_calibration"] = {"selection_sha256": digest(args.selection), "method": method}
        if method != baseline:
            baseline_path = Path(plan["methods"][baseline]["artifacts"])
            baseline_settings = read_json(baseline_path)["variants"]["hybrid"]
            clone["retrieval_fallback_policy"] = {"method": "fixed", "variant": "hybrid",
                "aspect_threshold": baseline_settings["aspect_threshold"],
                "runner_up_threshold": baseline_settings["runner_up_threshold"],
                "source_artifacts_sha256": digest(baseline_path),
                "source_artifacts_path": str(baseline_path.resolve())}
        else:
            clone.pop("retrieval_fallback_policy", None)
        artifact_path = output / "inputs" / f"artifacts_{method}.json"
        save_json(artifact_path, clone)
        plan["methods"][method] = {"artifacts": str(artifact_path),
            "artifacts_sha256": digest(artifact_path), "retrieval_method": method,
            "prefix_cache": "on" if method == baseline else "off"}
    copy_immutable(args.replay, output / "inputs" / "replay.jsonl")
    copy_immutable(args.capacity_replay, output / "inputs" / "capacity_replay.jsonl")
    copy_immutable(args.selection, output / "inputs" / "selection.json")
    copy_immutable(development_replay, output / "inputs" / "development_replay.jsonl")
    for item in score_inputs.values():
        copy_immutable(Path(item["source"]), Path(item["frozen_path"]))
    save_json(output / "plan.json", plan)
    return plan


def freeze_common_rates(plan: dict, output: Path) -> dict:
    medians, hashes = {}, {}
    for method in plan["methods"]:
        values = []
        for repeat in range(plan["repeats"]):
            run = output / "runs" / f"capacity__{method}__c1__r{repeat + 1}"
            complete = read_json(run / "complete.json")
            if complete["summary_sha256"] != digest(run / "summary.json"):
                raise ValueError("C1 reference summary changed after completion")
            summary = read_json(run / "summary.json")
            validate_run(summary, {"mode": "closed", "requests": plan["closed_requests"]})
            if summary.get("success_rate", 0) < .99:
                raise ValueError("C1 capacity reference has >1% failures; inspect before open load")
            value = summary["successful_reviews_per_second"]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid measured capacity")
            values.append(value)
            hashes[f"{method}_r{repeat + 1}"] = digest(run / "summary.json")
        medians[method] = statistics.median(values)
    minimum = min(medians.values())
    rates = {"schema_version": "rag_http_common_rates_v1", "reference_summary_hashes": hashes,
             "c1_median_reviews_per_second": medians, "minimum_capacity": minimum,
             "common_rates": {str(f): f * minimum for f in plan["open_rate_fractions"]},
             "burst_peak_rate": 2 * medians[plan["dynamic_candidate"]],
             "definition": "same_absolute_arrival_rates_for_both_methods"}
    save_json(output / "common_rates.json", rates)
    return rates


def stage_runs(stage: str, method: str, plan: dict, rates: dict | None) -> list[dict]:
    base = {"replay": plan["replay"], "seed": plan["seed"], "shuffle": True}
    runs = []
    if stage in ("capacity", "closed"):
        for concurrency in ([1] if stage == "capacity" else plan["concurrencies"]):
            for repeat in range(plan["repeats"]):
                runs.append({**base, "replay": plan["capacity_replay"] if stage == "capacity" else plan["replay"],
                    "name": f"{stage}__{method}__c{concurrency}__r{repeat + 1}",
                    "mode": "closed", "concurrency": concurrency, "requests": plan["closed_requests"],
                    "seed": plan["seed"] + repeat})
    else:
        if rates is None:
            raise ValueError("Open, soak and burst stages require frozen closed-loop capacities")
        if stage == "open":
            for fraction in plan["open_rate_fractions"]:
                for repeat in range(plan["repeats"]):
                    runs.append({**base, "name": f"open__{method}__f{int(fraction * 100)}__r{repeat + 1}",
                        "mode": "open", "rate": rates["common_rates"][str(fraction)],
                        "duration": plan["open_seconds"], "seed": plan["seed"] + repeat})
        elif stage == "soak":
            runs.append({**base, "name": f"soak__{method}", "mode": "open",
                "rate": .5 * rates["minimum_capacity"], "duration": plan["soak_seconds"]})
        elif stage == "burst":
            for repeat in range(plan["burst_cycles"]):
                phases = [{"name": name, "duration": duration,
                           "rate": rates["burst_peak_rate"] if name == "burst" else
                           fraction * rates["minimum_capacity"]}
                          for name, duration, fraction in zip(("normal", "burst", "recovery"),
                              plan["burst_phase_seconds"], plan["burst_rate_fractions"])]
                runs.append({**base, "name": f"burst__{method}__r{repeat + 1}", "mode": "open",
                             "phases": phases, "seed": plan["seed"] + repeat})
        else:
            raise ValueError("Unknown study stage")
    return runs


def execution_blocks(stage: str, plan: dict, rates: dict | None) -> list[dict]:
    if stage in plan["paired_repeat_order"]:
        blocks = []
        for paired in plan["paired_repeat_order"][stage]:
            repeat = paired["repeat"]
            for method in paired["method_order"]:
                blocks.append({"name": f"{stage}__r{repeat}__{method}", "method": method,
                    "repeat": repeat, "runs": [run for run in stage_runs(stage, method, plan, rates)
                                               if run["seed"] == plan["seed"] + repeat - 1]})
        return blocks
    method = plan["dynamic_candidate"]
    return [{"name": f"{stage}__{method}", "method": method,
             "runs": stage_runs(stage, method, plan, rates)}]


def endpoint(base: str, suffix: str, *, post: bool = False) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(base + suffix, data=b"{}" if post else None,
                                     method="POST" if post else "GET")
    with opener.open(request, timeout=5) as response:
        return json.loads(response.read())


def ensure_port_free(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if os.name == "nt":
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        probe.bind(("127.0.0.1", port))


def owned_port(process: subprocess.Popen, port: int) -> bool:
    import psutil

    remember_owned_family(process)
    if process.poll() is not None:
        return False
    try:
        pids = set()
        for pid, created in _OWNED_FAMILIES.get(id(process), {}).items():
            try:
                current = psutil.Process(pid)
                if current.create_time() == created and current.is_running():
                    pids.add(pid)
            except psutil.Error:
                pass
        return any(connection.status == psutil.CONN_LISTEN and connection.laddr and
                   connection.laddr.port == port and connection.pid in pids
                   for connection in psutil.net_connections(kind="tcp"))
    except (psutil.Error, OSError):
        return False


def remember_owned_family(process: subprocess.Popen) -> None:
    """Pin process creation times while the owned root still exists."""
    import psutil

    family = _OWNED_FAMILIES.setdefault(id(process), {})
    if process.poll() is not None:
        return
    try:
        root = psutil.Process(process.pid)
        created = root.create_time()
        if process.pid in family and family[process.pid] != created:
            return
        family[process.pid] = created
        for child in root.children(recursive=True):
            if child.is_running():
                family.setdefault(child.pid, child.create_time())
    except psutil.Error:
        pass


def stop_owned(process: subprocess.Popen | None, timeout: float = 20) -> None:
    if process is None:
        return
    import psutil

    remember_owned_family(process)
    family = _OWNED_FAMILIES.pop(id(process), {})
    children = []
    for pid, created in family.items():
        if pid == process.pid:
            continue
        try:
            child = psutil.Process(pid)
            if child.create_time() == created and child.is_running():
                children.append(child)
        except psutil.Error:
            pass
    for child in reversed(children):
        try:
            child.terminate()
        except psutil.Error:
            pass
    if process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=timeout)
        except ProcessLookupError:
            pass
    _, alive = psutil.wait_procs(children, timeout=timeout)
    for child in alive:
        try:
            child.kill()
        except psutil.Error:
            pass


def append_command(output: Path, argv: list[str], description: str) -> None:
    with (output / "commands.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"utc": utc(), "description": description, "argv": argv}) + "\n")


def run_client(argv: list[str], log: Path, output: Path, env: dict, description: str) -> None:
    append_command(output, argv, description)
    child = None
    with log.open("x", encoding="utf-8") as stream:
        try:
            child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stream,
                stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            remember_owned_family(child)
            while child.poll() is None:
                remember_owned_family(child)
                try:
                    child.wait(timeout=.5)
                except subprocess.TimeoutExpired:
                    pass
            code = child.returncode
            if code:
                raise RuntimeError(f"{description} failed with exit{code}; see {log}")
        finally:
            stop_owned(child)


def wait_idle(base: str, process: subprocess.Popen, timeout: float = 300) -> dict:
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if process.poll() is not None:
            raise RuntimeError("Owned service exited during queue drain")
        metrics = endpoint(base, "/metrics")
        if metrics.get("unfinished_tasks") == 0:
            return metrics
        time.sleep(.25)
    raise RuntimeError("Service queue did not drain before next run")


def client_command(run: dict, directory: Path, plan: dict) -> list[str]:
    argv = [sys.executable, "-u", str(ROOT / "scripts/benchmark_serving.py"),
        "--url", f"http://127.0.0.1:{plan['port']}", "--output", str(directory),
        "--expected-variant", "hybrid", "--timeout", str(plan["client"]["timeout"]),
        "--max-inflight", str(plan["client"]["max_inflight"]),
        "--metrics-interval", str(plan["client"]["metrics_interval"]), "--tag", run["name"]]
    for key in ("replay", "mode", "requests", "duration", "concurrency", "rate", "seed", "phase_plan"):
        if key in run:
            argv.extend(["--" + key.replace("_", "-"), str(run[key])])
    if run.get("shuffle"):
        argv.append("--shuffle")
    return argv


def resolved_run(run: dict, directory: Path) -> dict:
    resolved = dict(run)
    phases = resolved.pop("phases", None)
    if phases:
        phase_path = directory / "phases.json"
        save_json(phase_path, {"phases": phases})
        resolved["phase_plan"] = str(phase_path)
    return resolved


def scheduled_attempts(duration: float, rate: float) -> int:
    """Match benchmark_serving's i/rate < duration rule, including float endpoints."""
    count = max(1, math.ceil(duration * rate))
    while (count - 1) / rate >= duration:
        count -= 1
    while count / rate < duration:
        count += 1
    return count


def validate_run(summary: dict, run: dict) -> None:
    validate_completed_run(summary, run)
    if run.get("mode") != "open":
        return
    if run.get("phase_plan"):
        phases = read_json(Path(run["phase_plan"]))["phases"]
        expected = sum(scheduled_attempts(p["duration"], p["rate"]) for p in phases)
    else:
        expected = scheduled_attempts(run["duration"], run["rate"])
    if summary.get("attempts") != expected:
        raise RuntimeError("Open-loop attempt count differs from registered offered arrivals")


def completed_run(output: Path, run: dict) -> bool:
    directory = output / "runs" / run["name"]
    if not directory.exists():
        return False
    if not (directory / "complete.json").exists():
        raise RuntimeError(f"Incomplete prior run preserved at {directory}; use a new study output")
    resolved = resolved_run(run, directory)
    save_json(directory / "registered_run.json", resolved)
    complete = read_json(directory / "complete.json")
    if complete["summary_sha256"] != digest(directory / "summary.json"):
        raise RuntimeError("Completed run summary changed")
    validate_run(read_json(directory / "summary.json"), resolved)
    return True


def run_benchmark(run: dict, directory: Path, plan: dict, output: Path, env: dict,
                  process: subprocess.Popen, *, headline: bool = True) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    resolved = resolved_run(run, directory)
    save_json(directory / "registered_run.json", resolved)
    base = f"http://127.0.0.1:{plan['port']}"
    before = wait_idle(base, process)
    save_json(directory / "server_metrics_before.json", before)
    run_client(client_command(resolved, directory, plan), directory / "client.log", output,
               env, run["name"])
    after = wait_idle(base, process)
    save_json(directory / "server_metrics_after.json", after)
    if after.get("log_write_failures", 0) > before.get("log_write_failures", 0):
        raise RuntimeError("Worker evidence log failure")
    summary = read_json(directory / "summary.json")
    validate_run(summary, resolved)
    save_json(directory / "complete.json", {"completed_utc": utc(),
        "summary_sha256": digest(directory / "summary.json"), "headline": headline})
    return summary


def compare_api_response(reference: dict, response: dict, aspects: list[str],
                         thresholds: dict) -> dict:
    """Compare all presence scores and available conditional sentiment scores."""
    presence, sentiment = score_maps(reference, aspects)
    expected = decode_scores(presence, sentiment, thresholds["aspect_threshold"],
                             thresholds["runner_up_threshold"])
    if response.get("request_id") != reference["request_id"] or response.get("variant") != "hybrid":
        raise ValueError("API identity or model variant differs from offline reference")
    actual_presence = {}
    for item in response["aspect_scores"]:
        aspect, value = item["aspect"], float(item["score"])
        if aspect in actual_presence or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid or duplicate API presence score")
        actual_presence[aspect] = value
    if set(actual_presence) != set(aspects):
        raise ValueError("API response lacks the complete presence grid")
    actual, seen, sentiment_deltas = set(), set(), []
    for item in response["predictions"]:
        aspect = item["aspect"]
        labels = item["sentiments"]
        if aspect not in sentiment or aspect in seen or not labels or set(labels) - set(SENTIMENTS):
            raise ValueError("Invalid API aspect or sentiment labels")
        seen.add(aspect)
        actual.update((aspect, label) for label in labels)
        if set(item["sentiment_scores"]) != set(SENTIMENTS):
            raise ValueError("API conditional sentiment labels differ from offline contract")
        scores = [float(item["sentiment_scores"][label]) for label in SENTIMENTS]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores) or \
                not math.isclose(sum(scores), 1, abs_tol=1e-4):
            raise ValueError("Invalid API conditional sentiment probabilities")
        sentiment_deltas.extend(abs(a - b) for a, b in zip(scores, sentiment[aspect]))
    expected_present = {aspect for aspect, value in actual_presence.items()
                        if value >= thresholds["aspect_threshold"]}
    if seen != expected_present:
        raise ValueError("API predictions do not agree with its own frozen presence threshold")
    def prompt_contract(value):
        retrieval = value.get("diagnostics", {}).get("batch_aspect", {}).get("retrieval", [])
        return [{"aspect": row.get("candidate_aspect"),
                 "demonstration_sha256": row.get("demonstration_sha256"),
                 "truncation": row.get("truncation")} for row in retrieval]
    offline_contract = prompt_contract(reference["response"])
    api_contract = prompt_contract(response)
    contract_match = offline_contract == api_contract if offline_contract else None
    return {"exact_label_set_match": expected == actual,
            "retrieval_prompt_contract_match": contract_match,
            "offline_pair_count": len(expected), "api_pair_count": len(actual),
            "offline_only": sorted(expected - actual), "api_only": sorted(actual - expected),
            "presence_probability_max_abs_delta": max(abs(presence[a] - actual_presence[a]) for a in aspects),
            "sentiment_probability_max_abs_delta": max(sentiment_deltas) if sentiment_deltas else None,
            "conditional_sentiment_probability_comparisons": len(sentiment_deltas)}


def correctness_completed(output: Path, method: str) -> bool:
    directory = output / "correctness" / method
    if not directory.exists():
        return False
    if not (directory / "complete.json").exists():
        raise RuntimeError(f"Incomplete correctness evidence preserved at {directory}; use a new output")
    complete = read_json(directory / "complete.json")
    for name in ("summary.json", "requests.jsonl"):
        if complete["sha256"][name] != digest(directory / name):
            raise RuntimeError("Correctness evidence changed after completion")
    return True


def verify_correctness(output: Path, plan: dict) -> dict:
    summaries = {}
    for method in plan["methods"]:
        if not correctness_completed(output, method):
            raise RuntimeError("Both methods must pass --stage correctness before service timing")
        summary = read_json(output / "correctness" / method / "summary.json")
        if summary["plan_sha256"] != digest(output / "plan.json") or \
                summary["attempts"] != plan["correctness"]["reviews"]:
            raise RuntimeError("Correctness gate does not match the frozen service plan")
        summaries[method] = summary
    gate = {"passed": all(item["passed"] for item in summaries.values()),
            "methods": summaries, "timing_promotion_blocked":
            any(not item["passed"] for item in summaries.values())}
    save_json(output / "correctness" / "gate.json", gate)
    if not gate["passed"]:
        raise RuntimeError("Offline/API label-set mismatch or request failure; audit preserved correctness evidence before timing promotion")
    return gate


def run_correctness(method: str, plan: dict, output: Path, process: subprocess.Popen) -> dict:
    config = plan["correctness"]
    population = {row["request_id"]: row for row in replay_population(Path(config["replay"]))}
    references = records_by_id(read_jsonl(config["offline_scores"][method]["frozen_path"]))
    thresholds = read_json(Path(plan["methods"][method]["artifacts"]))["variants"]["hybrid"]
    directory = output / "correctness" / method
    directory.mkdir(parents=True, exist_ok=False)
    save_json(directory / "registered_run.json", {"method": method, "configuration": config,
              "thresholds": thresholds, "plan_sha256": digest(output / "plan.json")})
    base = f"http://127.0.0.1:{plan['port']}"
    before = wait_idle(base, process)
    save_json(directory / "server_metrics_before.json", before)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    records = []
    with (directory / "requests.jsonl").open("x", encoding="utf-8") as stream:
        for key in config["request_ids"]:
            started = time.monotonic()
            record = {"request_id": key, "text_characters": len(population[key]["text"]),
                      "success": False, "exact_label_set_match": False}
            request = urllib.request.Request(base + "/predict",
                data=json.dumps(population[key], ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST")
            try:
                with opener.open(request, timeout=plan["client"]["timeout"]) as reply:
                    record["http_status"] = reply.status
                    response = json.loads(reply.read())
                record["response"] = response
                record.update(compare_api_response(references[key], response, config["aspects"], thresholds))
                record["success"] = True
            except (OSError, urllib.error.URLError, ValueError, KeyError, TypeError) as error:
                record["error"] = f"{type(error).__name__}: {error}"
            record["elapsed_ms_excluded_from_timing_study"] = (time.monotonic() - started) * 1000
            records.append(record)
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
    after = wait_idle(base, process)
    save_json(directory / "server_metrics_after.json", after)
    evidence_ok = after.get("log_write_failures", 0) <= before.get("log_write_failures", 0)
    matches = sum(row["success"] and row["exact_label_set_match"] for row in records)
    prompt_mismatches = sum(row["success"] and
        row.get("retrieval_prompt_contract_match") is False for row in records)
    summary = {"method": method, "attempts": len(records),
        "successful": sum(row["success"] for row in records), "exact_matches": matches,
        "mismatched_or_failed_reviews": len(records) - matches,
        "exact_review_label_set_parity": matches / len(records),
        "presence_probability_max_abs_delta": max((row["presence_probability_max_abs_delta"]
            for row in records if row["success"]), default=None),
        "sentiment_probability_max_abs_delta": max((row["sentiment_probability_max_abs_delta"]
            for row in records if row["success"] and row["sentiment_probability_max_abs_delta"] is not None), default=None),
        "retrieval_prompt_contract_mismatch_count": prompt_mismatches,
        "passed": matches == len(records) and not prompt_mismatches and evidence_ok,
        "worker_evidence_logging_ok": evidence_ok,
        "plan_sha256": digest(output / "plan.json"), "timing_scope": "correctness_only_excluded_from_service_timings"}
    save_json(directory / "summary.json", summary)
    save_json(directory / "complete.json", {"completed_utc": utc(), "sha256": {
        name: digest(directory / name) for name in ("summary.json", "requests.jsonl")}})
    return summary


@contextmanager
def server_session(stage: str, method: str, plan: dict, output: Path):
    ensure_port_free(plan["port"])
    directory = output / "sessions" / f"{stage}_{method}_{uuid.uuid4().hex[:12]}"
    directory.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, **plan["environment"])
    method_cfg = plan["methods"][method]
    argv = [sys.executable, "-u", str(ROOT / "scripts/serve_chattermill.py"),
            "--artifacts", method_cfg["artifacts"], "--output-dir", str(directory),
            "--port", str(plan["port"]), "--prefix-cache", method_cfg["prefix_cache"],
            "--retrieval-index", plan["index"], "--retrieval-method", method,
            "--retrieval-seed", "13"]
    for key, value in plan["server"].items():
        argv.extend(["--" + key.replace("_", "-"), str(value)])
    if plan["encoder_path"]:
        argv.extend(["--retrieval-encoder", plan["encoder_path"]])
    save_json(directory / "launch.json", {"argv": argv, "environment": plan["environment"],
              "plan_sha256": digest(output / "plan.json"), "registered_utc": utc()})
    append_command(output, argv, f"server {stage}/{method}")
    process = telemetry = None
    base = f"http://127.0.0.1:{plan['port']}"
    with (directory / "server.log").open("x", encoding="utf-8") as server_log, \
         (directory / "telemetry.log").open("x", encoding="utf-8") as telemetry_log:
        try:
            launch_started = time.monotonic()
            process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=server_log,
                stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            remember_owned_family(process)
            telemetry_argv = [sys.executable, "-u", str(ROOT / "scripts/collect_serving_telemetry.py"),
                "--pid", str(process.pid), "--output", str(directory / "telemetry.jsonl"), "--interval", "5"]
            append_command(output, telemetry_argv, f"telemetry {stage}/{method}")
            telemetry = subprocess.Popen(telemetry_argv, cwd=ROOT, env=env, stdout=telemetry_log,
                stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            remember_owned_family(telemetry)
            save_json(directory / "owned_processes.json", {"server_pid": process.pid,
                "telemetry_pid": telemetry.pid, "started_utc": utc()})
            start = launch_started
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Service failed before readiness; inspect {directory / 'server.log'}")
                if owned_port(process, plan["port"]):
                    try:
                        if endpoint(base, "/health").get("ready"):
                            break
                    except (OSError, urllib.error.URLError, ValueError):
                        pass
                if time.monotonic() - start > plan["startup_timeout_seconds"]:
                    raise TimeoutError("Service did not become ready on its owned port")
                time.sleep(.5)
            save_json(directory / "ready.json", {"elapsed_seconds": time.monotonic() - start,
                "health": endpoint(base, "/health"), "ownership_verified": True,
                "scope": "process_spawn_to_healthy_including_monitor_startup_excluded_from_requests"})
            remember_owned_family(telemetry)
            save_json(directory / "owned_family_at_ready.json", {
                "server": _OWNED_FAMILIES.get(id(process), {}),
                "telemetry": _OWNED_FAMILIES.get(id(telemetry), {})})
            warmup = {"name": f"warmup_{stage}_{method}", "replay": plan["replay"],
                      "mode": "closed", "requests": plan["warmup_requests"], "concurrency": 1,
                      "seed": plan["seed"], "shuffle": True}
            warmup_summary = run_benchmark(warmup, directory / "warmup", plan, output, env,
                                           process, headline=False)
            if warmup_summary["successful"] != plan["warmup_requests"]:
                raise RuntimeError("Warmup had failed requests")
            yield process, env, directory
        finally:
            if process is not None:
                remember_owned_family(process)
            if telemetry is not None:
                remember_owned_family(telemetry)
            if process is not None and process.poll() is None:
                if owned_port(process, plan["port"]):
                    try:
                        endpoint(base, "/shutdown", post=True)
                        process.wait(timeout=150)
                    except (OSError, urllib.error.URLError, ValueError, subprocess.TimeoutExpired):
                        pass
            stop_owned(process)
            if telemetry is not None:
                try:
                    telemetry.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass
                stop_owned(telemetry)
            telemetry_rows = []
            if (directory / "telemetry.jsonl").exists():
                telemetry_rows = [json.loads(line) for line in
                    (directory / "telemetry.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            measurements = [r for r in telemetry_rows if r.get("event") == "measurement"]
            save_json(directory / "closed.json", {"closed_utc": utc(),
                "server_exit": process.returncode if process else None,
                "telemetry_exit": telemetry.returncode if telemetry else None,
                "telemetry_measurements": len(measurements),
                "gpu_measurements": sum(bool((r.get("gpu") or {}).get("name")) for r in measurements)})


def retrieval_observations(records: list[dict]) -> dict:
    fallback_reviews = fallback_aspects = observed = 0
    errors = Counter()
    for record in records:
        response = record.get("response") or {}
        retrieval = response.get("diagnostics", {}).get("batch_aspect", {}).get("retrieval", [])
        observed += bool(retrieval)
        fallbacks = [entry for entry in retrieval if entry.get("fallback") is True]
        fallback_reviews += bool(fallbacks)
        fallback_aspects += len(fallbacks)
        errors.update({entry.get("error_type", "unspecified") for entry in fallbacks})
    return {"responses_with_retrieval_diagnostics": observed,
            "fallback_review_count": fallback_reviews, "fallback_aspect_count": fallback_aspects,
            "fallback_error_review_counts": dict(errors),
            "scope": "fallbacks_are_retained_successful_HTTP_responses_not_silently_counted_as_normal_dynamic_retrieval"}


def aggregate(output: Path, plan: dict) -> dict:
    rows, burst = [], []
    for complete in sorted((output / "runs").glob("*/complete.json")):
        directory = complete.parent
        run, summary = read_json(directory / "registered_run.json"), read_json(directory / "summary.json")
        validate_run(summary, run)
        records = read_jsonl(directory / "requests.jsonl")
        rows.append({"run": directory.name, "mode": run["mode"],
            "measurement_role": "independent_capacity_pilot" if directory.name.startswith("capacity__")
                                else "performance_benchmark",
            "offered_rate": run.get("rate"), "concurrency": run.get("concurrency"),
            "attempts": summary["attempts"], "successful": summary["successful"],
            "sent": summary["sent"], "unsent": summary["attempts"] - summary["sent"],
            "failed": summary["failed"], "success_rate": summary["success_rate"],
            "successful_reviews_per_second": summary["successful_reviews_per_second"],
            "successful_latency": summary["successful_latency"],
            "http_status_counts": summary["http_status_counts"],
            "failure_counts": summary["failure_counts"],
            "response_budgets_seconds": summary["response_budgets_seconds"],
            "wall_seconds_including_drain": summary["wall_seconds_including_drain"],
            "retrieval": retrieval_observations(records),
            "phases": [{"name": p["name"], "duration": p["duration"], "offered_rate": p["rate"],
                "attempts": p["arrival_cohort_performance"]["attempts"],
                "sent": p["arrival_cohort_performance"]["sent"],
                "successful": p["arrival_cohort_performance"]["successful"],
                "failed": p["arrival_cohort_performance"]["failed"],
                "failure_counts": p["arrival_cohort_performance"]["failure_counts"],
                "successful_completed_inside_phase_window": p["successful_completed_inside_phase_window"],
                "completed_after_phase_boundary": p["completed_after_phase_boundary"]}
                for p in summary.get("phases", [])]})
        if directory.name.startswith("burst__"):
            from analyse_serving_resilience import recover_from_burst

            # Same-run baseline required; no implicit switch to a different method.
            burst.append({"run": directory.name,
                          **recover_from_burst(records, summary["phases"], [])})
    result = {"schema_version": "rag_http_service_summary_v1", "scope": plan["scope"],
              "baseline": plan["baseline"], "dynamic_candidate": plan["dynamic_candidate"],
              "candidate_promoted_by_quality": plan["candidate_promoted_by_quality"],
              "warmups_excluded": True, "all_failure_attempts_retained": True,
              "correctness_gate": read_json(output / "correctness" / "gate.json")
                  if (output / "correctness" / "gate.json").exists() else None,
              "runs": rows, "burst_recovery": burst}
    # The rolling aggregate is derivable from immutable completed runs. Timestamped
    # snapshots remain immutable; this convenience index can update as stages finish.
    save_json(output / "aggregate.json", result, mutable=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    plan = prepare_plan(args)
    output = args.output.resolve()
    if args.plan_only:
        print(json.dumps({"registered_plan": str(output / "plan.json"), "executed": False}))
        return 0
    lock_path = output / "running.lock"
    try:
        lock = lock_path.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise RuntimeError("Study already has an execution lock; inspect its owner before retrying") from error
    with lock:
        lock.write(json.dumps({"pid": os.getpid(), "started_utc": utc()}))
        lock.flush()
    invocation = uuid.uuid4().hex[:12]
    state_path = output / "invocations" / f"{invocation}.json"
    state = {"invocation": invocation, "stage": args.stage, "status": "running",
             "started_utc": utc(), "completed_runs": [], "skipped_completed_runs": []}
    save_json(state_path, state)
    try:
        stages = ["correctness", "capacity", "closed", "open", "soak", "burst"] if args.stage == "all" else [args.stage]
        for stage in stages:
            if stage == "correctness":
                for method in plan["methods"]:
                    if correctness_completed(output, method):
                        state["skipped_completed_runs"].append(f"correctness__{method}")
                        continue
                    with server_session(stage, method, plan, output) as (process, env, session):
                        state.update(current_session=session.name, current_method=method,
                                     current_run=f"correctness__{method}", updated_utc=utc())
                        save_json(state_path, state, mutable=True)
                        summary = run_correctness(method, plan, output, process)
                        state["completed_runs"].append(f"correctness__{method}")
                        save_json(state_path, state, mutable=True)
                        print(json.dumps({"completed": f"correctness__{method}",
                            "attempts": summary["attempts"], "exact_matches": summary["exact_matches"],
                            "passed": summary["passed"]}), flush=True)
                verify_correctness(output, plan)
                continue
            verify_correctness(output, plan)
            rates = None if stage in ("capacity", "closed") else freeze_common_rates(plan, output)
            for block in execution_blocks(stage, plan, rates):
                method, runs = block["method"], block["runs"]
                pending = []
                for run in runs:
                    if completed_run(output, run):
                        state["skipped_completed_runs"].append(run["name"])
                    else:
                        pending.append(run)
                if not pending:
                    continue
                with server_session(block["name"], method, plan, output) as (process, env, session):
                    state.update(current_session=session.name, current_method=method,
                                 current_block=block["name"])
                    for run in pending:
                        state.update(current_run=run["name"], updated_utc=utc())
                        save_json(state_path, state, mutable=True)
                        print(json.dumps({"starting": run["name"]}), flush=True)
                        summary = run_benchmark(run, output / "runs" / run["name"], plan,
                                                output, env, process, headline=stage != "capacity")
                        state["completed_runs"].append(run["name"])
                        save_json(state_path, state, mutable=True)
                        print(json.dumps({"completed": run["name"], "attempts": summary["attempts"],
                            "successes": summary["successful"], "p95_ms": summary["successful_latency"]["p95_ms"]}), flush=True)
                aggregate(output, plan)
            if stage == "capacity":
                freeze_common_rates(plan, output)
        result = aggregate(output, plan)
        save_json(output / "summaries" / f"{invocation}.json", result)
        state.update(status="completed", updated_utc=utc())
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     error=f"{type(error).__name__}: {error}", updated_utc=utc())
        raise
    finally:
        save_json(state_path, state, mutable=True)
        lock_path.unlink(missing_ok=True)
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--artifacts", type=Path, required=True)
    result.add_argument("--index", type=Path, required=True)
    result.add_argument("--encoder-path", type=Path)
    result.add_argument("--replay", type=Path, required=True)
    result.add_argument("--capacity-replay", type=Path, required=True)
    result.add_argument("--development-replay", type=Path, required=True,
                        help="Label-free development replay for the correctness gate")
    result.add_argument("--development-score-root", type=Path, required=True)
    result.add_argument("--correctness-reviews", type=int, default=100)
    result.add_argument("--selection", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--port", type=int, default=8766)
    result.add_argument("--stage", choices=["correctness", "capacity", "closed", "open", "soak", "burst", "all"], default="all")
    result.add_argument("--plan-only", action="store_true")
    result.add_argument("--seed", type=int, default=20261111)
    result.add_argument("--repeats", type=int, default=3)
    result.add_argument("--closed-requests", type=int, default=30)
    result.add_argument("--open-seconds", type=float, default=90)
    result.add_argument("--soak-seconds", type=float, default=3600)
    result.add_argument("--baseline-seconds", type=float, default=60)
    result.add_argument("--burst-seconds", type=float, default=30)
    result.add_argument("--recovery-seconds", type=float, default=180)
    result.add_argument("--startup-timeout", type=float, default=300)
    result.add_argument("--retrieval-timeout-s", type=float, default=10)
    return result


if __name__ == "__main__":
    with keep_system_awake():
        raise SystemExit(main())
