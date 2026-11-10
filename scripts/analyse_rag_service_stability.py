"""CPU-only stability evidence for one registered RAG HTTP campaign.

No labels, model loading, GPU calls or earlier campaign results are read. Only
completed, internally consistent open-arrival runs are promoted to aggregates.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import csv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from msc_project.serving.monitoring import latency_summary, prediction_pairs
from summarise_serving_campaign import numeric_stats, telemetry_summary

WINDOW_SECONDS = 300
METHODS = {"fixed", "random", "bm25", "e5", "rrf"}
SELECTION_POLICIES = {"global", "same_aspect_negative_seen"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def stamp(value):
    date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if date.tzinfo is None:
        raise ValueError("Evidence timestamps must include a UTC offset")
    return date.timestamp()


def internal_path(value, campaign):
    """Prevent this analysis from silently importing another campaign's replay."""
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    if not path.is_relative_to(campaign.resolve()):
        raise ValueError("Referenced replay/phase evidence is outside this campaign")
    return path


def safe_name(value):
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError("Run name is not a safe aggregate identifier")
    return value


def error_counts(rows):
    counts = Counter()
    for row in rows:
        if row.get("success"):
            continue
        value = str(row.get("error") or "unspecified")
        # Exception messages and machine paths are never exported as categories.
        category = value if re.fullmatch(r"[A-Za-z0-9_]{1,80}", value) else "other_error"
        counts[category] += 1
    return dict(sorted(counts.items()))


def fallback_counts(rows):
    observed = fallbacks = successful_fallbacks = 0
    errors = Counter()
    for row in rows:
        diag = (row.get("response") or {}).get("diagnostics") or {}
        flag = diag.get("retrieval_fallback")
        entries = (diag.get("batch_aspect") or {}).get("retrieval") or []
        if not isinstance(flag, bool) and entries:
            flag = any(item.get("fallback") is True for item in entries)
        if isinstance(flag, bool):
            observed += 1
            fallbacks += flag
            successful_fallbacks += flag and row.get("success") is True
            if flag:
                categories = {str(item.get("error_type") or "unspecified") for item in entries
                              if item.get("fallback") is True} or {"unspecified"}
                errors.update(value if re.fullmatch(r"[A-Za-z0-9_]{1,80}", value) else "other_error"
                              for value in categories)
    return {"attempts_with_fallback_observation": observed,
            "fallback_attempts": fallbacks,
            "successful_fallback_attempts": successful_fallbacks,
            "fallback_error_review_counts": dict(sorted(errors.items())),
            "retrieval_timeout_fallback_attempts": errors.get("RetrievalTimeoutError", 0),
            "fallback_fraction_of_observed_attempts": fallbacks / observed if observed else None,
            "unobserved_fallback_attempts": len(rows) - observed}


def repeat_consistency(rows, input_hashes):
    """Compare final sets only within identical observed model/threshold policy."""
    groups = defaultdict(list)
    input_policies = defaultdict(set)
    successful = [r for r in rows if r.get("success") is True]
    unqualified = 0
    for row in successful:
        response = row.get("response") or {}
        diag = response.get("diagnostics") or {}
        thresholds = diag.get("effective_thresholds")
        method = diag.get("effective_retrieval_method")
        selection_policy = diag.get("effective_retrieval_selection_policy")
        if (not response.get("model_version") or not isinstance(thresholds, dict) or
                not thresholds or method not in METHODS or
                selection_policy not in SELECTION_POLICIES or
                (selection_policy == "same_aspect_negative_seen" and method != "e5") or
                not isinstance(response.get("predictions"), list)):
            unqualified += 1
            continue
        policy = {"model_version": response["model_version"], "fold": response.get("fold_id"),
                  "variant": response.get("variant"), "thresholds": thresholds,
                  "method": method, "selection_policy": selection_policy,
                  "fallback": diag.get("retrieval_fallback"),
                  "fallback_policy_source_sha256": diag.get("fallback_policy_source_sha256")}
        key = fingerprint(policy)
        input_key = input_hashes[row["sample_id"]]
        input_policies[input_key].add(key)
        groups[key].append((row, input_key, frozenset(prediction_pairs(response)), method,
                            selection_policy))
    result = []
    for key, items in sorted(groups.items()):
        first, variants = {}, defaultdict(set)
        counts = Counter(item[1] for item in items)
        disagreements = repeats = 0
        for row, input_key, pairs, method, selection_policy in sorted(
                items, key=lambda item: item[0]["attempt_index"]):
            if input_key in first:
                repeats += 1
                disagreements += pairs != first[input_key]
            else:
                first[input_key] = pairs
            variants[input_key].add(pairs)
        result.append({"observed_policy_sha256": key, "effective_retrieval_method": method,
            "effective_retrieval_selection_policy": selection_policy,
            "successful_attempts": len(items), "unique_inputs": len(first),
            "inputs_with_repeated_successes": sum(n > 1 for n in counts.values()),
            "repeated_successful_deliveries_after_first": repeats,
            "inputs_with_differing_prediction_sets": sum(len(v) > 1 for v in variants.values()),
            "repeated_deliveries_differing_from_first": disagreements})
    return {"attempts": len(rows), "successful_attempts": len(successful),
        "failed_attempts": len(rows) - len(successful),
        "unique_attempted_inputs": len({input_hashes[r["sample_id"]] for r in rows}),
        "successful_attempts_missing_policy_identity": unqualified,
        "same_input_observed_under_multiple_policies": sum(len(p) > 1 for p in input_policies.values()),
        "policy_groups": result,
        "interpretation": "Within this run only, exact input text and observed model/threshold/retrieval method/selection/fallback policy define a group. Different policies are never pooled. Missing identity, including an absent effective retrieval selection policy, is excluded from consistency claims, but remains in delivery counts. Agreement of final label sets is not correctness, score equivalence, independent sample growth or universal determinism."}


def planned_schedule(registered, config, campaign):
    if registered.get("mode") != "open":
        raise ValueError("Stability analysis requires an independently scheduled open run")
    if registered.get("phase_plan"):
        phases = read_json(internal_path(registered["phase_plan"], campaign))["phases"]
    else:
        phases = [{"duration": registered["duration"], "rate": registered["rate"]}]
    schedule, offset = [], 0.0
    for phase in phases:
        duration, rate = float(phase["duration"]), float(phase["rate"])
        if not all(math.isfinite(v) and v > 0 for v in (duration, rate)):
            raise ValueError("Invalid registered arrival schedule")
        index = 0
        while index / rate < duration:
            schedule.append(offset + index / rate)
            index += 1
        offset += duration
    if config.get("mode") != "open":
        raise ValueError("Client mode differs from the registered open schedule")
    return schedule, offset


def verify_run(directory, campaign):
    complete = read_json(directory / "complete.json")
    if complete["summary_sha256"] != sha(directory / "summary.json"):
        raise ValueError("Completed summary fingerprint differs")
    registered, config = read_json(directory / "registered_run.json"), read_json(directory / "config.json")
    summary, rows = read_json(directory / "summary.json"), read_jsonl(directory / "requests.jsonl")
    schedule, horizon = planned_schedule(registered, config, campaign)
    wall = float(summary["wall_seconds_including_drain"])
    if (summary.get("interrupted") is not False or not math.isfinite(wall) or wall < horizon or
            len(rows) != len(schedule) or summary.get("attempts") != len(rows)):
        raise ValueError("Run is incomplete, interrupted, or missing registered attempts")
    if sorted(r["attempt_index"] for r in rows) != list(range(len(schedule))):
        raise ValueError("Duplicate or missing attempt indices")
    if len({r["request_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate attempt request identities")
    for row in rows:
        expected = schedule[row["attempt_index"]]
        if (not math.isfinite(float(row["scheduled_s"])) or
                abs(row["scheduled_s"] - expected) > 1e-6 or
                not math.isfinite(float(row["finished_s"])) or
                row["finished_s"] < expected or row["finished_s"] > wall + 1e-6):
            raise ValueError("Persisted timing differs from the registered schedule/window")
        started = row.get("started_s")
        if started is not None:
            if not math.isfinite(float(started)) or started < expected - 1e-6 or started > row["finished_s"]:
                raise ValueError("Invalid actual request start time")
            timings = {"latency_ms": (row["finished_s"] - started) * 1000,
                       "scheduled_latency_ms": (row["finished_s"] - row["scheduled_s"]) * 1000,
                       "schedule_lag_ms": max(0, started - row["scheduled_s"]) * 1000}
            for key, expected_ms in timings.items():
                value = row.get(key)
                if (not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or
                        not math.isclose(value, expected_ms, rel_tol=1e-9, abs_tol=1e-3)):
                    raise ValueError("Persisted latency differs from request timestamps")
        elif row.get("success"):
            raise ValueError("An unsent request cannot be successful")
        if row.get("success") is True and ((row.get("response") or {}).get("request_id") != row["request_id"] or
                not isinstance((row.get("response") or {}).get("predictions"), list)):
            raise ValueError("Successful response identity or prediction contract differs")
    successes = sum(r.get("success") is True for r in rows)
    if summary.get("successful") != successes or summary.get("failed") != len(rows) - successes:
        raise ValueError("Summary delivery counts differ from raw completed outcomes")
    if summary.get("sent") != sum(r.get("started_s") is not None for r in rows):
        raise ValueError("Summary sent count differs from persisted sent/drop outcomes")
    replay_path = internal_path(registered["replay"], campaign)
    if config.get("replay_sha256") != sha(replay_path):
        raise ValueError("Frozen replay differs from client replay fingerprint")
    inputs = read_jsonl(replay_path)
    if any(set(r) != {"request_id", "text"} for r in inputs):
        raise ValueError("Frozen replay must have request_id/text only; labels are not inputs")
    input_hashes = {r["request_id"]: hashlib.sha256(r["text"].encode()).hexdigest() for r in inputs}
    if len(input_hashes) != len(inputs) or any(r.get("sample_id") not in input_hashes for r in rows):
        raise ValueError("Request sample identity does not map uniquely to this frozen replay")
    return registered, config, summary, rows, schedule, horizon, wall, input_hashes


def session_telemetry(campaign, begin, end):
    matches = []
    for directory in sorted((campaign / "sessions").glob("*")):
        if not (directory / "launch.json").exists() or not (directory / "closed.json").exists():
            continue
        launch, closed = read_json(directory / "launch.json"), read_json(directory / "closed.json")
        if stamp(launch["registered_utc"]) <= begin and stamp(closed["closed_utc"]) >= end:
            matches.append(directory)
    if len(matches) != 1:
        return [], {"available": False, "reason": "No unambiguous completed session encloses this run"}
    path = matches[0] / "telemetry.jsonl"
    if not path.exists():
        return [], {"available": False, "reason": "Matched session has no telemetry file"}
    rows = [r for r in read_jsonl(path) if r.get("event") == "measurement" and
            begin <= stamp(r["utc"]) <= end]
    return rows, {"available": bool(rows), "telemetry_sha256": sha(path),
        "scope": "Approximately aligned by client config.utc_start, recorded just before its monotonic epoch. Only this run's own enclosing session is used. No session-wide or old-campaign fallback.",
        "measurement_count": len(rows)}


def cohort_metrics(rows, planned_count):
    successful = [r for r in rows if r.get("success") is True]
    failures = error_counts(rows)
    within_2s = sum(r["scheduled_latency_ms"] <= 2000 for r in successful)
    return {"planned_attempts": planned_count, "observed_attempts": len(rows),
        "successful": len(successful), "failed": len(rows) - len(successful),
        "sent": sum(r.get("started_s") is not None for r in rows),
        "unsent_client_drops": sum(r.get("started_s") is None for r in rows),
        "success_fraction_all_planned_attempts": len(successful) / planned_count if planned_count else None,
        "completed_within_2s": within_2s,
        "within_2s_fraction_all_planned_attempts": within_2s / planned_count if planned_count else None,
        "timeout_failures": sum(n for k, n in failures.items() if k in {"timeout", "http_504"}),
        "http_429_failures": failures.get("http_429", 0),
        "http_503_failures": failures.get("http_503", 0), "failure_counts": failures,
        "successful_http_latency": latency_summary(r.get("latency_ms") for r in successful),
        "successful_scheduled_latency": latency_summary(r.get("scheduled_latency_ms") for r in successful),
        "successful_queue_wait": latency_summary((r.get("response") or {}).get("timings_ms", {}).get("queue_wait") for r in successful),
        "sent_schedule_lag": latency_summary(r.get("schedule_lag_ms") for r in rows if r.get("started_s") is not None),
        "input_character_length": numeric_stats(r.get("input_chars") for r in rows),
        **fallback_counts(rows)}


def time_windows(rows, schedule, horizon, wall, telemetry, begin):
    intervals = []
    start = 0.0
    while start < horizon:
        end = min(horizon, start + WINDOW_SECONDS)
        intervals.append((start, end, "full_arrival_window" if end - start == WINDOW_SECONDS else "partial_arrival_tail"))
        start = end
    start = horizon
    while start < wall:
        end = min(wall, start + WINDOW_SECONDS)
        intervals.append((start, end, "drain_only_tail"))
        start = end
    result = []
    for start, end, kind in intervals:
        cohort = [r for r in rows if start <= r["scheduled_s"] < end]
        planned = sum(start <= t < end for t in schedule)
        completions = [r for r in rows if start <= r["finished_s"] < end or
                       end == wall and r["finished_s"] == wall]
        resource = [r for r in telemetry if begin + start <= stamp(r["utc"]) < begin + end]
        result.append({"window_start_s": start, "window_end_s": end, "observed_wall_seconds": end - start,
            "window_kind": kind, "is_full_five_minute_arrival_window": kind == "full_arrival_window",
            **cohort_metrics(cohort, planned),
            "cohort_completions_after_window_end": sum(r["finished_s"] >= end for r in cohort),
            "successful_completions_in_wall_window": sum(r.get("success") is True for r in completions),
            "failed_completions_in_wall_window": sum(r.get("success") is not True for r in completions),
            "completion_throughput_reviews_per_second": sum(r.get("success") is True for r in completions) / (end - start),
            "sampled_resources": telemetry_summary(resource)})
    return result


def analyse_run(directory, campaign):
    registered, config, summary, rows, schedule, horizon, wall, input_hashes = verify_run(directory, campaign)
    begin = stamp(config["utc_start"])
    telemetry, resource_scope = session_telemetry(campaign, begin, begin + wall)
    windows = time_windows(rows, schedule, horizon, wall, telemetry, begin)
    full = [w for w in windows if w["is_full_five_minute_arrival_window"]]
    peaks = [w["sampled_resources"]["metrics"]["gpu_memory_used_mib"]["max"] for w in full]
    peaks = [v for v in peaks if v is not None]
    return {"run": safe_name(directory.name), "mode": "open", "registered_arrival_seconds": horizon,
        "wall_seconds_including_drain": wall, "full_five_minute_window_count": len(full),
        "summary": cohort_metrics(rows, len(schedule)),
        "successful_reviews_per_second_including_drain": summary["successful"] / wall,
        "repeat_consistency": repeat_consistency(rows, input_hashes),
        "resource_alignment": resource_scope, "whole_run_sampled_resources": telemetry_summary(telemetry),
        "range_of_full_window_gpu_memory_maxima_mib": [min(peaks), max(peaks)] if peaks else None,
        "windows": windows,
        "input_hashes": {name: sha(directory / name) for name in
            ("registered_run.json", "config.json", "summary.json", "requests.jsonl", "complete.json")}}


def analyse_campaign(campaign, prefixes=("soak__",)):
    campaign = campaign.resolve()
    plan = read_json(campaign / "plan.json")
    if plan.get("schema_version") != "rag_http_service_study_v1":
        raise ValueError("Expected this RAG HTTP service schema; earlier serving campaigns are not accepted")
    runs, incomplete = [], []
    for directory in sorted((campaign / "runs").glob("*")):
        if not directory.is_dir() or not any(directory.name.startswith(p) for p in prefixes):
            continue
        if not (directory / "complete.json").exists():
            incomplete.append(safe_name(directory.name))
        else:
            runs.append(analyse_run(directory, campaign))
    if not runs:
        raise ValueError("No completed matching runs; no stability result can be produced")
    return {"schema_version": "rag_service_stability_v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": sha(campaign / "plan.json"), "labels_read": False,
        "analysis_source_sha256": sha(Path(__file__)), "window_seconds": WINDOW_SECONDS,
        "analysed_complete_run_count": len(runs), "incomplete_matching_runs": incomplete,
        "scope": "Only selected completed runs in this RAG campaign; not a declaration that every campaign stage is complete.",
        "window_scope": "Arrival cohorts retain every scheduled attempt, including 429/503/504/timeouts and unsent drops. Successful latency quantiles exclude failures; success and response-budget fractions retain all planned attempts. Completion-window throughput includes earlier backlog. Partial arrival and drain-only tails are separate, never described as full five-minute windows.",
        "interpretation": "Descriptive delivery, queue, latency, fallback, resource and repeat-consistency evidence. No gold labels, quality F1, drift threshold, automatic SLA pass or proof of no memory leak. Resource samples are device-wide; process-tree RSS can double-count shared pages. A stable response set can still be wrong.",
        "privacy": "Aggregates and evidence hashes only; no input text, sample/request IDs, per-input output sets, process IDs or machine paths.",
        "runs": runs}


def write_outputs(result, output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "stability.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    flat = []
    for run in result["runs"]:
        for window in run["windows"]:
            row = {"run": run["run"], **{k: window[k] for k in
                ("window_start_s", "window_end_s", "observed_wall_seconds", "window_kind", "planned_attempts", "observed_attempts", "sent", "unsent_client_drops", "successful", "failed", "success_fraction_all_planned_attempts", "within_2s_fraction_all_planned_attempts", "timeout_failures", "http_429_failures", "http_503_failures", "fallback_attempts", "retrieval_timeout_fallback_attempts", "attempts_with_fallback_observation", "unobserved_fallback_attempts", "successful_completions_in_wall_window", "failed_completions_in_wall_window", "completion_throughput_reviews_per_second")}}
            for name in ("successful_http_latency", "successful_scheduled_latency", "successful_queue_wait"):
                for q in ("p50_ms", "p95_ms", "p99_ms"):
                    row[name + "_" + q] = window[name][q]
            row["resource_measurement_count"] = window["sampled_resources"]["measurement_count"]
            for name in ("gpu_memory_used_mib", "server_process_tree_rss_bytes", "gpu_temperature_c"):
                for metric in ("min", "max", "p50"):
                    row[name + "_" + metric] = window["sampled_resources"]["metrics"][name][metric]
            flat.append(row)
    with (output / "windows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-prefix", action="append", help="Default soak__; use open__ or burst__ explicitly if needed")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("Refusing to overwrite stability analysis evidence")
    result = analyse_campaign(args.campaign, tuple(args.run_prefix or ["soak__"]))
    write_outputs(result, args.output_dir)
    print(json.dumps({"analysed_complete_runs": result["analysed_complete_run_count"],
                      "incomplete_matching_runs": len(result["incomplete_matching_runs"]), "labels_read": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
