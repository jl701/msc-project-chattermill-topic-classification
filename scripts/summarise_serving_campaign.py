"""Create publication-safe aggregates from a local HTTP serving campaign.

Only known metric fields are exported. Review text, request/sample identities,
labels, command lines, and absolute workstation paths are never copied. Quality
joins are limited to runs explicitly marked quality_evaluation=true.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from msc_project.serving.monitoring import evaluate_quality, latency_summary, percentile, summarise_load


SERVER_KEYS = ["variant", "pair_batch_size", "max_batch_tokens", "request_batch_size",
               "batch_wait_ms", "queue_size", "timeout_s", "logits_mode", "prefix_cache"]
ENGINE_KEYS = ["variant", "fold_id", "model_id", "model_revision", "candidate_count",
               "pair_batch_size", "max_batch_tokens", "logits_mode", "quantization",
               "prediction_cache", "kv_cache", "prefix_cache", "artifacts_sha256"]
COUNTERS = ["received", "accepted", "rejected", "inference_completed", "late_inference_completed",
            "deadline_expired", "failed", "batches", "log_write_failures"]


def read_json(path, default=None):
    if not Path(path).exists():
        return {} if default is None else default
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    if not Path(path).exists():
        return []
    rows = []
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A live writer may be completing its final line. Earlier corruption
            # must fail rather than silently remove requests from denominators.
            if i != len(lines) - 1:
                raise
    return rows


def utc_timestamp(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def safe_identifier(value):
    """Names are identifiers, never evidence paths or arbitrary response text."""
    if value is None:
        return None
    text = str(value)
    if re.match(r"^[A-Za-z]:[\\/]", text) or text.startswith(("/", "\\")):
        return "[redacted_path]"
    return text.replace("\\", "_")[:200]


def numeric_stats(values):
    values = [float(v) for v in values if isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v)]
    return {"n": len(values), "mean": sum(values) / len(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None,
            "p50": percentile(values, 50), "p95": percentile(values, 95)}


def telemetry_summary(rows):
    rows = [r for r in rows if r.get("event") == "measurement"]
    fields = {
        "gpu_temperature_c": ("gpu", "temperature.gpu"),
        "gpu_utilisation_percent": ("gpu", "utilization.gpu"),
        "gpu_memory_used_mib": ("gpu", "memory.used"),
        "gpu_power_draw_w": ("gpu", "power.draw"),
        "gpu_sm_clock_mhz": ("gpu", "clocks.sm"),
        "gpu_memory_clock_mhz": ("gpu", "clocks.mem"),
        "server_process_tree_rss_bytes": ("process", "rss_bytes"),
        "server_process_tree_cpu_percent": ("process", "cpu_percent"),
        "server_process_tree_process_count": ("process", "process_count"),
    }
    values = {label: numeric_stats(r.get(group, {}).get(field) for r in rows)
              for label, (group, field) in fields.items()}
    states = {}
    for field in ["sw_power_cap", "hw_thermal_slowdown", "sw_thermal_slowdown"]:
        key = "clocks_event_reasons." + field
        observed = [r.get("gpu", {}).get(key) for r in rows]
        observed = [value for value in observed if isinstance(value, bool)]
        counter = "clocks_event_reasons_counters." + field
        counts = [r.get("gpu", {}).get(counter) for r in rows]
        counts = [value for value in counts if isinstance(value, (int, float)) and math.isfinite(value)]
        deltas = [b - a for a, b in zip(counts, counts[1:])]
        states[field] = {"observations": len(observed), "active_observations": sum(observed),
                         "active_sample_fraction": sum(observed) / len(observed) if observed else None,
                         "cumulative_counter_delta_raw": sum(v for v in deltas if v >= 0) if deltas else None,
                         "counter_unit": "microseconds",
                         "cumulative_counter_delta_seconds": sum(v for v in deltas if v >= 0) / 1_000_000 if deltas else None,
                         "counter_reset_count": sum(v < 0 for v in deltas)}
    return {"measurement_count": len(rows), "metrics": values, "clock_limit_states": states,
            "gpu_query_error_records": sum("gpu_query_error" in row for row in rows),
            "rss_note": "Sum of root process and recursive-child working sets, potentially counting shared pages more than once. Not unique physical memory.",
            "clock_counter_note": "NVIDIA reports these cumulative counters in microseconds. Only nonnegative increments between included measurements are attributed to the observation interval; the initial absolute value includes past activity. Counter increments and fractions of sampled instantaneous flags are different measurements: a false flag at each sample does not prove the counter recorded no limiting time between samples. These are driver-reported observations, not a separate physical validation of the limiting cause.",
            "sampling_note": "Sampled telemetry maxima are not continuous high-water marks; NVIDIA measurements are device-wide, not exclusively the server process."}


def inspect_sessions(campaign, plan):
    names = [s["name"] for s in plan.get("sessions", [])]
    output = []
    for directory in sorted((campaign / "server_sessions").glob("*")):
        if not directory.is_dir():
            continue
        matches = [name for name in names if directory.name.startswith(name + "_")]
        session = max(matches, key=len) if matches else directory.name.rsplit("_", 1)[0]
        metadata = read_json(directory / "server_metadata.json")
        startup = read_json(directory / "startup_observation.json")
        telemetry = read_jsonl(directory / "telemetry.jsonl")
        stamps = [utc_timestamp(row.get("utc")) for row in telemetry]
        stamps = [value for value in stamps if value is not None]
        planned = next((s["server"] for s in plan.get("sessions", []) if s["name"] == session), {})
        server = {**planned, **metadata.get("arguments", {})}
        public = {
            "session": safe_identifier(session), "instance": safe_identifier(directory.name),
            "configuration": {k: server[k] for k in SERVER_KEYS if k in server},
            "engine": {k: safe_identifier(v) if isinstance(v, str) else v for k, v in metadata.get("engine", {}).items() if k in ENGINE_KEYS},
            "source_sha256": {Path(k).name: v for k, v in metadata.get("source_sha256", {}).items()},
            "device": safe_identifier(metadata.get("device")),
            "gpu_bytes": metadata.get("gpu_bytes"),
            "python": safe_identifier(metadata.get("python")),
            "package_versions": {safe_identifier(k): safe_identifier(v) for k, v in metadata.get("versions", {}).items()},
            "model_load_s": metadata.get("model_load_s"),
            "launch_to_http_ready_s": startup.get("launch_to_http_ready_s"),
            "telemetry_scope": "Entire server session including loading, warm-up, all runs and idle intervals",
            "telemetry": telemetry_summary(telemetry),
        }
        output.append({"name": session, "directory": directory, "public": public, "telemetry": telemetry,
                       "utc_min": min(stamps) if stamps else None, "utc_max": max(stamps) if stamps else None})
    return output


def match_session(sessions, name, begin, end):
    candidates = [s for s in sessions if s["name"] == name]
    if begin is not None and end is not None:
        aligned = [s for s in candidates if s["utc_min"] is not None and
                   s["utc_min"] <= end and s["utc_max"] >= begin]
        if len(aligned) == 1:
            session = aligned[0]
            rows = [r for r in session["telemetry"] if r.get("event") == "measurement" and
                    (t := utc_timestamp(r.get("utc"))) is not None and begin <= t <= end]
            if rows:
                return session, telemetry_summary(rows), "Approximately run-aligned by UTC: config.utc_start is recorded just before the client monotonic epoch; telemetry is sampled every few seconds. Includes client drain."
            return session, session["public"]["telemetry"], "Session-level only: no telemetry measurement falls inside this short run's UTC window."
    if len(candidates) == 1:
        session = candidates[0]
        return session, session["public"]["telemetry"], "Session-level only: exact run alignment is unavailable."
    return None, None, "Unavailable: cannot unambiguously associate a server session."


def aggregate_run(campaign, name, session_name, plan_run, sessions, labels, taxonomy):
    directory = campaign / "runs" / name
    registered = {**plan_run, **read_json(directory / "registered_run.json")}
    config = read_json(directory / "config.json")
    existing = read_json(directory / "summary.json")
    rows = read_jsonl(directory / "requests.jsonl")
    if existing and existing.get("attempts") != len(rows):
        raise ValueError("Final summary attempt count differs from the persisted request records")
    if not directory.exists():
        status = "not_started"
    elif existing:
        status = "interrupted" if existing.get("interrupted") else "completed"
    else:
        status = "partial_no_final_summary"
    perf = summarise_load(rows, existing.get("wall_seconds_including_drain"))
    # Error keys are known categorical codes, not arbitrary exception messages.
    perf["failure_counts"] = {safe_identifier(k): v for k, v in perf["failure_counts"].items()}
    begin = utc_timestamp(config.get("utc_start"))
    end = begin + perf["wall_seconds_including_drain"] if begin is not None else None
    session, telemetry, telemetry_scope = match_session(sessions, session_name, begin, end)
    server = session["public"]["configuration"] if session else {}
    engine = session["public"]["engine"] if session else {}
    successful = [r for r in rows if r.get("success")]
    variants = {r.get("response", {}).get("variant") for r in successful}
    variants.discard(None)
    folds = {r.get("response", {}).get("fold_id") for r in successful}
    folds.discard(None)
    version = {r.get("response", {}).get("model_version") for r in successful}
    version.discard(None)
    if len(variants) > 1 or len(folds) > 1 or len(version) > 1:
        raise ValueError("Mixed model variants/folds/versions within one run")
    queue_wait = latency_summary(r.get("response", {}).get("timings_ms", {}).get("queue_wait")
                                for r in successful)
    service_time = latency_summary(r.get("response", {}).get("timings_ms", {}).get("worker_batch")
                                  for r in successful)
    observed_batch = Counter(str(r.get("response", {}).get("request_batch_size", "unknown")) for r in successful)
    ids = {r.get("sample_id", r.get("request_id")) for r in rows}
    successful_ids = {r.get("sample_id", r.get("request_id")) for r in successful}
    mode = config.get("mode", registered.get("mode"))
    before = read_json(directory / "server_metrics_before.json")
    after = read_json(directory / "server_metrics_after.json")
    deltas = {k: after[k] - before[k] for k in COUNTERS if isinstance(after.get(k), (int, float)) and isinstance(before.get(k), (int, float))}
    intervals = [max(0, r["started_s"] - r["scheduled_s"]) for r in rows if r.get("started_s") is not None]
    public = {
        "run": safe_identifier(name), "session": safe_identifier(session_name),
        "session_instance": session["public"]["instance"] if session else None,
        "status": status,
        "purpose": safe_identifier(registered.get("purpose", registered.get("tag", "unspecified"))),
        "quality_evaluation": registered.get("quality_evaluation") is True,
        "variant": next(iter(variants), engine.get("variant", server.get("variant"))),
        "fold_id": next(iter(folds), engine.get("fold_id")),
        "model_version": next(iter(version), engine.get("model_revision")),
        "model_id": engine.get("model_id"),
        "server_configuration": server,
        "client_mode": mode,
        "arrival_schedule": config.get("arrival_schedule", "constant" if mode == "open" else "closed"),
        "client_concurrency": config.get("concurrency", registered.get("concurrency")) if mode == "closed" else None,
        "offered_reviews_per_second": config.get("rate", registered.get("rate")) if mode == "open" else None,
        "registered_rate_fraction": registered.get("rate_fraction"),
        "reference_capacity_reviews_per_second": registered.get("reference_capacity_reviews_per_second"),
        "registered_arrival_duration_s": config.get("duration", registered.get("duration")),
        "independent_attempted_review_count": len(ids),
        "independent_successful_review_count": len(successful_ids),
        "request_repeats_after_first_sample": len(rows) - len(ids),
        "performance": perf,
        "queue_wait_successful_requests": queue_wait,
        "worker_batch_time_attached_to_successful_requests": service_time,
        "observed_request_batch_size_counts": dict(observed_batch),
        "engine_timing_scope": "Worker batch time is shared by every response in that batch and is response-weighted here. It is not per-request exclusive GPU time and must not be summed across responses.",
        "client_schedule_lag_over_100ms_count": sum(v > 0.1 for v in intervals),
        "server_counter_deltas": deltas,
        "server_unfinished_tasks_after": after.get("unfinished_tasks"),
        "cuda_process_counters_after": {k: v for k, v in after.get("runtime", {}).items() if k in ["cuda_allocated_bytes", "cuda_reserved_bytes", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"]},
        "cuda_peak_scope": "Peak counters are cumulative since server model load, not isolated run peaks unless explicitly reset by the server.",
        "telemetry_scope": telemetry_scope, "telemetry": telemetry,
    }
    if existing.get("phases"):
        public["phase_scope"] = "One continuous timeline with no intermediate drain. Latencies are scheduled-arrival cohorts; completion-window counts include earlier backlog. Cohort drain spans overlap and their QPS cannot be summed."
        public["phases"] = []
        for phase in existing["phases"]:
            clean_phase = {k: phase[k] for k in [
                "name", "duration", "rate", "scheduled_start_s", "scheduled_end_s",
                "actual_first_start_s", "actual_last_finish_s", "completed_after_phase_boundary",
                "successful_completed_inside_phase_window", "arrival_cohort_performance"] if k in phase}
            clean_phase["name"] = safe_identifier(clean_phase.get("name"))
            public["phases"].append(clean_phase)
    if registered.get("quality_evaluation") is True and rows:
        if not labels or not taxonomy:
            raise ValueError("Explicit quality run requires --labels and --artifacts")
        heldout = taxonomy.get("heldout_aspects", [])
        aspects = taxonomy.get("canonical_order", taxonomy.get("all_aspects", []))
        if not aspects:
            aspects = taxonomy.get("seen_aspects", []) + heldout
        public["quality"] = evaluate_quality(rows, labels, heldout, aspects)
    return public


def csv_write(path, rows, columns):
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarise(campaign, output, labels=None, taxonomy=None):
    campaign, output = Path(campaign), Path(output)
    plan = read_json(campaign / "plan.json")
    if not plan.get("sessions"):
        raise ValueError("Campaign plan.json must define sessions")
    state = read_json(campaign / "state.json")
    sessions = inspect_sessions(campaign, plan)
    plans = {session["name"] + "__" + run["name"]: (session["name"], run)
             for session in plan["sessions"] for run in session["runs"]}
    for directory in (campaign / "runs").glob("*"):
        if directory.is_dir() and directory.name not in plans:
            plans[directory.name] = (directory.name.split("__", 1)[0], {})
    runs = [aggregate_run(campaign, name, session, row, sessions, labels or [], taxonomy or {})
            for name, (session, row) in plans.items()]
    result = {
        "schema_version": "serving_campaign_aggregate_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": hashlib.sha256((campaign / "plan.json").read_bytes()).hexdigest(),
        "campaign_status": safe_identifier(state.get("status", "unknown")),
        "current_run": safe_identifier(state.get("current_run")),
        "registered_run_count": len(plans),
        "completed_run_count": sum(r["status"] == "completed" for r in runs),
        "scope": "Local single-GPU genuine-inference HTTP replay. One request is one complete review, not a review-candidate forward. Same-host client overhead included.",
        "quality_scope": "Only explicitly marked quality runs join delayed labels. Single frozen fold, unique reviews, all F1 values 0-1. Existing validation engineering replay is not a new blind test or the historical 12-fold average.",
        "privacy": "Aggregate-only export: no review text, row/sample/request IDs, label records, process IDs, command lines or absolute workstation paths.",
        "sessions": [s["public"] for s in sessions], "runs": runs,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "campaign_aggregate.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    performance_rows, quality_rows, telemetry_rows, phase_rows = [], [], [], []
    for row in runs:
        perf, cfg = row["performance"], row["server_configuration"]
        item = {k: row[k] for k in ["run", "status", "variant", "fold_id", "client_mode", "client_concurrency", "offered_reviews_per_second", "independent_attempted_review_count", "independent_successful_review_count"]}
        item.update({k: cfg.get(k) for k in ["request_batch_size", "pair_batch_size", "max_batch_tokens", "batch_wait_ms", "prefix_cache"]})
        item.update({k: perf[k] for k in ["attempts", "successful", "failed", "success_rate", "error_rate", "wall_seconds_including_drain", "successful_reviews_per_second"]})
        item["actual_sent_reviews_per_second_including_drain"] = perf["sent_reviews_per_second"]
        for key in ["p50_ms", "p95_ms", "p99_ms"]:
            item[key] = perf["successful_latency"][key]
            item["scheduled_" + key] = perf["successful_latency_from_scheduled_arrival"][key]
            item["queue_wait_" + key] = row["queue_wait_successful_requests"][key]
        for budget, value in perf["response_budgets_seconds"].items():
            item[f"completed_within_{budget}s_fraction_all_attempts"] = value["fraction_of_all_attempts"]
        item["failure_counts_json"] = json.dumps(perf["failure_counts"], sort_keys=True)
        item["telemetry_scope"] = row["telemetry_scope"]
        performance_rows.append(item)
        for phase in row.get("phases", []):
            phase_perf = phase["arrival_cohort_performance"]
            phase_rows.append({"run": row["run"], "phase": phase["name"],
                **{k: phase.get(k) for k in ["duration", "rate", "scheduled_start_s", "scheduled_end_s",
                    "actual_first_start_s", "actual_last_finish_s", "completed_after_phase_boundary",
                    "successful_completed_inside_phase_window"]},
                **{k: phase_perf[k] for k in ["attempts", "successful", "failed", "success_rate"]},
                **{k: phase_perf["successful_latency"][k] for k in ["p50_ms", "p95_ms", "p99_ms"]},
                "scope": "Scheduled-arrival cohort. Completion-window count may include prior backlog."})
        if "quality" in row:
            for view in ["successful_unique_quality", "delivered_quality_failure_as_empty"]:
                for scope, values in row["quality"][view].items():
                    qrow = {"run": row["run"], "variant": row["variant"], "fold_id": row["fold_id"],
                            "view": view, "scope": scope, "f1_scale": "0-1",
                            "unique_reviews": values["unique_reviews"],
                            "gold_presence_support": values["gold_presence_support"],
                            "gold_pair_support": values["gold_pair_support"],
                            "positive_review_count": values["positive_review_count"]}
                    for metric in ["presence", "pair"]:
                        qrow.update({metric + "_" + k: v for k, v in values[metric].items()})
                    quality_rows.append(qrow)
        if row["telemetry"]:
            for metric, stats in row["telemetry"]["metrics"].items():
                telemetry_rows.append({"run": row["run"], "scope": row["telemetry_scope"], "metric": metric, **stats})
    csv_write(output / "performance.csv", performance_rows, list(performance_rows[0]) if performance_rows else ["run"])
    csv_write(output / "quality.csv", quality_rows, list(quality_rows[0]) if quality_rows else ["run", "scope", "f1_scale"])
    csv_write(output / "telemetry.csv", telemetry_rows, list(telemetry_rows[0]) if telemetry_rows else ["run", "scope", "metric"])
    csv_write(output / "phases.csv", phase_rows, list(phase_rows[0]) if phase_rows else ["run", "phase", "scope"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, action="append", default=[])
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()
    if args.labels and not args.artifacts:
        parser.error("--artifacts required when joining labels")
    labels = [row for path in args.labels for row in read_jsonl(path)]
    taxonomy = read_json(args.artifacts).get("taxonomy", {}) if args.artifacts else {}
    result = summarise(args.campaign, args.output_dir, labels, taxonomy)
    print(json.dumps({"campaign_status": result["campaign_status"], "runs": len(result["runs"]),
                      "completed": result["completed_run_count"], "quality_runs": sum("quality" in r for r in result["runs"])}))


if __name__ == "__main__":
    main()
