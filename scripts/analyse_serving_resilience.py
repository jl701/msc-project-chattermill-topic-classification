"""CPU-only aggregate analysis of soak, burst recovery and input-length shift.

Recovery rules and the descriptive JS threshold are fixed in code before any
resilience results. Raw comments and sample/request identities are not exported.
Delayed labels are joined only for completed runs, and only for diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from msc_project.serving.monitoring import (
    evaluate_quality, js_divergence, latency_summary, summarise_load, unlabelled_summary,
)
from summarise_serving_campaign import (
    inspect_sessions, match_session, read_json, read_jsonl, safe_identifier, utc_timestamp,
)

RECOVERY_CONSECUTIVE_REQUESTS = 3
BASELINE_MIN_SUCCESS = 10
LATENCY_MARGIN_MULTIPLIER = 1.10
QUEUE_MIN_BAND_MS = 100.0
LENGTH_JS_THRESHOLD = 0.1
SHIFT_WINDOW_SECONDS = 60
SOAK_WINDOW_SECONDS = 300


def quality(rows, labels, taxonomy):
    result = evaluate_quality(rows, labels, taxonomy["heldout_aspects"], taxonomy["canonical_order"])
    result["purpose"] = "Post-run diagnostic only; repeated development reviews, not primary evaluation or a new independent quality sample."
    return result


def queue_waits(rows):
    return [r.get("response", {}).get("timings_ms", {}).get("queue_wait") for r in rows if r.get("success")]


def cohort_summary(rows):
    sent = [r for r in rows if r.get("started_s") is not None]
    successful = [r for r in rows if r.get("success")]
    attempts = len(rows)
    return {
        "attempts": attempts, "sent": len(sent), "successful": len(successful),
        "failed": attempts - len(successful),
        "success_rate": len(successful) / attempts if attempts else None,
        "unique_attempted_review_count": len({r.get("sample_id", r.get("request_id")) for r in rows}),
        "successful_latency": latency_summary(r.get("latency_ms") for r in successful),
        "successful_scheduled_latency": latency_summary(r.get("scheduled_latency_ms") for r in successful),
        "successful_queue_wait": latency_summary(queue_waits(rows)),
        "failure_counts": summarise_load(rows)["failure_counts"],
    }


def recover_from_burst(rows, phase_spec, primary_reference):
    """Report a declared request-level recovery band, never an inferred SLA."""
    phase_map = {p["name"]: p for p in phase_spec}
    if not {"normal", "burst", "recovery"} <= phase_map.keys():
        raise ValueError("Burst run must contain normal, burst and recovery phases")
    normal = [r for r in rows if r.get("phase") == "normal"]
    base_rows = [r for r in normal if r.get("success")]
    normal_successes = len(base_rows)
    baseline_source = "same_run_normal_phase"
    if len(base_rows) < BASELINE_MIN_SUCCESS:
        base_rows = [r for r in primary_reference if r.get("success")]
        baseline_source = ("primary_hybrid_closed_c1_three_repeats_fallback"
                           if primary_reference else "no_baseline")
    base_latency = latency_summary(r.get("latency_ms") for r in base_rows)
    base_queue = latency_summary(queue_waits(base_rows))
    enough_baseline = len(base_rows) >= BASELINE_MIN_SUCCESS
    threshold_latency = base_latency["p95_ms"] * LATENCY_MARGIN_MULTIPLIER if enough_baseline and base_latency["p95_ms"] is not None else None
    threshold_queue = max(QUEUE_MIN_BAND_MS, base_queue["p95_ms"] * LATENCY_MARGIN_MULTIPLIER) if enough_baseline and base_queue["p95_ms"] is not None else None
    recovery_start = float(phase_map["recovery"]["scheduled_start_s"])
    recovery_rows = sorted([r for r in rows if r.get("phase") == "recovery"],
                           key=lambda r: (r["scheduled_s"], r.get("attempt_index", 0)))
    first_in_band = None
    confirmed = None
    streak = []
    in_band_count = 0
    for row in recovery_rows:
        queue_wait = row.get("response", {}).get("timings_ms", {}).get("queue_wait")
        within = (row.get("success") and threshold_latency is not None and threshold_queue is not None
                  and isinstance(row.get("latency_ms"), (int, float)) and isinstance(queue_wait, (int, float))
                  and row["latency_ms"] <= threshold_latency and queue_wait <= threshold_queue)
        if within:
            in_band_count += 1
            if first_in_band is None:
                first_in_band = {"scheduled_arrival_after_burst_s": row["scheduled_s"] - recovery_start,
                                 "observed_completion_after_burst_s": row["finished_s"] - recovery_start}
            streak.append(row)
            if len(streak) == RECOVERY_CONSECUTIVE_REQUESTS and confirmed is None:
                confirmed = {
                    "first_request_scheduled_after_burst_s": streak[0]["scheduled_s"] - recovery_start,
                    "confirmed_by_completion_after_burst_s": max(r["finished_s"] for r in streak) - recovery_start,
                    "consecutive_requests": RECOVERY_CONSECUTIVE_REQUESTS,
                }
        else:
            streak = []
    queue_missing = sum(r.get("success") and r.get("response", {}).get("timings_ms", {}).get("queue_wait") is None for r in recovery_rows)
    if not enough_baseline:
        status = "not_evaluable_insufficient_baseline_requests"
        confirmed = None
    elif threshold_latency is None or threshold_queue is None:
        status = "not_evaluable_missing_baseline_timing"
    elif confirmed:
        status = "confirmed_under_declared_request_rule"
    elif len(recovery_rows) < RECOVERY_CONSECUTIVE_REQUESTS:
        status = "not_confirmed_insufficient_recovery_requests"
    elif queue_missing:
        status = "not_confirmed_missing_queue_measurements"
    else:
        status = "not_confirmed_within_observed_recovery_phase"
    phases = []
    for phase in phase_spec:
        subset = [r for r in rows if r.get("phase") == phase["name"]]
        phases.append({"phase": safe_identifier(phase["name"]),
                       "scheduled_start_s": phase["scheduled_start_s"],
                       "scheduled_end_s": phase["scheduled_end_s"],
                       "offered_reviews_per_second": phase["rate"], **cohort_summary(subset),
                       "completed_after_phase_end": sum(r["finished_s"] > phase["scheduled_end_s"] for r in subset)})
    return {
        "status": status, "baseline_source": baseline_source, "baseline_successful_requests": len(base_rows),
        "normal_phase_successful_requests": normal_successes,
        "baseline_latency": base_latency, "baseline_queue": base_queue,
        "latency_band_ms": threshold_latency, "queue_band_ms": threshold_queue,
        "first_request_in_band": first_in_band, "confirmation": confirmed,
        "recovery_attempts": len(recovery_rows), "recovery_in_band_attempts": in_band_count,
        "phases": phases,
        "rule": "Three consecutive scheduled recovery requests must all return successfully with client latency <= 1.10 x baseline P95 and queue wait <= max(100 ms, 1.10 x baseline queue P95). Failures reset the sequence. Baseline requires at least 10 successful requests; otherwise no confirmation.",
        "interpretation": "This is an operational observation rule, not a business SLA, confidence interval, or proof of permanent recovery. Resolution depends on arrival spacing. A first in-band request alone does not establish recovery; later deterioration remains visible in phase/window metrics.",
    }


def time_windows(rows, horizon, seconds, *, labels=None, taxonomy=None, reference=None,
                 sessions=None, session_name=None, epoch=None):
    ref_summary = unlabelled_summary(reference) if reference is not None else None
    windows = []
    for index in range(math.ceil(horizon / seconds)):
        begin, end = index * seconds, min((index + 1) * seconds, horizon)
        cohort = [r for r in rows if begin <= r["scheduled_s"] < end]
        completed = [r for r in rows if r.get("success") and begin <= r["finished_s"] < end]
        row = {"window_start_s": begin, "window_end_s": end,
               **cohort_summary(cohort), "successful_completions_in_wall_window": len(completed),
               "completion_throughput_reviews_per_second": len(completed) / (end - begin),
               "cohort_completions_after_window_end": sum(r["finished_s"] >= end for r in cohort)}
        if sessions is not None:
            _, telemetry, scope = match_session(sessions, session_name,
                epoch + begin if epoch is not None else None, epoch + end if epoch is not None else None)
            row["telemetry"] = telemetry
            row["telemetry_scope"] = scope
        if reference is not None:
            now = unlabelled_summary(cohort)
            js = js_divergence(ref_summary["length_bin_counts"], now["length_bin_counts"])
            row["unlabelled"] = now
            row["input_length_js_bits"] = js
            row["input_length_shift_signal"] = js is not None and js > LENGTH_JS_THRESHOLD
            row["signal_threshold_bits"] = LENGTH_JS_THRESHOLD
            row["minimum_sample_warning"] = len(cohort) < 20
        if labels is not None and taxonomy is not None and cohort:
            row["delayed_label_diagnostic"] = quality(cohort, labels, taxonomy)
        windows.append(row)
    return windows


def analyse(campaign, primary, output, labels, taxonomy, primary_session=None):
    campaign, primary, output = Path(campaign), Path(primary), Path(output)
    plan, state = read_json(campaign / "plan.json"), read_json(campaign / "state.json")
    primary_plan = read_json(primary / "plan.json")
    targets = [s for s in primary_plan.get("sessions", []) if s["server"].get("variant") == "hybrid"]
    if primary_session:
        targets = [s for s in targets if s["name"] == primary_session]
    if len(targets) != 1:
        raise ValueError("Select exactly one primary hybrid reference session")
    reference = []
    reference_sources = []
    for repeat in range(1, 4):
        name = targets[0]["name"] + f"__closed_c1_r{repeat}"
        directory = primary / "runs" / name
        summary = read_json(directory / "summary.json")
        if not summary or summary.get("interrupted") is not False:
            raise ValueError("Primary reference run is not completed: " + name)
        rows = read_jsonl(directory / "requests.jsonl")
        if len(rows) != summary.get("attempts"):
            raise ValueError("Reference request count does not match summary")
        reference.extend(rows)
        reference_sources.append({"run": safe_identifier(name), "attempts": len(rows),
                                  "requests_sha256": hashlib.sha256((directory / "requests.jsonl").read_bytes()).hexdigest()})
    sessions = inspect_sessions(campaign, plan)
    result = {
        "schema_version": "serving_resilience_analysis_v1", "campaign_status": safe_identifier(state.get("status")),
        "plan_sha256": hashlib.sha256((campaign / "plan.json").read_bytes()).hexdigest(),
        "reference_sources": reference_sources, "reference_unlabelled": unlabelled_summary(reference),
        "soak": [], "burst_recovery": [], "input_shift": [], "unfinished_runs": [],
        "rules": {"soak_window_seconds": SOAK_WINDOW_SECONDS, "shift_window_seconds": SHIFT_WINDOW_SECONDS,
                  "input_length_js_threshold_bits": LENGTH_JS_THRESHOLD,
                  "recovery_consecutive_successes": RECOVERY_CONSECUTIVE_REQUESTS,
                  "baseline_min_successful_requests": BASELINE_MIN_SUCCESS,
                  "latency_band_multiplier": LATENCY_MARGIN_MULTIPLIER,
                  "queue_band_floor_ms": QUEUE_MIN_BAND_MS},
        "interpretation": "Descriptive local replay analysis. Input drift is a proxy signal and is not equivalent to F1 degradation. Post-run label diagnostics deduplicate source reviews within each scope, but the same development reviews can recur across windows and phases. Phase differences can reflect input mix, not changes in model parameters. HTTP failures remain in attempt denominators.",
        "privacy": "Aggregate-only: no review text, request/sample identities, process IDs, command lines or absolute workstation paths.",
    }
    for session in plan["sessions"]:
        for run in session["runs"]:
            name = session["name"] + "__" + run["name"]
            if run["name"] == "warmup":
                continue
            directory = campaign / "runs" / name
            summary = read_json(directory / "summary.json")
            if not summary or summary.get("interrupted") is not False:
                result["unfinished_runs"].append(safe_identifier(name))
                continue
            rows = read_jsonl(directory / "requests.jsonl")
            if len(rows) != summary.get("attempts"):
                raise ValueError("Completed run request count differs from persisted records")
            config = read_json(directory / "config.json")
            epoch = utc_timestamp(config.get("utc_start"))
            if run["name"] == "soak_60min":
                horizon = float(config.get("duration", run.get("duration", 3600)))
                if summary["wall_seconds_including_drain"] < horizon:
                    raise ValueError("Soak did not cover its complete registered observation window")
                result["soak"].append({"run": safe_identifier(name),
                    "registered_duration_s": horizon, "summary": cohort_summary(rows),
                    "windows": time_windows(rows, horizon, SOAK_WINDOW_SECONDS, sessions=sessions,
                                            session_name=session["name"], epoch=epoch),
                    "window_scope": "Scheduled-arrival cohorts for latency/errors/queue; actual completion-time windows for throughput. Last cohort may drain after the registered observation window; all attempt outcomes remain included."})
            elif run["name"].startswith("burst_recovery"):
                phases = config.get("phases", [])
                if not phases:
                    raise ValueError("Burst run has no persisted phase boundaries")
                result["burst_recovery"].append({"run": safe_identifier(name),
                    **recover_from_burst(rows, phases, reference)})
            elif run["name"] == "input_length_shift":
                phases = config.get("phases", [])
                if not phases:
                    raise ValueError("Shift run has no persisted phase boundaries")
                horizon = float(phases[-1]["scheduled_end_s"])
                phase_quality = []
                for phase in phases:
                    cohort = [row for row in rows if row.get("phase") == phase["name"]]
                    phase_quality.append({"phase": safe_identifier(phase["name"]),
                        **cohort_summary(cohort), "delayed_label_diagnostic": quality(cohort, labels, taxonomy)})
                result["input_shift"].append({"run": safe_identifier(name),
                    "windows": time_windows(rows, horizon, SHIFT_WINDOW_SECONDS, reference=reference,
                                            labels=labels, taxonomy=taxonomy),
                    "phase_label_diagnostics": phase_quality,
                    "unlabelled_reference_rule": "Primary hybrid closed_c1_r1/r2/r3 development arrivals. Fixed character-length decile bins; JS > 0.1 bits is a declared descriptive signal, never tuned on labels.",
                    "small_sample_rule": "Windows below 20 attempts are flagged; signals are still displayed but cannot support precise false-alarm or recovery-rate claims."})
    output.mkdir(parents=True, exist_ok=True)
    (output / "resilience_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    window_rows, recovery_rows, phase_rows = [], [], []
    for kind in ["soak", "input_shift"]:
        for run in result[kind]:
            for row in run["windows"]:
                flat = {"run": run["run"], "kind": kind,
                        **{k: row.get(k) for k in ["window_start_s", "window_end_s", "attempts", "successful", "failed", "success_rate", "unique_attempted_review_count", "successful_completions_in_wall_window", "completion_throughput_reviews_per_second", "input_length_js_bits", "input_length_shift_signal", "minimum_sample_warning"]}}
                for measure in ["successful_latency", "successful_queue_wait"]:
                    flat.update({measure + "_" + p: row[measure][p] for p in ["p50_ms", "p95_ms", "p99_ms"]})
                telemetry = row.get("telemetry") or {}
                metrics = telemetry.get("metrics", {})
                flat["telemetry_scope"] = row.get("telemetry_scope")
                flat["rss_tree_max_bytes"] = metrics.get("server_process_tree_rss_bytes", {}).get("max")
                flat["gpu_used_max_mib"] = metrics.get("gpu_memory_used_mib", {}).get("max")
                flat["gpu_temperature_max_c"] = metrics.get("gpu_temperature_c", {}).get("max")
                if "delayed_label_diagnostic" in row:
                    for scope, values in row["delayed_label_diagnostic"]["successful_unique_quality"].items():
                        flat[scope + "_pair_micro_f1_0_1"] = values["pair"]["micro_f1"]
                        flat[scope + "_gold_pair_support"] = values["gold_pair_support"]
                window_rows.append(flat)
    for run in result["burst_recovery"]:
        recovery_rows.append({"run": run["run"], "status": run["status"], "baseline_source": run["baseline_source"],
            "baseline_successful_requests": run["baseline_successful_requests"],
            "latency_band_ms": run["latency_band_ms"], "queue_band_ms": run["queue_band_ms"],
            "recovery_attempts": run["recovery_attempts"], "recovery_in_band_attempts": run["recovery_in_band_attempts"],
            "first_request_in_band_arrival_after_burst_s": (run["first_request_in_band"] or {}).get("scheduled_arrival_after_burst_s"),
            "confirmed_by_completion_after_burst_s": (run["confirmation"] or {}).get("confirmed_by_completion_after_burst_s")})
        for phase in run["phases"]:
            phase_rows.append({"run": run["run"], **{k: phase[k] for k in ["phase", "scheduled_start_s", "scheduled_end_s", "offered_reviews_per_second", "attempts", "successful", "failed", "success_rate", "completed_after_phase_end"]},
                               "p50_ms": phase["successful_latency"]["p50_ms"], "p95_ms": phase["successful_latency"]["p95_ms"],
                               "queue_p95_ms": phase["successful_queue_wait"]["p95_ms"]})
    for name, values in [("time_windows.csv", window_rows), ("burst_recovery.csv", recovery_rows), ("burst_phases.csv", phase_rows)]:
        columns = list(dict.fromkeys(k for row in values for k in row)) or ["run"]
        with (output / name).open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(values)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--primary-campaign", type=Path, required=True)
    parser.add_argument("--development-labels", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-session")
    args = parser.parse_args()
    result = analyse(args.campaign, args.primary_campaign, args.output_dir,
                     read_jsonl(args.development_labels), read_json(args.artifacts)["taxonomy"], args.primary_session)
    print(json.dumps({"soaks": len(result["soak"]), "bursts": len(result["burst_recovery"]),
                      "shifts": len(result["input_shift"]), "unfinished_runs": len(result["unfinished_runs"])}))


if __name__ == "__main__":
    main()
