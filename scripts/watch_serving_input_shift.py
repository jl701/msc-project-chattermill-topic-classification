"""Label-free CPU sidecar for future input-length-shift replay monitoring.

Waits for the resilience campaign and watches completed client outcomes. Emits
60-second completion-time windows and local drift signals with actual wall-clock
emission timestamps. No GPU execution, labels, external notifications or edits
to the server/client are involved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from msc_project.serving.monitoring import js_divergence, latency_summary, unlabelled_summary

WINDOW_SECONDS = 60
JS_THRESHOLD = 0.1
LOW_SAMPLE_BELOW = 20
EXPECTED_STUDY = "serving_resilience_20261109"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, json.JSONDecodeError):
        # Controllers write new state/config files in place; try again next poll.
        return None


def read_complete_jsonl(path):
    if not Path(path).exists():
        return []
    text = Path(path).read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    rows = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
    return rows


def primary_reference(study_root):
    """Read only aggregate config and completed request outcomes, never labels."""
    primary = Path(study_root) / "primary"
    plan = read_json(primary / "plan.json")
    if not plan:
        return None
    matches = [s for s in plan.get("sessions", []) if s.get("server", {}).get("variant") == "hybrid"]
    if len(matches) != 1:
        raise ValueError("Exactly one primary hybrid session is required")
    reference, sources = [], []
    for repeat in range(1, 4):
        name = matches[0]["name"] + f"__closed_c1_r{repeat}"
        run = primary / "runs" / name
        summary = read_json(run / "summary.json")
        if not summary:
            return None
        if summary.get("interrupted") is not False:
            raise ValueError("Primary reference was interrupted")
        rows = read_complete_jsonl(run / "requests.jsonl")
        if len(rows) != summary.get("attempts"):
            raise ValueError("Primary reference attempt count differs from persisted outcomes")
        reference.extend(rows)
        sources.append({"run": name, "attempts": len(rows),
                        "requests_sha256": hashlib.sha256((run / "requests.jsonl").read_bytes()).hexdigest()})
    return {"reference": unlabelled_summary(reference), "sources": sources}


def find_shift_run(study_root):
    matches = []
    for plan_path in sorted(Path(study_root).glob("*/plan.json")):
        campaign = plan_path.parent
        if not (campaign / "runs").is_dir():
            continue
        plan = read_json(plan_path)
        if not plan or plan.get("study_id") != EXPECTED_STUDY:
            continue
        for session in plan.get("sessions", []):
            for run in session.get("runs", []):
                if run.get("name") == "input_length_shift":
                    name = session["name"] + "__" + run["name"]
                    directory = campaign / "runs" / name
                    config = read_json(directory / "config.json")
                    if config:
                        matches.append((campaign, directory, config))
    if len(matches) > 1:
        raise ValueError("Multiple resilience input-shift runs exist; use a study root containing one target campaign")
    return matches[0] if matches else None


def waiting_failure_reason(study_root):
    """Stop waiting if an upstream controller failed before the target exists."""
    root = Path(study_root)
    for name in ("primary", "followup_controller"):
        state = read_json(root / name / "state.json") or {}
        if state.get("status") == "failed":
            return name + "_failed_before_input_shift"
    for path in sorted(root.glob("*/plan.json")):
        plan = read_json(path) or {}
        state = read_json(path.parent / "state.json") or {}
        if plan.get("study_id") == EXPECTED_STUDY and state.get("status") == "failed":
            return "resilience_failed_before_input_shift"
    return None


def completed_window(rows, index, reference):
    begin, end = index * WINDOW_SECONDS, (index + 1) * WINDOW_SECONDS
    selected = [r for r in rows if begin <= float(r["finished_s"]) < end]
    current = unlabelled_summary(selected)
    divergence = js_divergence(reference["length_bin_counts"], current["length_bin_counts"])
    n = len(selected)
    return {
        "window_start_s": begin, "window_end_s": end,
        "completed_attempts": n,
        "completed_successful": sum(r.get("success") is True for r in selected),
        "completed_failed": sum(r.get("success") is not True for r in selected),
        "success_fraction_of_completed_attempts": sum(r.get("success") is True for r in selected) / n if n else None,
        "input_length_js_bits": divergence,
        "input_length_shift_signal": divergence is not None and divergence > JS_THRESHOLD,
        "low_sample_warning": n < LOW_SAMPLE_BELOW,
        "unlabelled": current,
        "successful_latency": latency_summary(r.get("latency_ms") for r in selected if r.get("success")),
        "scope": "Completed client outcome cohort, including failures and client-capacity drops. Still-in-flight requests are not yet observable. Repeated requests count as traffic, not independent quality samples.",
    }


class OnlineWindows:
    """Emit new windows and explicit revisions when late persistence is observed."""
    def __init__(self, reference, attachment_elapsed_s):
        self.reference = reference
        self.attachment_elapsed_s = attachment_elapsed_s
        self.emitted = {}

    def update(self, rows, elapsed_s, emitted_utc, *, final=False, final_run_elapsed_s=None):
        # Completed regular windows use real observer time. Final partial windows
        # include every completed outcome, even if draining exceeded phase duration.
        last_record = max((r["finished_s"] for r in rows), default=0)
        observation_horizon = elapsed_s
        if final and final_run_elapsed_s is not None:
            observation_horizon = min(elapsed_s, max(float(final_run_elapsed_s), last_record))
        full_windows = max(0, math.floor(observation_horizon / WINDOW_SECONDS))
        final_windows = math.floor(last_record / WINDOW_SECONDS) + 1 if final and rows else 0
        count = max(full_windows, final_windows)
        events = []
        for index in range(count):
            metrics = completed_window(rows, index, self.reference)
            fingerprint = json.dumps(metrics, sort_keys=True)
            prior = self.emitted.get(index)
            if prior and prior["fingerprint"] == fingerprint:
                continue
            revision = 0 if prior is None else prior["revision"] + 1
            window_end = (index + 1) * WINDOW_SECONDS
            is_partial = final and window_end > observation_horizon
            backfill = window_end <= self.attachment_elapsed_s
            event = {
                "event": "completion_window" if prior is None else "completion_window_revision",
                "emitted_utc": emitted_utc, "observer_elapsed_s": elapsed_s,
                "revision": revision, "partial_final_window": is_partial,
                "attachment_was_after_window_end": backfill,
                "emission_lag_after_window_end_s": max(0, elapsed_s - window_end),
                "live_observation": not backfill,
                **metrics,
            }
            self.emitted[index] = {"fingerprint": fingerprint, "revision": revision, "latest": event}
            events.append(event)
        return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        raise SystemExit("Poll interval must be positive")
    if args.output_dir.exists():
        raise SystemExit("Refusing to overwrite sidecar observation evidence")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    stop = threading.Event()
    def interrupted(signum, frame):
        stop.set()
    previous = signal.signal(signal.SIGINT, interrupted)
    started = utc_now()
    metadata = {
        "observer_started_utc": started,
        "window_seconds": WINDOW_SECONDS,
        "input_length_js_threshold_bits": JS_THRESHOLD,
        "low_sample_warning_below": LOW_SAMPLE_BELOW,
        "source_time": "Client finished_s completion time, not scheduled-arrival time",
        "labels_read": False,
        "notification_destination": "Local JSONL only; no external messages",
        "scope": "CPU-only sidecar. Observes persisted completed client outcomes and never changes inference or workload.",
    }
    (args.output_dir / "observer_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    tracker = None
    reason = "observer_stopped"
    try:
        with (args.output_dir / "events.jsonl").open("x", encoding="utf-8") as output:
            def emit(event):
                output.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
                output.flush()
            emit({"event": "observer_started", "emitted_utc": started})
            reference = None
            target = None
            while not stop.is_set():
                failure = waiting_failure_reason(args.study_root)
                if failure:
                    reason = failure
                    break
                # Completed reference runs are immutable evidence. Cache once
                # while the much later resilience target is still pending.
                if reference is None:
                    reference = primary_reference(args.study_root)
                target = find_shift_run(args.study_root)
                if reference is not None and target is not None:
                    break
                stop.wait(args.poll_seconds)
            if target and reference and not stop.is_set():
                campaign, run, config = target
                attach_utc, attach_mono = utc_now(), time.perf_counter()
                attached_elapsed = max(0, timestamp(attach_utc) - timestamp(config["utc_start"]))
                tracker = OnlineWindows(reference["reference"], attached_elapsed)
                already_finished = (run / "summary.json").exists()
                emit({"event": "attached", "emitted_utc": attach_utc,
                      "run": run.name, "run_started_utc": config["utc_start"],
                      "attachment_elapsed_s": attached_elapsed,
                      "run_already_finished_at_attachment": already_finished,
                      "reference_sources": reference["sources"], "reference_unlabelled": reference["reference"],
                      "epoch_note": "At attachment, run elapsed time is approximated from config.utc_start; subsequent observer time advances monotonically. The client wrote config just before its monotonic start."})
                print(json.dumps({"attached_run": run.name, "elapsed_s": attached_elapsed,
                                  "already_finished": already_finished}), flush=True)
                while not stop.is_set():
                    rows = read_complete_jsonl(run / "requests.jsonl")
                    summary = read_json(run / "summary.json")
                    elapsed = attached_elapsed + time.perf_counter() - attach_mono
                    final = summary is not None
                    if final:
                        # The last outcome may have been persisted between the
                        # preceding requests snapshot and this summary read.
                        rows = read_complete_jsonl(run / "requests.jsonl")
                        if len(rows) != summary.get("attempts"):
                            raise ValueError("Completed input-shift summary differs from persisted outcome count")
                        # Flush final partial outcome cohort. Emission UTC remains
                        # actual observation time; no historical emission is forged.
                    for event in tracker.update(rows, elapsed, utc_now(), final=final,
                                                final_run_elapsed_s=summary.get("wall_seconds_including_drain") if summary else None):
                        emit(event)
                    if final:
                        reason = "run_interrupted" if summary.get("interrupted") else "run_completed"
                        emit({"event": "run_finished_observed", "emitted_utc": utc_now(),
                              "reason": reason, "completed_outcomes": len(rows)})
                        break
                    state = read_json(campaign / "state.json") or {}
                    if state.get("status") == "failed":
                        reason = "campaign_failed_before_run_summary"
                        break
                    stop.wait(args.poll_seconds)
            emit({"event": "observer_finished", "emitted_utc": utc_now(), "reason": reason})
    except Exception as error:
        reason = "observer_error_" + type(error).__name__
        raise
    finally:
        signal.signal(signal.SIGINT, previous)
        windows = [v["latest"] for _, v in sorted(tracker.emitted.items())] if tracker else []
        summary = {
            "observer_started_utc": started, "observer_finished_utc": utc_now(), "reason": reason,
            "labels_read": False, "window_count": len(windows),
            "signal_windows": sum(w["input_length_shift_signal"] for w in windows),
            "late_attachment_windows": sum(w["attachment_was_after_window_end"] for w in windows),
            "window_revision_count": sum(v["revision"] for v in tracker.emitted.values()) if tracker else 0,
            "windows": windows,
            "interpretation": "Signals are based on completed requests available at emission time. Queueing and service duration can move a long-input signal into a later completion window than the posthoc scheduled-arrival analysis. Emission timestamps are actual observer wall time. Low-count windows are descriptive only; JS signals do not prove quality decline. This sidecar never reads labels; delayed-label quality remains a separate later analysis.",
        }
        (args.output_dir / "observer_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"reason": reason, "windows": len(windows)}), flush=True)
    return 0 if reason == "run_completed" else 130 if stop.is_set() else 1


if __name__ == "__main__":
    raise SystemExit(main())
