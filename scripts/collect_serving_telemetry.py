"""Read-only GPU and server-process telemetry during a local serving campaign.

Stops when the selected PID exits or is reused. No GPU load is generated. The
JSONL starts with configuration, followed by time-stamped measurement records.
Unsupported optional NVIDIA counters are omitted or retained as unavailable.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import threading
import time

import psutil


BASE_FIELDS = ["index", "name", "temperature.gpu", "utilization.gpu", "memory.used",
               "memory.total", "power.draw", "power.limit", "clocks.sm", "clocks.mem"]
OPTIONAL_FIELDS = ["clocks_event_reasons.sw_power_cap", "clocks_event_reasons.hw_thermal_slowdown",
                   "clocks_event_reasons.sw_thermal_slowdown",
                   "clocks_event_reasons_counters.sw_power_cap",
                   "clocks_event_reasons_counters.sw_thermal_slowdown",
                   "clocks_event_reasons_counters.hw_thermal_slowdown"]


def command(arguments, timeout=5):
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=timeout,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def parse_value(value):
    value = value.strip()
    if value in ["[N/A]", "N/A", "[Not Supported]", "Not Supported"]:
        return None
    if value.lower() in ["active", "not active"]:
        return value.lower() == "active"
    try:
        return float(value) if "." in value else int(value)
    except ValueError:
        return value


def gpu_record(fields, index):
    text = command(["nvidia-smi", "--id=" + str(index), "--query-gpu=" + ",".join(fields),
                    "--format=csv,noheader,nounits"])
    values = next(csv.reader(text.strip().splitlines()))
    if len(values) != len(fields):
        raise ValueError("Unexpected NVIDIA query field count")
    return {key: parse_value(value) for key, value in zip(fields, values)}


def process_tree_record(root, process_cache):
    """Aggregate the launcher and recursive workers without missing Windows venv children.

    Per-process RSS is summed (shared pages may be represented more than once).
    CPU uses persistent Process objects so successive cpu_percent() calls measure
    intervals rather than returning the initial zero on every newly found child.
    """
    processes = [root] + root.children(recursive=True)
    members = []
    live_keys = set()
    for candidate in processes:
        try:
            key = (candidate.pid, candidate.create_time())
            live_keys.add(key)
            first_observation = key not in process_cache
            if first_observation:
                process_cache[key] = candidate
            process = process_cache[key]
            with process.oneshot():
                cpu = process.cpu_percent(None)
                item = {"pid": process.pid, "parent_pid": process.ppid(),
                        "executable_basename": Path(process.exe()).name,
                        "rss_bytes": process.memory_info().rss,
                        "cpu_percent": None if first_observation else cpu,
                        "num_threads": process.num_threads()}
            members.append(item)
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            members.append({"pid": candidate.pid, "error": "AccessDenied"})
    for key in list(process_cache):
        if key not in live_keys:
            del process_cache[key]
    available = [item for item in members if "rss_bytes" in item]
    cpu_complete = bool(available) and len(available) == len(members) and all(
        item["cpu_percent"] is not None for item in available)
    return {
        "pid": root.pid, "root_pid": root.pid, "process_count": len(members),
        "rss_bytes": sum(item["rss_bytes"] for item in available),
        "cpu_percent": sum(item["cpu_percent"] for item in available) if cpu_complete else None,
        "cpu_measurement_complete": cpu_complete,
        "num_threads": sum(item["num_threads"] for item in available),
        "members": members,
        "aggregation": "Given root PID plus recursive children. RSS sums working sets and may double-count shared pages; CPU sums one-core percentages. First observation of each process has no CPU interval.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pid", type=int, required=True, help="Server process PID, never repurposed")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--interval", type=float, default=5)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--max-seconds", type=float, help="Optional finite read-only observation")
    args = p.parse_args()
    if args.interval <= 0 or args.max_seconds is not None and args.max_seconds <= 0:
        raise SystemExit("Interval and duration must be positive")
    proc = psutil.Process(args.pid)
    created = proc.create_time()
    process_cache = {}
    try:
        help_text = command(["nvidia-smi", "--help-query-gpu"])
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        help_text = ""
    fields = BASE_FIELDS + [f for f in OPTIONAL_FIELDS if '"' + f + '"' in help_text]
    optional_enabled = len(fields) > len(BASE_FIELDS)
    stop = threading.Event()
    def interrupt(signum, frame):
        stop.set()
    previous = signal.signal(signal.SIGINT, interrupt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    count = 0
    status = "interrupted"
    utc = lambda: datetime.now(timezone.utc).isoformat()
    try:
        with args.output.open("x", encoding="utf-8") as output:
            def emit(row):
                output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                output.flush()
            emit({"event": "configuration", "utc": utc(), "pid": args.pid,
                  "process_create_time": created, "interval_s": args.interval,
                  "gpu_index": args.gpu_index, "gpu_fields": fields,
                  "units": {"temperature.gpu": "C", "utilization.gpu": "%",
                            "memory.used": "MiB", "memory.total": "MiB", "power.draw": "W",
                            "power.limit": "W", "clocks.sm": "MHz", "clocks.mem": "MHz",
                            "rss_bytes": "bytes", "cpu_percent": "one CPU core = 100%, may exceed 100%"},
                  "scope": "Device-level GPU metrics and selected server PID plus recursive children CPU/RSS; no model execution",
                  "process_aggregation": "Sum process working-set RSS, potentially counting shared pages more than once. Include Windows venv launcher and actual interpreter child.",
                  "counter_scope": "NVIDIA counters can be cumulative across earlier activity. Interpret differences over this run, not the initial absolute count."})
            while not stop.is_set():
                elapsed = time.perf_counter() - start
                if args.max_seconds is not None and elapsed >= args.max_seconds:
                    status = "duration_complete"
                    break
                row = {"event": "measurement", "utc": utc(), "elapsed_s": elapsed}
                try:
                    if not proc.is_running() or proc.create_time() != created or proc.status() == psutil.STATUS_ZOMBIE:
                        status = "server_process_ended"
                        break
                    row["process"] = process_tree_record(proc, process_cache)
                except psutil.NoSuchProcess:
                    status = "server_process_ended"
                    break
                except psutil.AccessDenied as error:
                    row["process_error"] = type(error).__name__
                try:
                    row["gpu"] = gpu_record(fields, args.gpu_index)
                except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
                    row["gpu_query_error"] = type(error).__name__ + ": " + str(error)
                    if optional_enabled:
                        fields = BASE_FIELDS[:]
                        optional_enabled = False
                        row["optional_counter_fallback"] = True
                        try:
                            row["gpu"] = gpu_record(fields, args.gpu_index)
                        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as retry:
                            row["gpu_retry_error"] = type(retry).__name__ + ": " + str(retry)
                row["collection_duration_s"] = time.perf_counter() - start - elapsed
                emit(row)
                count += 1
                stop.wait(max(0, args.interval - row["collection_duration_s"]))
            emit({"event": "finished", "utc": utc(), "elapsed_s": time.perf_counter() - start,
                  "measurements": count, "reason": status})
    finally:
        signal.signal(signal.SIGINT, previous)
    print(json.dumps({"measurements": count, "reason": status, "output": str(args.output)}))


if __name__ == "__main__":
    main()
