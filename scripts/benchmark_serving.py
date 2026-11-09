"""Local HTTP replay load test with independent-arrival and fixed-concurrency modes.

This driver uses Python's standard library and executes no model itself. Raw
responses belong in ignored local outputs. All attempts, including rejected,
timed-out and client-capacity-dropped requests, stay in the summary denominator.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import signal
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from msc_project.serving.monitoring import summarise_load


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def load_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8765")
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--metadata", type=Path, help="Optional request_id to length_decile map")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["closed", "open"], default="closed")
    p.add_argument("--phase-plan", type=Path, help="Independent-arrival phases: {phases:[{name,duration,rate,replay?,metadata?}]}; only final phase drains")
    p.add_argument("--requests", type=int, help="Attempt count; default replay length unless duration set")
    p.add_argument("--duration", type=float, help="Arrival period in seconds, drain follows")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--rate", type=float, default=1.0, help="Open-model scheduled arrivals per second")
    p.add_argument("--max-inflight", type=int, default=64)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--seed", type=int, default=20261109)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--expected-variant")
    p.add_argument("--metrics-interval", type=float, default=5.0)
    p.add_argument("--tag", default="serving-replay")
    return p.parse_args()


def main():
    args = load_args()
    if args.concurrency < 1 or args.max_inflight < 1 or args.rate <= 0 or args.timeout <= 0:
        raise SystemExit("Concurrency, rate, timeout and inflight limit must be positive")
    if args.duration is not None and args.duration <= 0 or args.requests is not None and args.requests <= 0:
        raise SystemExit("Duration/request count must be positive")
    replay = read_jsonl(args.replay)
    if not replay or any(not isinstance(x.get("text"), str) or not x.get("request_id") for x in replay):
        raise SystemExit("Replay requires nonempty rows with request_id and text")
    if args.shuffle:
        random.Random(args.seed).shuffle(replay)
    if args.requests is None and args.duration is None:
        args.requests = len(replay)
    metadata = {r["request_id"]: r for r in read_jsonl(args.metadata)} if args.metadata else {}
    phases = []
    if args.phase_plan:
        if args.duration is not None:
            raise SystemExit("--duration cannot be combined with --phase-plan; phase durations define the schedule")
        args.mode = "open"
        phase_spec = json.loads(args.phase_plan.read_text(encoding="utf-8-sig"))
        offset = 0.0
        names = set()
        for spec in phase_spec.get("phases", []):
            name = str(spec["name"])
            if name in names or not name:
                raise SystemExit("Phase names must be nonempty and unique")
            names.add(name)
            duration, rate = float(spec["duration"]), float(spec["rate"])
            if duration <= 0 or rate <= 0:
                raise SystemExit("Phase durations and rates must be positive")
            samples = read_jsonl(spec["replay"]) if spec.get("replay") else replay
            if not samples or any(not isinstance(x.get("text"), str) or not x.get("request_id") for x in samples):
                raise SystemExit("Phase replay requires request_id/text rows")
            if spec.get("metadata"):
                metadata.update({r["request_id"]: r for r in read_jsonl(spec["metadata"])})
            phases.append({"name": name, "duration": duration, "rate": rate,
                           "scheduled_start_s": offset, "scheduled_end_s": offset + duration,
                           "samples": samples})
            offset += duration
        if not phases:
            raise SystemExit("Phase plan must contain at least one phase")
        args.requests = None
    args.output.mkdir(parents=True, exist_ok=True)
    records_file = args.output / "requests.jsonl"
    if records_file.exists():
        raise SystemExit("Refusing to overwrite an existing run")
    run_id = uuid.uuid4().hex[:12]
    base = args.url.rstrip("/")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def read_endpoint(endpoint):
        try:
            with opener.open(base + endpoint, timeout=3) as response:
                return json.loads(response.read())
        except Exception as error:
            return {"collection_error": type(error).__name__ + ": " + str(error)}
    health = read_endpoint("/health")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(run_id=run_id, utc_start=datetime.now(timezone.utc).isoformat(),
                  replay_sha256=hashlib.sha256(args.replay.read_bytes()).hexdigest(),
                  health_before=health, host=socket.gethostname(),
                  client="Python urllib HTTP/1.1 new connection per request; proxy disabled; same-host loopback by default",
                  replay_rows=len(replay), raw_text_logged=False,
                  model_response_cache="No client result cache. Server must disable result cache.")
    if phases:
        config["rate"] = None
        config["duration"] = phases[-1]["scheduled_end_s"]
        config["arrival_schedule"] = "phased"
        config["phase_plan_sha256"] = hashlib.sha256(args.phase_plan.read_bytes()).hexdigest()
        config["phases"] = [{k: v for k, v in phase.items() if k != "samples"} for phase in phases]
    (args.output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    start = time.perf_counter()
    lock = threading.Lock()
    records = []
    stop_metrics = threading.Event()
    stop_arrivals = threading.Event()
    interrupted = False
    def interrupt_arrivals(signum, frame):
        nonlocal interrupted
        interrupted = True
        stop_arrivals.set()
        print("Stopping arrivals; waiting for in-flight requests and preserving results.", file=sys.stderr)
    previous_interrupt = signal.signal(signal.SIGINT, interrupt_arrivals)
    output = records_file.open("x", encoding="utf-8")
    def append(record):
        with lock:
            records.append(record)
            output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
    def collect_metrics():
        with (args.output / "metrics.jsonl").open("w", encoding="utf-8") as f:
            while not stop_metrics.is_set():
                value = {"elapsed_s": time.perf_counter() - start, "metrics": read_endpoint("/metrics")}
                f.write(json.dumps(value, ensure_ascii=False) + "\n")
                f.flush()
                stop_metrics.wait(max(0.5, args.metrics_interval))
    monitor = threading.Thread(target=collect_metrics, daemon=True)
    monitor.start()
    def template(index, scheduled, sample=None, phase=None):
        sample = sample if sample is not None else replay[index % len(replay)]
        meta = metadata.get(sample["request_id"], {})
        chars = len(sample["text"])
        return {
            "run_id": run_id, "request_id": f"{run_id}-{index:07d}", "sample_id": sample["request_id"],
            "attempt_index": index, "scheduled_s": scheduled,
            "phase": phase,
            "input_chars": chars, "length_bin": meta.get("length_decile", "short" if chars < 200 else "medium" if chars < 600 else "long"),
            "status_code": None, "success": False, "response": {},
        }
    def request(index, scheduled, sample=None, phase=None):
        sample = sample if sample is not None else replay[index % len(replay)]
        record = template(index, scheduled, sample, phase)
        actual = time.perf_counter() - start
        record.update(started_s=actual, schedule_lag_ms=max(0.0, (actual - scheduled) * 1000))
        body = json.dumps({"request_id": record["request_id"], "text": sample["text"]}).encode("utf-8")
        req = urllib.request.Request(base + "/predict", data=body, method="POST", headers={"Content-Type": "application/json"})
        try:
            with opener.open(req, timeout=args.timeout) as response:
                record["status_code"] = response.status
                record["response"] = json.loads(response.read())
            if record["response"].get("request_id") != record["request_id"]:
                record["error"] = "response_request_id_mismatch"
            elif not isinstance(record["response"].get("predictions"), list):
                record["error"] = "response_missing_predictions"
            elif args.expected_variant and record["response"].get("variant") != args.expected_variant:
                record["error"] = "response_variant_mismatch"
            else:
                record["success"] = 200 <= record["status_code"] < 300
                record["error"] = None
        except urllib.error.HTTPError as error:
            record.update(status_code=error.code, error="http_" + str(error.code))
            try:
                record["response"] = json.loads(error.read())
            except (ValueError, OSError):
                pass
        except (TimeoutError, socket.timeout):
            record["error"] = "timeout"
        except urllib.error.URLError as error:
            record["error"] = "timeout" if isinstance(error.reason, (TimeoutError, socket.timeout)) else "connection_error"
        except (ValueError, UnicodeError):
            record["error"] = "invalid_response_json"
        except Exception as error:
            record["error"] = "client_" + type(error).__name__
        end = time.perf_counter() - start
        record.update(finished_s=end, latency_ms=(end - actual) * 1000,
                      scheduled_latency_ms=(end - scheduled) * 1000)
        append(record)
    def allowed(index):
        return (not stop_arrivals.is_set() and (args.requests is None or index < args.requests) and
                (args.duration is None or time.perf_counter() - start < args.duration))
    def wait_until(deadline):
        # Windows timed waits may return a fraction of a millisecond early.
        # Recheck the monotonic deadline, including at the final observation
        # boundary, instead of weakening duration validation or rounding time.
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return True
            if stop_arrivals.wait(remaining):
                return False
    futures = []
    def schedule():
        if phases:
            index = 0
            for phase in phases:
                local_index = 0
                while local_index / phase["rate"] < phase["duration"]:
                    yield (index, phase["scheduled_start_s"] + local_index / phase["rate"],
                           phase["samples"][local_index % len(phase["samples"])], phase["name"])
                    local_index += 1
                    index += 1
        else:
            index = 0
            while ((args.requests is None or index < args.requests) and
                   (args.duration is None or index / args.rate < args.duration)):
                yield index, index / args.rate, None, None
                index += 1
    try:
        if args.mode == "closed":
            next_index = 0
            def worker():
                nonlocal next_index
                while True:
                    with lock:
                        if not allowed(next_index):
                            return
                        index = next_index
                        next_index += 1
                    request(index, time.perf_counter() - start)
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = [pool.submit(worker) for _ in range(args.concurrency)]
                for future in futures:
                    future.result()
        else:
            with ThreadPoolExecutor(max_workers=args.max_inflight) as pool:
                active = set()
                for index, scheduled, sample, phase in schedule():
                    if stop_arrivals.is_set():
                        break
                    if not wait_until(start + scheduled):
                        break
                    active = {future for future in active if not future.done()}
                    if len(active) >= args.max_inflight:
                        record = template(index, scheduled, sample, phase)
                        record.update(started_s=None, finished_s=time.perf_counter() - start,
                                      latency_ms=None, scheduled_latency_ms=None, schedule_lag_ms=None,
                                      error="client_capacity_drop")
                        append(record)
                    else:
                        future = pool.submit(request, index, scheduled, sample, phase)
                        active.add(future)
                        futures.append(future)
                # Keep the declared arrival observation window even when the
                # final scheduled request finishes before its last time bin ends.
                # Otherwise sparse arrivals inflate reviews/second by shortening
                # a duration-based experiment to the last completion.
                horizon = phases[-1]["scheduled_end_s"] if phases else args.duration if args.requests is None else None
                if horizon is not None:
                    wait_until(start + horizon)
                for future in futures:
                    future.result()
    except KeyboardInterrupt:
        interrupted = True
    finally:
        wall = time.perf_counter() - start
        signal.signal(signal.SIGINT, previous_interrupt)
        stop_metrics.set()
        monitor.join(timeout=4)
        output.close()
        summary = summarise_load(records, wall)
        summary.update(run_id=run_id, interrupted=interrupted,
                       health_after=read_endpoint("/health"),
                       note="Client drain includes awaiting all submitted requests up to their HTTP timeouts. Server work after a client timeout is outside this client window and must be checked before the next run.")
        if phases:
            summaries = []
            for phase in phases:
                subset = [r for r in records if r.get("phase") == phase["name"]]
                sent = [r for r in subset if r.get("started_s") is not None]
                span = max((r["finished_s"] for r in subset), default=phase["scheduled_start_s"]) - phase["scheduled_start_s"]
                summaries.append({
                    **{k: v for k, v in phase.items() if k != "samples"},
                    "actual_first_start_s": min((r["started_s"] for r in sent), default=None),
                    "actual_last_finish_s": max((r["finished_s"] for r in subset), default=None),
                    "completed_after_phase_boundary": sum(r["finished_s"] > phase["scheduled_end_s"] for r in subset),
                    "successful_completed_inside_phase_window": sum(r.get("success") and phase["scheduled_start_s"] <= r["finished_s"] < phase["scheduled_end_s"] for r in records),
                    "arrival_cohort_performance": summarise_load(subset, span),
                })
            summary["phases"] = summaries
            summary["phase_scope"] = "One continuous arrival timeline; no inter-phase drain. Per-phase latency belongs to scheduled-arrival cohorts; completion-window counts include earlier phase backlog. Per-cohort drain spans overlap and their QPS must not be added."
        (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: summary[k] for k in ["attempts", "successful", "failed", "wall_seconds_including_drain", "successful_reviews_per_second", "successful_latency"]}, indent=2))
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
