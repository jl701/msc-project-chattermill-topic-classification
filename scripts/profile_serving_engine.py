"""Development-only throughput/memory screening for real two-stage inference.

The default two-step screen tests pair microbatch sizes with one review at a
time, then tests review batching only for the selected pair size. No labels or
evaluation split are read. Amortized batch cost is not a response-time metric.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from check_serving_gpu import select_by_length
from msc_project.serving.inference import QwenServingEngine, VARIANTS


def integer_list(value: str) -> list[int]:
    values = list(dict.fromkeys(int(item) for item in value.split(",")))
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers.")
    return values


def prediction_fingerprints(responses) -> dict[str, str]:
    result = {}
    for response in responses:
        pairs = sorted((item["aspect"], sentiment)
                       for item in response["predictions"] for sentiment in item["sentiments"])
        result[response["request_id"]] = hashlib.sha256(
            json.dumps(pairs, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    return result


def aggregate_configuration(records, *, expected_reviews, memory_limit, baseline) -> dict[str, Any]:
    measured = [row for row in records if row["phase"] == "measure"]
    successes = [row for row in measured if row["status"] == "ok"]
    failures = [row for row in records if row["status"] != "ok"]
    reviews = sum(row["request_count"] for row in successes)
    seconds = sum(row["wall_seconds"] for row in successes)
    durations = [row["wall_seconds"] * 1000 for row in successes]
    amortized = [row["wall_seconds"] * 1000 / row["request_count"] for row in successes]
    input_tokens = sum(row["input_tokens"] for row in successes)
    padded_tokens = sum(row["padded_tokens"] for row in successes)
    fingerprints = {}
    unstable = set()
    for row in successes:
        for uid, value in row["prediction_fingerprints"].items():
            if uid in fingerprints and fingerprints[uid] != value:
                unstable.add(uid)
            fingerprints[uid] = value
    changed = sorted(uid for uid, fingerprint in fingerprints.items()
                     if uid in baseline and baseline[uid] != fingerprint)
    peak_allocated = max((row.get("peak_allocated_bytes", 0) for row in records), default=0)
    peak_reserved = max((row.get("peak_reserved_bytes", 0) for row in records), default=0)
    complete = reviews == expected_reviews and not failures
    return {
        "completed_reviews": reviews, "expected_reviews": expected_reviews,
        "completed_measured_batches": len(successes), "failure_count": len(failures),
        "failures": [{key: row.get(key) for key in ["phase", "repeat", "batch", "status", "error"]} for row in failures],
        "complete": complete, "reviews_per_second": reviews / seconds if seconds else None,
        "measured_seconds": seconds,
        "batch_wall_p50_ms": float(np.percentile(durations, 50)) if durations else None,
        "batch_wall_p95_ms": float(np.percentile(durations, 95)) if durations else None,
        "amortized_cost_p50_ms_per_review": float(np.percentile(amortized, 50)) if amortized else None,
        "amortized_cost_p95_ms_per_review": float(np.percentile(amortized, 95)) if amortized else None,
        "amortized_cost_warning": "Batch wall time divided by review count is cost allocation, NOT a measured request response time.",
        "stage_encode_seconds": sum(row["encode_ms"] for row in successes) / 1000,
        "stage_compute_seconds": sum(row["compute_ms"] for row in successes) / 1000,
        "input_tokens": input_tokens, "padded_tokens": padded_tokens,
        "padding_overhead_ratio": padded_tokens / input_tokens if input_tokens else None,
        "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
        "memory_limit_bytes": memory_limit,
        "memory_within_limit": peak_allocated <= memory_limit,
        "label_changed_ids_vs_first_successful_pair_config": changed,
        "repeat_unstable_ids": sorted(unstable),
        "prediction_fingerprints": fingerprints,
        "eligible_for_automatic_selection": complete and peak_allocated <= memory_limit and not changed and not unstable,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "outputs/serving_20261109/artifacts.json")
    parser.add_argument("--replay", type=Path, default=ROOT / "outputs/serving_20261109/replay_development.jsonl")
    parser.add_argument("--correctness-summary", type=Path, default=ROOT / "outputs/serving_20261109/correctness/summary.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/serving_20261109/profile")
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--pair-batches", type=integer_list, default=[1, 2, 4, 8])
    parser.add_argument("--request-batches", type=integer_list, default=[1, 2, 3])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--max-batch-tokens", type=int, default=8192)
    parser.add_argument("--memory-fraction", type=float, default=0.90)
    parser.add_argument("--logits-mode", choices=["full", "last_token"], default="last_token")
    parser.add_argument("--strategy", choices=["screen", "grid"], default="screen")
    args = parser.parse_args()
    variants = args.variants.split(",")
    if any(variant not in VARIANTS for variant in variants):
        parser.error("Unknown variant.")
    if args.repeats < 1 or args.warmup_batches < 0 or not 0 < args.memory_fraction <= 1:
        parser.error("Invalid repeat, warmup or memory setting.")
    if not 1 <= args.max_batch_tokens <= 8192:
        parser.error("Padded-token budget must be between 1 and 8192.")
    if args.strategy == "screen" and 1 not in args.request_batches:
        parser.error("The screen strategy needs request batch 1.")
    correctness = json.loads(args.correctness_summary.read_text(encoding="utf-8"))
    if args.logits_mode == "last_token":
        if correctness.get("status") != "completed" or correctness.get("label_gate_pass") is not True:
            parser.error("last_token requires a completed, passing GPU correctness summary.")
        comparisons = correctness.get("comparisons", [])
        for variant in variants:
            if not any(row.get("comparison") == variant + "/full_logits_vs_last_token"
                       and row.get("label_gate_pass") is True for row in comparisons):
                parser.error(f"Missing successful last-token gate for {variant}.")
    requests = select_by_length(args.replay, args.samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    journal = args.output_dir / "batches.jsonl"
    if journal.exists():
        parser.error("Output journal exists; choose a fresh output directory.")
    summary = {
        "status": "running", "selection_split": "development only; no ground truth",
        "samples": [{"request_id": row["request_id"], "characters": len(row["text"])} for row in requests],
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "correctness_summary_sha256": hashlib.sha256(args.correctness_summary.read_bytes()).hexdigest(),
        "configurations": [], "selections": {},
        "selection_rule": "Highest measured complete-review throughput within memory budget and identical development labels to first successful pair configuration; no quality labels used.",
        "interpretation": "Screening results select engineering settings. This is not a request-queue/HTTP latency benchmark.",
    }

    def persist():
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def record(row):
        with journal.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    engine = None
    persist()
    try:
        print("Loading one resident model for development screening", flush=True)
        engine = QwenServingEngine.from_artifacts(args.artifacts, logits_mode=args.logits_mode,
                                                 max_batch_tokens=args.max_batch_tokens)
        summary["engine"] = engine.metadata()
        total_memory = int(torch.cuda.get_device_properties(0).total_memory)
        memory_limit = int(total_memory * args.memory_fraction)
        summary["gpu_total_memory_bytes"] = total_memory
        summary["gpu_memory_limit_bytes"] = memory_limit

        def run_config(variant, pair_batch, request_batch, baseline):
            engine.set_variant(variant)
            engine.pair_batch_size = pair_batch
            config = {"variant": variant, "pair_batch_size": pair_batch,
                      "request_batch_size": request_batch, "logits_mode": args.logits_mode,
                      "max_batch_tokens": args.max_batch_tokens}
            config_id = f"{variant}-pairs{pair_batch}-reviews{request_batch}"
            batches = [requests[index:index + request_batch] for index in range(0, len(requests), request_batch)]
            schedule = [("warmup", -1, index, batches[index % len(batches)]) for index in range(args.warmup_batches)]
            schedule += [("measure", repeat, index, batch) for repeat in range(args.repeats)
                         for index, batch in enumerate(batches)]
            records = []
            print(f"CONFIG {config_id}: {args.warmup_batches} warmup batches, {len(requests)*args.repeats} measured reviews", flush=True)
            # Clear unused cached blocks left by a previous larger configuration.
            gc.collect()
            torch.cuda.empty_cache()
            for phase, repeat, index, batch in schedule:
                row = {"config_id": config_id, "config": config, "phase": phase,
                       "repeat": repeat, "batch": index, "request_count": len(batch),
                       "request_ids": [request["request_id"] for request in batch]}
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                oom = False
                try:
                    responses = engine.predict_batch(batch)
                    torch.cuda.synchronize()
                    diag = responses[0]["diagnostics"]
                    stages = [diag["batch_aspect"], diag["batch_sentiment"]]
                    row.update({
                        "status": "ok", "wall_seconds": time.perf_counter() - started,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                        "input_tokens": sum(stage["input_tokens"] for stage in stages),
                        "padded_tokens": sum(stage["padded_tokens"] for stage in stages),
                        "encode_ms": sum(stage["encode_ms"] for stage in stages),
                        "compute_ms": sum(stage["compute_ms"] for stage in stages),
                        "batch_aspect": stages[0], "batch_sentiment": stages[1],
                        "selected_aspects": sum(len(response["predictions"]) for response in responses),
                        "prediction_fingerprints": prediction_fingerprints(responses),
                    })
                except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                    oom = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()
                    row.update({"status": "oom" if oom else "fatal", "wall_seconds": time.perf_counter() - started,
                                "error": {"type": type(error).__name__, "message": str(error)},
                                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
                    record(row)
                    records.append(row)
                    if not oom:
                        raise
                else:
                    record(row)
                    records.append(row)
                print(f"  {phase} repeat={repeat} batch={index} {row['status']} {row['wall_seconds']:.2f}s", flush=True)
                if oom:
                    # The exception scope has ended; its traceback no longer owns tensors.
                    gc.collect()
                    torch.cuda.empty_cache()
                    print("  Config stopped after recorded OOM; continuing other configurations", flush=True)
                    break
            result = {"config_id": config_id, **config, **aggregate_configuration(
                records, expected_reviews=len(requests) * args.repeats,
                memory_limit=memory_limit, baseline=baseline,
            )}
            summary["configurations"].append(result)
            persist()
            print(f"RESULT {config_id}: throughput={result['reviews_per_second']}, eligible={result['eligible_for_automatic_selection']}, changed_labels={len(result['label_changed_ids_vs_first_successful_pair_config'])}", flush=True)
            return result

        for variant in variants:
            baseline = {}
            configs = []
            if args.strategy == "grid":
                coordinates = [(p, r) for p in args.pair_batches for r in args.request_batches]
            else:
                coordinates = [(p, 1) for p in args.pair_batches]
            for pair_batch, request_batch in coordinates:
                result = run_config(variant, pair_batch, request_batch, baseline)
                configs.append(result)
                if not baseline and result["complete"]:
                    baseline = result["prediction_fingerprints"]
            eligible = [config for config in configs if config["eligible_for_automatic_selection"]]
            if not eligible:
                summary["selections"][variant] = {"status": "no_eligible_configuration"}
                persist()
                continue
            selected_pair = max(eligible, key=lambda value: value["reviews_per_second"])
            if args.strategy == "screen":
                for request_batch in args.request_batches:
                    if request_batch != 1:
                        configs.append(run_config(variant, selected_pair["pair_batch_size"], request_batch, baseline))
            eligible = [config for config in configs if config["eligible_for_automatic_selection"]]
            selected = max(eligible, key=lambda value: value["reviews_per_second"])
            summary["selections"][variant] = {
                "status": "selected", "pair_screen_config": selected_pair["config_id"],
                "config_id": selected["config_id"], "pair_batch_size": selected["pair_batch_size"],
                "request_batch_size": selected["request_batch_size"],
                "reviews_per_second": selected["reviews_per_second"],
                "peak_allocated_bytes": selected["peak_allocated_bytes"],
            }
            persist()
        summary["status"] = "completed"
    except Exception as error:
        summary["status"] = "failed"
        summary["exception"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        print(summary["exception"]["traceback"], flush=True)
    finally:
        persist()
        if engine is not None:
            del engine
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps({"status": summary["status"], "selections": summary["selections"],
                      "summary": str(args.output_dir / 'summary.json')}), flush=True)
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
