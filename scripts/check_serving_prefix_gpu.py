"""Development-only GPU gate for frozen training-prefix KV reuse.

Cache-off and cache-on score the identical development requests. All aspect
and common-selected sentiment score differences are retained, even if labels
match. A three-review combined batch additionally checks cache state isolation.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from check_serving_gpu import compare_outputs, select_by_length
from msc_project.serving.inference import QwenServingEngine, SENTIMENTS


def complete_score_differences(left, right):
    old = {item["aspect"]: item["score"] for item in left["aspect_scores"]}
    new = {item["aspect"]: item["score"] for item in right["aspect_scores"]}
    aspects = [{"aspect": name, "cache_off": value, "cache_on": new[name],
                "difference": new[name] - value} for name, value in old.items()]
    old_p = {item["aspect"]: item for item in left["predictions"]}
    new_p = {item["aspect"]: item for item in right["predictions"]}
    sentiments = []
    for name in sorted(old_p.keys() & new_p.keys()):
        for sentiment in SENTIMENTS:
            a = old_p[name]["sentiment_scores"][sentiment]
            b = new_p[name]["sentiment_scores"][sentiment]
            sentiments.append({"aspect": name, "sentiment": sentiment,
                               "cache_off": a, "cache_on": b, "difference": b - a})
    return {"all_aspect_scores": aspects, "common_selected_sentiment_scores": sentiments}


def prefix_tensor_hashes(cache):
    result = {}
    for mode, prefix in cache._prefixes.items():
        digest = hashlib.sha256()
        for key, value in prefix.key_values:
            for tensor in [key, value]:
                digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        result[mode] = {"tokens": len(prefix.ids), "sha256": digest.hexdigest()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "outputs/serving_20261109/artifacts.json")
    parser.add_argument("--replay", type=Path, default=ROOT / "outputs/serving_20261109/replay_development.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/serving_20261109/prefix_correctness")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--pair-batch-size", type=int, default=4)
    parser.add_argument("--max-batch-tokens", type=int, default=4096)
    parser.add_argument("--variants", default="few_shot,hybrid")
    args = parser.parse_args()
    variants = args.variants.split(",")
    if any(value not in {"few_shot", "hybrid"} for value in variants):
        parser.error("This gate compares few_shot and hybrid only; QLoRA never uses this cache.")
    requests = select_by_length(args.replay, args.samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    journal = args.output_dir / "samples.jsonl"
    if journal.exists():
        parser.error("Output journal exists; choose a fresh output directory.")
    summary = {"status": "running", "label_gate_pass": False,
               "samples": [{"request_id": request["request_id"], "characters": len(request["text"])} for request in requests],
               "selection": "development only, evenly spaced character-length ranks; no gold labels",
               "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
               "variants": {}, "combined_batch_checks": [], "official_test_accessed": False,
               "timing_scope": "Correctness-gate descriptive timing; cache-on warmed once. Cache-off phase precedes cache-on phase."}

    def persist():
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def append(record):
        with journal.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    engine = None
    try:
        print("Loading selected adapter and shared quantized base", flush=True)
        engine = QwenServingEngine.from_artifacts(
            args.artifacts, pair_batch_size=args.pair_batch_size,
            max_batch_tokens=args.max_batch_tokens, logits_mode="last_token",
            enable_prefix_cache=False,
        )
        summary["engine"] = engine.metadata()
        persist()

        def measure(variant, phase, rows):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = engine.prefix_cache.stats()
            started = time.perf_counter()
            outputs = engine.predict_batch(rows)
            torch.cuda.synchronize()
            record = {"variant": variant, "phase": phase,
                      "request_ids": [row["request_id"] for row in rows],
                      "seconds": time.perf_counter() - started,
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                      "cache_before": before, "cache_after": engine.prefix_cache.stats(),
                      "outputs": outputs}
            append(record)
            return outputs, record

        for variant in variants:
            engine.set_variant(variant)
            engine.enable_prefix_cache = False
            engine.prefix_cache.clear()
            gc.collect()
            torch.cuda.empty_cache()
            thresholds = engine.artifacts["variants"][variant]
            off = {}
            off_records = []
            print(f"START {variant} cache-off: {len(requests)} reviews", flush=True)
            for index, request in enumerate(requests):
                outputs, record = measure(variant, "cache_off", [request])
                off[request["request_id"]] = outputs[0]
                off_records.append(record)
                if index % 10 == 0 or index == len(requests)-1:
                    print(f"{variant} cache-off {index+1}/{len(requests)} {record['seconds']:.2f}s", flush=True)
            engine.enable_prefix_cache = True
            print(f"WARMUP {variant} prefix build", flush=True)
            _, warm = measure(variant, "cache_on_warmup", [requests[0]])
            immutable_before = prefix_tensor_hashes(engine.prefix_cache)
            on = {}
            on_records = []
            comparisons = []
            print(f"START {variant} cache-on: {len(requests)} reviews", flush=True)
            for index, request in enumerate(requests):
                outputs, record = measure(variant, "cache_on", [request])
                uid = request["request_id"]
                on[uid] = outputs[0]
                on_records.append(record)
                comparison = {"request_id": uid,
                              **compare_outputs([off[uid]], outputs, thresholds)}
                append({"variant": variant, "phase": "comparison", **comparison,
                        **complete_score_differences(off[uid], outputs[0])})
                comparisons.append(comparison)
                if index % 10 == 0 or index == len(requests)-1 or not comparison["label_gate_pass"]:
                    print(f"{variant} cache-on {index+1}/{len(requests)} {record['seconds']:.2f}s "
                          f"labels={'PASS' if comparison['label_gate_pass'] else 'FAIL'} "
                          f"aspectdiff={comparison['aspect_max_abs_difference']:.8g} "
                          f"sentimentdiff={comparison['common_selected_sentiment_max_abs_difference']:.8g}", flush=True)
            positions = sorted(set([0, len(requests)//2, len(requests)-1]))
            combined = [requests[i] for i in positions]
            print(f"CHECK {variant} combined batch and stored-prefix immutability", flush=True)
            batch_outputs, _ = measure(variant, "cache_on_combined", combined)
            expected = [on[request["request_id"]] for request in combined]
            combined_check = {"variant": variant,
                              **compare_outputs(expected, batch_outputs, thresholds)}
            immutable_after = prefix_tensor_hashes(engine.prefix_cache)
            # A mode not invoked during warmup can be initialized by later rows;
            # every previously built prefix must remain byte-for-byte unchanged.
            combined_check["stored_prefix_immutable"] = all(
                immutable_after.get(mode) == value for mode, value in immutable_before.items()
            )
            combined_check["prefix_hashes_before"] = immutable_before
            combined_check["prefix_hashes_after"] = immutable_after
            summary["combined_batch_checks"].append(combined_check)
            append({"phase": "combined_check", **combined_check})
            summary["variants"][variant] = {
                "request_count": len(requests),
                "label_gate_pass": all(row["label_gate_pass"] for row in comparisons),
                "label_changed_ids": [row["request_id"] for row in comparisons if not row["label_gate_pass"]],
                "aspect_max_abs_difference": max(row["aspect_max_abs_difference"] for row in comparisons),
                "sentiment_max_abs_difference": max(row["common_selected_sentiment_max_abs_difference"] for row in comparisons),
                "closest_aspect_threshold_margin": min(row["closest_aspect_threshold_margin"] for row in comparisons),
                "closest_runner_up_threshold_margin": min((row["closest_runner_up_threshold_margin"] for row in comparisons if row["closest_runner_up_threshold_margin"] is not None), default=None),
                "cache_off_total_seconds": sum(row["seconds"] for row in off_records),
                "cache_on_total_seconds": sum(row["seconds"] for row in on_records),
                "cache_off_p50_seconds": float(np.median([row["seconds"] for row in off_records])),
                "cache_on_p50_seconds": float(np.median([row["seconds"] for row in on_records])),
                "cache_off_peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in off_records),
                "cache_on_peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in on_records),
                "cache_on_warmup_seconds": warm["seconds"],
                "cache_statistics": engine.prefix_cache.stats(),
            }
            persist()
        summary["status"] = "completed"
        summary["label_gate_pass"] = (
            all(value["label_gate_pass"] for value in summary["variants"].values())
            and all(value["label_gate_pass"] and value["stored_prefix_immutable"]
                    for value in summary["combined_batch_checks"])
        )
    except Exception as error:
        summary["status"] = "failed"
        summary["exception"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        print(summary["exception"]["traceback"], flush=True)
    finally:
        persist()
        if engine is not None:
            engine.prefix_cache.clear()
            del engine
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps({"status": summary["status"], "label_gate_pass": summary["label_gate_pass"],
                      "summary": str(args.output_dir / 'summary.json')}), flush=True)
    return 0 if summary["status"] == "completed" and summary["label_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
