"""Real-engine dynamic retrieval fault checks; run only with the GPU reserved.

This uses three length-stratified development inputs without reading gold labels.
It is a resilience/correctness gate, not a quality or throughput experiment.
Evidence stays in a fresh private output directory; source indexes are unchanged.
Requires the verified fixed-fallback policy integration in the serving engine.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from msc_project.serving.presence_policy import validated_presence_policy


def sha_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def select_inputs(path: Path) -> list[dict[str, str]]:
    """Read only the input replay; no label or prediction-based selection."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if (len(rows) < 3 or len({r["request_id"] for r in rows}) != len(rows) or
            any(set(r) != {"request_id", "text"} or not r["text"].strip() for r in rows)):
        raise ValueError("Need at least three unique input-only development records")
    ordered = sorted(rows, key=lambda row: (len(row["text"]), row["request_id"]))
    return [ordered[i] for i in (0, (len(ordered) - 1) // 2, len(ordered) - 1)]


def validate_fixed_source(dynamic: dict, fixed: dict, fixed_path: Path) -> dict:
    """Fail before model loading if the claimed fixed fallback is not verified."""
    policy = dynamic.get("retrieval_fallback_policy")
    if not isinstance(policy, dict):
        raise ValueError("Dynamic artifacts need an explicit verified fixed fallback policy")
    if policy.get("method") != "fixed" or policy.get("variant") != "hybrid":
        raise ValueError("Fallback policy must name fixed hybrid inference")
    if Path(policy.get("source_artifacts_path", "")).resolve() != fixed_path.resolve():
        raise ValueError("Fallback source path differs from the supplied fixed artifacts")
    if policy.get("source_artifacts_sha256") != sha_file(fixed_path):
        raise ValueError("Fixed fallback artifact SHA mismatch")
    if fixed.get("service_calibration", {}).get("method") != "fixed":
        raise ValueError("Fallback source is not a calibrated fixed comparator")
    for key in ("fold_id", "training_scope_id", "condition", "model", "adapter", "taxonomy", "few_shot", "qlora"):
        if dynamic.get(key) != fixed.get(key):
            raise ValueError(f"Fixed fallback changes frozen inference resource: {key}")
    for key in ("aspect_threshold", "runner_up_threshold"):
        if policy.get(key) != fixed["variants"]["hybrid"][key]:
            raise ValueError(f"Fallback {key} differs from its hashed source")
    if validated_presence_policy(policy) != validated_presence_policy(fixed["variants"]["hybrid"]):
        raise ValueError("Fallback score space or margin threshold differs from hashed fixed source")
    return policy


def make_small_versioned_index(original, target: Path) -> dict:
    """An isolated real-training subset, with matching existing embedding rows.

    Used only to test swapping index identity and restoring it. This never enters
    reported quality selection, and it never modifies the campaign index.
    """
    import numpy as np
    from msc_project.serving.retrieval import digest

    target.mkdir(parents=True, exist_ok=False)
    fixed = {json.dumps(asdict(d), sort_keys=True) for d in original.fixed}
    indices = {i for i, row in enumerate(original.pool) if json.dumps(row, sort_keys=True) in fixed}
    for answer in ("Y", "N"):
        candidates = [i for i, row in enumerate(original.pool) if row["answer"] == answer]
        indices.update(candidates[:8])
    indices = sorted(indices)
    if len(indices) >= len(original.pool):
        raise ValueError("Source pool is too small to test a distinct index version")
    pool = [original.pool[i] for i in indices]
    (target / "pool.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pool), encoding="utf-8")
    manifest = deepcopy(original.manifest)
    manifest.update(pool_sha256=digest(pool), pool_pairs=len(pool),
                    schema_version=str(manifest["schema_version"]) + "/fault_subset_v1",
                    fault_check_only=True, source_pool_sha256=original.fingerprint,
                    selection="fixed demos plus first eight legal pool rows per answer class")
    if (original.index_dir / "embeddings.npy").exists():
        values = np.load(original.index_dir / "embeddings.npy", mmap_mode="r")
        np.save(target / "embeddings.npy", np.asarray(values[indices]))
        manifest["embeddings_sha256"] = sha_file(target / "embeddings.npy")
    write_json(target / "manifest.json", manifest)
    return manifest


class FaultInjector:
    """Local fault fixture; the real retriever remains unchanged on disk."""
    def __init__(self, inner, *, error=None, release=None):
        self.inner = inner
        self.error = error
        self.release = release
        self.started = threading.Event()

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def prepare(self, texts, cards):
        self.started.set()
        if self.release is not None and not self.release.wait(120):
            raise TimeoutError("Fault fixture was not released")
        if self.error is not None:
            raise self.error
        return self.inner.prepare(texts, cards)

    def select(self, *args):
        return self.inner.select(*args)


def check_policy(responses, *, fallback: bool, method: str, thresholds: dict,
                 source_sha: str | None = None, error_type: str | None = None) -> None:
    for response in responses:
        diagnostics = response["diagnostics"]
        if diagnostics.get("retrieval_fallback") is not fallback:
            raise AssertionError("Wrong effective fallback flag")
        if diagnostics.get("effective_retrieval_method") != method:
            raise AssertionError("Wrong effective retrieval method")
        actual_thresholds = diagnostics.get("effective_thresholds",{})
        if (validated_presence_policy(actual_thresholds) != validated_presence_policy(thresholds) or
                actual_thresholds.get("runner_up_threshold") != thresholds["runner_up_threshold"]):
            raise AssertionError("Wrong effective inference thresholds")
        if fallback and diagnostics.get("fallback_policy_source_sha256") != source_sha:
            raise AssertionError("Fallback did not identify its verified source")
        selections = diagnostics["batch_aspect"]["retrieval"]
        if not selections or any(item.get("fallback", False) != fallback for item in selections):
            raise AssertionError("Mixed or missing retrieval policy within a request batch")
        if error_type and any(item.get("error_type") != error_type for item in selections):
            raise AssertionError("Fault type was not preserved in fallback evidence")


def compare_sentiment_anchors(reference: dict, candidate: dict, runner_thresholds: list[float]) -> dict:
    """Compare the complete adapter score grid, independent of presence gates."""
    from msc_project.serving.inference import decode_sentiments, SENTIMENTS

    old = {r["request_id"]: r for r in reference["responses"]}
    new = {r["request_id"]: r for r in candidate["responses"]}
    if set(old) != set(new) or len(old) != len(reference["responses"]) or len(new) != len(candidate["responses"]):
        raise ValueError("Sentiment anchor response identities differ or repeat")
    differences, label_changes, aspect_count = [], 0, 0
    for identity in old:
        left_grid = old[identity]["full_grid_scores"]
        right_grid = new[identity]["full_grid_scores"]
        left = {v["aspect"]: v["sentiment_scores"] for v in left_grid}
        right = {v["aspect"]: v["sentiment_scores"] for v in right_grid}
        if list(left) != list(right) or len(left) != len(left_grid) or len(right) != len(right_grid):
            raise ValueError("Sentiment candidate order/coverage differs")
        for aspect in left:
            if set(left[aspect]) != set(SENTIMENTS) or set(right[aspect]) != set(SENTIMENTS):
                raise ValueError("Sentiment score-vector coverage differs")
            a = [float(left[aspect][s]) for s in SENTIMENTS]
            b = [float(right[aspect][s]) for s in SENTIMENTS]
            if not all(math.isfinite(v) for v in a+b):
                raise ValueError("Non-finite sentiment anchor score")
            differences.extend(abs(x-y) for x,y in zip(a,b))
            label_changes += any(decode_sentiments(a,t) != decode_sentiments(b,t) for t in runner_thresholds)
            aspect_count += 1
    old_inputs = [r for r in reference["encoded_inputs"] if r["mode"] == "sentiment"]
    new_inputs = [r for r in candidate["encoded_inputs"] if r["mode"] == "sentiment"]
    if any(r["mode"] == "aspect" for r in candidate["encoded_inputs"]):
        raise AssertionError("Sentiment-only path unexpectedly scored Stage1")
    return {"request_count": len(old), "scored_aspects": aspect_count,
            "max_abs_score_difference": max(differences, default=0.0),
            "nonidentical_score_count": sum(d != 0 for d in differences),
            "label_changed_aspects": label_changes, "runner_thresholds": sorted(set(runner_thresholds)),
            "encoded_sentiment_inputs_equal": old_inputs == new_inputs,
            "sentiment_truncation_equal": all(old[i]["sentiment_truncation_by_aspect"] ==
                                               new[i]["sentiment_truncation_by_aspect"] for i in old)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--fixed-artifacts", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--encoder", type=Path)
    parser.add_argument("--method", choices=["random", "bm25", "e5", "rrf"], required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--development-replay", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--normal-timeout-s", type=float, default=10.0)
    parser.add_argument("--fault-timeout-s", type=float, default=.05)
    args = parser.parse_args(argv)
    if any(not math.isfinite(v) or v <= 0 for v in (args.normal_timeout_s, args.fault_timeout_s)):
        parser.error("Timeouts must be finite and positive")
    dynamic_artifacts, fixed_artifacts = read_json(args.artifacts), read_json(args.fixed_artifacts)
    policy = validate_fixed_source(dynamic_artifacts, fixed_artifacts, args.fixed_artifacts)
    requests = select_inputs(args.development_replay)
    manifest = read_json(args.index / "manifest.json")
    if sha_file(args.development_replay) not in manifest.get("exclusion_source_hashes", {}).values():
        raise ValueError("Development inputs were not excluded from the retrieval index")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    import torch
    from check_serving_gpu import compare_outputs
    from msc_project.serving.inference import QwenServingEngine
    from msc_project.serving.retrieval import DemonstrationRetriever
    from msc_project.serving.timed_retrieval import TimedDemonstrationRetriever

    sources = ["scripts/check_rag_faults_gpu.py", "src/msc_project/serving/inference.py",
               "src/msc_project/serving/retrieval.py", "src/msc_project/serving/timed_retrieval.py",
               "src/msc_project/serving/fallback_policy.py", "src/msc_project/serving/presence_policy.py",
               "src/msc_project/serving/prefix_cache.py"]
    input_hashes = {"dynamic_artifacts": sha_file(args.artifacts), "fixed_artifacts": sha_file(args.fixed_artifacts),
                    "development_replay": sha_file(args.development_replay),
                    "index": {p.name: sha_file(p) for p in args.index.iterdir() if p.is_file()}}
    summary = {"status": "running", "all_gates_pass": False, "uses_gold": False,
               "scope": "real_model_resilience_gate_not_quality_or_HTTP_latency",
               "selection": "three evenly spaced development character-length ranks; ID breaks ties",
               "samples": [{"request_id": r["request_id"], "characters": len(r["text"]),
                            "text_sha256": hashlib.sha256(r["text"].encode()).hexdigest()} for r in requests],
               "method": args.method, "normal_timeout_s": args.normal_timeout_s,
               "injected_timeout_s": args.fault_timeout_s, "input_sha256": input_hashes,
               "source_sha256": {p: sha_file(ROOT / p) for p in sources},
               "policy": policy, "steps": [], "comparisons": [], "rejection_gates": []}
    write_json(args.output_dir / "preregistration.json", summary)
    wrappers, release_events, engines = [], [], []
    current_trace: list[dict] = []

    def persist():
        write_json(args.output_dir / "summary.json", summary)

    def attach_trace(engine):
        original = engine._score_encoded
        def traced(items, token_ids, **kwargs):
            encoded_sha = hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()
            current_trace.append({"mode": kwargs.get("mode"), "adapter": kwargs.get("use_adapter", True),
                                  "encoded_inputs_sha256": encoded_sha, "prompt_count": len(items)})
            return original(items, token_ids, **kwargs)
        engine._score_encoded = traced

    def measure(name, fn):
        print(f"START {name}", flush=True)
        current_trace.clear()
        torch.cuda.synchronize()
        start = time.perf_counter()
        responses = fn()
        torch.cuda.synchronize()
        entry = {"step": name, "seconds": time.perf_counter() - start,
                 "encoded_inputs": deepcopy(current_trace), "responses": responses}
        with (args.output_dir / "steps.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
        summary["steps"].append({k: v for k, v in entry.items() if k != "responses"})
        persist()
        print(f"DONE {name}", flush=True)
        return entry

    def compare(name, reference, candidate, thresholds):
        result = {"name": name, **compare_outputs(reference["responses"], candidate["responses"], thresholds),
                  "encoded_inputs_equal": reference["encoded_inputs"] == candidate["encoded_inputs"]}
        summary["comparisons"].append(result)
        persist()
        if not result["label_gate_pass"] or not result["encoded_inputs_equal"]:
            raise AssertionError(f"Prediction or model-input gate failed: {name}")

    def reject(name, function, expected_name):
        before = len(current_trace)
        try:
            function()
        except Exception as exc:
            gate = {"name": name, "error_type": type(exc).__name__, "expected": expected_name,
                    "passed": type(exc).__name__ == expected_name,
                    "model_scoring_calls": len(current_trace) - before}
            summary["rejection_gates"].append(gate)
            persist()
            if not gate["passed"] or gate["model_scoring_calls"]:
                raise AssertionError(f"Unsafe request was not rejected before scoring: {name}") from exc
        else:
            raise AssertionError(f"Expected rejection did not happen: {name}")

    try:
        engine = QwenServingEngine.from_artifacts(args.artifacts, variant="hybrid", pair_batch_size=4,
            max_batch_tokens=4096, logits_mode="last_token", enable_prefix_cache=False,
            device="cuda", local_files_only=True)
        engines.append(engine)
        attach_trace(engine)
        summary["engine"] = engine.metadata()
        summary["device"] = torch.cuda.get_device_name()
        persist()
        fixed_engine = QwenServingEngine(fixed_artifacts, engine.tokenizer, engine.model, variant="hybrid",
            pair_batch_size=4, max_batch_tokens=4096, logits_mode="last_token", enable_prefix_cache=False)
        fixed_engine.artifacts_sha256 = input_hashes["fixed_artifacts"]
        engines.append(fixed_engine)
        attach_trace(fixed_engine)
        cards = {str(c["text"]): str(c["aspect"]) for c in engine.candidates}
        dynamic_thresholds = {**validated_presence_policy(dynamic_artifacts["variants"]["hybrid"]),
                              "runner_up_threshold":float(dynamic_artifacts["variants"]["hybrid"]["runner_up_threshold"])}
        fixed_thresholds = {**validated_presence_policy(fixed_artifacts["variants"]["hybrid"]),
                            "runner_up_threshold":float(fixed_artifacts["variants"]["hybrid"]["runner_up_threshold"])}
        legacy_anchor = measure("legacy_full_grid_sentiment_anchor",
                                lambda: engine.predict_batch(requests, full_grid=True))
        sentiment_only = measure("sentiment_only_anchor", lambda: engine.score_full_sentiment(requests))
        anchor_gate = compare_sentiment_anchors(legacy_anchor, sentiment_only,
            [dynamic_thresholds["runner_up_threshold"], fixed_thresholds["runner_up_threshold"]])
        summary["sentiment_anchor_gate"] = anchor_gate
        persist()
        if (not anchor_gate["encoded_sentiment_inputs_equal"] or
                not anchor_gate["sentiment_truncation_equal"] or anchor_gate["label_changed_aspects"]):
            raise AssertionError("Sentiment-only optimization changed inputs, truncation or decoded labels")
        raw = DemonstrationRetriever(args.index, method=args.method, seed=args.seed,
            fixed_demonstrations=engine.demonstrations["aspect"], encoder_path=args.encoder)
        fixed = DemonstrationRetriever(args.index, method="fixed", fixed_demonstrations=engine.demonstrations["aspect"])
        fixed_engine.set_retriever(fixed, fallback_on_error=False)
        baseline = measure("fixed_reference", lambda: fixed_engine.predict_batch(requests))
        engine.set_retriever(raw, fallback_on_error=False)
        normal = measure("raw_dynamic_reference", lambda: engine.predict_batch(requests))
        timed = TimedDemonstrationRetriever(raw, candidate_aspects=cards, timeout_s=args.normal_timeout_s)
        wrappers.append(timed)
        engine.set_retriever(timed, fallback_on_error=True)
        wrapped = measure("timed_dynamic_normal", lambda: engine.predict_batch(requests))
        check_policy(wrapped["responses"], fallback=False, method=args.method, thresholds=dynamic_thresholds)
        compare("raw_vs_timed_normal", normal, wrapped, dynamic_thresholds)

        injected = FaultInjector(raw, error=OSError("Injected unavailable retrieval"))
        broken = TimedDemonstrationRetriever(injected, candidate_aspects=cards, timeout_s=args.normal_timeout_s)
        wrappers.append(broken)
        engine.set_retriever(broken, fallback_on_error=True)
        failed = measure("retrieval_exception_fallback", lambda: engine.predict_batch(requests))
        check_policy(failed["responses"], fallback=True, method="fixed", thresholds=fixed_thresholds,
                     source_sha=policy["source_artifacts_sha256"], error_type="OSError")
        compare("exception_vs_fixed_reference", baseline, failed, fixed_thresholds)

        # Fail closed before any model scoring if fixed fallback would expose the
        # current request as one of its own labelled demonstrations.
        unsafe = [{"request_id": "fixed-self-match-fault-check", "text": raw.fixed[0].review_text}]
        reject("query_matches_fixed_example", lambda: engine.predict_batch(unsafe), "UnsafeFixedFallbackError")

        release = threading.Event()
        release_events.append(release)
        delayed = FaultInjector(raw, release=release)
        deadline = TimedDemonstrationRetriever(delayed, candidate_aspects=cards, timeout_s=args.fault_timeout_s)
        wrappers.append(deadline)
        engine.set_retriever(deadline, fallback_on_error=True)
        expired = measure("retrieval_timeout_fallback", lambda: engine.predict_batch(requests))
        check_policy(expired["responses"], fallback=True, method="fixed", thresholds=fixed_thresholds,
                     source_sha=policy["source_artifacts_sha256"], error_type="RetrievalTimeoutError")
        compare("timeout_vs_fixed_reference", baseline, expired, fixed_thresholds)
        busy = measure("prior_work_busy_fallback", lambda: engine.predict_batch(requests))
        check_policy(busy["responses"], fallback=True, method="fixed", thresholds=fixed_thresholds,
                     source_sha=policy["source_artifacts_sha256"], error_type="RetrievalBusyError")
        compare("busy_vs_fixed_reference", baseline, busy, fixed_thresholds)
        if deadline.metadata()["timed_retrieval_counters"]["submitted"] != 1:
            raise AssertionError("Busy fallback incorrectly queued another retrieval batch")
        release.set()
        deadline._future.result(timeout=120)
        deadline.timeout_s = args.normal_timeout_s
        recovered = measure("recovered_after_late_completion", lambda: engine.predict_batch(requests))
        check_policy(recovered["responses"], fallback=False, method=args.method, thresholds=dynamic_thresholds)
        compare("recovered_vs_dynamic_reference", normal, recovered, dynamic_thresholds)
        if deadline.metadata()["timed_retrieval_counters"]["late_discarded"] != 1:
            raise AssertionError("Late result was not discarded")
        summary["timeout_counters"] = deadline.metadata()["timed_retrieval_counters"]

        tampered = deepcopy(dynamic_artifacts)
        tampered["retrieval_fallback_policy"]["source_artifacts_sha256"] = "0" * 64
        reject("fallback_source_hash_tampering", lambda: QwenServingEngine(
            tampered, engine.tokenizer, engine.model, variant="hybrid"), "ValueError")
        tampered = deepcopy(dynamic_artifacts)
        tampered["retrieval_fallback_policy"]["aspect_threshold"] = -0.5
        reject("fallback_threshold_tampering", lambda: QwenServingEngine(
            tampered, engine.tokenizer, engine.model, variant="hybrid"), "ValueError")

        version_path = args.output_dir / "fault_subset_index"
        new_manifest = make_small_versioned_index(raw, version_path)
        versioned = DemonstrationRetriever(version_path, method=args.method, seed=args.seed,
            fixed_demonstrations=engine.demonstrations["aspect"], encoder_path=args.encoder)
        changed = TimedDemonstrationRetriever(versioned, candidate_aspects=cards, timeout_s=args.normal_timeout_s)
        wrappers.append(changed)
        previous_version = engine.model_version()
        engine.set_retriever(changed, fallback_on_error=True)
        versioned_result = measure("changed_legal_index_version", lambda: engine.predict_batch(requests))
        check_policy(versioned_result["responses"], fallback=False, method=args.method, thresholds=dynamic_thresholds)
        if engine.model_version() == previous_version:
            raise AssertionError("Changing the index content did not change model version")
        for response in versioned_result["responses"]:
            for entry in response["diagnostics"]["batch_aspect"]["retrieval"]:
                if entry["pool_sha256"] != new_manifest["pool_sha256"]:
                    raise AssertionError("Response used a stale index version")
        new_manifest["pool_sha256"] = "f" * 64
        write_json(version_path / "manifest.json", new_manifest)
        reject("tampered_index_rejected", lambda: DemonstrationRetriever(version_path,
            method=args.method, fixed_demonstrations=engine.demonstrations["aspect"], encoder_path=args.encoder), "ValueError")
        engine.set_retriever(timed, fallback_on_error=True)
        restored = measure("original_index_restored", lambda: engine.predict_batch(requests))
        check_policy(restored["responses"], fallback=False, method=args.method, thresholds=dynamic_thresholds)
        compare("restored_vs_dynamic_reference", normal, restored, dynamic_thresholds)
        if {p.name: sha_file(p) for p in args.index.iterdir() if p.is_file()} != input_hashes["index"]:
            raise AssertionError("Source campaign index changed during fault checks")
        summary.update(status="completed", all_gates_pass=True)
    except Exception as error:
        summary.update(status="failed", exception={"type": type(error).__name__,
            "message": str(error), "traceback": traceback.format_exc()})
    finally:
        for event in release_events:
            event.set()
        for wrapper in wrappers:
            wrapper.close(wait=False)
        persist()
        engines.clear()
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps({"status": summary["status"], "all_gates_pass": summary["all_gates_pass"],
                      "summary": str(args.output_dir / "summary.json")}), flush=True)
    return 0 if summary["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
