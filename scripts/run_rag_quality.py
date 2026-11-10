"""Registered offline detection matrix with a common frozen sentiment score grid.

Outputs are private, append-only and resumable only under identical inputs and
inference code. Timings are offline profiling, never HTTP service latency.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_ACTIVE_OUTPUT: Path | None = None


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def select_requests(rows: list[dict], limit: int | None) -> list[dict]:
    if any(set(row) != {"request_id", "text"} for row in rows):
        raise ValueError("Inference input must contain only request_id and text.")
    if len({row["request_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate request IDs.")
    if limit is None or limit >= len(rows):
        return rows
    if limit < 2:
        raise ValueError("Smoke limit must be at least two.")
    ordered = sorted(rows, key=lambda row: (len(row["text"]), row["request_id"]))
    indices = [round(i * (len(ordered) - 1) / (limit - 1)) for i in range(limit)]
    return [ordered[index] for index in indices]


def response_from_scores(presence: dict, grid: list[dict], artifacts: dict, method: str,
                         sentiment_truncation: dict | None = None) -> dict:
    from msc_project.serving.inference import decode_sentiments
    thresholds = artifacts["variants"]["hybrid"]
    sentiment = {row["aspect"]: row["sentiment_scores"] for row in grid}
    predictions = []
    for row in presence["aspect_scores"]:
        if row["score"] >= thresholds["aspect_threshold"]:
            scores = sentiment[row["aspect"]]
            predictions.append({"aspect": row["aspect"], "aspect_score": row["score"],
                                "sentiment_scores": scores,
                                "sentiments": decode_sentiments(
                                    [scores[name] for name in ("negative", "neutral", "positive")],
                                    thresholds["runner_up_threshold"])})
    return {**presence, "variant": "hybrid", "retrieval_method": method,
            "fold_id": artifacts["fold_id"], "predictions": predictions,
            "full_grid_scores": grid,
            "sentiment_truncation_by_aspect": sentiment_truncation or {},
            "score_scope": "offline_presence_with_identical_frozen_full_sentiment_grid"}


def validate_registration(config: dict, artifacts: dict, manifest: dict, replay_sha256: str) -> None:
    checks = {"fold": artifacts["fold_id"], "condition": artifacts["condition"],
              "model_id": artifacts["model"]["model_id"],
              "model_revision": artifacts["model"]["revision"], "variant": "hybrid"}
    for name, actual in checks.items():
        if config[name] != actual:
            raise ValueError(f"Registered {name} differs from artifacts/runtime")
    for name, actual in (("fewshot_max_length", artifacts["few_shot"]["max_length"]),
                         ("qlora_max_length", artifacts["qlora"]["max_length"])):
        if config["inference"][name] != actual:
            raise ValueError(f"Registered {name} differs from artifacts")
    for name, actual in {"bm25_k1": 1.2, "bm25_b": .75, "rrf_c": 60,
                         "rrf_weights": [1,1], "shortlist_per_class": 256,
                         "model_max_length": 256, "e5_device": "cpu"}.items():
        if config["retrieval"][name] != actual:
            raise ValueError(f"Registered retrieval {name} differs from implementation")
    if config["retrieval"]["e5_revision"] != manifest["e5_revision"]:
        raise ValueError("E5 revision differs from registered index")
    if config["heldout_aspect"] != manifest["heldout_aspect"]:
        raise ValueError("Held-out aspect differs from index")
    if replay_sha256 not in manifest["exclusion_source_hashes"].values():
        raise ValueError("Replay is not covered by index duplicate-exclusion audit; rebuild/audit index first")
    if config["inference"]["quality_prefix_cache"] is not False:
        raise ValueError("Offline quality comparison requires prefix cache off")


def checked_anchors(rows: list[dict], expected: set[str], aspects: set[str]) -> dict:
    result = {}
    for row in rows:
        key = row.get("request_id")
        response = row.get("response", {})
        if key not in expected or key in result or response.get("request_id") != key:
            raise ValueError("Sentiment anchor has unexpected, duplicate or mismatched IDs")
        grid = response.get("full_grid_scores", [])
        if row.get("status") != "ok" or len(grid) != len(aspects) or {r["aspect"] for r in grid} != aspects:
            raise ValueError("Incomplete sentiment anchor")
        result[key] = row
    return result


def record_failure(error: BaseException) -> None:
    if _ACTIVE_OUTPUT is not None:
        write_json(_ACTIVE_OUTPUT / "state.json", {"status":"failed", "error_type":type(error).__name__,
                   "error":str(error), "utc":datetime.now(timezone.utc).isoformat()})


def main() -> int:
    global _ACTIVE_OUTPUT
    _ACTIVE_OUTPUT = None
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--encoder-path", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/serving_rag_20261111.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", default="fixed,random_13,random_37,random_73,bm25,e5,rrf")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    definitions = {entry["id"]: entry for entry in config["methods"]}
    methods = args.methods.split(",")
    if len(set(methods)) != len(methods) or set(methods) - definitions.keys():
        raise ValueError("Unknown or duplicated method.")
    requests = select_requests(read_lines(args.replay), args.limit)
    if not requests:
        raise ValueError("Empty inference input.")
    artifacts = json.loads(args.artifacts.read_text(encoding="utf-8"))
    manifest = json.loads((args.index / "manifest.json").read_text(encoding="utf-8"))
    validate_registration(config, artifacts, manifest, digest(args.replay))
    args.output.mkdir(parents=True, exist_ok=True)
    index_files = sorted(path for path in args.index.rglob("*") if path.is_file())
    fingerprint = {
        "artifacts": digest(args.artifacts), "replay": digest(args.replay),
        "config": digest(args.config), "methods": methods,
        "request_ids": [row["request_id"] for row in requests],
        "index_files": {str(path.relative_to(args.index)): digest(path) for path in index_files},
        "inference_code": {name: digest(ROOT / name) for name in [
            "src/msc_project/serving/inference.py", "src/msc_project/serving/retrieval.py",
            "src/msc_project/llm/qwen_two_stage_classifier.py", "scripts/run_rag_quality.py"]},
    }
    identity_path = args.output / "identity.json"
    if identity_path.exists():
        if not args.resume or json.loads(identity_path.read_text(encoding="utf-8")) != fingerprint:
            raise ValueError("Existing run identity differs, or --resume was not provided.")
    else:
        if any(args.output.glob("*.jsonl")):
            raise ValueError("Found scores without a registered identity.")
        write_json(identity_path, fingerprint)
    _ACTIVE_OUTPUT = args.output
    started = time.perf_counter()
    write_json(args.output / "state.json", {"status": "loading", "utc": datetime.now(timezone.utc).isoformat(),
                                           "reviews": len(requests), "methods": methods})
    from msc_project.serving.inference import QwenServingEngine
    from msc_project.serving.retrieval import DemonstrationRetriever
    engine = QwenServingEngine.from_artifacts(
        args.artifacts, variant="hybrid", pair_batch_size=config["inference"]["pair_batch_size"],
        max_batch_tokens=config["inference"]["max_batch_tokens"],
        logits_mode=config["inference"]["logits_mode"], enable_prefix_cache=False)
    write_json(args.output / "runtime.json", {"engine": engine.metadata(),
        "python": sys.version, "executable": sys.executable,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scope": "offline_scores_not_service_latency", "run_start_utc": datetime.now(timezone.utc).isoformat()})
    expected = {row["request_id"] for row in requests}
    anchor_path = args.output / "_sentiment_anchor.jsonl"
    anchors = checked_anchors(read_lines(anchor_path), expected, {c["aspect"] for c in engine.candidates})
    engine.set_retriever(None)
    for position, request in enumerate(requests, 1):
        if request["request_id"] in anchors:
            continue
        call_start = time.perf_counter()
        response = engine.predict_batch([request], full_grid=True)[0]
        if len(response.get("full_grid_scores", [])) != len(engine.candidates):
            raise RuntimeError("Full sentiment grid is missing; cannot fairly rethreshold.")
        row = {"request_id": request["request_id"], "response": response,
               "elapsed_ms": 1000 * (time.perf_counter() - call_start), "status": "ok"}
        append(anchor_path, row)
        anchors[request["request_id"]] = row
        if position % 10 == 0 or position == len(requests):
            progress = {"status":"running", "phase":"common_sentiment_grid", "completed":position,
                        "total":len(requests), "elapsed_s":time.perf_counter()-started}
            write_json(args.output / "state.json", progress)
            print(json.dumps(progress), flush=True)
    summaries = {}
    for method in methods:
        definition = definitions[method]
        retriever = DemonstrationRetriever(args.index, method=definition["method"],
            seed=definition["seed"], fixed_demonstrations=engine.demonstrations["aspect"],
            encoder_path=args.encoder_path)
        engine.set_retriever(retriever)
        output_path = args.output / f"{method}.jsonl"
        rows = read_lines(output_path)
        completed = {row["request_id"] for row in rows}
        if len(completed) != len(rows) or completed - expected:
            raise ValueError("Method output has unexpected or duplicate IDs.")
        for position, request in enumerate(requests, 1):
            if request["request_id"] in completed:
                continue
            call_start = time.perf_counter()
            presence = engine.score_presence([request])[0]
            grid = anchors[request["request_id"]]["response"]["full_grid_scores"]
            response = response_from_scores(presence, grid, artifacts, method,
                anchors[request["request_id"]]["response"].get("sentiment_truncation_by_aspect"))
            row = {"request_id":request["request_id"], "status":"ok", "response":response,
                   "elapsed_ms":1000*(time.perf_counter()-call_start),
                   "timing_scope":"presence_and_retrieval_only_reused_sentiment_not_http"}
            append(output_path, row)
            rows.append(row)
            if position % 10 == 0 or position == len(requests):
                progress = {"status":"running", "phase":method, "completed":position,
                            "total":len(requests), "elapsed_s":time.perf_counter()-started}
                write_json(args.output / "state.json", progress)
                print(json.dumps(progress), flush=True)
        summaries[method] = {"successful":len(rows), "file_sha256":digest(output_path),
                             "mean_offline_presence_ms":sum(row["elapsed_ms"] for row in rows)/len(rows)}
        write_json(args.output / "summary.json", {"methods":summaries, "reviews":len(requests),
                    "scope":"historical_replay" if "serving_20261109" in str(args.replay) else "new_inputs_without_gold",
                    "no_labels_read":True, "elapsed_s":time.perf_counter()-started})
    write_json(args.output / "state.json", {"status":"complete", "reviews":len(requests),
                                           "methods":methods, "elapsed_s":time.perf_counter()-started})
    return 0


if __name__ == "__main__":
    # Keep this unattended process running without keeping the display awake.
    if sys.platform == "win32":
        import ctypes
        if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):
            raise OSError("Could not keep Windows awake for this experiment process")
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as error:
        try:
            record_failure(error)
        except Exception as state_error:
            print(json.dumps({"state_write_error":type(state_error).__name__}), flush=True)
        print(json.dumps({"status":"failed","error_type":type(error).__name__,
                          "error":str(error)}), flush=True)
        raise
    finally:
        if sys.platform == "win32":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
