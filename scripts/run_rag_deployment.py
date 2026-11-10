"""One-attempt-per-review HTTP quality replay under an existing frozen service plan.

No gold labels or classifier selection are read. Warmups remain separate.
The existing service launcher, owned-process cleanup and HTTP client are reused.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path

import run_rag_service_study as study

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_METHODS = ["fixed", "random_13", "random_37", "random_73", "bm25", "e5", "rrf", "e5_negative_seen"]
DEFAULT_OFFLINE_CONFIG = ROOT / "configs/experiments/serving_rag_negative_20261111.json"


def check_sources(plan: dict) -> None:
    for name, expected in plan["source_hashes"].items():
        if study.digest(ROOT / name) != expected:
            raise ValueError(f"Frozen service source changed: {name}")
    if study.digest(Path(plan["replay"])) != plan["input_hashes"]["replay"]:
        raise ValueError("Frozen warmup replay changed")
    current_index = {p.name: study.digest(p) for p in sorted(Path(plan["index"]).iterdir()) if p.is_file()}
    if current_index != plan["input_hashes"]["index_files"]:
        raise ValueError("Frozen retrieval index changed")


def check_correctness(service_root: Path, plan: dict, method: str, plan_sha: str) -> dict:
    if not study.correctness_completed(service_root, method):
        raise ValueError("Selected deployment method lacks a completed API correctness gate")
    directory = service_root / "correctness" / method
    summary = study.read_json(directory / "summary.json")
    if summary.get("passed") is not True or summary.get("plan_sha256") != plan_sha or \
            summary.get("attempts") != plan["correctness"]["reviews"]:
        raise ValueError("API correctness gate failed or differs from the frozen plan")
    return {name: study.digest(directory / name) for name in ("summary.json", "requests.jsonl", "complete.json")}


def prepare_registration(args) -> tuple[dict, dict]:
    source_plan, output = args.service_plan.resolve(), args.output.resolve()
    if output == source_plan.parent or output.is_relative_to(source_plan.parent) or output == args.offline_root.resolve():
        raise ValueError("Use a separate new deployment output, preserving the original studies")
    plan = study.read_json(source_plan)
    if plan.get("schema_version") != "rag_http_service_study_v1" or args.method not in plan["methods"]:
        raise ValueError("Deployment method must belong to the frozen RAG service plan")
    if plan["server"]["request_batch_size"] != 1:
        raise ValueError("Registered deployment replay requires single-review HTTP requests")
    check_sources(plan)
    plan_sha = study.digest(source_plan)
    correctness = check_correctness(source_plan.parent, plan, args.method, plan_sha)
    method_config = plan["methods"][args.method]
    if args.method == "fixed" and method_config["prefix_cache"] != "on":
        raise ValueError("Fixed production-quality evidence requires the frozen cache-on policy")
    clone_path = Path(method_config["artifacts"])
    if study.digest(clone_path) != method_config["artifacts_sha256"]:
        raise ValueError("Frozen service clone changed")
    clone = study.read_json(clone_path)
    if clone.get("service_calibration", {}).get("method") != args.method or \
            clone.get("service_calibration", {}).get("selection_sha256") != plan["input_hashes"]["selection"]:
        raise ValueError("Clone calibration identity differs from frozen selection")
    if clone.get("service_execution", {}).get("sentiment_execution") != plan["server"]["sentiment_execution"] or \
            clone.get("service_execution", {}).get("sentiment_group_contract") != plan["sentiment_group_contract"]:
        raise ValueError("Clone execution contract differs from frozen service plan")
    offline_config = study.read_json(args.offline_config)
    methods = [row["id"] for row in offline_config["methods"]]
    if methods != EXPECTED_METHODS:
        raise ValueError("Offline campaign must register the exact final eight-method comparison")
    inference = offline_config["inference"]
    if inference.get("presence_score_space", "probability") != plan["presence_score_space"] or \
            inference.get("quality_prefix_cache") is not False or any(
                inference.get(key) != plan["server"][key]
                for key in ("pair_batch_size", "max_batch_tokens", "request_batch_size", "logits_mode")):
        raise ValueError("Offline campaign inference controls differ from frozen comparison")
    replay = study.replay_population(args.replay)
    if len(replay) != args.expected_reviews or args.expected_reviews < 1:
        raise ValueError("Evaluation replay differs from the registered review count")
    replay_sha = study.digest(args.replay)
    manifest = study.read_json(Path(plan["index"]) / "manifest.json")
    exclusions = [name for name, value in manifest.get("exclusion_source_hashes", {}).items() if value == replay_sha]
    if not exclusions:
        raise ValueError("Evaluation replay SHA is not covered by frozen index duplicate exclusions")
    source_hashes = dict(plan["source_hashes"])
    auxiliary = {name: study.digest(ROOT / name) for name in (
        "scripts/run_rag_deployment.py", "scripts/run_serving_study.py", "scripts/collect_serving_telemetry.py")}
    frozen_replay = "inputs/replay_evaluation.jsonl"
    registration = {"schema_version": "rag_deployment_quality_registration_v1",
        "scope": "frozen_production_configuration_quality_not_classifier_selection",
        "service_plan": {"source": str(source_plan), "sha256": plan_sha}, "method": args.method,
        "method_config": method_config,
        "clone": {"source": str(clone_path.resolve()), "frozen_copy": f"inputs/artifacts_{args.method}.json",
                  "sha256": method_config["artifacts_sha256"]},
        "evaluation_replay": {"source": str(args.replay.resolve()), "frozen_copy": frozen_replay,
            "sha256": replay_sha, "reviews": len(replay), "request_ids": [row["request_id"] for row in replay]},
        "source_hashes": source_hashes, "auxiliary_source_hashes": auxiliary,
        "correctness_gate_sha256": correctness, "index_files": plan["input_hashes"]["index_files"],
        "index_exclusion": {"manifest_sha256": plan["input_hashes"]["index_files"]["manifest.json"],
                            "replay_sha256": replay_sha, "matching_exclusion_sources": exclusions},
        "offline_root": str(args.offline_root.resolve()),
        "offline_campaign": {"config_source": str(args.offline_config.resolve()),
            "config_frozen_copy": "inputs/offline_config.json", "config_sha256": study.digest(args.offline_config),
            "methods": methods, "presence_score_space": plan["presence_score_space"]},
        "run": {"name": f"deployment_quality__{args.method}", "mode": "closed", "concurrency": 1,
                "requests": len(replay), "replay": str(output / frozen_replay), "seed": plan["seed"], "shuffle": False},
        "warmup": {"requests": plan["warmup_requests"], "replay_sha256": plan["input_hashes"]["replay"],
                   "included_in_evaluation": False},
        "inference_precondition": "all registered main offline evaluation methods complete with matching inputs",
        "no_labels_read": True, "no_threshold_search": True, "retry_policy": "none_one_attempt_per_input",
        "failure_scope": "retain_all_failed_and_timed_out_attempts_no_success_only_filter",
        "timing_scope": "HTTP_delivery_times_not_comparable_as_speedup_to_offline_component_times"}
    path = output / "registration.json"
    if output.exists() and not path.is_file():
        raise FileExistsError("Output already exists without this immutable registration")
    if path.exists() and study.read_json(path) != registration:
        raise FileExistsError("Deployment registration changed; preserve existing evidence")
    output.mkdir(parents=True, exist_ok=True)
    study.copy_immutable(source_plan, output / "plan.json")
    study.copy_immutable(args.replay, output / frozen_replay)
    study.copy_immutable(clone_path, output / registration["clone"]["frozen_copy"])
    study.copy_immutable(args.offline_config, output / "inputs/offline_config.json")
    study.save_json(path, registration)
    return registration, plan


def validate_offline_records(path: Path, expected: set[str], aspects: list[str], score_space: str) -> dict:
    rows = study.read_jsonl(path)
    identities = [row.get("request_id") for row in rows]
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError("Offline method has missing or duplicate review IDs")
    for row in rows:
        response = row.get("response", {})
        if row.get("status") != "ok" or response.get("request_id") != row["request_id"] or response.get("variant") != "hybrid":
            raise ValueError("Offline method lacks successful complete response identity")
        if response.get("presence_score_space", "probability") != score_space:
            raise ValueError("Offline method score space differs from the registered campaign")
        presence = response.get("aspect_scores", [])
        full_grid = response.get("full_grid_scores", [])
        if [item.get("aspect") for item in presence] != aspects or [item.get("aspect") for item in full_grid] != aspects:
            raise ValueError("Offline method lacks the complete ordered candidate grids")
        for item in presence:
            value = item.get("score")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Offline method has invalid presence probability")
            if score_space == "logit_margin":
                margin = item.get("presence_logit_margin")
                if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin):
                    raise ValueError("Offline method lacks a finite raw presence margin")
        for item in full_grid:
            values = item.get("sentiment_scores", {})
            if set(values) != {"negative", "neutral", "positive"} or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1
                for value in values.values()) or not math.isclose(sum(values.values()), 1.0, abs_tol=1e-4):
                raise ValueError("Offline method has an incomplete or invalid sentiment vector")
    return {"reviews": len(rows), "unique_review_ids": len(set(identities)), "successful": len(rows),
            "candidate_count": len(aspects), "presence_score_space": score_space, "complete_outputs": True}


def offline_dependency(registration: dict, plan: dict) -> dict:
    root = Path(registration["offline_root"])
    state = study.read_json(root / "state.json")
    if state.get("status") != "complete":
        raise RuntimeError("Main offline evaluation matrix is not complete; do not start deployment replay")
    identity, summary = study.read_json(root / "identity.json"), study.read_json(root / "summary.json")
    expected = set(registration["evaluation_replay"]["request_ids"])
    campaign = registration["offline_campaign"]
    if (study.digest(Path(campaign["config_source"])) != campaign["config_sha256"] or
            identity.get("config") != campaign["config_sha256"] or identity.get("methods") != campaign["methods"] or
            state.get("methods") != campaign["methods"] or set(summary.get("methods", {})) != set(campaign["methods"])):
        raise ValueError("Offline matrix differs from the registered campaign config or exact eight methods")
    if (set(identity.get("request_ids", [])) != expected or len(identity.get("request_ids", [])) != len(expected) or
            identity.get("artifacts") != plan["input_hashes"]["artifacts"] or
            identity.get("replay") != registration["evaluation_replay"]["sha256"] or
            identity.get("index_files") != registration["index_files"] or
            summary.get("reviews") != len(expected) or state.get("reviews") != len(expected) or
            set(summary.get("methods", {})) != set(identity.get("methods", [])) or
            set(state.get("methods", [])) != set(identity.get("methods", []))):
        raise ValueError("Offline matrix is incomplete or has different registered inputs")
    method = registration["method"]
    if method not in identity["methods"]:
        raise ValueError("Deployment method lacks a matching offline classifier output")
    clone = study.read_json(Path(registration["clone"]["source"]))
    aspects = [row["aspect"] for row in clone["taxonomy"]["candidates"]]
    validated, score_hashes = {}, {}
    for name in campaign["methods"]:
        if study.digest(root / f"{name}.jsonl") != summary["methods"][name]["file_sha256"]:
            raise ValueError("Offline completed method score SHA changed")
        validated[name] = validate_offline_records(root / f"{name}.jsonl", expected, aspects, campaign["presence_score_space"])
        if summary["methods"][name].get("successful") != len(expected):
            raise ValueError("Offline method successful count differs from its complete outputs")
        score_hashes[name] = summary["methods"][name]["file_sha256"]
    result = {"root": str(root), "identity_sha256": study.digest(root / "identity.json"),
        "summary_sha256": study.digest(root / "summary.json"), "state_sha256": study.digest(root / "state.json"),
        "method": method, "score_path": str(root / f"{method}.jsonl"),
        "score_sha256": summary["methods"][method]["file_sha256"],
        "replay_sha256": identity["replay"], "artifacts_sha256": identity["artifacts"], "review_count": len(expected)}
    result.update(config_sha256=campaign["config_sha256"], methods=campaign["methods"],
                  method_score_sha256=score_hashes, per_method_validation=validated)
    return result


def coverage(rows: list[dict], expected: set[str]) -> dict:
    counts = Counter(row.get("sample_id") for row in rows)
    missing = expected - set(counts)
    duplicate_ids = {key for key, count in counts.items() if count > 1}
    unexpected = set(counts) - expected
    return {"expected_reviews": len(expected), "attempts": len(rows),
        "successful_attempts": sum(row.get("success") is True for row in rows),
        "failed_attempts": sum(row.get("success") is not True for row in rows),
        "unique_expected_reviews": len(set(counts) & expected), "missing_reviews": len(missing),
        "duplicate_review_ids": len(duplicate_ids), "extra_duplicate_attempts": sum(counts[k] - 1 for k in duplicate_ids),
        "unexpected_review_ids": len(unexpected),
        "valid": not missing and not duplicate_ids and not unexpected and len(rows) == len(expected)}


def run(args) -> dict:
    registration, plan = prepare_registration(args)
    output = args.output.resolve()
    if args.plan_only:
        return {"registered": True, "executed": False, "registration_sha256": study.digest(output / "registration.json")}
    state_path = output / "deployment_state.json"
    if state_path.exists():
        raise FileExistsError("An execution already exists; preserve it and never retry its review attempts")
    dependency = offline_dependency(registration, plan)
    study.save_json(output / "offline_dependency_receipt.json", dependency)
    lock_path = output / "deployment_running.lock"
    with lock_path.open("x", encoding="utf-8") as lock:
        json.dump({"pid": os.getpid(), "started_utc": study.utc()}, lock)
    state = {"status": "running", "method": args.method, "started_utc": study.utc()}
    study.save_json(state_path, state, mutable=True)
    try:
        run_config = registration["run"]
        directory = output / "runs" / run_config["name"]
        with study.server_session("deployment_quality", args.method, plan, output) as (process, env, _):
            study.run_benchmark(run_config, directory, plan, output, env, process, headline=False)
        # Verify all frozen references again after shutdown, without reading gold.
        prepare_registration(args)
        if offline_dependency(registration, plan) != dependency:
            raise ValueError("Offline dependency changed during deployment replay")
        rows = study.read_jsonl(directory / "requests.jsonl")
        observed = coverage(rows, set(registration["evaluation_replay"]["request_ids"]))
        completion = {"schema_version": "rag_deployment_quality_complete_v1", "status": "complete",
            "method": args.method, "registration_sha256": study.digest(output / "registration.json"),
            "service_plan_sha256": study.digest(output / "plan.json"),
            "offline_dependency_receipt_sha256": study.digest(output / "offline_dependency_receipt.json"),
            "run_directory": str(Path("runs") / run_config["name"]),
            "files_sha256": {name: study.digest(directory / name) for name in (
                "requests.jsonl", "summary.json", "complete.json", "registered_run.json", "config.json")},
            "coverage": observed, "completed_utc": study.utc(), "no_labels_read": True,
            "warmup_in_evaluation": False}
        study.save_json(output / "deployment_completion.json", completion)
        state.update(status="complete", completion_sha256=study.digest(output / "deployment_completion.json"),
                     coverage_valid=observed["valid"])
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["finished_utc"] = study.utc()
        study.save_json(state_path, state, mutable=True)
        lock_path.unlink(missing_ok=True)
    return {"status": "complete", "coverage": observed, "output": str(output)}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--service-plan", type=Path, required=True)
    result.add_argument("--replay", type=Path, required=True)
    result.add_argument("--offline-root", type=Path, required=True)
    result.add_argument("--offline-config", type=Path, default=DEFAULT_OFFLINE_CONFIG)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--method", default="fixed")
    result.add_argument("--expected-reviews", type=int, default=957)
    result.add_argument("--plan-only", action="store_true")
    return result


if __name__ == "__main__":
    with study.keep_system_awake():
        print(json.dumps(run(parser().parse_args()), ensure_ascii=False))
