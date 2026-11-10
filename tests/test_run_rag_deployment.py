"""Frozen-plan HTTP replay orchestration tested without model loading."""
from contextlib import contextmanager
import json
import math

import pytest

import run_rag_deployment as deployment
import run_rag_service_study as study
from test_rag_service_study import make_args
from test_rag_evaluation import record


def fixture(tmp_path):
    source_args = make_args(tmp_path)
    source_args.sentiment_execution = "canonical_full_grid_groups"
    evaluation = tmp_path / "evaluation.jsonl"
    evaluation.write_text(''.join(json.dumps({"request_id": f"eval{i}", "text": f"Evaluation synthetic input {i}"})+'\n'
                                   for i in range(3)))
    (source_args.index / "manifest.json").write_text(json.dumps({"exclusion_source_hashes": {
        "evaluation": study.digest(evaluation)}}))
    plan = study.prepare_plan(source_args)
    gate = source_args.output / "correctness/fixed"
    study.save_json(gate / "summary.json", {"passed": True, "plan_sha256": study.digest(source_args.output / "plan.json"),
                                           "attempts": plan["correctness"]["reviews"]})
    study.save_json(gate / "requests.jsonl", {})
    study.save_json(gate / "complete.json", {"sha256": {name: study.digest(gate / name)
                                                       for name in ("summary.json", "requests.jsonl")}})
    offline = tmp_path / "offline"
    offline_config = tmp_path / "offline_config.json"
    campaign = study.read_json(deployment.DEFAULT_OFFLINE_CONFIG)
    campaign["inference"]["presence_score_space"] = plan["presence_score_space"]
    offline_config.write_text(json.dumps(campaign), encoding="utf-8")
    args = deployment.parser().parse_args(["--service-plan", str(source_args.output / "plan.json"),
        "--replay", str(evaluation), "--offline-root", str(offline), "--expected-reviews", "3",
        "--offline-config", str(offline_config),
        "--output", str(tmp_path / "deployment")])
    return args, plan


def finish_offline(args, plan):
    root = args.offline_root
    root.mkdir()
    ids = [row["request_id"] for row in study.replay_population(args.replay)]
    for method in deployment.EXPECTED_METHODS:
        rows = []
        for key in ids:
            row = record(key)
            row["response"].update(variant="hybrid", presence_score_space=plan["presence_score_space"])
            for item in row["response"]["aspect_scores"]:
                item["presence_logit_margin"] = math.log(item["score"] / (1-item["score"]))
            rows.append(row)
        (root / f"{method}.jsonl").write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding="utf-8")
    study.save_json(root / "identity.json", {"request_ids": ids, "methods": deployment.EXPECTED_METHODS,
        "config": study.digest(args.offline_config),
        "artifacts": plan["input_hashes"]["artifacts"], "replay": study.digest(args.replay),
        "index_files": plan["input_hashes"]["index_files"]})
    study.save_json(root / "summary.json", {"reviews": len(ids), "methods": {
        method: {"file_sha256": study.digest(root / f"{method}.jsonl"), "successful": len(ids)}
        for method in deployment.EXPECTED_METHODS}})
    study.save_json(root / "state.json", {"status": "complete", "reviews": len(ids),
                                         "methods": deployment.EXPECTED_METHODS})


def test_plan_only_binds_exact_plan_clone_index_and_replay_without_offline_or_process(tmp_path, monkeypatch):
    args, plan = fixture(tmp_path)
    args.plan_only = True
    monkeypatch.setattr(study, "server_session", lambda *a: pytest.fail("No server in registration"))
    result = deployment.run(args)
    assert result["executed"] is False
    registered = study.read_json(args.output / "registration.json")
    assert registered["source_hashes"] == plan["source_hashes"]
    assert registered["index_exclusion"]["replay_sha256"] == study.digest(args.replay)
    assert registered["method_config"]["prefix_cache"] == "on"
    assert registered["run"]["requests"] == 3 and registered["run"]["concurrency"] == 1
    assert registered["run"]["shuffle"] is False
    assert registered["offline_campaign"]["methods"] == deployment.EXPECTED_METHODS
    assert registered["offline_campaign"]["config_sha256"] == study.digest(args.offline_config)
    assert (args.output / "inputs/offline_config.json").read_bytes() == args.offline_config.read_bytes()
    assert (args.output / "plan.json").read_bytes() == args.service_plan.read_bytes()
    assert not args.offline_root.exists()
    assert not (args.output / "deployment_state.json").exists()
    assert deployment.run(args) == result


def test_missing_index_exclusion_rejected_before_output(tmp_path):
    args, plan = fixture(tmp_path)
    args.replay.write_text(args.replay.read_text()+'\n')  # same reviews, different unexcluded byte hash
    with pytest.raises(ValueError, match="duplicate exclusions"):
        deployment.prepare_registration(args)
    assert not args.output.exists()


def test_frozen_source_or_clone_change_is_rejected(tmp_path):
    args, plan = fixture(tmp_path)
    plan["source_hashes"]["scripts/run_rag_service_study.py"] = "bad"
    with pytest.raises(ValueError, match="Frozen service source"):
        deployment.check_sources(plan)
    clone = study.Path(plan["methods"]["fixed"]["artifacts"])
    clone.write_text(clone.read_text()+'\n')
    with pytest.raises(ValueError, match="clone changed"):
        deployment.prepare_registration(args)


def test_incomplete_main_matrix_prevents_launch_and_leaves_registration_reusable(tmp_path, monkeypatch):
    args, plan = fixture(tmp_path)
    args.offline_root.mkdir()
    study.save_json(args.offline_root / "state.json", {"status": "running"})
    monkeypatch.setattr(study, "server_session", lambda *a: pytest.fail("No server before offline completion"))
    with pytest.raises(RuntimeError, match="not complete"):
        deployment.run(args)
    assert (args.output / "registration.json").exists()
    assert not (args.output / "deployment_state.json").exists()


def test_execution_uses_one_c1_attempt_per_input_preserves_failure_and_never_retries(tmp_path, monkeypatch):
    args, plan = fixture(tmp_path)
    finish_offline(args, plan)
    launched, closed = [], []
    @contextmanager
    def session(stage, method, supplied_plan, output):
        launched.append((stage, method, supplied_plan))
        try:
            yield object(), {}, output / "synthetic_session"
        finally:
            closed.append(True)
    def benchmark(run, directory, supplied_plan, output, env, process, headline):
        assert run["requests"] == 3 and run["concurrency"] == 1 and not run["shuffle"]
        assert headline is False
        directory.mkdir(parents=True)
        rows = [{"sample_id": f"eval{i}", "attempt_index": i, "request_id": f"op{i}",
                 "success": i != 1, "error": "timeout" if i == 1 else None} for i in range(3)]
        (directory / "requests.jsonl").write_text(''.join(json.dumps(row)+'\n' for row in rows))
        for name in ("summary.json", "complete.json", "registered_run.json", "config.json"):
            study.save_json(directory / name, run if name == "registered_run.json" else {})
    monkeypatch.setattr(study, "server_session", session)
    monkeypatch.setattr(study, "run_benchmark", benchmark)
    result = deployment.run(args)
    assert result["coverage"]["valid"] and result["coverage"]["failed_attempts"] == 1
    assert len(launched) == len(closed) == 1
    assert launched[0][2] == plan  # exact original plan; helper keeps its separate warmup
    complete = study.read_json(args.output / "deployment_completion.json")
    assert complete["files_sha256"]["requests.jsonl"] == study.digest(
        args.output / "runs/deployment_quality__fixed/requests.jsonl")
    assert study.read_json(args.output / "deployment_state.json")["completion_sha256"] == study.digest(
        args.output / "deployment_completion.json")
    with pytest.raises(FileExistsError, match="never retry"):
        deployment.run(args)
    assert len(launched) == 1


def test_coverage_keeps_duplicate_missing_unknown_and_failed_counts():
    rows = [{"sample_id": "a", "success": False}, {"sample_id": "a", "success": True},
            {"sample_id": "other", "success": True}]
    result = deployment.coverage(rows, {"a", "b", "c"})
    assert result["missing_reviews"] == 2 and result["duplicate_review_ids"] == 1
    assert result["extra_duplicate_attempts"] == 1 and result["unexpected_review_ids"] == 1
    assert result["failed_attempts"] == 1 and not result["valid"]


def test_aborted_execution_retains_partial_evidence_and_cleans_own_session(tmp_path, monkeypatch):
    args, plan = fixture(tmp_path)
    finish_offline(args, plan)
    closed = []
    @contextmanager
    def session(*unused):
        try:
            yield object(), {}, args.output / "session"
        finally:
            closed.append(True)
    def fail(run, directory, *unused, **kwargs):
        directory.mkdir(parents=True)
        (directory / "requests.jsonl").write_text('preserved partial journal\n')
        raise RuntimeError("Synthetic interrupted client")
    monkeypatch.setattr(study, "server_session", session)
    monkeypatch.setattr(study, "run_benchmark", fail)
    with pytest.raises(RuntimeError, match="interrupted client"):
        deployment.run(args)
    assert closed == [True]
    assert study.read_json(args.output / "deployment_state.json")["status"] == "failed"
    assert not (args.output / "deployment_completion.json").exists()
    assert not (args.output / "deployment_running.lock").exists()
    assert (args.output / "runs/deployment_quality__fixed/requests.jsonl").read_text() == 'preserved partial journal\n'


def test_complete_dependency_receipt_binds_all_eight_validated_methods(tmp_path):
    args, plan = fixture(tmp_path)
    finish_offline(args, plan)
    registration, _ = deployment.prepare_registration(args)
    receipt = deployment.offline_dependency(registration, plan)
    assert receipt["config_sha256"] == study.digest(args.offline_config)
    assert receipt["methods"] == deployment.EXPECTED_METHODS
    assert set(receipt["method_score_sha256"]) == set(deployment.EXPECTED_METHODS)
    assert all(value["successful"] == 3 and value["complete_outputs"]
               for value in receipt["per_method_validation"].values())


@pytest.mark.parametrize("fault", ["only_fixed", "wrong_config", "bad_successful"])
def test_self_consistent_but_wrong_campaign_cannot_launch(tmp_path, monkeypatch, fault):
    args, plan = fixture(tmp_path)
    finish_offline(args, plan)
    identity = study.read_json(args.offline_root / "identity.json")
    summary = study.read_json(args.offline_root / "summary.json")
    state = study.read_json(args.offline_root / "state.json")
    if fault == "only_fixed":
        identity["methods"] = state["methods"] = ["fixed"]
        summary["methods"] = {"fixed": summary["methods"]["fixed"]}
    elif fault == "wrong_config":
        identity["config"] = "wrong-source-campaign"
    else:
        summary["methods"]["rrf"]["successful"] = 2
    for name, value in (("identity", identity), ("summary", summary), ("state", state)):
        (args.offline_root / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(study, "server_session", lambda *a: pytest.fail("Invalid offline evidence must not launch"))
    with pytest.raises(ValueError):
        deployment.run(args)
    assert not (args.output / "deployment_state.json").exists()


@pytest.mark.parametrize("fault", ["duplicate", "failed", "stub", "wrong_identity", "missing_grid",
                                  "wrong_order", "bad_probability", "bad_sentiment"])
def test_nonfixed_incomplete_outputs_rejected_even_with_matching_file_sha(tmp_path, monkeypatch, fault):
    args, plan = fixture(tmp_path)
    finish_offline(args, plan)
    path = args.offline_root / "e5_negative_seen.jsonl"
    rows = study.read_jsonl(path)
    if fault == "duplicate":
        rows[2] = rows[0]
    elif fault == "failed":
        rows[0]["status"] = "error"
    elif fault == "stub":
        rows[0] = {"request_id": rows[0]["request_id"], "status": "ok"}
    elif fault == "wrong_identity":
        rows[0]["response"]["request_id"] = "other"
    elif fault == "missing_grid":
        rows[0]["response"]["full_grid_scores"].pop()
    elif fault == "wrong_order":
        rows[0]["response"]["aspect_scores"].reverse()
    elif fault == "bad_probability":
        rows[0]["response"]["aspect_scores"][0]["score"] = float("nan")
    else:
        rows[0]["response"]["full_grid_scores"][0]["sentiment_scores"] = {"negative": .2, "neutral": .1, "positive": .1}
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding="utf-8")
    summary_path = args.offline_root / "summary.json"
    summary = study.read_json(summary_path)
    summary["methods"]["e5_negative_seen"]["file_sha256"] = study.digest(path)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr(study, "server_session", lambda *a: pytest.fail("Incomplete matrix must not launch"))
    with pytest.raises(ValueError):
        deployment.run(args)
    assert not (args.output / "deployment_state.json").exists()


@pytest.mark.parametrize("margin", [None, float("nan"), float("inf"), True])
def test_margin_campaign_rejects_missing_or_nonfinite_native_scores(tmp_path, margin):
    row = record("synthetic")
    row["response"].update(variant="hybrid", presence_score_space="logit_margin")
    for item in row["response"]["aspect_scores"]:
        item["presence_logit_margin"] = 2.
    row["response"]["aspect_scores"][0]["presence_logit_margin"] = margin
    path = tmp_path / "native_scores.jsonl"
    path.write_text(json.dumps(row)+'\n', encoding="utf-8")
    with pytest.raises(ValueError, match="finite raw presence margin"):
        deployment.validate_offline_records(path, {"synthetic"}, ["seen", "heldout"], "logit_margin")
