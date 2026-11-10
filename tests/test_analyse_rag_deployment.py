"""Deployment metrics and gold-access gates using synthetic local HTTP journals."""
from contextlib import contextmanager
from copy import deepcopy
import json
import math

import pytest

import analyse_rag_deployment as analysis
import run_rag_deployment as deployment
import run_rag_service_study as study
from test_run_rag_deployment import fixture as runner_fixture, finish_offline
from test_rag_evaluation import record


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def write_lines(path, rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding="utf-8")


def fixture(tmp_path, monkeypatch, *, perfect=False, bad_coverage=False, saturated=False,
            mutate_requests=None):
    prepare = study.prepare_plan
    def prepare_with_taxonomy(args):
        artifacts = study.read_json(args.artifacts)
        artifacts["taxonomy"]["heldout_aspects"] = ["heldout"]
        artifacts.update(model={"revision": "synthetic-revision"}, fold_id="synthetic-fold")
        dump(args.artifacts, artifacts)
        selection = study.read_json(args.selection)
        selection["presence_score_space"] = "logit_margin"
        for threshold in selection["thresholds"].values():
            threshold.update(presence_score_space="logit_margin", presence_margin_threshold=2.)
        dump(args.selection, selection)
        for path in args.development_score_root.glob("*.jsonl"):
            rows = study.read_jsonl(path)
            for row in rows:
                row["response"]["presence_score_space"] = "logit_margin"
                for item in row["response"]["aspect_scores"]:
                    item["presence_logit_margin"] = math.log(item["score"] / (1 - item["score"]))
            write_lines(path, rows)
        return prepare(args)
    with monkeypatch.context() as fixture_patch:
        fixture_patch.setattr(study, "prepare_plan", prepare_with_taxonomy)
        run_args, plan = runner_fixture(tmp_path)
    finish_offline(run_args, plan)
    offline = []
    for i in range(3):
        row = record(f"eval{i}", seen=.1 if i == 2 else .9, heldout=.9 if i == 2 else .1,
                     sentiment={"negative": .1, "neutral": .1, "positive": .8} if i == 2 else None)
        for item in row["response"]["aspect_scores"]:
            item["presence_logit_margin"] = math.log(item["score"] / (1 - item["score"]))
            if saturated:
                item["score"] = 1.0  # intentional p32 ties, raw margins retain separation
        row["response"].update(variant="hybrid", presence_score_space="logit_margin", presence_margin_threshold=2.)
        offline.append(row)
    score_path = run_args.offline_root / "fixed.jsonl"
    write_lines(score_path, offline)
    summary = study.read_json(run_args.offline_root / "summary.json")
    summary["methods"]["fixed"].update(file_sha256=study.digest(score_path), successful=3)
    dump(run_args.offline_root / "summary.json", summary)
    clone = study.read_json(study.Path(plan["methods"]["fixed"]["artifacts"]))
    requests = []
    for i, reference in enumerate(offline):
        aspect = "heldout" if i == 2 else "seen"
        label = "positive" if i == 2 or (i == 0 and not perfect) else "negative"
        version = (f"{clone['model']['revision'][:8]}-{clone['fold_id']}-"
                   f"{plan['methods']['fixed']['artifacts_sha256'][:12]}-hybrid-prefixon-rag-fixed-synthetic")
        response = {"request_id": f"op{i}", "variant": "hybrid", "model_version": version,
            "presence_score_space": "logit_margin",
            "aspect_scores": deepcopy(reference["response"]["aspect_scores"]),
            "predictions": [{"aspect": aspect, "sentiments": [label]}],
            "diagnostics": {"retrieval_fallback": False, "effective_retrieval_method": "fixed",
                "effective_retrieval_selection_policy": "global",
                "effective_thresholds": clone["variants"]["hybrid"],
                "sentiment_execution": plan["server"]["sentiment_execution"],
                "batch_aspect": {"prefix_cache_configured": True},
                "sentiment_group_contract": plan["sentiment_group_contract"]}}
        if i == 2 and not perfect:
            response["aspect_scores"][1].update(score=.95, presence_logit_margin=math.log(.95/.05))
        success = perfect or i != 1
        requests.append({"sample_id": f"eval{i}", "attempt_index": i, "request_id": f"op{i}",
            "success": success, "status_code": 200 if success else None,
            "error": None if success else "timeout", "response": response if success else {}})
    if bad_coverage:
        requests.pop()  # eval2 missing, later eval1 retry must not replace its failure
        retry = deepcopy(requests[0])
        retry.update(sample_id="eval1", attempt_index=3, request_id="op3")
        retry["response"]["request_id"] = "op3"
        requests.append(retry)
    if mutate_requests is not None:
        mutate_requests(requests, clone, plan)
    @contextmanager
    def session(*args):
        yield object(), {}, tmp_path / "fake-session"
    def benchmark(run, directory, *args, **kwargs):
        directory.mkdir(parents=True)
        write_lines(directory / "requests.jsonl", requests)
        dump(directory / "summary.json", {"attempts": len(requests),
            "successful": sum(row["success"] for row in requests), "interrupted": False})
        dump(directory / "complete.json", {"summary_sha256": analysis.sha(directory / "summary.json")})
        dump(directory / "registered_run.json", run)
        dump(directory / "config.json", {k: run[k] for k in ("mode", "concurrency", "requests", "shuffle", "seed", "replay")})
    monkeypatch.setattr(study, "server_session", session)
    monkeypatch.setattr(study, "run_benchmark", benchmark)
    deployment.run(run_args)
    labels = tmp_path / "synthetic_labels.jsonl"
    write_lines(labels, [{"request_id": f"eval{i}", "pairs": [{"aspect": "heldout" if i == 2 else "seen",
        "sentiment": "positive" if i == 2 else "negative"}]} for i in range(3)])
    args = analysis.parser().parse_args(["--deployment-root", str(run_args.output),
        "--offline-scores", str(score_path), "--labels", str(labels), "--output", str(tmp_path / "analysis")])
    return args, requests


def reject_gold_reads(monkeypatch, args):
    original = analysis.read_jsonl
    def checked(path):
        if study.Path(path).resolve() == args.labels.resolve():
            pytest.fail("Gold was opened before evidence checks completed")
        return original(path)
    monkeypatch.setattr(analysis, "read_jsonl", checked)


def test_actual_outputs_failures_and_scope_counts_are_preserved(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path, monkeypatch)
    result = analysis.analyse(args)
    assert result["coverage"]["attempts"] == 3
    assert result["coverage"]["first_attempt_failed_reviews"] == 1
    assert result["API"]["scopes"]["overall"]["pair"]["micro_f1"] == .4
    assert result["API"]["scopes"]["overall"]["presence"]["micro_f1"] == .8
    assert result["API"]["scopes"]["seen"]["pair"]["fn"] == 2
    assert result["API"]["scopes"]["heldout"]["pair"]["tp"] == 1
    assert result["actual_API_exact_set_accuracy_all_expected_reviews"] == pytest.approx(1/3)
    assert result["review_label_set_changes_all_expected_reviews"] == {
        "sentiment_only_changed": 1, "presence_changed": 1, "unchanged": 1}
    assert result["presence_score_deltas"]["candidate_pairs"] == 4
    assert result["presence_score_deltas"]["probability_max_abs_delta"] == pytest.approx(.05)
    assert result["bootstrap"]["draws"] == 2000 and result["bootstrap"]["seed"] == 20261111
    assert result["API_minus_offline"]["scopes"]["overall"]["pair"]["f1_difference_points"] == -60
    assert "HTTP" in result["timing_scope"]
    assert result["population_scope"] == "historically_exposed_validation_replay"
    assert "eval0" not in (args.output / "quality.json").read_text()
    analysis.analyse(args)  # identical evidence is idempotent


def test_identical_api_offline_has_zero_paired_intervals(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path, monkeypatch, perfect=True)
    result = analysis.analyse(args)
    for scope in ("overall", "seen", "heldout"):
        for metric in ("pair", "presence"):
            contrast = result["API_minus_offline"]["scopes"][scope][metric]
            assert contrast["f1_difference_points"] == 0
            assert contrast["f1_difference_points_percentile_95_interval"] == [0, 0]


def test_explicit_margin_policy_decodes_saturated_probability_ties(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path, monkeypatch, perfect=True, saturated=True)
    result = analysis.analyse(args)
    assert result["frozen_thresholds"]["presence_score_space"] == "logit_margin"
    assert result["frozen_thresholds"]["presence_margin_threshold"] == 2.
    assert result["frozen_offline"]["scopes"]["overall"]["presence"]["fp"] == 0
    assert result["API_minus_offline"]["scopes"]["overall"]["pair"]["f1_difference_points"] == 0


def test_failed_request_is_not_exactly_correct_even_when_gold_is_empty(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path, monkeypatch)
    labels = analysis.read_jsonl(args.labels)
    labels[1]["pairs"] = []
    write_lines(args.labels, labels)
    result = analysis.analyse(args)
    assert result["actual_API_exact_set_accuracy_all_expected_reviews"] == pytest.approx(1/3)


@pytest.mark.parametrize("mismatch", ["threshold", "cache", "version", "group"])
def test_successful_api_wrong_runtime_identity_refuses_gold(tmp_path, monkeypatch, mismatch):
    def mutate(rows, clone, plan):
        response = rows[0]["response"]
        if mismatch == "threshold":
            response["diagnostics"]["effective_thresholds"] = dict(clone["variants"]["hybrid"], presence_margin_threshold=123.)
        elif mismatch == "cache":
            response["diagnostics"]["batch_aspect"]["prefix_cache_configured"] = False
        elif mismatch == "version":
            response["model_version"] = "unbound-previous-model"
        else:
            response["diagnostics"]["sentiment_group_contract"] = {"contract_sha256": "wrong"}
    args, _ = fixture(tmp_path, monkeypatch, mutate_requests=mutate)
    reject_gold_reads(monkeypatch, args)
    with pytest.raises(ValueError, match="API"):
        analysis.analyse(args)


def test_fixed_fallback_uses_own_frozen_threshold_and_source(tmp_path, monkeypatch):
    def mutate(rows, clone, plan):
        rows[0]["response"]["diagnostics"].update(retrieval_fallback=True,
            fallback_policy_source_sha256=plan["methods"]["fixed"]["artifacts_sha256"])
    args, _ = fixture(tmp_path, monkeypatch, mutate_requests=mutate)
    result = analysis.analyse(args)
    assert result["coverage"]["first_attempt_successful_reviews"] == 2


def test_incomplete_inference_refuses_gold(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path, monkeypatch)
    state = study.read_json(args.deployment_root / "deployment_state.json")
    state["status"] = "running"
    dump(args.deployment_root / "deployment_state.json", state)
    reject_gold_reads(monkeypatch, args)
    with pytest.raises(ValueError, match="must be complete"):
        analysis.analyse(args)


@pytest.mark.parametrize("target", ["requests", "offline", "other_method", "clone", "registration", "dependency"])
def test_changed_completed_evidence_refuses_gold(tmp_path, monkeypatch, target):
    args, _ = fixture(tmp_path, monkeypatch)
    paths = {"requests": args.deployment_root / "runs/deployment_quality__fixed/requests.jsonl",
        "offline": args.offline_scores, "clone": args.deployment_root / "inputs/artifacts_fixed.json",
        "other_method": args.offline_scores.parent / "e5_negative_seen.jsonl",
        "registration": args.deployment_root / "registration.json",
        "dependency": args.deployment_root / "offline_dependency_receipt.json"}
    with paths[target].open("a") as stream:
        stream.write(" ")
    reject_gold_reads(monkeypatch, args)
    with pytest.raises(ValueError, match="hash"):
        analysis.analyse(args)


def test_bad_coverage_outputs_diagnostic_before_gold(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path, monkeypatch, bad_coverage=True)
    reject_gold_reads(monkeypatch, args)
    with pytest.raises(ValueError, match="coverage is incomplete"):
        analysis.analyse(args)
    result = study.read_json(args.output / "coverage.json")
    assert result["missing_reviews"] == 1 and result["duplicate_review_ids"] == 1
    assert not (args.output / "quality.json").exists()


def test_explicit_audit_keeps_earliest_failed_attempt_and_missing_empty(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path, monkeypatch, bad_coverage=True)
    args.allow_incomplete_audit = True
    result = analysis.analyse(args)
    assert result["scope"] == "coverage_audit_not_complete_deployment"
    assert result["coverage"]["first_attempt_successful_reviews"] == 1
    assert result["coverage"]["failed_or_missing_reviews_in_quality_denominator"] == 2
    assert result["API"]["scopes"]["overall"]["pair"]["micro_f1"] == 0


def test_attempt_order_is_explicit_and_never_success_preference():
    rows = [{"sample_id": "a", "attempt_index": 2, "success": True},
            {"sample_id": "a", "attempt_index": 0, "success": False, "error": "timeout"},
            {"sample_id": "outside", "attempt_index": 3, "success": True}]
    result, first = analysis.coverage(rows, ["a", "b"])
    assert first["a"]["success"] is False
    assert result["unexpected_attempts"] == result["missing_reviews"] == 1
    rows[-1]["attempt_index"] = 0
    with pytest.raises(ValueError, match="Repeated attempt"):
        analysis.coverage(rows, ["a", "b"])
