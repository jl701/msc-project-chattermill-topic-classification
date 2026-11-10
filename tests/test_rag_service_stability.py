"""Synthetic, label-free checks for the RAG campaign stability analysis."""
from datetime import datetime, timedelta, timezone
import json

import pytest

from analyse_rag_service_stability import (
    analyse_campaign, fallback_counts, repeat_consistency, sha, time_windows, write_outputs,
)
from msc_project.serving.sentiment_execution import build_sentiment_group_contract


def execution_contract(*, execution="canonical_full_grid_groups", pair_batch_size=4,
                       max_batch_tokens=4096, logits_mode="last_token", candidates=None):
    return build_sentiment_group_contract(
        [{"aspect": "a"}, {"aspect": "b"}] if candidates is None else candidates,
        execution=execution, pair_batch_size=pair_batch_size,
        max_batch_tokens=max_batch_tokens, logits_mode=logits_mode)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def outcome(index, scheduled, *, success=True, finish=None, error=None, drop=False,
            sample="private-input", threshold=.4, label="positive", fallback=False):
    finish = scheduled + .1 if finish is None else finish
    response = {"request_id": f"private-attempt-{index}", "model_version": "synthetic-model", "variant": "hybrid", "fold_id": "synthetic-fold",
        "predictions": [{"aspect": "aspect", "sentiments": [label]}],
        "diagnostics": {"effective_retrieval_method": "fixed" if fallback else "bm25",
                        "effective_retrieval_selection_policy": "global",
                        "sentiment_execution": "canonical_full_grid_groups",
                        "sentiment_group_contract": execution_contract(),
                        "effective_thresholds": {"aspect_threshold": threshold},
                        "retrieval_fallback": fallback},
        "timings_ms": {"queue_wait": 2}}
    return {"request_id": f"private-attempt-{index}", "sample_id": sample,
        "attempt_index": index, "scheduled_s": scheduled, "finished_s": finish,
        "started_s": None if drop else scheduled, "success": success,
        "error": error, "status_code": None if drop else 200 if success else 429,
        "latency_ms": None if drop else (finish - scheduled) * 1000,
        "scheduled_latency_ms": None if drop else (finish - scheduled) * 1000,
        "schedule_lag_ms": None if drop else 0, "input_chars": 45,
        "response": response if success else {}}


def test_windows_keep_all_planned_attempts_and_separate_arrivals_from_completion():
    rows = [outcome(0, 0), outcome(1, 100, success=False, error="http_429"),
            outcome(2, 200, success=False, drop=True, error="client_capacity_drop"),
            outcome(3, 299, finish=301), outcome(4, 300, finish=721, fallback=True)]
    windows = time_windows(rows, [0, 100, 200, 299, 300], 720, 722, [], 0)
    assert [w["window_kind"] for w in windows] == [
        "full_arrival_window", "full_arrival_window", "partial_arrival_tail", "drain_only_tail"]
    first, second, partial, drain = windows
    assert first["planned_attempts"] == first["observed_attempts"] == 4
    assert first["successful"] == 2 and first["failed"] == 2
    assert first["http_429_failures"] == first["unsent_client_drops"] == 1
    assert first["success_fraction_all_planned_attempts"] == .5
    assert first["successful_completions_in_wall_window"] == 1
    assert first["completion_throughput_reviews_per_second"] == pytest.approx(1 / 300)
    assert second["successful_completions_in_wall_window"] == 1  # earlier backlog
    assert partial["observed_wall_seconds"] == 120
    assert not partial["is_full_five_minute_arrival_window"]
    assert drain["planned_attempts"] == drain["observed_attempts"] == 0
    assert drain["successful_completions_in_wall_window"] == 1
    assert drain["success_fraction_all_planned_attempts"] is None
    assert drain["completion_throughput_reviews_per_second"] == .5


def test_repeat_sets_never_pool_different_thresholds_or_fallback_policies():
    rows = [outcome(0, 0), outcome(1, 1), outcome(2, 2, threshold=.7, label="negative"),
            outcome(3, 3, threshold=.7, label="negative"), outcome(4, 4, label="negative"),
            outcome(5, 5, fallback=True), outcome(6, 6, success=False, error="timeout")]
    rows.append(outcome(7, 7))
    del rows[-1]["response"]["diagnostics"]["effective_thresholds"]
    result = repeat_consistency(rows, {"private-input": "private-text-hash"})
    assert result["attempts"] == 8 and result["failed_attempts"] == 1
    assert result["successful_attempts_missing_policy_identity"] == 1
    assert len(result["policy_groups"]) == 3
    assert result["same_input_observed_under_multiple_policies"] == 1
    assert sum(g["repeated_successful_deliveries_after_first"] for g in result["policy_groups"]) == 3
    assert sum(g["repeated_deliveries_differing_from_first"] for g in result["policy_groups"]) == 1
    assert "private-" not in json.dumps(result)
    observed = fallback_counts(rows)
    assert observed["fallback_attempts"] == 1
    assert observed["unobserved_fallback_attempts"] == 1


def test_repeat_sets_separate_e5_global_from_negative_seen_selection():
    rows = [outcome(i, i, label="positive" if i < 2 else "negative") for i in range(4)]
    for i, row in enumerate(rows):
        diag = row["response"]["diagnostics"]
        diag["effective_retrieval_method"] = "e5"
        if i >= 2:
            diag["effective_retrieval_selection_policy"] = "same_aspect_negative_seen"
            # Held-out candidates intentionally retain global selection within
            # this policy. Entry ordering must not create another policy group.
            diag["batch_aspect"] = {"retrieval": [
                {"candidate_aspect": "seen", "effective_selection_policy": "same_aspect_negative_seen"},
                {"candidate_aspect": "heldout", "effective_selection_policy": "global"},
            ][::1 if i == 2 else -1]}
    fallback = outcome(4, 4, fallback=True)
    rows.extend([fallback, outcome(5, 5, fallback=True)])
    result = repeat_consistency(rows, {"private-input": "private-text-hash"})
    assert result["successful_attempts_missing_policy_identity"] == 0
    assert result["same_input_observed_under_multiple_policies"] == 1
    assert len(result["policy_groups"]) == 3
    by_policy = {(g["effective_retrieval_method"], g["effective_retrieval_selection_policy"]): g
                 for g in result["policy_groups"]}
    assert set(by_policy) == {("e5", "global"), ("e5", "same_aspect_negative_seen"),
                              ("fixed", "global")}
    for group in by_policy.values():
        assert group["repeated_successful_deliveries_after_first"] == 1
        assert group["repeated_deliveries_differing_from_first"] == 0
    assert "private-" not in json.dumps(result)


@pytest.mark.parametrize("selection", [None, "unknown", "same_aspect_negative_seen"])
def test_missing_or_invalid_selection_identity_is_not_inferred_as_global(selection):
    rows = [outcome(0, 0), outcome(1, 1)]
    diag = rows[-1]["response"]["diagnostics"]
    if selection is None:
        del diag["effective_retrieval_selection_policy"]
    else:
        diag["effective_retrieval_selection_policy"] = selection  # bm25 cannot use E5-only policy
    result = repeat_consistency(rows, {"private-input": "private-text-hash"})
    assert result["successful_attempts"] == 2
    assert result["successful_attempts_missing_policy_identity"] == 1
    assert len(result["policy_groups"]) == 1
    assert result["policy_groups"][0]["repeated_successful_deliveries_after_first"] == 0


@pytest.mark.parametrize("changes", [
    {"execution": "selected_only"}, {"pair_batch_size": 8}, {"max_batch_tokens": 2048},
    {"logits_mode": "full"}, {"candidates": [{"aspect": "b"}, {"aspect": "a"}]},
])
def test_repeat_sets_bind_execution_parameters_and_ordered_candidates(changes):
    rows = [outcome(i, i, label="positive" if i < 2 else "negative") for i in range(4)]
    changed = execution_contract(**changes)
    for row in rows[2:]:
        row["response"]["diagnostics"].update(
            sentiment_execution=changed["execution"], sentiment_group_contract=changed)
    # Keep model_version identical deliberately: the contract must independently
    # separate execution policies/limits/order rather than rely on a name suffix.
    result = repeat_consistency(rows, {"private-input": "private-text-hash"})
    assert result["successful_attempts_missing_policy_identity"] == 0
    assert len(result["policy_groups"]) == 2
    assert len({g["sentiment_group_contract"]["contract_sha256"] for g in result["policy_groups"]}) == 2
    assert all(g["repeated_successful_deliveries_after_first"] == 1 and
               g["repeated_deliveries_differing_from_first"] == 0 for g in result["policy_groups"])


@pytest.mark.parametrize("damage,reason", [
    ("execution", "missing_execution"), ("contract", "missing_contract"),
    ("hash", "invalid_contract"), ("malformed_field", "invalid_contract"),
    ("contradiction", "execution_contract_mismatch"),
])
def test_unqualified_stage2_identity_keeps_success_denominator(damage, reason):
    rows = [outcome(0, 0), outcome(1, 1)]
    diag = rows[-1]["response"]["diagnostics"]
    if damage == "execution":
        del diag["sentiment_execution"]
    elif damage == "contract":
        del diag["sentiment_group_contract"]
    elif damage == "hash":
        diag["sentiment_group_contract"]["pair_batch_size"] = 8
    elif damage == "malformed_field":
        diag["sentiment_group_contract"]["logits_mode"] = []
    else:
        diag["sentiment_execution"] = "selected_only"
    result = repeat_consistency(rows, {"private-input": "private-text-hash"})
    assert result["successful_attempts"] == 2 and result["failed_attempts"] == 0
    assert result["successful_attempts_missing_policy_identity"] == 1
    assert result["successful_attempts_excluded_for_sentiment_identity"] == {reason: 1}
    assert result["policy_groups"][0]["repeated_successful_deliveries_after_first"] == 0


def synthetic_campaign(tmp_path):
    campaign = tmp_path / "campaign"
    dump(campaign / "plan.json", {"schema_version": "rag_http_service_study_v1",
        "server": {"sentiment_execution": "canonical_full_grid_groups", "pair_batch_size": 4,
                   "max_batch_tokens": 4096, "logits_mode": "last_token"},
        "sentiment_group_contract": execution_contract()})
    run = campaign / "runs" / "soak__bm25"
    run.mkdir(parents=True)
    replay = campaign / "inputs" / "replay.jsonl"
    replay.parent.mkdir()
    replay.write_text(json.dumps({"request_id": "private-input", "text": "PRIVATE REVIEW THAT MUST NEVER APPEAR IN OUTPUT"}) + "\n", encoding="utf-8")
    begin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [outcome(i, i * 60) for i in range(12)]
    rows[3] = outcome(3, 180, success=False, error="timeout", finish=310)
    rows[6] = outcome(6, 360, success=False, error="client_capacity_drop", drop=True)
    rows[-1] = outcome(11, 660, finish=721, fallback=True)
    registered = {"name": "soak__bm25", "mode": "open", "duration": 720, "rate": 1/60,
                  "replay": str(replay)}
    config = {"mode": "open", "utc_start": begin.isoformat(), "replay_sha256": sha(replay)}
    summary = {"interrupted": False, "attempts": 12, "sent": 11, "successful": 10, "failed": 2,
               "wall_seconds_including_drain": 722}
    dump(run / "registered_run.json", registered)
    dump(run / "config.json", config)
    dump(run / "summary.json", summary)
    (run / "requests.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    dump(run / "complete.json", {"summary_sha256": sha(run / "summary.json")})
    session = campaign / "sessions" / "synthetic"
    dump(session / "launch.json", {"registered_utc": (begin - timedelta(seconds=5)).isoformat()})
    dump(session / "closed.json", {"closed_utc": (begin + timedelta(seconds=725)).isoformat()})
    telemetry = [{"event": "measurement", "utc": (begin + timedelta(seconds=t)).isoformat(),
                  "gpu": {"memory.used": memory, "temperature.gpu": 60},
                  "process": {"pid": 99999, "rss_bytes": 1000 + t}}
                 for t, memory in [(1, 3000), (299, 4000), (301, 4001), (700, 4002), (721, 4003)]]
    (session / "telemetry.jsonl").write_text("\n".join(json.dumps(r) for r in telemetry) + "\n", encoding="utf-8")
    return campaign, run


def test_complete_campaign_resources_privacy_and_outputs(tmp_path, monkeypatch):
    from pathlib import Path

    campaign, _ = synthetic_campaign(tmp_path)
    original = Path.read_text

    def guarded_read(path, *args, **kwargs):
        assert "label" not in path.name and "gold" not in path.name
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    result = analyse_campaign(campaign)
    run = result["runs"][0]
    assert result["labels_read"] is False
    from analyse_rag_service_stability import ROOT
    assert result["sentiment_contract_source_sha256"] == sha(ROOT / "src/msc_project/serving/sentiment_execution.py")
    assert result["registered_sentiment_identity"]["sentiment_group_contract"] == execution_contract()
    assert run["repeat_consistency"]["successful_attempts_excluded_for_sentiment_identity"] == {}
    assert run["full_five_minute_window_count"] == 2
    assert run["summary"]["planned_attempts"] == 12
    assert run["summary"]["timeout_failures"] == 1
    assert run["resource_alignment"]["available"]
    assert run["whole_run_sampled_resources"]["metrics"]["gpu_memory_used_mib"]["min"] == 3000
    assert run["whole_run_sampled_resources"]["metrics"]["gpu_memory_used_mib"]["max"] == 4003
    assert run["range_of_full_window_gpu_memory_maxima_mib"] == [4000, 4001]
    encoded = json.dumps(result)
    assert "PRIVATE REVIEW" not in encoded and "private-input" not in encoded
    assert "private-attempt" not in encoded and str(tmp_path) not in encoded and '"pid"' not in encoded
    output = tmp_path / "analysis"
    write_outputs(result, output)
    assert (output / "stability.json").exists() and (output / "windows.csv").exists()
    with pytest.raises(FileExistsError):
        write_outputs(result, output)


@pytest.mark.parametrize("damage,reason", [
    ("execution", "missing_execution"), ("contract", "missing_contract"),
    ("order", "registered_contract_mismatch"), ("limits", "registered_contract_mismatch"),
    ("mode", "registered_contract_mismatch"),
])
def test_plan_binding_reports_response_disagreement_without_erasing_delivery(tmp_path, damage, reason):
    campaign, run = synthetic_campaign(tmp_path)
    rows = [json.loads(line) for line in (run / "requests.jsonl").read_text().splitlines()]
    diag = rows[0]["response"]["diagnostics"]
    if damage in {"execution", "contract"}:
        del diag["sentiment_execution" if damage == "execution" else "sentiment_group_contract"]
    else:
        changes = {"order": {"candidates": [{"aspect": "b"}, {"aspect": "a"}]},
                   "limits": {"pair_batch_size": 8}, "mode": {"execution": "selected_only"}}[damage]
        contract = execution_contract(**changes)
        diag.update(sentiment_execution=contract["execution"], sentiment_group_contract=contract)
    (run / "requests.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    result = analyse_campaign(campaign)["runs"][0]
    assert result["summary"]["planned_attempts"] == 12 and result["summary"]["successful"] == 10
    consistency = result["repeat_consistency"]
    assert consistency["successful_attempts_missing_policy_identity"] == 1
    assert consistency["successful_attempts_excluded_for_sentiment_identity"] == {reason: 1}
    assert sum(g["successful_attempts"] for g in consistency["policy_groups"]) == 9


@pytest.mark.parametrize("damage", ["execution", "parameters", "hash"])
def test_plan_internal_stage2_contradiction_prevents_analysis(tmp_path, damage):
    campaign, _ = synthetic_campaign(tmp_path)
    plan = json.loads((campaign / "plan.json").read_text())
    if damage == "execution":
        plan["server"]["sentiment_execution"] = "selected_only"
    elif damage == "parameters":
        plan["server"]["max_batch_tokens"] = 2048
    else:
        plan["sentiment_group_contract"]["contract_sha256"] = "0" * 64
    dump(campaign / "plan.json", plan)
    with pytest.raises(ValueError, match="Registered sentiment execution/group contract"):
        analyse_campaign(campaign)


def test_legacy_plan_missing_identity_is_explicit_and_never_assumed_canonical(tmp_path):
    campaign, _ = synthetic_campaign(tmp_path)
    dump(campaign / "plan.json", {"schema_version": "rag_http_service_study_v1"})
    result = analyse_campaign(campaign)
    assert result["registered_sentiment_identity"]["available"] is False
    assert result["runs"][0]["repeat_consistency"]["registered_sentiment_identity"]["available"] is False


@pytest.mark.parametrize("damage", ["missing_attempt", "duplicate_attempt", "wrong_schedule", "wrong_scheduled_latency", "summary_changed", "interrupted"])
def test_completed_marker_does_not_promote_damaged_or_short_evidence(tmp_path, damage):
    campaign, run = synthetic_campaign(tmp_path)
    if damage in {"summary_changed", "interrupted"}:
        summary = json.loads((run / "summary.json").read_text())
        summary["interrupted"] = True
        dump(run / "summary.json", summary)
        if damage == "interrupted":
            dump(run / "complete.json", {"summary_sha256": sha(run / "summary.json")})
    else:
        rows = [json.loads(line) for line in (run / "requests.jsonl").read_text().splitlines()]
        if damage == "missing_attempt":
            rows.pop()
        elif damage == "duplicate_attempt":
            rows[-1]["attempt_index"] = 0
        elif damage == "wrong_scheduled_latency":
            rows[-1]["scheduled_latency_ms"] = 0  # Would falsely turn a 61-second success into an SLA success.
        else:
            rows[-1]["scheduled_s"] += 1
        (run / "requests.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError):
        analyse_campaign(campaign)


def test_no_old_campaign_schema_or_cross_campaign_replay(tmp_path):
    campaign, run = synthetic_campaign(tmp_path)
    registered = json.loads((run / "registered_run.json").read_text())
    registered["replay"] = str(tmp_path / "other_campaign" / "replay.jsonl")
    dump(run / "registered_run.json", registered)
    with pytest.raises(ValueError, match="outside this campaign"):
        analyse_campaign(campaign)
    dump(campaign / "plan.json", {"study_id": "serving_20261109"})
    with pytest.raises(ValueError, match="earlier serving"):
        analyse_campaign(campaign)


def test_missing_telemetry_is_unavailable_not_zero_memory(tmp_path):
    campaign, _ = synthetic_campaign(tmp_path)
    (campaign / "sessions" / "synthetic" / "closed.json").unlink()
    result = analyse_campaign(campaign)["runs"][0]
    assert not result["resource_alignment"]["available"]
    assert result["whole_run_sampled_resources"]["metrics"]["gpu_memory_used_mib"]["max"] is None


def test_phase_schedule_cli_and_incomplete_other_run(tmp_path):
    import subprocess
    import sys

    campaign, run = synthetic_campaign(tmp_path)
    registered = json.loads((run / "registered_run.json").read_text())
    phase_path = run / "phases.json"
    dump(phase_path, {"phases": [{"name": "first", "duration": 300, "rate": 1/60},
                                {"name": "second", "duration": 420, "rate": 1/60}]})
    registered.pop("duration")
    registered.pop("rate")
    registered["phase_plan"] = str(phase_path)
    dump(run / "registered_run.json", registered)
    (campaign / "runs" / "soak__pending").mkdir()
    result = analyse_campaign(campaign)
    assert result["runs"][0]["summary"]["planned_attempts"] == 12
    assert result["incomplete_matching_runs"] == ["soak__pending"]
    output = tmp_path / "cli_output"
    from analyse_rag_service_stability import ROOT

    process = subprocess.run([sys.executable, str(ROOT / "scripts/analyse_rag_service_stability.py"),
        "--campaign", str(campaign), "--output-dir", str(output)], capture_output=True, text=True, timeout=15,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    assert process.returncode == 0, process.stderr
    assert json.loads((output / "stability.json").read_text())["incomplete_matching_runs"] == ["soak__pending"]


def test_retrieval_timeout_fallback_is_success_not_http_timeout_and_counts_once():
    row = outcome(0, 0, fallback=True)
    row["response"]["diagnostics"]["batch_aspect"] = {"retrieval": [
        {"fallback": True, "error_type": "RetrievalTimeoutError"} for _ in range(12)]}
    result = fallback_counts([row])
    assert result["retrieval_timeout_fallback_attempts"] == 1
    assert result["successful_fallback_attempts"] == 1
    assert result["fallback_error_review_counts"] == {"RetrievalTimeoutError": 1}


def test_exact_one_hour_has_twelve_full_windows_and_no_invented_tail():
    rows = [outcome(i, i * 100) for i in range(36)]
    windows = time_windows(rows, [i * 100 for i in range(36)], 3600, 3600, [], 0)
    assert len(windows) == 12
    assert all(w["is_full_five_minute_arrival_window"] for w in windows)
    assert sum(w["planned_attempts"] for w in windows) == 36
    assert sum(w["successful_completions_in_wall_window"] for w in windows) == 36
