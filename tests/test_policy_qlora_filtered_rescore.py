"""CPU-only safeguards for the frozen filtered-reference inference handoff."""
import copy
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_policy_qlora_filtered_rescore as rescore


@pytest.mark.parametrize("relative", ["../escape", "/absolute", "C:/drive", "a\\b"])
def test_safe_path(tmp_path, relative):
    with pytest.raises(ValueError):
        rescore.safe_path(tmp_path, relative)


def test_copy_never_overwrites_or_accepts_tamper(tmp_path):
    source, target = tmp_path / "source", tmp_path / "copy"
    source.write_bytes(b"original frozen adapter")
    digest = rescore.masked.sha256_file(source)
    rescore.copy_verified(source, target, digest)
    rescore.copy_verified(source, target, digest)
    target.write_bytes(b"changed")
    with pytest.raises(FileExistsError):
        rescore.copy_verified(source, target, digest)
    source.write_bytes(b"different")
    with pytest.raises(ValueError):
        rescore.copy_verified(source, target, digest)


def valid_audit():
    selected = [{"training_scope_id": scope} for scope in rescore.masked.l2_scopes()]
    return {"status": "PASS_ADAPTER_TRAINING_REUSE", "artifact_origin_commit": rescore.ORIGIN,
            "failed_checks": [], "candidate_count": 36, "scope_count": 12,
            "scopes": copy.deepcopy(selected)}, selected


def test_complete_audit_required():
    audit, selected = valid_audit()
    rescore.validate_audit(audit, selected)
    for name, value in [("status", "partial"), ("candidate_count", 12),
                        ("failed_checks", ["failure"]), ("artifact_origin_commit", "wrong")]:
        with pytest.raises(ValueError):
            rescore.validate_audit({**audit, name: value}, selected)
    with pytest.raises(ValueError):
        rescore.validate_audit(audit, selected[:-1])


def test_duplicate_scope_rejected():
    audit, selected = valid_audit()
    selected[-1] = selected[0]
    audit["scopes"] = selected
    with pytest.raises(ValueError):
        rescore.validate_audit(audit, selected)


def test_scoring_cannot_train_or_reselect():
    import inspect
    source = inspect.getsource(rescore.score)
    assert "executor.select_scope = lambda unused: selection" in source
    assert "train_candidate(" not in source
    assert "select_second_sentiment_threshold(" not in source
    assert "select_two_stage_threshold(" not in source
    assert "CanonicalNDScorer" in source
    assert "local_files_only = True" in source


def test_prepare_missing_inputs_fails_before_mutation(tmp_path):
    with pytest.raises(SystemExit):
        rescore.main(["--phase", "prepare", "--output-root", str(tmp_path / "prepared")])
    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize("field,value", [
    ("study_id", "other"), ("configuration_sha256", "changed"),
    ("recipe", {}), ("retraining", True), ("reselection", True), ("test_contract_count", 1)])
def test_complete_run_identity_binding(field, value):
    executor = rescore.masked.load_executor()
    config = rescore.read(ROOT / "configs/experiments/taxonomy_policy_qlora_completion_v1.json")
    expected = rescore.expected_identity(executor, {"manifest_sha256": "a" * 64}, config)
    normal = rescore.json.loads(rescore.json.dumps(expected))
    rescore.validate_identity(normal, expected)
    with pytest.raises(ValueError):
        rescore.validate_identity({**normal, field: value}, expected)
