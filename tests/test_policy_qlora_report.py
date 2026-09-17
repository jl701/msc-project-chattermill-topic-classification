"""All fixture values are invented layout-test data, never experiment evidence."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location("policy_report", Path(__file__).resolve().parents[1] / "scripts/render_policy_qlora_completion_report.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def fixture(root):
    root.mkdir()
    audit = {"status": "PASS", "source_fold_units": 48, "source_condition_grids": 96,
             "derived_composition_grids": 48, "reported_fold_condition_systems": 144,
             "failure_count": 0, "test_contract_count": 0, "selection_retuned": False,
             "same_policy_components": True, "N_D_seen_scores_identical": True, "evidence_role": report.ROLE}
    (root / "audit.json").write_text(json.dumps(audit))
    (root / "input_manifest.json").write_text(json.dumps({"include_official_test": False, "test_contract_count": 0,
                                                          "allowed_splits": ["validation"], "synthetic_fixture": True}))
    rows = [{"method": m, "policy": p, "condition": c, "folds": 12, "heldout_pair_f1": .5,
             "heldout_presence_f1": .6, "oracle_stage2_f1": .8} for p in report.POLICIES for m in report.METHODS for c in ("N", "D")]
    pd.DataFrame(rows).to_csv(root / "aggregate_metrics.csv", index=False)
    pd.DataFrame([{**row, "fold_id": f"l2-a{i:02d}"} for row in rows for i in range(1, 13)]).to_csv(root / "fold_metrics.csv", index=False)
    names = [f"{m}_{c}_masked_minus_filtered" for c in ("N", "D") for m in report.METHODS]
    names += [f"{p}_{c}_composition_minus_{m}" for c in ("N", "D") for p in report.POLICIES for m in report.METHODS[:2]]
    names += [f"{c}_change_in_composition_gain_over_{m}" for c in ("N", "D") for m in report.METHODS[:2]]
    pd.DataFrame([{"comparison": n, "estimate": .01 * i, "nominal_ci_low": .01 * i - .02,
                   "nominal_ci_high": .01 * i + .02, "draws": 20000, "seed": 13} for i, n in enumerate(names)]).to_csv(root / "paired_contrasts.csv", index=False)
    (root / "review_confusion_and_bootstrap.npz").write_bytes(b"SYNTHETIC fixture placeholder, renderer does not load bootstrap arrays")
    seal(root)


def seal(root):
    (root / "output_hashes.json").write_text(json.dumps({n: report.digest(root / n) for n in report.FILES}))


def test_synthetic_cannot_be_presented_as_results(tmp_path):
    source = tmp_path / "fixture"
    fixture(source)
    with pytest.raises(ValueError, match="synthetic"):
        report.render(source, tmp_path / "report")
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize("field,value", [("status", "PENDING"), ("source_fold_units", 47), ("test_contract_count", 1), ("same_policy_components", False)])
def test_incomplete_or_unsafe_audit_blocks_before_output(tmp_path, field, value):
    source = tmp_path / "fixture"
    fixture(source)
    audit = report.read(source / "audit.json")
    audit[field] = value
    (source / "audit.json").write_text(json.dumps(audit))
    seal(source)
    with pytest.raises(ValueError):
        report.render(source, tmp_path / "report", True)
    assert not (tmp_path / "report").exists()


def test_tamper_blocks(tmp_path):
    source = tmp_path / "fixture"
    fixture(source)
    (source / "aggregate_metrics.csv").write_text("changed")
    with pytest.raises(ValueError, match="hash"):
        report.validate(source, True)


def test_full_synthetic_render_and_immutable_output(tmp_path):
    source, target = tmp_path / "fixture", tmp_path / "preview"
    fixture(source)
    receipt = report.render(source, target, True)
    assert receipt["status"] == "SYNTHETIC_QA_ONLY"
    assert len(pd.read_csv(target / "training_policy_core.csv")) == 6
    assert "SYNTHETIC PREVIEW" in (target / "training_policy_core.tex").read_text()
    assert all(report.digest(target / name) == h for name, h in receipt["files"].items())
    for extension in ("svg", "png", "pdf"):
        assert (target / f"training_policy_core_effects.{extension}").stat().st_size > 1000
    with pytest.raises(FileExistsError):
        report.render(source, target, True)
