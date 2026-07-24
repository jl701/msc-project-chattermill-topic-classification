from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.experiments.taxonomy_audit import (
    TestUseLedger as FormalTestUseLedger,
    assert_formal_run_gates,
    audit_strict_calibration,
    audit_taxonomy_fold,
)
from msc_project.experiments.taxonomy_execution import RunContract, canonical_sha256
from msc_project.experiments.taxonomy_protocol import (
    build_budgeted_training_manifest,
    build_candidate_table,
    build_taxonomy_eval_grid,
    build_taxonomy_fold_splits,
    pair_identity_hash,
    registered_folds,
)
from msc_project.experiments.taxonomy_resources import load_minimal_descriptions


def source_frame() -> pd.DataFrame:
    aspects = registered_folds("L1")[0].seen_aspects
    heldout = registered_folds("L1")[0].heldout_aspects[0]
    rows = [
        ("train", 1, [(heldout, "positive")]),
        ("train", 2, [(aspects[1], "negative")]),
        ("validation", 3, [(heldout, "negative")]),
        ("validation", 4, []),
        ("test", 5, [(aspects[1], "positive")]),
        ("test", 6, []),
    ]
    frame = pd.DataFrame(
        [
            {
                "id": index,
                "original_split": split,
                "row_uid": f"{split}:{index}",
                "text": f"synthetic {index}",
                "labels": labels,
                "org_index": index % 2,
                "industry": "synthetic",
                "data_source": "synthetic",
            }
            for split, index, labels in rows
        ]
    )
    frame["pair_labels"] = frame["labels"].apply(
        lambda labels: [f"{aspect} | {sentiment}" for aspect, sentiment in labels]
    )
    frame["aspect_labels"] = frame["labels"].apply(
        lambda labels: sorted({aspect for aspect, _ in labels})
    )
    return frame


def valid_audit_inputs():
    frame = source_frame()
    fold = registered_folds("L3")[0]
    splits = build_taxonomy_fold_splits(frame, fold)
    resource = load_minimal_descriptions(require_approved=False)
    training = build_budgeted_training_manifest(
        splits["train"],
        fold,
        resource,
        total_budget=32,
        positive_budget=16,
    )
    candidates = {}
    validation = {}
    test = {}
    for condition in fold.conditions:
        candidates[condition] = build_candidate_table(fold, condition, resource)
        validation[condition] = build_taxonomy_eval_grid(
            splits["validation"],
            candidates[condition],
            fold_id=fold.fold_id,
            condition=condition,
        )
        test[condition] = build_taxonomy_eval_grid(
            splits["test"],
            candidates[condition],
            fold_id=fold.fold_id,
            condition=condition,
        )
    return frame, fold, splits, training, candidates, validation, test, resource


def test_fold_audit_passes_and_reports_allowed_organisation_overlap() -> None:
    values = valid_audit_inputs()
    report = audit_taxonomy_fold(*values)
    assert report.passed
    report.assert_passed()
    organisation = next(
        check for check in report.checks if check.name == "organisation_overlap_reported"
    )
    assert organisation.passed
    assert "not prohibited" in organisation.details["note"]


def test_fold_audit_rejects_pair_identity_or_training_leakage() -> None:
    values = list(valid_audit_inputs())
    validation = {key: value.copy() for key, value in values[5].items()}
    validation["DN"].loc[validation["DN"].index[0], "target"] ^= 1
    values[5] = validation
    report = audit_taxonomy_fold(*values)
    assert not report.passed
    with pytest.raises(ValueError, match="condition_pair_identities_match"):
        report.assert_passed()

    values = list(valid_audit_inputs())
    training = values[3].copy()
    training.loc[training.index[0], "candidate_aspect"] = values[1].heldout_aspects[0]
    values[3] = training
    report = audit_taxonomy_fold(*values)
    assert not report.passed


def test_calibration_audit_requires_exact_seen_set() -> None:
    fold = registered_folds("L2")[0]
    assert audit_strict_calibration(list(fold.seen_aspects), fold).passed
    assert not audit_strict_calibration(
        [*fold.seen_aspects, *fold.heldout_aspects], fold
    ).passed


def test_formal_gate_rejects_pending_resource_and_statistics() -> None:
    resource = load_minimal_descriptions(require_approved=False)
    with pytest.raises(ValueError, match="descriptions"):
        assert_formal_run_gates(resource, {"statistics": {"status": "approved_and_frozen"}})
    approved = dict(resource)
    approved["status"] = "approved_and_frozen"
    with pytest.raises(ValueError, match="statistical"):
        assert_formal_run_gates(approved, {"statistics": {"status": "proposed"}})


def formal_contract() -> RunContract:
    return RunContract(
        protocol_id="p",
        scientific_protocol_sha256="0" * 64,
        method_id="m",
        method_spec_sha256="a" * 64,
        method_registry_sha256="b" * 64,
        description_resource_sha256="c" * 64,
        candidate_representation_sha256="d" * 64,
        level="L2",
        fold_id="f1",
        condition="D",
        split="test",
        seed=13,
        training_manifest_sha256="e" * 64,
        evaluation_data_sha256="f" * 64,
        evaluation_pair_identity_sha256="0" * 64,
        scientific_parameters_sha256=canonical_sha256({"lr": 1}),
        shard_count=2,
        formal=True,
    )


def test_test_use_ledger_allows_resume_but_rejects_new_contract(tmp_path: Path) -> None:
    ledger = FormalTestUseLedger(tmp_path / "test_use.json")
    run = formal_contract()
    assert ledger.claim(run)["status"] == "started"
    assert ledger.claim(run)["contract_sha256"] == run.contract_sha256
    assert ledger.complete(run)["status"] == "complete"
    changed = replace(
        run,
        scientific_parameters_sha256=canonical_sha256({"lr": 2}),
    )
    with pytest.raises(ValueError, match="different formal contract"):
        ledger.claim(changed)

    other_seed = replace(run, seed=23)
    assert ledger.claim(other_seed)["status"] == "started"
    assert ledger.complete(other_seed)["status"] == "complete"
