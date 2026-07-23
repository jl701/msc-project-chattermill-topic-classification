"""Leakage and one-test-use gates for taxonomy-generalisation runs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from msc_project.experiments.taxonomy_execution import RunContract
from msc_project.experiments.taxonomy_protocol import (
    PAIR_KEY,
    TaxonomyFold,
    candidate_representation_variants,
    pair_identity_hash,
)
from msc_project.experiments.taxonomy_resources import (
    canonical_json_sha256,
    description_hash_payload,
)


@dataclass(frozen=True)
class AuditCheck:
    name: str
    passed: bool
    details: dict[str, object]


@dataclass(frozen=True)
class AuditReport:
    checks: tuple[AuditCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def assert_passed(self) -> None:
        failures = [check.name for check in self.checks if not check.passed]
        if failures:
            raise ValueError(f"Taxonomy leakage audit failed: {failures}")

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "checks": [
                {
                    "name": check.name,
                    "passed": check.passed,
                    "details": check.details,
                }
                for check in self.checks
            ],
        }


def _aspects_in_labels(frame: pd.DataFrame, column: str) -> set[str]:
    if column not in frame:
        return set()
    return {
        str(aspect)
        for labels in frame[column]
        for aspect, _ in labels
    }


def audit_taxonomy_fold(
    source_frame: pd.DataFrame,
    fold: TaxonomyFold,
    splits: Mapping[str, pd.DataFrame],
    training_manifest: pd.DataFrame,
    candidates_by_condition: Mapping[str, pd.DataFrame],
    validation_grids_by_condition: Mapping[str, pd.DataFrame],
    test_grids_by_condition: Mapping[str, pd.DataFrame],
    description_resource: Mapping[str, object],
) -> AuditReport:
    required_splits = {"train", "validation", "test"}
    if set(splits) != required_splits:
        raise ValueError("Fold audit requires train, validation, and test splits.")
    checks: list[AuditCheck] = []
    heldout = set(fold.heldout_aspects)

    row_sets = {
        name: set(frame["row_uid"].astype(str))
        for name, frame in splits.items()
    }
    overlap = {
        "train_validation": sorted(row_sets["train"] & row_sets["validation"]),
        "train_test": sorted(row_sets["train"] & row_sets["test"]),
        "validation_test": sorted(row_sets["validation"] & row_sets["test"]),
    }
    checks.append(
        AuditCheck(
            "no_row_overlap",
            not any(overlap.values()),
            {"overlap_counts": {key: len(value) for key, value in overlap.items()}},
        )
    )

    official_eval_match = {}
    for split_name in ("validation", "test"):
        expected = set(
            source_frame.loc[
                source_frame["original_split"] == split_name, "row_uid"
            ].astype(str)
        )
        observed = row_sets[split_name]
        official_eval_match[split_name] = {
            "expected_rows": len(expected),
            "observed_rows": len(observed),
            "missing": sorted(expected - observed),
            "extra": sorted(observed - expected),
        }
    checks.append(
        AuditCheck(
            "all_official_evaluation_rows",
            all(
                not value["missing"] and not value["extra"]
                for value in official_eval_match.values()
            ),
            official_eval_match,
        )
    )

    heldout_in_original = sorted(
        heldout & _aspects_in_labels(splits["train"], "labels")
    )
    heldout_in_supervision = sorted(
        heldout & _aspects_in_labels(splits["train"], "supervision_labels")
    )
    heldout_in_pairs = sorted(
        heldout & set(training_manifest["candidate_aspect"].astype(str))
    )
    checks.append(
        AuditCheck(
            "heldout_absent_from_training",
            not heldout_in_original
            and not heldout_in_supervision
            and not heldout_in_pairs,
            {
                "original_labels": heldout_in_original,
                "supervision_labels": heldout_in_supervision,
                "training_pairs": heldout_in_pairs,
            },
        )
    )

    train_eval_org_overlap = {
        split_name: sorted(
            set(splits["train"]["org_index"].astype(str))
            & set(splits[split_name]["org_index"].astype(str))
        )
        if "org_index" in splits["train"] and "org_index" in splits[split_name]
        else []
        for split_name in ("validation", "test")
    }
    checks.append(
        AuditCheck(
            "organisation_overlap_reported",
            True,
            {
                "note": "Organisation overlap is reported, not prohibited, because Levels 1-4 preserve official split boundaries.",
                "overlap": train_eval_org_overlap,
            },
        )
    )

    condition_sets_match = (
        set(candidates_by_condition)
        == set(validation_grids_by_condition)
        == set(test_grids_by_condition)
        == set(fold.conditions)
    )
    checks.append(
        AuditCheck(
            "registered_condition_coverage",
            condition_sets_match,
            {
                "expected": list(fold.conditions),
                "candidate_conditions": sorted(candidates_by_condition),
                "validation_conditions": sorted(validation_grids_by_condition),
                "test_conditions": sorted(test_grids_by_condition),
            },
        )
    )

    representation_errors: dict[str, object] = {}
    pair_hashes: dict[str, dict[str, str]] = {}
    for condition in fold.conditions:
        candidates = candidates_by_condition.get(condition)
        validation = validation_grids_by_condition.get(condition)
        test = test_grids_by_condition.get(condition)
        if candidates is None or validation is None or test is None:
            continue
        expected_variants = candidate_representation_variants(fold, condition)
        observed_variants = {
            str(aspect): str(group["representation_variant"].iloc[0])
            for aspect, group in candidates.groupby("candidate_aspect", sort=False)
        }
        candidate_pairs = set(
            zip(
                candidates["candidate_aspect"].astype(str),
                candidates["candidate_sentiment"].astype(str),
            )
        )
        expected_pairs = {
            (str(aspect), sentiment)
            for aspect in fold.evaluation_aspects
            for sentiment in ("negative", "neutral", "positive")
        }
        errors = {
            "variant_mismatch": observed_variants != expected_variants,
            "candidate_pair_mismatch": candidate_pairs != expected_pairs,
            "validation_duplicate_pairs": bool(
                validation.duplicated(list(PAIR_KEY)).any()
            ),
            "test_duplicate_pairs": bool(test.duplicated(list(PAIR_KEY)).any()),
        }
        if any(errors.values()):
            representation_errors[condition] = errors
        pair_hashes[condition] = {
            "validation": pair_identity_hash(validation),
            "test": pair_identity_hash(test),
        }
    shared_identities = all(
        len({values[split_name] for values in pair_hashes.values()}) <= 1
        for split_name in ("validation", "test")
    )
    checks.append(
        AuditCheck(
            "candidate_scope_and_representation",
            not representation_errors,
            {"errors": representation_errors},
        )
    )
    checks.append(
        AuditCheck(
            "condition_pair_identities_match",
            shared_identities,
            {"pair_identity_sha256": pair_hashes},
        )
    )

    expected_description_hash = canonical_json_sha256(
        description_hash_payload(description_resource)
    )
    checks.append(
        AuditCheck(
            "description_hash_matches",
            description_resource.get("content_sha256")
            == expected_description_hash,
            {
                "declared": description_resource.get("content_sha256"),
                "observed": expected_description_hash,
            },
        )
    )
    return AuditReport(tuple(checks))


def audit_strict_calibration(
    calibration_aspects: tuple[str, ...] | list[str],
    fold: TaxonomyFold,
) -> AuditCheck:
    observed = tuple(str(value) for value in calibration_aspects)
    passed = set(observed) == set(fold.seen_aspects) and not (
        set(observed) & set(fold.heldout_aspects)
    )
    return AuditCheck(
        "seen_only_threshold_calibration",
        passed,
        {
            "expected_seen_aspects": list(fold.seen_aspects),
            "observed_calibration_aspects": list(observed),
            "heldout_aspects": list(fold.heldout_aspects),
        },
    )


def assert_formal_run_gates(
    description_resource: Mapping[str, object],
    precloud_config: Mapping[str, object],
) -> None:
    if description_resource.get("status") != "approved_and_frozen":
        raise ValueError("Formal execution requires approved_and_frozen descriptions.")
    statistics = precloud_config.get("statistics")
    if not isinstance(statistics, Mapping) or statistics.get("status") != "approved_and_frozen":
        raise ValueError("Formal execution requires an approved_and_frozen statistical contract.")


def _test_use_key(contract: RunContract) -> str:
    return "/".join(
        (
            contract.protocol_id,
            contract.method_id,
            contract.level,
            contract.fold_id,
            contract.condition,
            contract.split,
        )
    )


class TestUseLedger:
    """Local fail-closed ledger preventing a second incompatible formal test use."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock_path = path.with_suffix(path.suffix + ".lock")

    def _acquire(self) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            return os.open(
                self.lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError as error:
            raise RuntimeError(f"Test-use ledger is locked: {self.lock_path}") from error

    def _release(self, descriptor: int) -> None:
        os.close(descriptor)
        self.lock_path.unlink()

    def _read(self) -> dict[str, object]:
        if not self.path.exists():
            return {"schema_version": "taxonomy_test_use_v1", "claims": {}}
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != "taxonomy_test_use_v1"
            or not isinstance(value.get("claims"), dict)
        ):
            raise ValueError("Test-use ledger is malformed.")
        return value

    def _write(self, value: Mapping[str, object]) -> None:
        temporary = self.path.with_name(f".{self.path.name}.tmp-{os.getpid()}")
        if temporary.exists():
            raise FileExistsError(f"Stale ledger temporary file exists: {temporary}")
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, self.path)

    def claim(self, contract: RunContract) -> dict[str, object]:
        contract.validate()
        if contract.split != "test" or not contract.formal:
            raise ValueError("Only formal test contracts may claim the test-use ledger.")
        descriptor = self._acquire()
        try:
            ledger = self._read()
            claims = ledger["claims"]
            key = _test_use_key(contract)
            existing = claims.get(key)
            if existing is not None:
                if existing.get("contract_sha256") != contract.contract_sha256:
                    raise ValueError(
                        "A different formal contract has already claimed this test endpoint."
                    )
                return dict(existing)
            claim = {
                "contract_sha256": contract.contract_sha256,
                "status": "started",
            }
            claims[key] = claim
            self._write(ledger)
            return claim
        finally:
            self._release(descriptor)

    def complete(self, contract: RunContract) -> dict[str, object]:
        descriptor = self._acquire()
        try:
            ledger = self._read()
            claims = ledger["claims"]
            key = _test_use_key(contract)
            existing = claims.get(key)
            if (
                existing is None
                or existing.get("contract_sha256") != contract.contract_sha256
            ):
                raise ValueError("Formal test contract was not claimed before completion.")
            existing["status"] = "complete"
            self._write(ledger)
            return dict(existing)
        finally:
            self._release(descriptor)
