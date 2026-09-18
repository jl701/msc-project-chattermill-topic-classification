"""Select one global two-stage router using validation-only formal-v2 scores."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for value in (PROJECT_ROOT, SRC_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import analyse_taxonomy_post_supervisor_formal_v2 as legacy
import analyse_taxonomy_scientific_freeze_v1 as inference
import build_taxonomy_scientific_freeze_formal_evidence as freeze_evidence

from msc_project.experiments.taxonomy_global_router import (
    build_component_arrays,
    confusion_additive_f1,
    policy_id,
    policy_metrics,
    routed_score_frame,
)
from msc_project.experiments.taxonomy_post_supervisor import (
    evaluate_l2_condition,
    post_supervisor_l2_folds,
)
from msc_project.experiments.taxonomy_scientific_freeze import (
    row_pair_confusions,
)

STUDY_ID = "taxonomy_two_stage_global_router_validation_v1"
ROUTER_METHOD_ID = "global_two_stage_uncertainty_router_v1"
REQUIRED_METHODS = (
    "distilbert_review_candidate_cross_encoder",
    "frozen_qwen_few_shot",
    "qwen_candidate_pair_qlora",
)
L2_FOLDS = tuple(f"l2-a{index:02d}" for index in range(1, 13))
CONDITIONS = ("D", "N")
IDENTITY = ["row_uid", "candidate_aspect", "candidate_sentiment"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def non_finite_numeric_count(frame: pd.DataFrame) -> int:
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty:
        return 0
    return int((~np.isfinite(numeric.to_numpy(dtype=float))).sum())


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError(f"Refusing to write an empty router table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def validate_config(config: Mapping[str, Any]) -> tuple[float, ...]:
    if (
        config.get("schema_version") != STUDY_ID
        or config.get("protocol_id") != legacy.PROTOCOL_ID
        or config.get("evaluation_partition") != "validation_only"
        or config.get("official_test_permitted") is not False
        or config.get("include_official_test") is not False
        or int(config.get("test_contract_count", -1)) != 0
        or tuple(config.get("folds", ())) != L2_FOLDS
        or tuple(config.get("router_input_methods", ())) != REQUIRED_METHODS
        or config.get("selection_condition") != "D"
    ):
        raise ValueError("Global-router configuration violates the frozen boundary.")
    bands = tuple(float(value) for value in config["rescue_and_confirmation_bands"])
    expected = (-1.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5)
    if bands != expected:
        raise ValueError("Router band grid differs from the pre-registration.")
    if int(config.get("total_policy_count", -1)) != 486:
        raise ValueError("Router policy count differs from the pre-registration.")
    return bands


def _entry_map(
    backup_root: Path,
) -> dict[tuple[str, str, str], tuple[Path, Path, dict[str, Any]]]:
    values: dict[tuple[str, str, str], tuple[Path, Path, dict[str, Any]]] = {}
    for worker_root, path, result in freeze_evidence._result_entries(backup_root):
        identity = (
            str(result.get("method_id", "")),
            str(result.get("fold_id", "")),
            str(result.get("condition", "")),
        )
        if (
            result.get("level") != "L2"
            or identity[0] not in REQUIRED_METHODS
            or identity[1] not in L2_FOLDS
            or identity[2] not in CONDITIONS
        ):
            continue
        if identity in values:
            raise ValueError(f"Duplicate router input result: {identity}")
        if (
            result.get("evaluation_partition") != "validation_only"
            or int(result.get("test_contract_count", -1)) != 0
            or result.get("protocol_id") != legacy.PROTOCOL_ID
        ):
            raise ValueError(f"Unsafe router result boundary: {path}")
        values[identity] = (worker_root, path, result)
    expected = {
        (method, fold, condition)
        for method in REQUIRED_METHODS
        for fold in L2_FOLDS
        for condition in CONDITIONS
    }
    if set(values) != expected:
        raise ValueError(f"Router input result set is incomplete: {sorted(expected-set(values))}")
    return values


def _selection_for_fold(
    selections: Mapping[tuple[str, str], Mapping[str, Any]],
    method: str,
    fold: str,
) -> Mapping[str, Any]:
    scope = f"heldout-a{int(fold[-2:]):02d}"
    value = selections[(method, scope)]
    if int(value.get("test_contract_count", -1)) != 0:
        raise ValueError("A router component selection touched a test contract.")
    return value


def _critical_score_paths(
    worker_root: Path,
    result_path: Path,
    result: Mapping[str, Any],
    selection_path: Path,
) -> list[Path]:
    score_dir = (
        worker_root
        / "scores"
        / str(result["method_id"])
        / "L2"
        / str(result["fold_id"])
        / str(result["condition"])
        / str(result["run_contract_sha256"])[:16]
    )
    paths = [result_path, selection_path]
    paths.extend(sorted(score_dir.glob("shard-*.csv")))
    paths.extend(sorted(score_dir.glob("shard-*.manifest.json")))
    if len(paths) != 18:
        raise ValueError(f"Router input artifact unit is incomplete: {score_dir}")
    return paths


def _load_frame(
    entries: Mapping[tuple[str, str, str], tuple[Path, Path, dict[str, Any]]],
    *,
    method: str,
    fold: str,
    condition: str,
    repair_audit: dict[str, Any],
) -> pd.DataFrame:
    worker_root, _, result = entries[(method, fold, condition)]
    scored = freeze_evidence._load_scores(worker_root, result)
    if set(scored["split"].astype(str)) != {"validation"}:
        raise ValueError("A non-validation score entered the router study.")
    if (
        method == freeze_evidence.REPAIR_METHOD
        and fold in freeze_evidence.REPAIR_FOLDS
        and condition == "D"
    ):
        n_worker, _, n_result = entries[(method, fold, "N")]
        source_n = freeze_evidence._load_scores(n_worker, n_result)
        scored, record = freeze_evidence._repair_seen_scores(
            scored,
            source_n,
            heldout_aspects=set(result["heldout_aspects"]),
        )
        repair_audit[fold] = record
    return scored


def _component(
    entries: Mapping[tuple[str, str, str], tuple[Path, Path, dict[str, Any]]],
    selections: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    method: str,
    fold: str,
    condition: str,
    repair_audit: dict[str, Any],
):
    scored = _load_frame(
        entries,
        method=method,
        fold=fold,
        condition=condition,
        repair_audit=repair_audit,
    )
    selected = _selection_for_fold(selections, method, fold)
    thresholds = selected["selected_thresholds"]
    return build_component_arrays(
        scored,
        aspect_threshold=float(thresholds["aspect"]),
        runner_up_sentiment_threshold=float(thresholds["runner_up_sentiment"]),
    )


def _complementarity(
    base,
    expert,
    *,
    fold: str,
    condition: str,
) -> dict[str, Any]:
    heldout = base.is_heldout
    target = base.target[heldout]
    base_pred = base.prediction[heldout]
    expert_pred = expert.prediction[heldout]
    base_exact = np.all(base_pred == target, axis=1)
    expert_exact = np.all(expert_pred == target, axis=1)
    base_presence = base_pred.any(axis=1)
    expert_presence = expert_pred.any(axis=1)
    return {
        "fold_id": fold,
        "condition": condition,
        "base_method_id": base.method_id,
        "expert_method_id": expert.method_id,
        "heldout_aspect_instances": int(heldout.sum()),
        "presence_disagreement_count": int(np.sum(base_presence != expert_presence)),
        "presence_disagreement_rate": float(np.mean(base_presence != expert_presence)),
        "expert_exact_when_base_wrong": int(np.sum(~base_exact & expert_exact)),
        "base_exact_when_expert_wrong": int(np.sum(base_exact & ~expert_exact)),
        "both_exact": int(np.sum(base_exact & expert_exact)),
        "both_wrong": int(np.sum(~base_exact & ~expert_exact)),
    }


def _select_policy(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or set(frame["condition"]) != {"D"}:
        raise ValueError("Router selection requires non-empty D-only policy rows.")
    aggregate = (
        frame.groupby("policy_id", sort=False)
        .agg(
            base_method_id=("base_method_id", "first"),
            expert_method_id=("expert_method_id", "first"),
            rescue_band=("rescue_band", "first"),
            confirmation_band=("confirmation_band", "first"),
            selection_fold_count=("fold_id", "nunique"),
            mean_D_heldout_pair_micro_f1=("heldout_pair_micro_f1", "mean"),
            mean_D_overall_pair_micro_f1=("overall_pair_micro_f1", "mean"),
            mean_D_route_rate=("route_rate", "mean"),
        )
        .reset_index()
    )
    selected = aggregate.sort_values(
        [
            "mean_D_heldout_pair_micro_f1",
            "mean_D_overall_pair_micro_f1",
            "mean_D_route_rate",
            "policy_id",
        ],
        ascending=[False, False, True, True],
        kind="stable",
    ).iloc[0]
    return selected.to_dict()


def _crossfit(candidate_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for evaluation_fold in L2_FOLDS:
        training = candidate_frame[
            candidate_frame["condition"].eq("D")
            & ~candidate_frame["fold_id"].eq(evaluation_fold)
        ]
        selected = _select_policy(training)
        policy = str(selected["policy_id"])
        for condition in CONDITIONS:
            evaluation = candidate_frame[
                candidate_frame["fold_id"].eq(evaluation_fold)
                & candidate_frame["condition"].eq(condition)
                & candidate_frame["policy_id"].eq(policy)
            ]
            if len(evaluation) != 1:
                raise ValueError("Cross-fitted router evaluation identity is not unique.")
            row = evaluation.iloc[0].to_dict()
            row.update(
                {
                    "evaluation_fold": evaluation_fold,
                    "selection_condition": "D",
                    "selection_fold_count": int(selected["selection_fold_count"]),
                    "selection_mean_D_heldout_pair_micro_f1": selected[
                        "mean_D_heldout_pair_micro_f1"
                    ],
                    "selection_mean_D_overall_pair_micro_f1": selected[
                        "mean_D_overall_pair_micro_f1"
                    ],
                    "selection_mean_D_route_rate": selected["mean_D_route_rate"],
                }
            )
            records.append(row)
    frame = pd.DataFrame.from_records(records)
    if len(frame) != 24:
        raise AssertionError("Cross-fitted router must produce 12 D and 12 N rows.")
    summary: dict[str, Any] = {}
    for condition in CONDITIONS:
        selected = frame[frame["condition"].eq(condition)]
        heldout_tp = int(selected["heldout_pair_tp"].sum())
        heldout_fp = int(selected["heldout_pair_fp"].sum())
        heldout_fn = int(selected["heldout_pair_fn"].sum())
        summary[condition] = {
            "fold_count": int(selected["fold_id"].nunique()),
            "aspect_balanced_heldout_pair_micro_f1": float(
                selected["heldout_pair_micro_f1"].mean()
            ),
            "aspect_balanced_overall_pair_micro_f1": float(
                selected["overall_pair_micro_f1"].mean()
            ),
            "pooled_heldout_pair_micro_f1": confusion_additive_f1(
                selected, "heldout"
            ),
            "pooled_overall_pair_micro_f1": confusion_additive_f1(
                selected, "overall"
            ),
            "pooled_heldout_pair_precision": float(
                heldout_tp / (heldout_tp + heldout_fp)
                if heldout_tp + heldout_fp
                else 0.0
            ),
            "pooled_heldout_pair_recall": float(
                heldout_tp / (heldout_tp + heldout_fn)
                if heldout_tp + heldout_fn
                else 0.0
            ),
            "mean_route_rate": float(selected["route_rate"].mean()),
        }
    summary["selected_policy_frequency"] = dict(
        sorted(Counter(frame[frame["condition"].eq("D")]["policy_id"]).items())
    )
    return frame, summary


def _final_summary(
    candidate_frame: pd.DataFrame,
    selected: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    policy = str(selected["policy_id"])
    frame = candidate_frame[candidate_frame["policy_id"].eq(policy)].copy()
    if len(frame) != 24:
        raise ValueError("Final global router does not cover all folds and conditions.")
    summary: dict[str, Any] = {"selected_policy": dict(selected), "conditions": {}}
    for condition in CONDITIONS:
        rows = frame[frame["condition"].eq(condition)]
        heldout_tp = int(rows["heldout_pair_tp"].sum())
        heldout_fp = int(rows["heldout_pair_fp"].sum())
        heldout_fn = int(rows["heldout_pair_fn"].sum())
        summary["conditions"][condition] = {
            "fold_count": int(rows["fold_id"].nunique()),
            "aspect_balanced_heldout_pair_micro_f1": float(
                rows["heldout_pair_micro_f1"].mean()
            ),
            "aspect_balanced_overall_pair_micro_f1": float(
                rows["overall_pair_micro_f1"].mean()
            ),
            "pooled_heldout_pair_micro_f1": confusion_additive_f1(rows, "heldout"),
            "pooled_overall_pair_micro_f1": confusion_additive_f1(rows, "overall"),
            "pooled_heldout_pair_precision": float(
                heldout_tp / (heldout_tp + heldout_fp)
                if heldout_tp + heldout_fp
                else 0.0
            ),
            "pooled_heldout_pair_recall": float(
                heldout_tp / (heldout_tp + heldout_fn)
                if heldout_tp + heldout_fn
                else 0.0
            ),
            "aspect_balanced_heldout_presence_f1": float(
                rows["heldout_presence_f1"].mean()
            ),
            "mean_route_rate": float(rows["route_rate"].mean()),
        }
    return frame, summary


def _comparison_table(
    source: Path,
    crossfit_summary: Mapping[str, Any],
    final_summary: Mapping[str, Any],
) -> pd.DataFrame:
    base = pd.read_csv(source)
    required = {
        "method_id",
        "method",
        "condition",
        "heldout_pair_micro_f1_mean",
        "overall_pair_micro_f1_mean",
        "heldout_presence_f1_mean",
    }
    if required - set(base.columns):
        raise ValueError("Scientific-freeze comparison table has changed schema.")
    base = base.copy()
    base["evidence_role"] = "single_system_formal_validation"
    base["pooled_heldout_pair_micro_f1"] = np.nan
    base["pooled_overall_pair_micro_f1"] = np.nan
    base["route_rate"] = 0.0
    additions = []
    for condition in CONDITIONS:
        for role, label, summary in (
            (
                "cross_fitted_router_estimate",
                "Global router (cross-fitted selection)",
                crossfit_summary[condition],
            ),
            (
                "all_validation_fitted_router_diagnostic",
                "Global router (final frozen validation fit)",
                final_summary["conditions"][condition],
            ),
        ):
            additions.append(
                {
                    "method_id": ROUTER_METHOD_ID,
                    "method": label,
                    "condition": condition,
                    "fold_count": 12,
                    "heldout_pair_micro_f1_mean": summary[
                        "aspect_balanced_heldout_pair_micro_f1"
                    ],
                    "overall_pair_micro_f1_mean": summary[
                        "aspect_balanced_overall_pair_micro_f1"
                    ],
                    "seen_pair_micro_f1_mean": np.nan,
                    "heldout_presence_f1_mean": (
                        summary.get("aspect_balanced_heldout_presence_f1", np.nan)
                    ),
                    "heldout_presence_average_precision_mean": summary.get(
                        "aspect_balanced_heldout_presence_average_precision",
                        np.nan,
                    ),
                    "oracle_gated_sentiment_set_micro_f1_mean": summary.get(
                        "aspect_balanced_oracle_gated_sentiment_set_micro_f1",
                        np.nan,
                    ),
                    "evidence_role": role,
                    "pooled_heldout_pair_micro_f1": summary[
                        "pooled_heldout_pair_micro_f1"
                    ],
                    "pooled_overall_pair_micro_f1": summary[
                        "pooled_overall_pair_micro_f1"
                    ],
                    "route_rate": summary["mean_route_rate"],
                }
            )
    return pd.concat([base, pd.DataFrame(additions)], ignore_index=True).sort_values(
        ["condition", "heldout_pair_micro_f1_mean", "evidence_role"],
        ascending=[True, False, True],
        kind="stable",
    )


def _evidence_cube_for_methods(
    evidence: pd.DataFrame,
    methods: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build one fully paired held-out evidence cube for router inference."""

    row_uids = tuple(sorted(evidence["row_uid"].astype(str).unique()))
    expected = {
        (method, fold, condition)
        for method in methods
        for fold in L2_FOLDS
        for condition in CONDITIONS
    }
    observed = set(
        evidence[["method_id", "fold_id", "condition"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if observed != expected or len(row_uids) != 1057:
        raise ValueError("Router bootstrap evidence is not exactly matched.")
    cube = np.zeros(
        (len(methods), len(CONDITIONS), len(L2_FOLDS), len(row_uids), 3),
        dtype=np.int16,
    )
    for method_index, method in enumerate(methods):
        for condition_index, condition in enumerate(CONDITIONS):
            for fold_index, fold in enumerate(L2_FOLDS):
                group = evidence[
                    evidence["method_id"].eq(method)
                    & evidence["condition"].eq(condition)
                    & evidence["fold_id"].eq(fold)
                ].copy()
                group["row_uid"] = group["row_uid"].astype(str)
                group = group.set_index("row_uid").loc[list(row_uids)]
                cube[method_index, condition_index, fold_index] = group[
                    list(inference.COUNT_COLUMNS)
                ].to_numpy(dtype=np.int16)
    return cube, row_uids


def _router_bootstrap_tables(
    router_evidence: pd.DataFrame,
    *,
    formal_evidence_root: Path,
    local_evidence_root: Path,
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Synchronously bootstrap cross-fitted/final routers and seven systems."""

    existing = inference._load_l2_evidence(
        formal_evidence_root, local_evidence_root
    )
    summaries: list[dict[str, Any]] = []
    pairwise: list[dict[str, Any]] = []
    audit: dict[str, Any] = {}
    for evidence_role, router_method in (
        ("cross_fitted_router_estimate", f"{ROUTER_METHOD_ID}__crossfit"),
        ("all_validation_fitted_router_diagnostic", f"{ROUTER_METHOD_ID}__final"),
    ):
        selected_router = router_evidence[
            router_evidence["evidence_role"].eq(evidence_role)
            & router_evidence["partition"].eq("heldout")
        ].copy()
        selected_router["method_id"] = router_method
        combined = pd.concat([existing, selected_router], ignore_index=True)
        methods = (*inference.METHODS, router_method)
        cube, row_uids = _evidence_cube_for_methods(combined, methods)
        balanced_boot, pooled_boot = inference._synchronised_bootstrap(
            cube, draws=draws, seed=seed
        )
        router_index = len(methods) - 1
        for condition_index, condition in enumerate(CONDITIONS):
            balanced, pooled, _ = inference._point_metrics(
                cube[router_index, condition_index]
            )
            balanced_low, balanced_high = inference._interval(
                balanced_boot[:, router_index, condition_index]
            )
            pooled_low, pooled_high = inference._interval(
                pooled_boot[:, router_index, condition_index]
            )
            summaries.append(
                {
                    "method_id": ROUTER_METHOD_ID,
                    "evidence_role": evidence_role,
                    "condition": condition,
                    "fold_count": 12,
                    "validation_review_clusters": len(row_uids),
                    "aspect_balanced_heldout_pair_micro_f1": balanced,
                    "aspect_balanced_review_bootstrap_95_ci_low": balanced_low,
                    "aspect_balanced_review_bootstrap_95_ci_high": balanced_high,
                    "pooled_heldout_pair_micro_f1_sensitivity": pooled,
                    "pooled_review_bootstrap_95_ci_low": pooled_low,
                    "pooled_review_bootstrap_95_ci_high": pooled_high,
                    "bootstrap_draws": draws,
                    "bootstrap_seed": seed,
                }
            )
        d_index = CONDITIONS.index("D")
        for reference_index, reference in enumerate(inference.METHODS):
            router_point, _, _ = inference._point_metrics(
                cube[router_index, d_index]
            )
            reference_point, _, _ = inference._point_metrics(
                cube[reference_index, d_index]
            )
            differences = (
                balanced_boot[:, router_index, d_index]
                - balanced_boot[:, reference_index, d_index]
            )
            low, high = inference._interval(differences)
            pairwise.append(
                {
                    "condition": "D",
                    "router_evidence_role": evidence_role,
                    "router_method_id": ROUTER_METHOD_ID,
                    "reference_method_id": reference,
                    "router_minus_reference_aspect_balanced_heldout_pair_micro_f1": (
                        router_point - reference_point
                    ),
                    "shared_review_bootstrap_95_ci_low": low,
                    "shared_review_bootstrap_95_ci_high": high,
                    "bootstrap_draws": draws,
                    "bootstrap_seed": seed,
                }
            )
        audit[evidence_role] = {
            "validation_review_cluster_count": len(row_uids),
            "validation_row_uid_sha256": hashlib.sha256(
                ("\n".join(row_uids) + "\n").encode("utf-8")
            ).hexdigest(),
        }
    return (
        pd.DataFrame.from_records(summaries),
        pd.DataFrame.from_records(pairwise),
        audit,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config = legacy.read_json(config_path)
    bands = validate_config(config)
    backup_root = args.backup_root.resolve()
    output_dir = args.output_dir.resolve()
    public_dir = args.public_dir.resolve()

    states = legacy.audit_states(backup_root)
    receipt_summary, published = legacy.audit_receipts(backup_root)
    selections_audit = legacy.audit_selections(backup_root, published)
    selections = selections_audit["values"]
    entries = _entry_map(backup_root)

    critical_hashes: dict[str, str] = {str(config_path): file_sha256(config_path)}
    for method in REQUIRED_METHODS:
        for fold in L2_FOLDS:
            selection = _selection_for_fold(selections, method, fold)
            scope = str(selection["training_scope_id"])
            selection_candidates = list(
                backup_root.glob(f"worker-*/selections/{method}/{scope}.json")
            )
            if len(selection_candidates) != 1:
                raise ValueError(f"Selection path is not unique for {method}/{scope}.")
            selection_path = selection_candidates[0]
            for condition in CONDITIONS:
                worker_root, result_path, result = entries[(method, fold, condition)]
                for path in _critical_score_paths(
                    worker_root, result_path, result, selection_path
                ):
                    if path.resolve() not in published:
                        raise ValueError(f"Router input lacks a verified receipt: {path}")
                    observed = file_sha256(path)
                    if published[path.resolve()] != observed:
                        raise ValueError(f"Router input receipt hash mismatch: {path}")
                    critical_hashes[str(path)] = observed

    policies = []
    for base, expert in itertools.permutations(REQUIRED_METHODS, 2):
        for rescue_band, confirmation_band in itertools.product(bands, repeat=2):
            record = {
                "base_method_id": base,
                "expert_method_id": expert,
                "rescue_band": rescue_band,
                "confirmation_band": confirmation_band,
            }
            record["policy_id"] = policy_id(record)
            policies.append(record)
    if len(policies) != 486 or len({row["policy_id"] for row in policies}) != 486:
        raise AssertionError("The pre-registered router policy grid is incomplete.")

    candidate_records: list[dict[str, Any]] = []
    complementarity_records: list[dict[str, Any]] = []
    component_records: list[dict[str, Any]] = []
    repair_audit: dict[str, Any] = {}
    for fold in L2_FOLDS:
        for condition in CONDITIONS:
            components = {
                method: _component(
                    entries,
                    selections,
                    method=method,
                    fold=fold,
                    condition=condition,
                    repair_audit=repair_audit,
                )
                for method in REQUIRED_METHODS
            }
            for method, component in components.items():
                metrics = policy_metrics(
                    component,
                    component,
                    rescue_band=-1.0,
                    confirmation_band=-1.0,
                )
                metrics.update(
                    {
                        "fold_id": fold,
                        "condition": condition,
                        "method_id": method,
                    }
                )
                component_records.append(metrics)
            for base_method, expert_method in itertools.permutations(
                REQUIRED_METHODS, 2
            ):
                base = components[base_method]
                expert = components[expert_method]
                complementarity_records.append(
                    _complementarity(
                        base,
                        expert,
                        fold=fold,
                        condition=condition,
                    )
                )
                for rescue_band, confirmation_band in itertools.product(
                    bands, repeat=2
                ):
                    metrics = policy_metrics(
                        base,
                        expert,
                        rescue_band=rescue_band,
                        confirmation_band=confirmation_band,
                    )
                    metrics.update(
                        {
                            "fold_id": fold,
                            "condition": condition,
                            "policy_id": policy_id(metrics),
                        }
                    )
                    candidate_records.append(metrics)

    candidates = pd.DataFrame.from_records(candidate_records)
    expected_rows = 486 * len(L2_FOLDS) * len(CONDITIONS)
    if len(candidates) != expected_rows or candidates.isna().any().any():
        raise ValueError("Router policy table is incomplete or non-finite.")
    crossfit_frame, crossfit_summary = _crossfit(candidates)
    final_selected = _select_policy(candidates[candidates["condition"].eq("D")])
    final_frame, final_summary = _final_summary(candidates, final_selected)

    folds_by_id = {fold.fold_id: fold for fold in post_supervisor_l2_folds()}
    base_method = str(final_selected["base_method_id"])
    expert_method = str(final_selected["expert_method_id"])
    rescue_band = float(final_selected["rescue_band"])
    confirmation_band = float(final_selected["confirmation_band"])
    crossfit_by_fold = {
        str(row["fold_id"]): row
        for row in crossfit_frame[crossfit_frame["condition"].eq("D")].to_dict(
            orient="records"
        )
    }
    evaluation_policies: dict[str, dict[str, Mapping[str, Any]]] = {
        "cross_fitted_router_estimate": crossfit_by_fold,
        "all_validation_fitted_router_diagnostic": {
            fold: final_selected for fold in L2_FOLDS
        },
    }
    full_records: list[dict[str, Any]] = []
    row_evidence_frames: list[pd.DataFrame] = []
    route_records: list[pd.DataFrame] = []
    for evidence_role, fold_policies in evaluation_policies.items():
        for fold in L2_FOLDS:
            policy = fold_policies[fold]
            policy_base = str(policy["base_method_id"])
            policy_expert = str(policy["expert_method_id"])
            policy_rescue = float(policy["rescue_band"])
            policy_confirmation = float(policy["confirmation_band"])
            taxonomy_fold = folds_by_id[fold]
            for condition in CONDITIONS:
                base = _component(
                    entries,
                    selections,
                    method=policy_base,
                    fold=fold,
                    condition=condition,
                    repair_audit=repair_audit,
                )
                expert = _component(
                    entries,
                    selections,
                    method=policy_expert,
                    fold=fold,
                    condition=condition,
                    repair_audit=repair_audit,
                )
                routed, prediction, route_frame = routed_score_frame(
                    base,
                    expert,
                    rescue_band=policy_rescue,
                    confirmation_band=policy_confirmation,
                    method_id=ROUTER_METHOD_ID,
                )
                result = evaluate_l2_condition(
                    routed,
                    taxonomy_fold,
                    aspect_threshold=0.5,
                    second_sentiment_threshold=0.5,
                )
                full_records.append(
                    {
                        "method_id": ROUTER_METHOD_ID,
                        "evidence_role": evidence_role,
                        "fold_id": fold,
                        "condition": condition,
                        "base_method_id": policy_base,
                        "expert_method_id": policy_expert,
                        "rescue_band": policy_rescue,
                        "confirmation_band": policy_confirmation,
                        "score_sha256": result["score_sha256"],
                        "heldout_pair_micro_f1": result["L2_E"]["partitions"]["heldout"]["pair_micro_f1"],
                        "overall_pair_micro_f1": result["L2_E"]["partitions"]["overall"]["pair_micro_f1"],
                        "seen_pair_micro_f1": result["L2_E"]["partitions"]["seen"]["pair_micro_f1"],
                        "heldout_presence_f1": result["L2_S"]["aspect_presence"]["f1"],
                        "heldout_presence_ap": result["L2_S"]["aspect_presence"]["average_precision"],
                        "oracle_gated_sentiment_set_micro_f1": result["L2_S"]["oracle_aspect_gated_sentiment"]["capped_two_pair_metrics"]["pair_micro_f1"],
                        "maximum_sentiments_per_selected_aspect": result["L2_E"]["maximum_sentiments_per_selected_aspect"],
                        "route_count": int(route_frame["routed"].sum()),
                        "route_rate": float(route_frame["routed"].mean()),
                        "test_contract_count": 0,
                    }
                )
                for partition, aspects in (
                    ("heldout", taxonomy_fold.heldout_aspects),
                    ("overall", taxonomy_fold.evaluation_aspects),
                ):
                    evidence = row_pair_confusions(
                        routed,
                        prediction,
                        aspects=aspects,
                        method_id=ROUTER_METHOD_ID,
                        fold_id=fold,
                        condition=condition,
                        level="L2",
                    )
                    evidence.insert(4, "evidence_role", evidence_role)
                    evidence.insert(5, "partition", partition)
                    row_evidence_frames.append(evidence)
                route_frame.insert(0, "evidence_role", evidence_role)
                route_frame.insert(1, "fold_id", fold)
                route_frame.insert(2, "condition", condition)
                route_records.append(route_frame)

    full_frame = pd.DataFrame.from_records(full_records)
    row_evidence = pd.concat(row_evidence_frames, ignore_index=True)
    route_instances = pd.concat(route_records, ignore_index=True)
    if (
        len(full_frame) != 48
        or int(full_frame["maximum_sentiments_per_selected_aspect"].max()) > 2
        or int(full_frame["test_contract_count"].sum()) != 0
    ):
        raise ValueError("Router full-metric evaluation failed its contract.")
    for condition in CONDITIONS:
        crossfit_rows = full_frame[
            full_frame["evidence_role"].eq("cross_fitted_router_estimate")
            & full_frame["condition"].eq(condition)
        ]
        crossfit_summary[condition].update(
            {
                "aspect_balanced_heldout_presence_f1": float(
                    crossfit_rows["heldout_presence_f1"].mean()
                ),
                "aspect_balanced_heldout_presence_average_precision": float(
                    crossfit_rows["heldout_presence_ap"].mean()
                ),
                "aspect_balanced_oracle_gated_sentiment_set_micro_f1": float(
                    crossfit_rows["oracle_gated_sentiment_set_micro_f1"].mean()
                ),
            }
        )
        final_rows = full_frame[
            full_frame["evidence_role"].eq(
                "all_validation_fitted_router_diagnostic"
            )
            & full_frame["condition"].eq(condition)
        ]
        final_summary["conditions"][condition].update(
            {
                "aspect_balanced_heldout_presence_average_precision": float(
                    final_rows["heldout_presence_ap"].mean()
                ),
                "aspect_balanced_oracle_gated_sentiment_set_micro_f1": float(
                    final_rows["oracle_gated_sentiment_set_micro_f1"].mean()
                ),
            }
        )
    bootstrap_summary, bootstrap_pairwise, bootstrap_audit = (
        _router_bootstrap_tables(
            row_evidence,
            formal_evidence_root=args.formal_evidence_root.resolve(),
            local_evidence_root=args.local_evidence_root.resolve(),
            draws=args.bootstrap_draws,
            seed=args.bootstrap_seed,
        )
    )
    comparison = _comparison_table(
        args.comparison_source.resolve(), crossfit_summary, final_summary
    )

    current_best_d = float(
        comparison[
            comparison["condition"].eq("D")
            & comparison["evidence_role"].eq("single_system_formal_validation")
        ]["heldout_pair_micro_f1_mean"].max()
    )
    crossfit_d = float(
        crossfit_summary["D"]["aspect_balanced_heldout_pair_micro_f1"]
    )
    selected_component_ids = {base_method, expert_method}
    component_table = pd.DataFrame.from_records(component_records)
    component_d = (
        component_table[
            component_table["condition"].eq("D")
            & component_table["method_id"].isin(selected_component_ids)
        ]
        .groupby("method_id")["heldout_pair_micro_f1"]
        .mean()
        .to_dict()
    )
    success = bool(
        crossfit_d > current_best_d
        and all(crossfit_d > float(value) for value in component_d.values())
    )

    final_full_frame = full_frame[
        full_frame["evidence_role"].eq(
            "all_validation_fitted_router_diagnostic"
        )
    ].copy()
    crossfit_full_frame = full_frame[
        full_frame["evidence_role"].eq("cross_fitted_router_estimate")
    ].copy()
    numeric_audit_frames = (
        candidates,
        component_table,
        crossfit_frame,
        final_frame,
        full_frame,
        row_evidence,
        route_instances,
        bootstrap_summary,
        bootstrap_pairwise,
    )
    non_finite_count = sum(
        non_finite_numeric_count(frame) for frame in numeric_audit_frames
    )
    if non_finite_count:
        raise ValueError("Router outputs contain non-finite numeric values.")
    duplicate_policy_identity_count = int(
        candidates.duplicated(["policy_id", "fold_id", "condition"]).sum()
    )
    prediction_collapse_count = int(
        row_evidence[row_evidence["partition"].eq("heldout")]
        .groupby(["evidence_role", "fold_id", "condition"], sort=False)[
            "pair_predicted_count"
        ]
        .sum()
        .eq(0)
        .sum()
    )
    if duplicate_policy_identity_count or prediction_collapse_count:
        raise ValueError("Router output identity or prediction-collapse audit failed.")

    output_files = {
        "candidate_policy_fold_metrics.csv": candidates,
        "component_fold_metrics.csv": component_table,
        "component_complementarity.csv": pd.DataFrame.from_records(
            complementarity_records
        ),
        "crossfit_selected_fold_metrics.csv": crossfit_frame,
        "final_router_fold_metrics.csv": final_frame,
        "crossfit_router_full_metrics.csv": crossfit_full_frame,
        "final_router_full_metrics.csv": final_full_frame,
        "router_row_evidence.csv": row_evidence,
        "router_route_instances.csv": route_instances,
        "router_review_bootstrap.csv": bootstrap_summary,
        "router_pairwise_review_bootstrap.csv": bootstrap_pairwise,
    }
    output_hashes: dict[str, str] = {}
    for name, frame in output_files.items():
        path = output_dir / name
        atomic_csv(path, frame)
        output_hashes[str(path)] = file_sha256(path)

    public_comparison = public_dir / "global_router_method_comparison.csv"
    public_folds = public_dir / "global_router_fold_results.csv"
    public_crossfit = public_dir / "global_router_crossfit_selection.csv"
    public_bootstrap = public_dir / "global_router_review_bootstrap.csv"
    public_pairwise = public_dir / "global_router_pairwise_review_bootstrap.csv"
    for path, frame in (
        (public_comparison, comparison),
        (public_folds, full_frame),
        (public_crossfit, crossfit_frame),
        (public_bootstrap, bootstrap_summary),
        (public_pairwise, bootstrap_pairwise),
    ):
        atomic_csv(path, frame)
        output_hashes[str(path)] = file_sha256(path)

    selected_router = {
        "schema_version": "taxonomy_two_stage_global_router_selected_v1",
        "study_id": STUDY_ID,
        "protocol_id": legacy.PROTOCOL_ID,
        "evaluation_partition": "validation_only",
        "official_test_opened": False,
        "include_official_test": False,
        "test_contract_count": 0,
        "selection_condition": "D",
        "base_method_id": base_method,
        "expert_method_id": expert_method,
        "rescue_band": rescue_band,
        "confirmation_band": confirmation_band,
        "normalised_aspect_threshold": 0.5,
        "normalised_runner_up_sentiment_threshold": 0.5,
        "maximum_sentiments_per_selected_aspect": 2,
        "n_condition_retuned": False,
        "crossfit_summary": crossfit_summary,
        "all_validation_fitted_summary": final_summary,
        "current_best_single_system_D_heldout_pair_micro_f1": current_best_d,
        "selected_component_D_heldout_pair_micro_f1": component_d,
        "pre_registered_success_rule_passed": success,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
    }
    selected_router["selected_payload_sha256"] = canonical_sha256(selected_router)
    selected_path = output_dir / "selected_router.json"
    atomic_json(selected_path, selected_router)
    output_hashes[str(selected_path)] = file_sha256(selected_path)

    audit = {
        "schema_version": "taxonomy_two_stage_global_router_audit_v1",
        "study_id": STUDY_ID,
        "protocol_id": legacy.PROTOCOL_ID,
        "official_test_opened": False,
        "include_official_test": False,
        "test_contract_count": 0,
        "config_sha256": file_sha256(config_path),
        "campaign_states": states,
        "receipt_summary": receipt_summary,
        "selection_count": selections_audit["selection_count"],
        "critical_verified_input_file_count": len(critical_hashes),
        "critical_verified_input_sha256": dict(sorted(critical_hashes.items())),
        "policy_count": len(policies),
        "policy_fold_record_count": len(candidates),
        "crossfit_fold_record_count": len(crossfit_frame),
        "final_router_fold_record_count": len(final_frame),
        "router_full_metric_record_count": len(full_frame),
        "non_finite_value_count": non_finite_count,
        "duplicate_policy_identity_count": duplicate_policy_identity_count,
        "prediction_collapse_count": prediction_collapse_count,
        "bootstrap": {
            "draws": args.bootstrap_draws,
            "seed": args.bootstrap_seed,
            "synchronisation": "same review resample for every method, fold, and condition",
            "evidence": bootstrap_audit,
        },
        "qlora_seen_invariant_repairs": repair_audit,
        "output_sha256": dict(sorted(output_hashes.items())),
        "selected_router_sha256": selected_router["selected_payload_sha256"],
        "pre_registered_success_rule_passed": success,
    }
    audit["audit_payload_sha256"] = canonical_sha256(audit)
    audit_path = output_dir / "audit.json"
    atomic_json(audit_path, audit)

    summary = {
        "selected_router": selected_router,
        "audit_path": str(audit_path),
        "comparison_path": str(public_comparison),
        "output_dir": str(output_dir),
    }
    atomic_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs/experiments/taxonomy_two_stage_global_router_validation_v1.json",
    )
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=Path(
            "C:/Msc_DSML/Msc_Project/cloud_backups/taxonomy_two_stage_formal_v2_r2"
        ),
    )
    parser.add_argument(
        "--comparison-source",
        type=Path,
        default=PROJECT_ROOT
        / "docs/thesis_figure_data/taxonomy_scientific_freeze_v1/l2_complete_model_condition_metrics.csv",
    )
    parser.add_argument(
        "--formal-evidence-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experimental/taxonomy_scientific_freeze_v1/formal_row_evidence",
    )
    parser.add_argument(
        "--local-evidence-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experimental/taxonomy_scientific_freeze_v1/local_row_evidence",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experimental/taxonomy_two_stage_global_router_validation_v1",
    )
    parser.add_argument(
        "--public-dir",
        type=Path,
        default=PROJECT_ROOT
        / "docs/thesis_figure_data/taxonomy_two_stage_global_router_validation_v1",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result["selected_router"], indent=2, sort_keys=True), flush=True)
