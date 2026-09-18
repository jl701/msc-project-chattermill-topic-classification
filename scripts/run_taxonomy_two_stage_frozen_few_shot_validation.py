"""Run one immutable frozen-Qwen few-shot validation training scope.

Only train and validation are loadable.  Demonstrations come from the exact
outer train fold, threshold selection uses seen-aspect validation evidence,
and held-out validation labels are evaluation-only.  Exact prompt contracts
are cached in independently sealed chunks so interrupted inference resumes
without borrowing zero-shot scores or another scope's demonstrations.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msc_project.data.fabsa import default_data_dir  # noqa: E402
from msc_project.data.splits import load_official_fabsa_splits  # noqa: E402
from msc_project.experiments.taxonomy_execution import (  # noqa: E402
    RunContract,
    canonical_sha256,
    dataframe_sha256,
    select_pair_shard,
)
from msc_project.experiments.taxonomy_few_shot_cache import (  # noqa: E402
    FormalFewShotCache,
)
from msc_project.experiments.taxonomy_methods import (  # noqa: E402
    method_registry_sha256,
    resolve_method_spec,
)
from msc_project.experiments.taxonomy_protocol import (  # noqa: E402
    build_taxonomy_fold_splits,
    pair_identity_hash,
)
from msc_project.experiments.taxonomy_post_supervisor import (  # noqa: E402
    evaluate_l2_condition,
)
from msc_project.experiments.taxonomy_resources import (  # noqa: E402
    load_description_bundle,
)
from msc_project.experiments.taxonomy_two_stage import (  # noqa: E402
    capped_two_sentiment_prediction_mask,
    conditional_sentiment_metrics,
    evaluate_prediction_mask,
    select_second_sentiment_threshold,
    select_two_stage_threshold,
)
from msc_project.experiments.taxonomy_two_stage_artifacts import (  # noqa: E402
    build_two_stage_score_artifact,
    merge_two_stage_score_shards,
    two_stage_shard_resume_state,
    write_two_stage_score_shard,
)
from msc_project.experiments.taxonomy_two_stage_formal import (  # noqa: E402
    PROTOCOL_ID,
    representative_fold,
    scope_folds,
    variant_map,
)
from msc_project.experiments.taxonomy_two_stage_runtime import (  # noqa: E402
    build_aspect_grid,
    build_sentiment_grid,
    join_two_stage_scores,
    select_qwen_two_stage_demonstrations,
)
from msc_project.experiments.verified_artifact_sync import (  # noqa: E402
    publish_artifact_unit,
)
from msc_project.llm.qwen_pair_classifier import (  # noqa: E402
    load_frozen_qwen_pair,
)
from msc_project.llm.qwen_two_stage_classifier import (  # noqa: E402
    demonstrations_sha256,
    qwen_two_stage_few_shot_contract_sha256,
    score_two_stage_prompts,
    validate_sentiment_verbalizer_token_ids,
)


def formal_conditions(fold):
    """Default legacy condition set; post-supervisor wrappers may narrow it."""

    return fold.conditions


METHOD_ID = "frozen_qwen_few_shot"
BASE_METHOD_ID = "frozen_qwen_candidate_pair"
MAX_LENGTH = 1024
BATCH_SIZE = 6
CACHE_CHUNK_PROMPTS = 1536
DEFAULT_SAFETY_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "taxonomy_two_stage_cloud_execution_safety_v1.json"
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Stale temporary JSON file: {temporary}")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _seal_payload(value: Mapping[str, object], field: str) -> dict[str, object]:
    payload = dict(value)
    payload.pop(field, None)
    payload[field] = canonical_sha256(payload)
    return payload


def _validate_sealed_payload(value: Mapping[str, object], field: str) -> None:
    expected = str(value.get(field, ""))
    payload = dict(value)
    payload.pop(field, None)
    if expected != canonical_sha256(payload):
        raise RuntimeError(f"JSON artifact content hash mismatch: {field}")


ALLOWED_SAFETY_STATUSES = {
    "preregistered_before_formal_execution",
    "preregistered_after_rich_gate_before_formal_execution",
}


def _validate_safety_config(config: Mapping[str, object]) -> None:
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected formal two-stage protocol ID.")
    if config.get("status") not in ALLOWED_SAFETY_STATUSES:
        raise ValueError("Formal execution safety contract is not preregistered.")
    data = config.get("sealed_data_contract")
    if not isinstance(data, Mapping):
        raise ValueError("Formal execution safety contract lacks its data boundary.")
    if list(data.get("allowed_splits", [])) != ["train", "validation"]:
        raise ValueError("Frozen-Qwen executor may load only train and validation.")
    if data.get("include_official_test") is not False:
        raise ValueError("Official test must remain disabled.")
    methods = config.get("methods")
    if not isinstance(methods, Mapping) or METHOD_ID not in methods:
        raise ValueError("Frozen-Qwen formal method is not preregistered.")


def _protocol_hash(config: Mapping[str, object]) -> str:
    scientific = dict(config)
    scientific.pop("status", None)
    scientific.pop("registered_at", None)
    return canonical_sha256(scientific)


def _frozen_method_spec_sha256() -> str:
    base = resolve_method_spec(BASE_METHOD_ID)
    return canonical_sha256(
        {
            "method_id": METHOD_ID,
            "base_method_id": BASE_METHOD_ID,
            "base_method_spec_sha256": base.spec_sha256,
            "prompt_contract_sha256": qwen_two_stage_few_shot_contract_sha256(
                max_length=MAX_LENGTH
            ),
            "demonstration_selection": (
                "stable_sha256_order_from_exact_outer_train_fold"
            ),
            "requires_pair_training": False,
        }
    )


def _load_scope_data(
    data_dir: Path, scope_id: str
) -> tuple[pd.DataFrame, pd.DataFrame, Any, tuple[Any, ...]]:
    folds = scope_folds().get(scope_id)
    if folds is None:
        raise ValueError(f"Unknown formal training scope: {scope_id!r}")
    fold = representative_fold(folds)
    frame = load_official_fabsa_splits(data_dir, ("train", "validation"))
    if set(frame["original_split"].astype(str)) != {"train", "validation"}:
        raise AssertionError("Frozen-Qwen executor loaded an unregistered split.")
    splits = build_taxonomy_fold_splits(
        frame,
        fold,
        evaluation_splits=("validation",),
    )
    resource = load_description_bundle(require_approved=True)
    return splits["train"], splits["validation"], resource, folds


def _augment_grids(
    aspect_grid: pd.DataFrame,
    sentiment_grid: pd.DataFrame,
    *,
    fold: Any,
    condition: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_uids = sorted(sentiment_grid["row_uid"].astype(str).unique())
    row_indices = {uid: index for index, uid in enumerate(row_uids)}
    for frame in (aspect_grid, sentiment_grid):
        frame["fold_id"] = fold.fold_id
        frame["condition"] = condition
        frame["row_index"] = frame["row_uid"].astype(str).map(row_indices).astype(int)
        frame["is_seen"] = frame["candidate_aspect"].astype(str).isin(
            fold.seen_aspects
        )
        frame["is_heldout"] = frame["candidate_aspect"].astype(str).isin(
            fold.heldout_aspects
        )
    return aspect_grid, sentiment_grid


def _demonstration_records(
    demonstrations: Mapping[str, Sequence[Any]],
) -> dict[str, object]:
    return {
        mode: {
            "count": len(values),
            "answers": [value.answer for value in values],
            "sha256": demonstrations_sha256(values),
            "row_uids": [value.row_uid for value in values],
            "candidate_aspects": [value.candidate_aspect for value in values],
        }
        for mode, values in demonstrations.items()
    }


def _scope_contract(
    *,
    config: Mapping[str, object],
    scope_id: str,
    demonstrations: Mapping[str, Sequence[Any]],
) -> tuple[str, dict[str, object]]:
    spec = resolve_method_spec(BASE_METHOD_ID)
    if not spec.model_id or not spec.model_revision:
        raise ValueError("Frozen-Qwen model ID and revision must be pinned.")
    records = _demonstration_records(demonstrations)
    payload = {
        "protocol_id": PROTOCOL_ID,
        "scientific_protocol_sha256": _protocol_hash(config),
        "method_id": METHOD_ID,
        "base_method_id": BASE_METHOD_ID,
        "method_spec_sha256": _frozen_method_spec_sha256(),
        "base_method_spec_sha256": spec.spec_sha256,
        "method_registry_sha256": method_registry_sha256(),
        "training_scope_id": scope_id,
        "seed": 13,
        "model_id": spec.model_id,
        "model_revision": spec.model_revision,
        "prompt_contract_sha256": qwen_two_stage_few_shot_contract_sha256(
            max_length=MAX_LENGTH
        ),
        "demonstrations": records,
        "batch_size": BATCH_SIZE,
        "maximum_length": MAX_LENGTH,
        "selection_partition": "seen_validation_only",
        "test_contract_count": 0,
    }
    return canonical_sha256(payload), payload


def _calibration_grids(
    validation_rows: pd.DataFrame,
    seen_aspects: Sequence[str],
    resource: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    variants = {str(aspect): "name_and_description" for aspect in seen_aspects}
    return (
        build_aspect_grid(validation_rows, tuple(seen_aspects), variants, resource),
        build_sentiment_grid(
            validation_rows, tuple(seen_aspects), variants, resource
        ),
    )


def _score_grids(
    cache: FormalFewShotCache,
    aspect_grid: pd.DataFrame,
    sentiment_grid: pd.DataFrame,
    *,
    model: Any,
    tokenizer: Any,
    aspect_ids: Any,
    sentiment_ids: Any,
    demonstrations: Mapping[str, Sequence[Any]],
) -> pd.DataFrame:
    reviews = aspect_grid["text"].astype(str).tolist()
    candidates = aspect_grid["candidate_text"].astype(str).tolist()
    aspect_matrix = cache.score(
        reviews,
        candidates,
        mode="aspect",
        scorer=lambda values, cards: score_two_stage_prompts(
            model,
            tokenizer,
            values,
            cards,
            mode="aspect",
            max_length=MAX_LENGTH,
            batch_size=BATCH_SIZE,
            aspect_verbalizer_ids=aspect_ids,
            demonstrations=demonstrations["aspect"],
        ),
    )
    sentiment_matrix = cache.score(
        reviews,
        candidates,
        mode="sentiment",
        scorer=lambda values, cards: score_two_stage_prompts(
            model,
            tokenizer,
            values,
            cards,
            mode="sentiment",
            max_length=MAX_LENGTH,
            batch_size=BATCH_SIZE,
            sentiment_verbalizer_ids=sentiment_ids,
            demonstrations=demonstrations["sentiment"],
        ),
    )
    if (
        aspect_matrix.shape != (len(aspect_grid), 2)
        or sentiment_matrix.shape != (len(aspect_grid), 3)
        or not np.isfinite(aspect_matrix).all()
        or not np.isfinite(sentiment_matrix).all()
    ):
        raise RuntimeError("Non-finite frozen-Qwen few-shot probability.")
    return join_two_stage_scores(
        aspect_grid,
        sentiment_grid,
        aspect_matrix[:, 0],
        sentiment_matrix.reshape(-1),
    )


def _run_contract(
    *,
    config: Mapping[str, object],
    fold: Any,
    condition: str,
    scope_contract_sha256: str,
    demonstration_sha256s: Mapping[str, str],
    thresholds: Mapping[str, float],
    sentiment_grid: pd.DataFrame,
    resource: Mapping[str, object],
) -> RunContract:
    candidate_rows = sentiment_grid.drop_duplicates(
        ["candidate_aspect", "candidate_sentiment"]
    )
    spec = resolve_method_spec(BASE_METHOD_ID)
    return RunContract(
        protocol_id=PROTOCOL_ID,
        scientific_protocol_sha256=_protocol_hash(config),
        method_id=METHOD_ID,
        method_spec_sha256=_frozen_method_spec_sha256(),
        method_registry_sha256=method_registry_sha256(),
        description_resource_sha256=canonical_sha256(resource),
        candidate_representation_sha256=dataframe_sha256(
            candidate_rows,
            [
                "candidate_aspect",
                "candidate_sentiment",
                "candidate_text",
                "representation_variant",
            ],
            sort_columns=["candidate_aspect", "candidate_sentiment"],
        ),
        level=fold.level,
        fold_id=fold.fold_id,
        condition=condition,
        split="validation",
        seed=13,
        training_manifest_sha256=canonical_sha256(demonstration_sha256s),
        evaluation_data_sha256=dataframe_sha256(
            sentiment_grid,
            ["row_uid", "candidate_aspect", "candidate_sentiment", "target"],
            sort_columns=["row_uid", "candidate_aspect", "candidate_sentiment"],
        ),
        evaluation_pair_identity_sha256=pair_identity_hash(sentiment_grid),
        scientific_parameters_sha256=canonical_sha256(
            {
                "scope_contract_sha256": scope_contract_sha256,
                "prompt_contract_sha256": qwen_two_stage_few_shot_contract_sha256(
                    max_length=MAX_LENGTH
                ),
                "demonstration_sha256s": demonstration_sha256s,
                "thresholds": thresholds,
                "decoder": "top_one_plus_thresholded_runner_up",
                "maximum_sentiments_per_aspect": 2,
            }
        ),
        shard_count=8,
        formal=True,
    )


def _release_model(model: Any) -> None:
    try:
        model.to("cpu")
    except (AttributeError, ValueError):
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run(args: argparse.Namespace) -> dict[str, object]:
    config = _read_object(args.config)
    _validate_safety_config(config)
    output_root = args.output_root.resolve()
    train_rows, validation_rows, resource, folds = _load_scope_data(
        args.data_dir, args.scope_id
    )
    fold = representative_fold(folds)
    demonstrations = select_qwen_two_stage_demonstrations(
        train_rows, fold.seen_aspects, resource, seed=13
    )
    heldout = set(fold.heldout_aspects)
    if any(
        example.candidate_aspect in heldout
        for values in demonstrations.values()
        for example in values
    ):
        raise RuntimeError("A held-out aspect entered formal demonstrations.")
    scope_contract_sha, scope_contract = _scope_contract(
        config=config,
        scope_id=args.scope_id,
        demonstrations=demonstrations,
    )
    demonstration_sha256s = {
        mode: demonstrations_sha256(values)
        for mode, values in demonstrations.items()
    }
    cache = FormalFewShotCache(
        output_root / "prompt_cache" / METHOD_ID / args.scope_id,
        protocol_id=PROTOCOL_ID,
        scope_id=args.scope_id,
        scope_contract_sha256=scope_contract_sha,
        demonstration_sha256s=demonstration_sha256s,
        artifact_root=output_root,
        chunk_size=CACHE_CHUNK_PROMPTS,
    )
    spec = resolve_method_spec(BASE_METHOD_ID)
    if not spec.model_id or not spec.model_revision:
        raise ValueError("Frozen-Qwen model ID and revision must be pinned.")
    tokenizer, model, aspect_ids = load_frozen_qwen_pair(
        spec.model_id,
        revision=spec.model_revision,
        load_in_4bit=True,
        local_files_only=args.local_files_only,
    )
    sentiment_ids = validate_sentiment_verbalizer_token_ids(tokenizer)
    try:
        selection_path = output_root / "selections" / METHOD_ID / f"{args.scope_id}.json"
        if selection_path.is_file() and args.resume:
            selection = _read_object(selection_path)
            _validate_sealed_payload(selection, "selection_payload_sha256")
            if (
                selection.get("scope_contract_sha256") != scope_contract_sha
                or selection.get("test_contract_count") != 0
                or selection.get("failure_count") != 0
            ):
                raise RuntimeError("Frozen-Qwen selection resume conflict.")
        else:
            if selection_path.exists():
                raise FileExistsError(selection_path)
            aspect_grid, sentiment_grid = _calibration_grids(
                validation_rows, fold.seen_aspects, resource
            )
            scored = _score_grids(
                cache,
                aspect_grid,
                sentiment_grid,
                model=model,
                tokenizer=tokenizer,
                aspect_ids=aspect_ids,
                sentiment_ids=sentiment_ids,
                demonstrations=demonstrations,
            )
            if (
                scored["aspect_score"].nunique() < 2
                or scored["sentiment_score"].nunique() < 2
            ):
                raise RuntimeError(
                    "Frozen-Qwen few-shot prediction collapse on the complete "
                    "seen-validation calibration grid."
                )
            aspect_selection = select_two_stage_threshold(scored)
            runner_selection = select_second_sentiment_threshold(
                scored, aspect_threshold=aspect_selection.threshold
            )
            mask = capped_two_sentiment_prediction_mask(
                scored,
                aspect_threshold=aspect_selection.threshold,
                second_sentiment_threshold=runner_selection.second_sentiment_threshold,
            )
            selection = _seal_payload(
                {
                    "schema_version": "taxonomy_frozen_qwen_few_shot_selection_v1",
                    "protocol_id": PROTOCOL_ID,
                    "method_id": METHOD_ID,
                    "training_scope_id": args.scope_id,
                    "scope_contract": scope_contract,
                    "scope_contract_sha256": scope_contract_sha,
                    "demonstration_sha256s": demonstration_sha256s,
                    "selection_partition": "seen_validation_only",
                    "thresholds": {
                        "aspect": aspect_selection.threshold,
                        "runner_up_sentiment": (
                            runner_selection.second_sentiment_threshold
                        ),
                    },
                    "selection_metrics": evaluate_prediction_mask(
                        scored, mask, aspects=fold.seen_aspects
                    ),
                    "conditional_sentiment_metrics": conditional_sentiment_metrics(
                        scored
                    ),
                    "calibration_pair_identity_sha256": pair_identity_hash(
                        sentiment_grid
                    ),
                    "calibration_score_sha256": dataframe_sha256(
                        scored,
                        [
                            "row_uid",
                            "candidate_aspect",
                            "candidate_sentiment",
                            "aspect_score",
                            "sentiment_score",
                        ],
                        sort_columns=[
                            "row_uid",
                            "candidate_aspect",
                            "candidate_sentiment",
                        ],
                    ),
                    "failure_count": 0,
                    "test_contract_count": 0,
                },
                "selection_payload_sha256",
            )
            _write_json_atomic(selection_path, selection)
            publish_artifact_unit(
                output_root,
                [selection_path.relative_to(output_root)],
                unit_id=f"selection-{METHOD_ID}-{args.scope_id}",
                protocol_id=PROTOCOL_ID,
                contract_sha256=scope_contract_sha,
            )

        thresholds = selection["thresholds"]
        if not isinstance(thresholds, Mapping):
            raise ValueError("Frozen-Qwen selection thresholds are invalid.")
        results: list[dict[str, object]] = []
        for current_fold in folds:
            for condition in formal_conditions(current_fold):
                variants = variant_map(current_fold, condition)
                aspect_grid = build_aspect_grid(
                    validation_rows,
                    current_fold.evaluation_aspects,
                    variants,
                    resource,
                )
                sentiment_grid = build_sentiment_grid(
                    validation_rows,
                    current_fold.evaluation_aspects,
                    variants,
                    resource,
                )
                aspect_grid, sentiment_grid = _augment_grids(
                    aspect_grid,
                    sentiment_grid,
                    fold=current_fold,
                    condition=condition,
                )
                contract = _run_contract(
                    config=config,
                    fold=current_fold,
                    condition=condition,
                    scope_contract_sha256=scope_contract_sha,
                    demonstration_sha256s=demonstration_sha256s,
                    thresholds={
                        "aspect": float(thresholds["aspect"]),
                        "runner_up_sentiment": float(
                            thresholds["runner_up_sentiment"]
                        ),
                    },
                    sentiment_grid=sentiment_grid,
                    resource=resource,
                )
                artifacts: dict[int, pd.DataFrame] = {}
                for shard_index in range(contract.shard_count):
                    pair_shard = select_pair_shard(
                        sentiment_grid, shard_index, contract.shard_count
                    )
                    if pair_shard.empty:
                        continue
                    state, artifact = two_stage_shard_resume_state(
                        output_root,
                        contract,
                        sentiment_grid,
                        shard_index=shard_index,
                    )
                    if state == "complete":
                        assert artifact is not None
                        artifacts[shard_index] = artifact
                        continue
                    shard_uids = set(pair_shard["row_uid"].astype(str))
                    aspect_shard = aspect_grid[
                        aspect_grid["row_uid"].astype(str).isin(shard_uids)
                    ].copy()
                    scored = _score_grids(
                        cache,
                        aspect_shard,
                        pair_shard,
                        model=model,
                        tokenizer=tokenizer,
                        aspect_ids=aspect_ids,
                        sentiment_ids=sentiment_ids,
                        demonstrations=demonstrations,
                    )
                    artifact = build_two_stage_score_artifact(
                        scored, contract, shard_index=shard_index
                    )
                    csv_path, manifest_path = write_two_stage_score_shard(
                        artifact,
                        output_root,
                        contract,
                        shard_index=shard_index,
                    )
                    publish_artifact_unit(
                        output_root,
                        [
                            csv_path.relative_to(output_root),
                            manifest_path.relative_to(output_root),
                        ],
                        unit_id=(
                            f"score-{METHOD_ID}-{current_fold.fold_id}-{condition}-"
                            f"s{shard_index:02d}-{contract.contract_sha256[:12]}"
                        ),
                        protocol_id=PROTOCOL_ID,
                        contract_sha256=contract.contract_sha256,
                    )
                    artifacts[shard_index] = artifact
                merged = merge_two_stage_score_shards(
                    artifacts, contract, sentiment_grid
                )
                if (
                    merged["aspect_score"].nunique() < 2
                    or merged["sentiment_score"].nunique() < 2
                ):
                    raise RuntimeError(
                        "Frozen-Qwen few-shot score collapse in a complete formal "
                        "condition."
                    )
                mask = capped_two_sentiment_prediction_mask(
                    merged,
                    aspect_threshold=float(thresholds["aspect"]),
                    second_sentiment_threshold=float(
                        thresholds["runner_up_sentiment"]
                    ),
                )
                per_group = (
                    pd.DataFrame(
                        {
                            "row_uid": merged["row_uid"].astype(str),
                            "candidate_aspect": merged["candidate_aspect"].astype(str),
                            "selected": mask.astype(int),
                        }
                    )
                    .groupby(["row_uid", "candidate_aspect"], sort=False)["selected"]
                    .sum()
                )
                if int(per_group.max()) > 2:
                    raise RuntimeError("Frozen-Qwen decoder emitted a third sentiment.")
                aspect_presence = per_group.gt(0).astype(int)
                if aspect_presence.nunique() < 2:
                    raise RuntimeError(
                        "Frozen-Qwen few-shot prediction collapse in a formal condition."
                    )
                post_supervisor_views = (
                    {
                        "post_supervisor_views": evaluate_l2_condition(
                            merged,
                            current_fold,
                            aspect_threshold=float(thresholds["aspect"]),
                            second_sentiment_threshold=float(
                                thresholds["runner_up_sentiment"]
                            ),
                        )
                    }
                    if PROTOCOL_ID == "taxonomy_two_stage_formal_v2"
                    and current_fold.level == "L2"
                    else {}
                )
                result_path = (
                    output_root
                    / "results"
                    / METHOD_ID
                    / current_fold.level
                    / current_fold.fold_id
                    / f"{condition}.json"
                )
                result = _seal_payload(
                    {
                        "schema_version": (
                            "taxonomy_frozen_qwen_few_shot_fold_condition_v1"
                        ),
                        "protocol_id": PROTOCOL_ID,
                        "method_id": METHOD_ID,
                        "training_scope_id": args.scope_id,
                        "fold_id": current_fold.fold_id,
                        "level": current_fold.level,
                        "condition": condition,
                        "heldout_aspects": list(current_fold.heldout_aspects),
                        "selection_partition": "seen_validation_only",
                        "evaluation_partition": "validation_only",
                        "scope_contract_sha256": scope_contract_sha,
                        "demonstration_sha256s": demonstration_sha256s,
                        "thresholds": dict(thresholds),
                        "run_contract_sha256": contract.contract_sha256,
                        "score_shards": len(artifacts),
                        "score_rows": len(merged),
                        "score_sha256": dataframe_sha256(
                            merged,
                            [
                                "row_uid",
                                "candidate_aspect",
                                "candidate_sentiment",
                                "aspect_score",
                                "sentiment_score",
                            ],
                            sort_columns=[
                                "row_uid",
                                "candidate_aspect",
                                "candidate_sentiment",
                            ],
                        ),
                        "partitions": {
                            "heldout": evaluate_prediction_mask(
                                merged, mask, aspects=current_fold.heldout_aspects
                            ),
                            "full": evaluate_prediction_mask(
                                merged, mask, aspects=current_fold.evaluation_aspects
                            ),
                        },
                        **post_supervisor_views,
                        "conditional_sentiment_metrics": (
                            conditional_sentiment_metrics(merged)
                        ),
                        "maximum_sentiments_per_aspect": int(per_group.max()),
                        "failure_count": 0,
                        "test_contract_count": 0,
                    },
                    "result_payload_sha256",
                )
                if result_path.exists() and args.resume:
                    observed = _read_object(result_path)
                    _validate_sealed_payload(observed, "result_payload_sha256")
                    if observed.get("run_contract_sha256") != contract.contract_sha256:
                        raise RuntimeError(
                            "Existing Frozen-Qwen result belongs to another contract."
                        )
                    result = observed
                elif result_path.exists():
                    raise FileExistsError(result_path)
                else:
                    _write_json_atomic(result_path, result)
                    publish_artifact_unit(
                        output_root,
                        [result_path.relative_to(output_root)],
                        unit_id=(
                            f"result-{METHOD_ID}-{current_fold.fold_id}-{condition}-"
                            f"{contract.contract_sha256[:12]}"
                        ),
                        protocol_id=PROTOCOL_ID,
                        contract_sha256=contract.contract_sha256,
                    )
                results.append(result)
        return {
            "schema_version": "taxonomy_frozen_qwen_few_shot_scope_result_v1",
            "status": "pass",
            "protocol_id": PROTOCOL_ID,
            "method_id": METHOD_ID,
            "training_scope_id": args.scope_id,
            "scope_contract_sha256": scope_contract_sha,
            "completed_fold_conditions": len(results),
            "cache": cache.summary(),
            "failure_count": 0,
            "test_contract_count": 0,
        }
    finally:
        _release_model(model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one validation-only formal Frozen-Qwen few-shot scope."
    )
    parser.add_argument("--scope-id", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_SAFETY_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs/experimental/taxonomy_two_stage_formal_v1"
            / "worker-distil-frozen"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.scope_id not in scope_folds():
        raise ValueError(f"Unknown Frozen-Qwen training scope: {args.scope_id!r}")
    value = run(args)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
