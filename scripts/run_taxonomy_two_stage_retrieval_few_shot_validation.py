"""Run the single preregistered retrieval-based Frozen-Qwen Stage 1.

The runner opens only official train and validation, retrieves all four
demonstrations from the exact outer training fold and seen aspects, and writes
resumable validation-only aspect probabilities. It never selects a threshold.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
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

from msc_project.baselines.candidate_similarity import (
    REGISTERED_CONFIGS,
    FrozenTransformerSentenceEncoder,
)
from msc_project.data.fabsa import default_data_dir
from msc_project.data.splits import load_official_fabsa_splits
from msc_project.experiments.taxonomy_execution import canonical_sha256
from msc_project.experiments.taxonomy_methods import resolve_method_spec
from msc_project.experiments.taxonomy_post_supervisor import post_supervisor_l2_folds
from msc_project.experiments.taxonomy_protocol import build_taxonomy_fold_splits
from msc_project.experiments.taxonomy_resources import load_description_bundle
from msc_project.experiments.taxonomy_retrieval_few_shot import (
    assemble_demonstrations,
    pair_embeddings,
    select_diverse_examples,
)
from msc_project.experiments.taxonomy_two_stage_formal import variant_map
from msc_project.experiments.taxonomy_two_stage_runtime import build_aspect_grid
from msc_project.llm.qwen_pair_classifier import load_frozen_qwen_pair
from msc_project.llm.qwen_two_stage_classifier import (
    demonstrations_sha256,
    score_two_stage_prompts,
)


STUDY_ID = "taxonomy_two_stage_no_retraining_extensions_v1"
METHOD_ID = "e5_retrieval_frozen_qwen_few_shot_stage1__qlora_stage2"
BASE_METHOD_ID = "frozen_qwen_candidate_pair"
CONDITIONS = ("D", "N")
FOLDS = tuple(f"l2-a{index:02d}" for index in range(1, 13))
MANIFEST_COLUMNS = (
    "row_uid",
    "candidate_aspect",
    "representation_variant",
    "target",
    "demo_1_row_uid",
    "demo_1_aspect",
    "demo_2_row_uid",
    "demo_2_aspect",
    "demo_3_row_uid",
    "demo_3_aspect",
    "demo_4_row_uid",
    "demo_4_aspect",
    "demonstrations_sha256",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_config(config: Mapping[str, Any]) -> None:
    section = config.get("retrieval_few_shot_stage_1", {})
    retriever = section.get("retriever", {})
    demonstrations = section.get("demonstrations", {})
    if (
        config.get("schema_version") != STUDY_ID
        or config.get("evaluation_partition") != "validation_only"
        or config.get("allowed_splits") != ["train", "validation"]
        or config.get("official_test_permitted") is not False
        or config.get("include_official_test") is not False
        or int(config.get("test_contract_count", -1)) != 0
        or tuple(config.get("folds", ())) != FOLDS
        or tuple(config.get("conditions", ())) != CONDITIONS
        or section.get("method_id") != METHOD_ID
        or section.get("base_model_training") is not False
        or retriever.get("model_id") != "intfloat/e5-base-v2"
        or demonstrations.get("order") != ["Y", "Y", "N", "N"]
        or int(demonstrations.get("shortlist_size_per_answer_class", -1)) != 256
        or section.get("selection_condition") != "D_only"
    ):
        raise ValueError("Retrieval few-shot configuration violates the boundary.")


def _normalise_rows(frame: pd.DataFrame) -> pd.DataFrame:
    value = frame.copy()
    value["row_uid"] = value["row_uid"].astype(str)
    if value["row_uid"].duplicated().any():
        raise ValueError("Review identities are not unique.")
    return value.sort_values("row_uid", kind="stable").reset_index(drop=True)


def _embedding_cache(
    *,
    output_root: Path,
    config_sha256: str,
    all_rows: pd.DataFrame,
    candidate_texts: Sequence[str],
    local_files_only: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    cache_dir = output_root / "retrieval_embedding_cache"
    cache_path = cache_dir / "e5_embeddings.npz"
    manifest_path = cache_dir / "manifest.json"
    row_uids = all_rows["row_uid"].astype(str).to_numpy()
    texts = all_rows["text"].fillna("").astype(str).tolist()
    cards = tuple(sorted(set(str(value) for value in candidate_texts)))
    contract = {
        "schema_version": "taxonomy_retrieval_embedding_cache_v1",
        "study_id": STUDY_ID,
        "config_sha256": config_sha256,
        "row_uid_sha256": canonical_sha256(row_uids.tolist()),
        "review_text_sha256": canonical_sha256(texts),
        "candidate_text_sha256": canonical_sha256(cards),
        "model_id": REGISTERED_CONFIGS["e5_base_v2"].model_id,
        "model_revision": REGISTERED_CONFIGS["e5_base_v2"].revision,
        "test_contract_count": 0,
    }
    contract_sha = canonical_sha256(contract)
    if cache_path.is_file() and manifest_path.is_file():
        manifest = _read_json(manifest_path)
        if (
            manifest.get("contract_sha256") != contract_sha
            or manifest.get("cache_sha256") != _file_sha256(cache_path)
        ):
            raise RuntimeError("Retrieval embedding cache resume conflict.")
        cached = np.load(cache_path, allow_pickle=False)
        if not np.array_equal(cached["row_uids"].astype(str), row_uids):
            raise RuntimeError("Retrieval review cache identities changed.")
        if not np.array_equal(cached["candidate_texts"].astype(str), np.asarray(cards)):
            raise RuntimeError("Retrieval candidate cache identities changed.")
        review_matrix = cached["review_embeddings"].astype(np.float32)
        candidate_matrix = cached["candidate_embeddings"].astype(np.float32)
    else:
        if cache_path.exists() or manifest_path.exists():
            raise RuntimeError("Retrieval embedding cache is incomplete.")
        encoder = FrozenTransformerSentenceEncoder(
            REGISTERED_CONFIGS["e5_base_v2"],
            device="auto",
            local_files_only=local_files_only,
        )
        try:
            review_matrix = encoder.encode(texts, role="review").astype(np.float32)
            candidate_matrix = encoder.encode(list(cards), role="candidate").astype(
                np.float32
            )
        finally:
            encoder.close()
        if not np.isfinite(review_matrix).all() or not np.isfinite(candidate_matrix).all():
            raise ValueError("E5 embedding cache contains non-finite values.")
        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name("e5_embeddings.tmp.npz")
        np.savez_compressed(
            temporary,
            row_uids=row_uids,
            candidate_texts=np.asarray(cards),
            review_embeddings=review_matrix,
            candidate_embeddings=candidate_matrix,
        )
        os.replace(temporary, cache_path)
        manifest = {
            **contract,
            "contract_sha256": contract_sha,
            "cache_sha256": _file_sha256(cache_path),
            "review_count": len(row_uids),
            "candidate_card_count": len(cards),
            "embedding_dimension": int(review_matrix.shape[1]),
            "official_test_opened": False,
            "include_official_test": False,
        }
        _atomic_json(manifest_path, manifest)
    review_lookup = {
        str(uid): review_matrix[index] for index, uid in enumerate(row_uids)
    }
    candidate_lookup = {
        str(card): candidate_matrix[index] for index, card in enumerate(cards)
    }
    return review_lookup, candidate_lookup, _read_json(manifest_path)


def _pair_matrix(
    frame: pd.DataFrame,
    review_lookup: Mapping[str, np.ndarray],
    candidate_lookup: Mapping[str, np.ndarray],
) -> np.ndarray:
    reviews = np.stack(
        [review_lookup[str(value)] for value in frame["row_uid"].astype(str)], axis=0
    )
    cards = np.stack(
        [candidate_lookup[str(value)] for value in frame["candidate_text"].astype(str)],
        axis=0,
    )
    return pair_embeddings(reviews, cards)


def _shortlists(
    query: np.ndarray,
    pool: np.ndarray,
    *,
    size: int,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    if query.ndim != 2 or pool.ndim != 2 or query.shape[1] != pool.shape[1]:
        raise ValueError("Retrieval query and pool matrices do not align.")
    if len(pool) < size:
        raise ValueError("Retrieval pool is smaller than the fixed shortlist.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool_tensor = torch.from_numpy(pool).to(device)
    index_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    try:
        for start in range(0, len(query), batch_size):
            values = torch.from_numpy(query[start : start + batch_size]).to(device)
            with torch.inference_mode():
                scores = values @ pool_tensor.T
                top_scores, top_indices = torch.topk(
                    scores, k=size, dim=1, largest=True, sorted=False
                )
            index_parts.append(top_indices.cpu().numpy())
            score_parts.append(top_scores.cpu().numpy())
    finally:
        del pool_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.concatenate(index_parts), np.concatenate(score_parts)


def _retrieve_manifest(
    query_grid: pd.DataFrame,
    train_grid: pd.DataFrame,
    *,
    review_lookup: Mapping[str, np.ndarray],
    candidate_lookup: Mapping[str, np.ndarray],
    shortlist_size: int,
) -> pd.DataFrame:
    query_matrix = _pair_matrix(query_grid, review_lookup, candidate_lookup)
    pools = {
        "Y": train_grid[train_grid["target"].astype(int).eq(1)].reset_index(drop=True),
        "N": train_grid[train_grid["target"].astype(int).eq(0)].reset_index(drop=True),
    }
    pool_matrices = {
        answer: _pair_matrix(pool, review_lookup, candidate_lookup)
        for answer, pool in pools.items()
    }
    shortlist = {
        answer: _shortlists(
            query_matrix, matrix, size=shortlist_size
        )
        for answer, matrix in pool_matrices.items()
    }
    records: list[dict[str, Any]] = []
    for query_index, row in enumerate(query_grid.itertuples(index=False)):
        selected: dict[str, tuple[Any, ...]] = {}
        for answer in ("Y", "N"):
            indices, scores = shortlist[answer]
            subset = pools[answer].iloc[indices[query_index]].reset_index(drop=True)
            selected[answer] = select_diverse_examples(
                scores[query_index], subset, answer=answer, count=2
            )
        demonstrations = assemble_demonstrations(selected["Y"], selected["N"])
        record: dict[str, Any] = {
            "row_uid": str(row.row_uid),
            "candidate_aspect": str(row.candidate_aspect),
            "representation_variant": str(row.representation_variant),
            "target": int(row.target),
            "demonstrations_sha256": demonstrations_sha256(demonstrations),
        }
        for position, demo in enumerate(demonstrations, start=1):
            record[f"demo_{position}_row_uid"] = demo.row_uid
            record[f"demo_{position}_aspect"] = demo.candidate_aspect
        records.append(record)
    result = pd.DataFrame.from_records(records)[list(MANIFEST_COLUMNS)]
    if result.duplicated(["row_uid", "candidate_aspect"]).any():
        raise ValueError("Retrieval manifest contains duplicate query identities.")
    return result


def _prepare_fold_manifests(
    *,
    fold: Any,
    train_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    resource: Mapping[str, Any],
    review_lookup: Mapping[str, np.ndarray],
    candidate_lookup: Mapping[str, np.ndarray],
    shortlist_size: int,
    output_root: Path,
    config_sha256: str,
) -> dict[str, Path]:
    variants_train = {aspect: "name_and_description" for aspect in fold.seen_aspects}
    train_grid = build_aspect_grid(
        train_rows, fold.seen_aspects, variants_train, resource
    )
    if set(train_grid["candidate_aspect"].astype(str)) & set(fold.heldout_aspects):
        raise RuntimeError("A held-out aspect entered the retrieval pool.")
    paths: dict[str, Path] = {}
    d_manifest: pd.DataFrame | None = None
    for condition in CONDITIONS:
        path = output_root / "retrieval_manifests" / fold.fold_id / f"{condition}.csv"
        metadata_path = path.with_suffix(".manifest.json")
        if path.is_file() and metadata_path.is_file():
            metadata = _read_json(metadata_path)
            if (
                metadata.get("config_sha256") != config_sha256
                or metadata.get("manifest_sha256") != _file_sha256(path)
                or int(metadata.get("test_contract_count", -1)) != 0
            ):
                raise RuntimeError("Retrieval manifest resume conflict.")
            manifest = pd.read_csv(path)
        else:
            if path.exists() or metadata_path.exists():
                raise RuntimeError("Retrieval manifest unit is incomplete.")
            grid = build_aspect_grid(
                validation_rows,
                fold.evaluation_aspects,
                variant_map(fold, condition),
                resource,
            )
            if condition == "D":
                manifest = _retrieve_manifest(
                    grid,
                    train_grid,
                    review_lookup=review_lookup,
                    candidate_lookup=candidate_lookup,
                    shortlist_size=shortlist_size,
                )
                d_manifest = manifest
            else:
                if d_manifest is None:
                    d_manifest = pd.read_csv(
                        output_root / "retrieval_manifests" / fold.fold_id / "D.csv"
                    )
                heldout = grid["candidate_aspect"].astype(str).isin(fold.heldout_aspects)
                heldout_manifest = _retrieve_manifest(
                    grid[heldout].reset_index(drop=True),
                    train_grid,
                    review_lookup=review_lookup,
                    candidate_lookup=candidate_lookup,
                    shortlist_size=shortlist_size,
                )
                seen_manifest = d_manifest[
                    ~d_manifest["candidate_aspect"].astype(str).isin(
                        fold.heldout_aspects
                    )
                ].copy()
                manifest = pd.concat([seen_manifest, heldout_manifest], ignore_index=True)
                expected_order = grid[["row_uid", "candidate_aspect"]].astype(str)
                manifest = expected_order.merge(
                    manifest,
                    on=["row_uid", "candidate_aspect"],
                    how="left",
                    validate="one_to_one",
                )
                manifest = manifest[list(MANIFEST_COLUMNS)]
            if len(manifest) != len(grid) or manifest.isna().any().any():
                raise ValueError("Retrieval manifest does not cover the validation grid.")
            _atomic_csv(path, manifest)
            metadata = {
                "schema_version": "taxonomy_retrieval_demonstration_manifest_v1",
                "study_id": STUDY_ID,
                "fold_id": fold.fold_id,
                "condition": condition,
                "config_sha256": config_sha256,
                "query_count": len(manifest),
                "training_pool_count": len(train_grid),
                "training_pool_positive_count": int(train_grid["target"].sum()),
                "training_pool_negative_count": int((1 - train_grid["target"]).sum()),
                "heldout_aspect_pool_instance_count": 0,
                "retrieval_similarity_target_read_count": 0,
                "manifest_sha256": _file_sha256(path),
                "official_test_opened": False,
                "include_official_test": False,
                "test_contract_count": 0,
            }
            metadata["metadata_payload_sha256"] = canonical_sha256(metadata)
            _atomic_json(metadata_path, metadata)
        if len(manifest) != len(validation_rows) * len(fold.evaluation_aspects):
            raise ValueError("Retrieved query count is incomplete.")
        paths[condition] = path
    return paths


def _demo_lookup(
    train_grid: pd.DataFrame,
) -> dict[tuple[str, str, int], Any]:
    lookup = {}
    for row in train_grid.itertuples(index=False):
        key = (str(row.row_uid), str(row.candidate_aspect), int(row.target))
        if key in lookup:
            raise ValueError("Training retrieval identity is duplicated.")
        lookup[key] = row
    return lookup


def _manifest_demonstrations(row: Any, lookup: Mapping[tuple[str, str, int], Any]):
    values = []
    for position, answer in enumerate(("Y", "Y", "N", "N"), start=1):
        target = 1 if answer == "Y" else 0
        key = (
            str(getattr(row, f"demo_{position}_row_uid")),
            str(getattr(row, f"demo_{position}_aspect")),
            target,
        )
        source = lookup.get(key)
        if source is None:
            raise RuntimeError(f"Retrieved demonstration is absent from train pool: {key}")
        from msc_project.llm.qwen_two_stage_classifier import TwoStageDemonstration

        values.append(
            TwoStageDemonstration(
                row_uid=key[0],
                candidate_aspect=key[1],
                review_text=str(source.text),
                aspect_candidate=str(source.candidate_text),
                answer=answer,
            )
        )
    demonstrations = tuple(values)
    if demonstrations_sha256(demonstrations) != str(row.demonstrations_sha256):
        raise RuntimeError("Retrieved demonstration hash changed before inference.")
    return demonstrations


def _score_manifest(
    *,
    fold: Any,
    condition: str,
    manifest_path: Path,
    train_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    resource: Mapping[str, Any],
    model: Any,
    tokenizer: Any,
    aspect_ids: Any,
    output_root: Path,
    config_sha256: str,
    max_length: int,
    batch_size: int,
    checkpoint_prompts: int,
    prompt_limit: int | None,
) -> Path:
    manifest = pd.read_csv(manifest_path)
    grid = build_aspect_grid(
        validation_rows,
        fold.evaluation_aspects,
        variant_map(fold, condition),
        resource,
    )
    identities = grid[["row_uid", "candidate_aspect"]].astype(str)
    manifest_identities = manifest[["row_uid", "candidate_aspect"]].astype(str)
    if not identities.equals(manifest_identities):
        raise RuntimeError("Retrieval manifest and validation grid order differ.")
    train_grid = build_aspect_grid(
        train_rows,
        fold.seen_aspects,
        {aspect: "name_and_description" for aspect in fold.seen_aspects},
        resource,
    )
    lookup = _demo_lookup(train_grid)
    output_path = output_root / "retrieval_scores" / fold.fold_id / f"{condition}.csv"
    metadata_path = output_path.with_suffix(".manifest.json")
    contract_path = output_path.with_suffix(".contract.json")
    contract = {
        "schema_version": "taxonomy_retrieval_score_resume_contract_v1",
        "study_id": STUDY_ID,
        "fold_id": fold.fold_id,
        "condition": condition,
        "config_sha256": config_sha256,
        "retrieval_manifest_sha256": _file_sha256(manifest_path),
        "official_test_opened": False,
        "include_official_test": False,
        "test_contract_count": 0,
    }
    contract["contract_payload_sha256"] = canonical_sha256(contract)
    if contract_path.is_file():
        if _read_json(contract_path) != contract:
            raise RuntimeError("Retrieval score resume contract conflict.")
    else:
        if output_path.exists() or metadata_path.exists():
            raise RuntimeError("Retrieval score cache exists without its contract.")
        _atomic_json(contract_path, contract)
    progress = (
        pd.read_csv(output_path)
        if output_path.is_file()
        else pd.DataFrame(
            columns=[
                "row_uid",
                "candidate_aspect",
                "representation_variant",
                "target",
                "aspect_score",
                "aspect_no_score",
                "demonstrations_sha256",
            ]
        )
    )
    copied_from_d = 0
    if condition == "N":
        d_path = output_root / "retrieval_scores" / fold.fold_id / "D.csv"
        d_metadata_path = d_path.with_suffix(".manifest.json")
        if not d_path.is_file() or not d_metadata_path.is_file():
            raise RuntimeError("N scoring requires the completed identical D seen scores.")
        d_metadata = _read_json(d_metadata_path)
        if d_metadata.get("status") != "complete":
            raise RuntimeError("N scoring cannot reuse an incomplete D score unit.")
        d_scores = pd.read_csv(d_path)
        seen = ~manifest["candidate_aspect"].astype(str).isin(fold.heldout_aspects)
        seen_manifest = manifest.loc[
            seen, ["row_uid", "candidate_aspect", "demonstrations_sha256"]
        ].copy()
        reusable = seen_manifest.merge(
            d_scores,
            on=["row_uid", "candidate_aspect", "demonstrations_sha256"],
            how="left",
            validate="one_to_one",
        )
        if reusable[["aspect_score", "aspect_no_score"]].isna().any().any():
            raise RuntimeError("D and N seen retrieval-score identities differ.")
        existing_keys = set(
            zip(
                progress["row_uid"].astype(str),
                progress["candidate_aspect"].astype(str),
            )
        )
        reusable = reusable[
            [
                (str(row.row_uid), str(row.candidate_aspect)) not in existing_keys
                for row in reusable.itertuples(index=False)
            ]
        ]
        if not reusable.empty:
            progress = (
                pd.concat([progress, reusable], ignore_index=True)
                if not progress.empty
                else reusable.reset_index(drop=True)
            )
            _atomic_csv(output_path, progress)
            copied_from_d = len(reusable)
    if progress.duplicated(["row_uid", "candidate_aspect"]).any():
        raise RuntimeError("Retrieval score resume identities are duplicated.")
    existing = set(
        zip(progress["row_uid"].astype(str), progress["candidate_aspect"].astype(str))
    )
    missing_indices = [
        index
        for index, row in enumerate(manifest.itertuples(index=False))
        if (str(row.row_uid), str(row.candidate_aspect)) not in existing
    ]
    if prompt_limit is not None:
        missing_indices = missing_indices[:prompt_limit]
    val_lookup = validation_rows.set_index(validation_rows["row_uid"].astype(str))
    new_records: list[dict[str, Any]] = []
    for start in range(0, len(missing_indices), batch_size):
        batch_indices = missing_indices[start : start + batch_size]
        rows = [manifest.iloc[index] for index in batch_indices]
        demonstrations = [
            _manifest_demonstrations(row, lookup)
            for row in manifest.iloc[batch_indices].itertuples(index=False)
        ]
        reviews = [str(val_lookup.loc[str(row.row_uid), "text"]) for row in rows]
        candidates = [str(grid.iloc[index]["candidate_text"]) for index in batch_indices]
        probabilities = score_two_stage_prompts(
            model,
            tokenizer,
            reviews,
            candidates,
            mode="aspect",
            max_length=max_length,
            batch_size=batch_size,
            aspect_verbalizer_ids=aspect_ids,
            demonstration_sets=demonstrations,
        )
        if probabilities.shape != (len(rows), 2) or not np.isfinite(probabilities).all():
            raise RuntimeError("Retrieval Frozen-Qwen returned invalid probabilities.")
        for row, probability in zip(rows, probabilities):
            new_records.append(
                {
                    "row_uid": str(row.row_uid),
                    "candidate_aspect": str(row.candidate_aspect),
                    "representation_variant": str(row.representation_variant),
                    "target": int(row.target),
                    "aspect_score": float(probability[0]),
                    "aspect_no_score": float(probability[1]),
                    "demonstrations_sha256": str(row.demonstrations_sha256),
                }
            )
        if len(new_records) >= checkpoint_prompts:
            addition = pd.DataFrame(new_records)
            progress = (
                pd.concat([progress, addition], ignore_index=True)
                if not progress.empty
                else addition
            )
            _atomic_csv(output_path, progress)
            new_records.clear()
    if new_records:
        addition = pd.DataFrame(new_records)
        progress = (
            pd.concat([progress, addition], ignore_index=True)
            if not progress.empty
            else addition
        )
        _atomic_csv(output_path, progress)
    complete = len(progress) == len(manifest)
    if complete:
        ordered = identities.merge(
            progress,
            on=["row_uid", "candidate_aspect"],
            how="left",
            validate="one_to_one",
        )
        _atomic_csv(output_path, ordered)
        progress = ordered
    if progress[["aspect_score", "aspect_no_score"]].apply(
        pd.to_numeric, errors="coerce"
    ).isna().any().any():
        raise RuntimeError("Retrieval score cache contains invalid probabilities.")
    metadata = {
        "schema_version": "taxonomy_retrieval_few_shot_aspect_scores_v1",
        "study_id": STUDY_ID,
        "method_id": METHOD_ID,
        "fold_id": fold.fold_id,
        "condition": condition,
        "config_sha256": config_sha256,
        "retrieval_manifest_sha256": _file_sha256(manifest_path),
        "completed_prompt_count": len(progress),
        "planned_prompt_count": len(manifest),
        "copied_identical_D_seen_prompt_count": copied_from_d,
        "new_Qwen_inference_prompt_count": len(missing_indices),
        "status": "complete" if complete else "smoke_partial",
        "score_sha256": _file_sha256(output_path) if output_path.is_file() else None,
        "non_finite_score_count": 0,
        "prediction_score_unique_count": int(progress["aspect_score"].nunique()),
        "failure_count": 0,
        "official_test_opened": False,
        "include_official_test": False,
        "test_contract_count": 0,
    }
    metadata["metadata_payload_sha256"] = canonical_sha256(metadata)
    _atomic_json(metadata_path, metadata)
    return output_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config = _read_json(config_path)
    validate_config(config)
    section = config["retrieval_few_shot_stage_1"]
    config_sha256 = _file_sha256(config_path)
    output_root = args.output_root.resolve()
    frame = load_official_fabsa_splits(args.data_dir, ("train", "validation"))
    if set(frame["original_split"].astype(str)) != {"train", "validation"}:
        raise AssertionError("Retrieval runner opened an unregistered split.")
    frame = _normalise_rows(frame)
    validation_rows = frame[frame["original_split"].eq("validation")].copy()
    resource = load_description_bundle(require_approved=True)
    folds_by_id = {fold.fold_id: fold for fold in post_supervisor_l2_folds()}
    requested = tuple(args.folds or FOLDS)
    if not requested or set(requested) - set(FOLDS):
        raise ValueError("Requested retrieval folds are invalid.")

    candidate_texts: list[str] = []
    split_rows: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for fold_id in requested:
        fold = folds_by_id[fold_id]
        splits = build_taxonomy_fold_splits(
            frame, fold, evaluation_splits=("validation",)
        )
        split_rows[fold_id] = (splits["train"], splits["validation"])
        for condition in CONDITIONS:
            grid = build_aspect_grid(
                splits["validation"],
                fold.evaluation_aspects,
                variant_map(fold, condition),
                resource,
            )
            candidate_texts.extend(grid["candidate_text"].astype(str))
        train_grid = build_aspect_grid(
            splits["train"],
            fold.seen_aspects,
            {aspect: "name_and_description" for aspect in fold.seen_aspects},
            resource,
        )
        candidate_texts.extend(train_grid["candidate_text"].astype(str))
    review_lookup, candidate_lookup, embedding_manifest = _embedding_cache(
        output_root=output_root,
        config_sha256=config_sha256,
        all_rows=frame,
        candidate_texts=candidate_texts,
        local_files_only=args.local_files_only,
    )

    manifest_paths: dict[tuple[str, str], Path] = {}
    for fold_id in requested:
        fold = folds_by_id[fold_id]
        train_rows, eval_rows = split_rows[fold_id]
        paths = _prepare_fold_manifests(
            fold=fold,
            train_rows=train_rows,
            validation_rows=eval_rows,
            resource=resource,
            review_lookup=review_lookup,
            candidate_lookup=candidate_lookup,
            shortlist_size=int(
                section["demonstrations"]["shortlist_size_per_answer_class"]
            ),
            output_root=output_root,
            config_sha256=config_sha256,
        )
        for condition, path in paths.items():
            manifest_paths[(fold_id, condition)] = path
    del review_lookup, candidate_lookup
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    if args.prepare_only:
        summary = {
            "schema_version": "taxonomy_retrieval_preparation_summary_v1",
            "study_id": STUDY_ID,
            "folds": list(requested),
            "manifest_count": len(manifest_paths),
            "embedding_manifest": embedding_manifest,
            "official_test_opened": False,
            "include_official_test": False,
            "test_contract_count": 0,
        }
        _atomic_json(output_root / "retrieval_preparation_summary.json", summary)
        return summary

    spec = resolve_method_spec(BASE_METHOD_ID)
    if not spec.model_id or not spec.model_revision:
        raise ValueError("Frozen Qwen model identity is not pinned.")
    tokenizer, model, aspect_ids = load_frozen_qwen_pair(
        spec.model_id,
        revision=spec.model_revision,
        load_in_4bit=True,
        local_files_only=args.local_files_only,
    )
    score_paths: list[Path] = []
    try:
        for fold_id in requested:
            fold = folds_by_id[fold_id]
            train_rows, eval_rows = split_rows[fold_id]
            for condition in CONDITIONS:
                score_paths.append(
                    _score_manifest(
                        fold=fold,
                        condition=condition,
                        manifest_path=manifest_paths[(fold_id, condition)],
                        train_rows=train_rows,
                        validation_rows=eval_rows,
                        resource=resource,
                        model=model,
                        tokenizer=tokenizer,
                        aspect_ids=aspect_ids,
                        output_root=output_root,
                        config_sha256=config_sha256,
                        max_length=int(section["max_length"]),
                        batch_size=int(section["batch_size"]),
                        checkpoint_prompts=args.checkpoint_prompts,
                        prompt_limit=args.prompt_limit,
                    )
                )
    finally:
        try:
            model.to("cpu")
        except (AttributeError, ValueError):
            pass
        del model, tokenizer
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    metadata = [
        _read_json(path.with_suffix(".manifest.json")) for path in score_paths
    ]
    complete = all(value["status"] == "complete" for value in metadata)
    summary = {
        "schema_version": "taxonomy_retrieval_few_shot_run_summary_v1",
        "study_id": STUDY_ID,
        "method_id": METHOD_ID,
        "folds": list(requested),
        "condition_count": len(score_paths),
        "completed_prompt_count": sum(
            int(value["completed_prompt_count"]) for value in metadata
        ),
        "planned_prompt_count": sum(int(value["planned_prompt_count"]) for value in metadata),
        "new_Qwen_inference_prompt_count": sum(
            int(value["new_Qwen_inference_prompt_count"]) for value in metadata
        ),
        "copied_identical_D_seen_prompt_count": sum(
            int(value["copied_identical_D_seen_prompt_count"]) for value in metadata
        ),
        "status": "complete" if complete else "smoke_partial",
        "failure_count": 0,
        "non_finite_score_count": 0,
        "official_test_opened": False,
        "include_official_test": False,
        "test_contract_count": 0,
    }
    summary["summary_payload_sha256"] = canonical_sha256(summary)
    _atomic_json(output_root / "retrieval_run_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs/experiments/taxonomy_two_stage_no_retraining_extensions_v1.json",
    )
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experimental/taxonomy_two_stage_no_retraining_extensions_v1",
    )
    parser.add_argument("--folds", nargs="*", default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prompt-limit", type=int, default=None)
    parser.add_argument("--checkpoint-prompts", type=int, default=256)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
