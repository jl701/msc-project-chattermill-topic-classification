"""Isolated, train/validation-only training-policy sensitivity primitives."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from msc_project.data.fabsa import aspect_labels, pair_labels
from msc_project.data.splits import load_official_fabsa_splits
from msc_project.experiments.taxonomy_execution import canonical_sha256
from msc_project.experiments.taxonomy_post_supervisor import evaluate_l2_condition
from msc_project.experiments.taxonomy_protocol import registered_folds
from msc_project.experiments.taxonomy_two_stage import (
    capped_two_sentiment_prediction_mask, evaluate_prediction_mask,
    select_second_sentiment_threshold, select_two_stage_threshold,
)
from msc_project.experiments.taxonomy_two_stage_runtime import (
    build_aspect_grid, build_sentiment_grid, join_two_stage_scores,
)

STUDY_ID = "taxonomy_training_policy_sensitivity_v1"
IDENTITY = ["row_uid", "candidate_aspect", "candidate_sentiment"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value):
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")
    if isinstance(value, (pd.Series, np.ndarray)):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                   ensure_ascii=False, allow_nan=False, default=_json_default) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_study(config_path: Path, data_dir: Path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (config["study_id"] != STUDY_ID or config["allowed_splits"] != ["train", "validation"]
            or config["include_official_test"] is not False or config["test_contract_count"] != 0):
        raise ValueError("Invalid sensitivity data boundary")
    if set(config["data_hashes"]) != {"train.csv", "validation.csv"}:
        raise ValueError("Only train/validation inputs may be hashed or loaded")
    for name, expected in config["data_hashes"].items():
        if sha256_file(data_dir / name) != expected:
            raise ValueError(f"Dataset hash mismatch: {name}")
    frame = load_official_fabsa_splits(data_dir, ("train", "validation"))
    if frame.row_uid.duplicated().any():
        raise ValueError("Duplicate official row identities")
    train = frame[frame.original_split.eq("train")].copy()
    validation = frame[frame.original_split.eq("validation")].copy()
    if len(train) != 7930 or len(validation) != 1057:
        raise ValueError("Registered row counts changed")
    validation["supervision_labels"] = validation["labels"]
    folds = tuple(registered_folds("L2"))
    if [f.fold_id for f in folds] != config["folds"]:
        raise ValueError("Registered fold identities changed")
    return config, train, validation, folds


def training_rows(original: pd.DataFrame, fold, policy: str, subset_seed: int | None = None):
    """Do not expose raw held-out labels to the model-facing training frame."""
    if set(original.original_split) != {"train"}:
        raise ValueError("Training pool must contain original train rows only")
    heldout = set(fold.heldout_aspects)
    has_target = original.labels.map(lambda labels: any(a in heldout for a, _ in labels))
    result = original.copy()
    if policy == "review_filtered":
        if subset_seed is not None:
            raise ValueError("Filtered arm has no subset seed")
        result = result.loc[~has_target].copy()
    elif policy == "label_masked_all_reviews":
        if subset_seed is not None:
            raise ValueError("Full masked arm has no subset seed")
    elif policy == "size_matched_label_masked":
        if subset_seed not in (13, 29, 47):
            raise ValueError("Unregistered subset seed")
        rank = sorted(original.row_uid.astype(str), key=lambda uid: (
            hashlib.sha256(f"{STUDY_ID}|{fold.fold_id}|{subset_seed}|{uid}".encode()).hexdigest(), uid))
        kept = set(rank[:int((~has_target).sum())])
        result = result.loc[result.row_uid.astype(str).isin(kept)].copy()
    else:
        raise ValueError(f"Unregistered training policy: {policy}")
    result["labels"] = result.labels.map(lambda labels: [(a, s) for a, s in labels if a not in heldout])
    # Strip all alternative raw annotation channels, not just supervision_labels.
    result = result[["row_uid", "text", "original_split", "labels"]].copy()
    result["supervision_labels"] = result.labels
    result["supervision_aspect_labels"] = result.labels.map(aspect_labels)
    result["supervision_pair_labels"] = result.labels.map(pair_labels)
    result["aspect_labels"] = result.supervision_aspect_labels
    result["pair_labels"] = result.supervision_pair_labels
    if any(a not in fold.seen_aspects for labels in result.labels for a, _ in labels):
        raise AssertionError("Held-out supervision survived masking")
    if result.empty or result.row_uid.duplicated().any():
        raise AssertionError("Invalid training pool identities")
    return result


def train_pool_hash(train: pd.DataFrame) -> str:
    records = train.sort_values("row_uid")[["row_uid", "text", "supervision_labels"]].to_dict("records")
    return canonical_sha256(records)


def grids(rows: pd.DataFrame, fold, resource, condition: str, *, seen_only=False):
    if condition not in ("N", "D"):
        raise ValueError("Only N/D are registered")
    aspects = fold.seen_aspects if seen_only else fold.evaluation_aspects
    variants = {a: "name_only" if condition == "N" and a in fold.heldout_aspects
                else "name_and_description" for a in aspects}
    return (build_aspect_grid(rows, aspects, variants, resource),
            build_sentiment_grid(rows, aspects, variants, resource))


def score(runtime, method: str, ag: pd.DataFrame, sg: pd.DataFrame):
    presence = runtime.score_aspects(ag)
    sentiment = (runtime.score_sentiments(sg) if method == "tfidf"
                 else runtime.score_sentiments(ag).reshape(-1))
    result = join_two_stage_scores(ag, sg, presence, sentiment)
    if not np.isfinite(result[["aspect_score", "sentiment_score"]].to_numpy()).all():
        raise ValueError("Non-finite sensitivity scores")
    return result


def select_seen(scored: pd.DataFrame, fold) -> dict:
    if set(scored.candidate_aspect) != set(fold.seen_aspects):
        raise ValueError("Threshold selection must contain exactly eleven seen aspects")
    aspect = select_two_stage_threshold(scored)
    second = select_second_sentiment_threshold(scored, aspect_threshold=aspect.threshold)
    prediction = capped_two_sentiment_prediction_mask(scored, aspect_threshold=aspect.threshold,
                                                     second_sentiment_threshold=second.second_sentiment_threshold)
    return {"aspect_threshold": float(aspect.threshold),
            "second_sentiment_threshold": float(second.second_sentiment_threshold),
            "selection_metrics": evaluate_prediction_mask(scored, prediction, aspects=fold.seen_aspects),
            "aspect_selection": asdict(aspect), "sentiment_selection": asdict(second)}


def evaluate_and_save(scored, fold, selection, root: Path, condition: str):
    scored = scored.sort_values(IDENTITY, kind="stable").reset_index(drop=True)
    expected = 1057 * 12 * 3
    if len(scored) != expected or scored.duplicated(IDENTITY).any():
        raise ValueError("Evaluation must contain the full 1057 x 12 x 3 grid")
    if not scored.row_uid.astype(str).str.startswith("validation:").all():
        raise ValueError("Only validation may be evaluated")
    thresholds = {key: selection[key] for key in ("aspect_threshold", "second_sentiment_threshold")}
    result = evaluate_l2_condition(scored, fold, **thresholds)
    predicted = capped_two_sentiment_prediction_mask(scored, **thresholds).to_numpy(bool)
    if not predicted.any() or predicted.all():
        raise RuntimeError("Unexpected whole-grid empty/constant prediction collapse")
    if scored.aspect_score.nunique() == 1 or scored.sentiment_score.nunique() == 1:
        raise RuntimeError("Unexpected constant raw score collapse")
    counts = scored.loc[predicted].groupby(["row_uid", "candidate_aspect"]).size()
    if (counts > 2).any():
        raise AssertionError("Third sentiment emitted")
    result["decoder_usage"] = {"selected_review_aspects": int(len(counts)),
                               "two_sentiment_instances": int((counts == 2).sum())}
    root.mkdir(parents=True, exist_ok=True)
    score_path = root / f"{condition}_scores.csv.gz"
    scored.to_csv(score_path, index=False, compression={"method": "gzip", "mtime": 0})
    evidence = scored[IDENTITY].copy()
    truth = scored.target.to_numpy(bool)
    for name, value in {"tp": truth & predicted, "fp": ~truth & predicted, "fn": truth & ~predicted}.items():
        evidence[name] = value.astype(np.int8)
    evidence["partition"] = np.where(evidence.candidate_aspect.isin(fold.heldout_aspects), "heldout", "seen")
    evidence = evidence.groupby(["row_uid", "partition"])[["tp", "fp", "fn"]].sum().reset_index()
    evidence.to_csv(root / f"{condition}_row_counts.csv", index=False)
    write_json(root / f"{condition}_metrics.json", result)
    return result


def receipt(root: Path, contract: dict) -> dict:
    files = {str(p.relative_to(root)).replace("\\", "/"): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
             for p in sorted(root.rglob("*")) if p.is_file() and p != root / "receipt.json" and p.name != "FAILED.json"}
    payload = {"status": "complete", "contract": contract, "files": files,
               "failure_count": 0, "test_contract_count": 0}
    write_json(root / "receipt.json", payload)
    return payload


def verify_receipt(root: Path, contract: dict | None = None):
    payload = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
    if payload["status"] != "complete" or payload["failure_count"] or payload["test_contract_count"]:
        raise ValueError("Invalid receipt status")
    if contract is not None and payload["contract"] != contract:
        raise ValueError("Resume contract conflict")
    for relative, entry in payload["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Unsafe receipt path")
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Receipt hash mismatch: {relative}")
    return payload
