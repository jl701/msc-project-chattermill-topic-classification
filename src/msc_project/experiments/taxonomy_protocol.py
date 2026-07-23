"""Strict fold, candidate-grid, calibration, and metric protocol for Levels 1-4."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from msc_project.data.fabsa import format_pair_label
from msc_project.data.splits import build_heldout_aspect_split
from msc_project.evaluation.metrics import evaluate_pair_and_aspect
from msc_project.experiments.taxonomy_resources import (
    REPRESENTATIONS,
    load_minimal_descriptions,
    mixed_representation_sha256,
    render_candidate_claim,
)
from msc_project.experiments.unified_candidate_pairs import (
    CANDIDATE_SENTIMENTS,
    budget_sample,
    manifest_hash,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRECLOUD_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "taxonomy_generalisation_precloud_v1.json"
)
PAIR_KEY = ("row_uid", "candidate_aspect", "candidate_sentiment")
LEVELS = ("L1", "L2", "L3", "L4")
L3_CONDITIONS = ("NN", "DN", "ND", "DD")


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing taxonomy protocol resource: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Taxonomy protocol resource must contain an object: {path}")
    return value


def load_precloud_config(path: Path | None = None) -> dict[str, object]:
    config = _read_json(path or PRECLOUD_CONFIG_PATH)
    if config.get("protocol_id") != "taxonomy_generalisation_precloud_v1":
        raise ValueError("Unexpected taxonomy-generalisation protocol_id.")
    if config.get("status") != "implementation_preregistered_before_new_validation_or_test":
        raise ValueError("Taxonomy-generalisation protocol is not preregistered.")
    aspects = config.get("canonical_aspects")
    if not isinstance(aspects, list) or len(aspects) != 12 or len(set(aspects)) != 12:
        raise ValueError("Protocol must contain twelve unique canonical aspects.")
    if tuple(config.get("candidate_sentiments", ())) != CANDIDATE_SENTIMENTS:
        raise ValueError("Protocol sentiment order does not match the candidate-pair task.")
    return config


def canonical_aspects(config: Mapping[str, object] | None = None) -> tuple[str, ...]:
    source = config or load_precloud_config()
    return tuple(str(value) for value in source["canonical_aspects"])  # type: ignore[index]


@dataclass(frozen=True)
class TaxonomyFold:
    level: str
    fold_id: str
    heldout_aspects: tuple[str, ...]
    seen_aspects: tuple[str, ...]
    evaluation_aspects: tuple[str, ...]
    evaluation_label_scope: str
    conditions: tuple[str, ...]
    heldout_group: str | None = None

    def validate(self, all_aspects: Sequence[str]) -> None:
        canonical = tuple(all_aspects)
        canonical_set = set(canonical)
        heldout = set(self.heldout_aspects)
        seen = set(self.seen_aspects)
        evaluation = set(self.evaluation_aspects)
        if self.level not in LEVELS:
            raise ValueError(f"Unknown taxonomy level: {self.level!r}")
        if not self.fold_id:
            raise ValueError("fold_id must be non-empty.")
        if not heldout or heldout - canonical_set:
            raise ValueError("heldout_aspects must be a non-empty canonical subset.")
        if seen != canonical_set - heldout:
            raise ValueError("seen_aspects must be the exact complement of heldout_aspects.")
        if not evaluation or evaluation - canonical_set:
            raise ValueError("evaluation_aspects must be a non-empty canonical subset.")
        if self.evaluation_label_scope not in {"heldout", "full"}:
            raise ValueError("evaluation_label_scope must be heldout or full.")
        if self.level == "L1" and evaluation != heldout:
            raise ValueError("Level 1 must evaluate only the supplied held-out candidate.")
        if self.level != "L1" and tuple(self.evaluation_aspects) != canonical:
            raise ValueError("Levels 2-4 must evaluate all canonical candidates.")


def _fold(
    *,
    level: str,
    fold_id: str,
    heldout: Sequence[str],
    aspects: Sequence[str],
    evaluation_aspects: Sequence[str],
    evaluation_label_scope: str,
    conditions: Sequence[str],
    heldout_group: str | None = None,
) -> TaxonomyFold:
    heldout_tuple = tuple(heldout)
    heldout_set = set(heldout_tuple)
    value = TaxonomyFold(
        level=level,
        fold_id=fold_id,
        heldout_aspects=heldout_tuple,
        seen_aspects=tuple(aspect for aspect in aspects if aspect not in heldout_set),
        evaluation_aspects=tuple(evaluation_aspects),
        evaluation_label_scope=evaluation_label_scope,
        conditions=tuple(conditions),
        heldout_group=heldout_group,
    )
    value.validate(aspects)
    return value


def registered_folds(
    level: str,
    config: Mapping[str, object] | None = None,
) -> list[TaxonomyFold]:
    if level not in LEVELS:
        raise ValueError(f"Unknown taxonomy level: {level!r}")
    source = config or load_precloud_config()
    aspects = canonical_aspects(source)
    if level in {"L1", "L2"}:
        return [
            _fold(
                level=level,
                fold_id=f"{level.lower()}-a{index:02d}",
                heldout=(aspect,),
                aspects=aspects,
                evaluation_aspects=(aspect,) if level == "L1" else aspects,
                evaluation_label_scope="heldout" if level == "L1" else "full",
                conditions=("N", "D"),
            )
            for index, aspect in enumerate(aspects, start=1)
        ]
    if level == "L3":
        return [
            _fold(
                level=level,
                fold_id=f"l3-a{index:02d}-a{(index % len(aspects)) + 1:02d}",
                heldout=(aspect, aspects[index % len(aspects)]),
                aspects=aspects,
                evaluation_aspects=aspects,
                evaluation_label_scope="full",
                conditions=L3_CONDITIONS,
            )
            for index, aspect in enumerate(aspects, start=1)
        ]

    levels = source.get("levels")
    if not isinstance(levels, Mapping):
        raise ValueError("Protocol is missing levels.")
    l4 = levels.get("L4_D")
    groups = l4.get("heldout_groups") if isinstance(l4, Mapping) else None
    if not isinstance(groups, Mapping):
        raise ValueError("Protocol is missing Level 4 held-out groups.")
    folds: list[TaxonomyFold] = []
    for index, (group, children) in enumerate(groups.items(), start=1):
        if not isinstance(children, list):
            raise ValueError(f"Level 4 group {group!r} must contain a list.")
        folds.append(
            _fold(
                level=level,
                fold_id=f"l4-g{index:02d}",
                heldout=tuple(str(value) for value in children),
                aspects=aspects,
                evaluation_aspects=aspects,
                evaluation_label_scope="full",
                conditions=("N", "D"),
                heldout_group=str(group),
            )
        )
    return folds


def build_taxonomy_fold_splits(
    frame: pd.DataFrame,
    fold: TaxonomyFold,
    *,
    all_aspects: Sequence[str] | None = None,
) -> dict[str, pd.DataFrame]:
    canonical = tuple(all_aspects or canonical_aspects())
    fold.validate(canonical)
    required = {"original_split", "row_uid", "labels"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"FABSA frame is missing split columns: {missing}")
    if frame["row_uid"].astype(str).duplicated().any():
        raise ValueError("FABSA row_uid values must be globally unique.")

    splits = build_heldout_aspect_split(
        frame,
        fold.heldout_aspects,
        strategy="example_filtered",
        eval_label_scope=fold.evaluation_label_scope,
        eval_row_scope="all",
    )
    heldout = set(fold.heldout_aspects)
    train_original = {
        aspect
        for labels in splits["train"]["labels"]
        for aspect, _ in labels
    }
    train_supervision = {
        aspect
        for labels in splits["train"]["supervision_labels"]
        for aspect, _ in labels
    }
    if heldout & train_original or heldout & train_supervision:
        raise AssertionError("Held-out aspects survived example-filtered training.")
    for split_name in ("validation", "test"):
        expected = set(
            frame.loc[frame["original_split"] == split_name, "row_uid"].astype(str)
        )
        observed = set(splits[split_name]["row_uid"].astype(str))
        if observed != expected:
            raise AssertionError(f"{split_name} does not preserve every official row.")
    return splits


def _heldout_variants(fold: TaxonomyFold, condition: str) -> dict[str, str]:
    if condition not in fold.conditions:
        raise ValueError(
            f"Condition {condition!r} is not registered for {fold.fold_id}: {fold.conditions}"
        )
    if fold.level == "L3":
        if len(fold.heldout_aspects) != 2 or len(condition) != 2:
            raise AssertionError("Level 3 requires two ordered held-out aspects.")
        return {
            aspect: "minimal" if marker == "D" else "name_only"
            for aspect, marker in zip(fold.heldout_aspects, condition)
        }
    variant = "minimal" if condition == "D" else "name_only"
    return {aspect: variant for aspect in fold.heldout_aspects}


def candidate_representation_variants(
    fold: TaxonomyFold,
    condition: str,
) -> dict[str, str]:
    variants = {aspect: "minimal" for aspect in fold.seen_aspects}
    variants.update(_heldout_variants(fold, condition))
    return {aspect: variants[aspect] for aspect in fold.evaluation_aspects}


def build_candidate_table(
    fold: TaxonomyFold,
    condition: str,
    resource: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    descriptions = resource or load_minimal_descriptions(require_approved=False)
    variants = candidate_representation_variants(fold, condition)
    records: list[dict[str, object]] = []
    for aspect in fold.evaluation_aspects:
        variant = variants[aspect]
        if variant not in REPRESENTATIONS:
            raise AssertionError(f"Unsupported candidate representation: {variant}")
        for sentiment in CANDIDATE_SENTIMENTS:
            records.append(
                {
                    "candidate_aspect": aspect,
                    "candidate_sentiment": sentiment,
                    "candidate_text": render_candidate_claim(
                        aspect,
                        sentiment,
                        variant,
                        descriptions,
                    ),
                    "representation_variant": variant,
                    "is_seen": aspect in fold.seen_aspects,
                    "is_heldout": aspect in fold.heldout_aspects,
                }
            )
    result = pd.DataFrame.from_records(records)
    if result.duplicated(["candidate_aspect", "candidate_sentiment"]).any():
        raise AssertionError("Candidate table contains duplicate aspect-sentiment claims.")
    return result


def _normalise_pair_labels(row: pd.Series) -> tuple[str, ...]:
    if "supervision_pair_labels" in row and isinstance(
        row["supervision_pair_labels"], (list, tuple, set, frozenset)
    ):
        return tuple(str(value) for value in row["supervision_pair_labels"])
    if "supervision_labels" in row and isinstance(
        row["supervision_labels"], (list, tuple, set, frozenset)
    ):
        return tuple(
            format_pair_label(str(aspect), str(sentiment))
            for aspect, sentiment in row["supervision_labels"]
        )
    raise ValueError("Evaluation rows require supervision pair labels.")


def build_taxonomy_eval_grid(
    frame: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    fold_id: str,
    condition: str,
) -> pd.DataFrame:
    required_rows = {"row_uid", "text"}
    required_candidates = {
        "candidate_aspect",
        "candidate_sentiment",
        "candidate_text",
        "representation_variant",
        "is_seen",
        "is_heldout",
    }
    missing_rows = sorted(required_rows - set(frame.columns))
    missing_candidates = sorted(required_candidates - set(candidates.columns))
    if missing_rows or missing_candidates:
        raise ValueError(
            f"Evaluation grid inputs are incomplete: rows={missing_rows}, "
            f"candidates={missing_candidates}."
        )
    if frame.empty or candidates.empty:
        raise ValueError("Evaluation rows and candidates must be non-empty.")
    if frame["row_uid"].astype(str).duplicated().any():
        raise ValueError("Evaluation row_uid values must be unique.")
    if candidates.duplicated(["candidate_aspect", "candidate_sentiment"]).any():
        raise ValueError("Candidate aspect-sentiment identities must be unique.")

    records: list[dict[str, object]] = []
    ordered_rows = frame.assign(_uid=frame["row_uid"].astype(str)).sort_values(
        "_uid", kind="stable"
    )
    ordered_candidates = candidates.sort_values(
        ["candidate_aspect", "candidate_sentiment"], kind="stable"
    )
    for row_index, (_, row) in enumerate(ordered_rows.iterrows()):
        gold = set(_normalise_pair_labels(row))
        for candidate in ordered_candidates.to_dict(orient="records"):
            pair = format_pair_label(
                str(candidate["candidate_aspect"]),
                str(candidate["candidate_sentiment"]),
            )
            records.append(
                {
                    "fold_id": fold_id,
                    "condition": condition,
                    "row_index": int(row_index),
                    "row_uid": str(row["row_uid"]),
                    "text": "" if pd.isna(row["text"]) else str(row["text"]),
                    **candidate,
                    "target": int(pair in gold),
                    "pair_label": pair,
                }
            )
    result = pd.DataFrame.from_records(records)
    if result.duplicated(list(PAIR_KEY)).any():
        raise AssertionError("Evaluation grid is not unique on the pair key.")
    expected = len(frame) * len(candidates)
    if len(result) != expected:
        raise AssertionError("Evaluation grid is not the full review-candidate product.")
    return result.reset_index(drop=True)


def pair_identity_hash(frame: pd.DataFrame) -> str:
    required = {*PAIR_KEY, "target"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Pair identity frame is missing columns: {missing}")
    identities = (
        frame[[*PAIR_KEY, "target"]]
        .sort_values(list(PAIR_KEY), kind="stable")
        .to_dict(orient="records")
    )
    payload = json.dumps(
        identities,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parent(aspect: str) -> str:
    return aspect.split(":", maxsplit=1)[0].strip()


def _stable_choice(
    values: Sequence[str],
    *,
    seed: int,
    parts: Sequence[str],
) -> str:
    if not values:
        raise ValueError("A deterministic choice requires at least one value.")
    return min(
        values,
        key=lambda value: hashlib.sha256(
            "\x1f".join([str(seed), *parts, value]).encode("utf-8")
        ).hexdigest(),
    )


def build_taxonomy_training_manifest(
    frame: pd.DataFrame,
    fold: TaxonomyFold,
    resource: Mapping[str, object] | None = None,
    *,
    seed: int = 13,
) -> pd.DataFrame:
    required = {"row_uid", "text"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Training frame is missing columns: {missing}")
    if frame["row_uid"].astype(str).duplicated().any():
        raise ValueError("Training row_uid values must be unique.")
    descriptions = resource or load_minimal_descriptions(require_approved=False)
    candidate_set = set(fold.seen_aspects)
    records: list[dict[str, object]] = []

    def add(
        row_uid: str,
        text: str,
        aspect: str,
        sentiment: str,
        target: int,
        negative_type: str,
    ) -> None:
        records.append(
            {
                "row_uid": row_uid,
                "text": text,
                "candidate_aspect": aspect,
                "candidate_sentiment": sentiment,
                "candidate_text": render_candidate_claim(
                    aspect,
                    sentiment,
                    "minimal",
                    descriptions,
                ),
                "representation_variant": "minimal",
                "target": target,
                "negative_type": negative_type,
            }
        )

    ordered = frame.assign(_uid=frame["row_uid"].astype(str)).sort_values(
        "_uid", kind="stable"
    )
    for _, row in ordered.iterrows():
        row_uid = str(row["row_uid"])
        text = "" if pd.isna(row["text"]) else str(row["text"])
        gold_pairs = {
            tuple(str(value) for value in pair)
            for pair in row["supervision_labels"]
            if str(pair[0]) in candidate_set
        }
        gold_aspects = {aspect for aspect, _ in gold_pairs}
        for aspect, sentiment in sorted(gold_pairs):
            if sentiment not in CANDIDATE_SENTIMENTS:
                raise ValueError(f"Unknown sentiment {sentiment!r} in row {row_uid}.")
            add(row_uid, text, aspect, sentiment, 1, "gold_positive")
            for other_sentiment in CANDIDATE_SENTIMENTS:
                if (
                    other_sentiment != sentiment
                    and (aspect, other_sentiment) not in gold_pairs
                ):
                    add(
                        row_uid,
                        text,
                        aspect,
                        other_sentiment,
                        0,
                        "wrong_sentiment_same_aspect",
                    )

            absent = [
                candidate
                for candidate in fold.seen_aspects
                if candidate not in gold_aspects
            ]
            same_parent = [
                candidate
                for candidate in absent
                if _parent(candidate) == _parent(aspect)
            ]
            hard_aspect: str | None = None
            if same_parent:
                hard_aspect = _stable_choice(
                    same_parent,
                    seed=seed,
                    parts=("same_parent", row_uid, aspect, sentiment),
                )
                add(
                    row_uid,
                    text,
                    hard_aspect,
                    sentiment,
                    0,
                    "same_parent_absent_same_sentiment",
                )
            random_pool = [
                candidate for candidate in absent if candidate != hard_aspect
            ] or absent
            if random_pool:
                random_aspect = _stable_choice(
                    random_pool,
                    seed=seed,
                    parts=("random_absent", row_uid, aspect, sentiment),
                )
                add(
                    row_uid,
                    text,
                    random_aspect,
                    sentiment,
                    0,
                    "random_absent_same_sentiment",
                )

    columns = [
        "row_uid",
        "text",
        "candidate_aspect",
        "candidate_sentiment",
        "candidate_text",
        "representation_variant",
        "target",
        "negative_type",
    ]
    if not records:
        return pd.DataFrame(columns=columns)
    manifest = pd.DataFrame.from_records(records, columns=columns)
    target_counts = manifest.groupby(list(PAIR_KEY), sort=False)["target"].nunique()
    if (target_counts > 1).any():
        raise AssertionError("Training manifest contains conflicting pair targets.")
    manifest = manifest.sort_values(
        [*PAIR_KEY, "target"], ascending=[True, True, True, False], kind="stable"
    )
    manifest = manifest.drop_duplicates(list(PAIR_KEY), keep="first")
    if set(manifest["candidate_aspect"]) - candidate_set:
        raise AssertionError("Held-out aspects entered the training manifest.")
    return manifest.reset_index(drop=True)


def build_budgeted_training_manifest(
    frame: pd.DataFrame,
    fold: TaxonomyFold,
    resource: Mapping[str, object] | None = None,
    *,
    total_budget: int = 4096,
    positive_budget: int = 2048,
    seed: int = 13,
) -> pd.DataFrame:
    full = build_taxonomy_training_manifest(frame, fold, resource, seed=seed)
    return budget_sample(
        full,
        total_budget=total_budget,
        positive_budget=positive_budget,
        seed=seed,
    )


def _prediction_sets(
    scored_grid: pd.DataFrame,
    threshold: float,
    *,
    allowed_aspects: set[str],
) -> tuple[list[str], list[list[str]], list[list[str]], list[str]]:
    required = {*PAIR_KEY, "target", "score"}
    missing = sorted(required - set(scored_grid.columns))
    if missing:
        raise ValueError(f"Scored grid is missing columns: {missing}")
    frame = scored_grid[
        scored_grid["candidate_aspect"].astype(str).isin(allowed_aspects)
    ].copy()
    if frame.empty:
        raise ValueError("No scored candidates remain in the requested partition.")
    if frame.duplicated(list(PAIR_KEY)).any():
        raise ValueError("Scored grid contains duplicate pair identities.")
    scores = frame["score"].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite.")
    frame["gold_pair"] = np.where(frame["target"].astype(int) == 1, frame["pair_label"], None)
    frame["pred_pair"] = np.where(scores >= threshold, frame["pair_label"], None)
    row_uids = sorted(frame["row_uid"].astype(str).unique())
    grouped = frame.groupby(frame["row_uid"].astype(str), sort=False)
    gold_by_uid = {
        str(uid): sorted(value for value in group["gold_pair"] if value is not None)
        for uid, group in grouped
    }
    pred_by_uid = {
        str(uid): sorted(value for value in group["pred_pair"] if value is not None)
        for uid, group in grouped
    }
    classes = [
        format_pair_label(aspect, sentiment)
        for aspect in sorted(allowed_aspects)
        for sentiment in CANDIDATE_SENTIMENTS
    ]
    return (
        row_uids,
        [gold_by_uid[uid] for uid in row_uids],
        [pred_by_uid[uid] for uid in row_uids],
        classes,
    )


def evaluate_scored_grid(
    scored_grid: pd.DataFrame,
    threshold: float,
    *,
    seen_aspects: Iterable[str],
    heldout_aspects: Iterable[str],
) -> dict[str, object]:
    seen = set(str(value) for value in seen_aspects)
    unseen = set(str(value) for value in heldout_aspects)
    if not seen or not unseen or seen & unseen:
        raise ValueError("Seen and held-out aspect partitions must be non-empty and disjoint.")
    available = set(scored_grid["candidate_aspect"].astype(str))
    partitions = {
        "overall": available,
        "seen": available & seen,
        "unseen": available & unseen,
    }
    result: dict[str, object] = {"threshold": float(threshold)}
    for name, aspects in partitions.items():
        if not aspects:
            result[name] = None
            continue
        _, gold, predicted, classes = _prediction_sets(
            scored_grid,
            threshold,
            allowed_aspects=aspects,
        )
        metrics = evaluate_pair_and_aspect(gold, predicted, classes)
        metrics["examples"] = len(gold)
        result[name] = metrics
    seen_metrics = result["seen"]
    unseen_metrics = result["unseen"]
    if isinstance(seen_metrics, Mapping) and isinstance(unseen_metrics, Mapping):
        seen_f1 = float(seen_metrics["pair_micro_f1"])
        unseen_f1 = float(unseen_metrics["pair_micro_f1"])
        result["seen_unseen_harmonic_pair_micro_f1"] = (
            2 * seen_f1 * unseen_f1 / (seen_f1 + unseen_f1)
            if seen_f1 + unseen_f1
            else 0.0
        )
    else:
        result["seen_unseen_harmonic_pair_micro_f1"] = None
    return result


def strict_threshold_candidates(scores: Iterable[float]) -> list[float]:
    unique = np.unique(np.asarray(list(scores), dtype=float))
    if not np.isfinite(unique).all():
        raise ValueError("Threshold candidate scores must be finite.")
    midpoints = (
        (unique[:-1] + unique[1:]) / 2
        if len(unique) > 1
        else np.asarray([], dtype=float)
    )
    regular = np.arange(0.01, 1.00, 0.01, dtype=float)
    values = np.concatenate([regular, midpoints])
    values = values[(values >= 0.0) & (values <= 1.0)]
    return sorted({round(float(value), 12) for value in values})


@dataclass(frozen=True)
class StrictThresholdSelection:
    threshold: float
    metrics: dict[str, object]
    sweep: pd.DataFrame
    calibration_aspects: tuple[str, ...]


def select_strict_seen_threshold(
    validation_scores: pd.DataFrame,
    *,
    seen_aspects: Sequence[str],
    heldout_aspects: Sequence[str],
) -> StrictThresholdSelection:
    seen = tuple(str(value) for value in seen_aspects)
    heldout = set(str(value) for value in heldout_aspects)
    if not seen or set(seen) & heldout:
        raise ValueError("Strict calibration requires disjoint seen and held-out aspects.")
    available = set(validation_scores["candidate_aspect"].astype(str))
    missing_seen = sorted(set(seen) - available)
    if missing_seen:
        raise ValueError(f"Strict calibration is missing seen candidates: {missing_seen}")
    calibration = validation_scores[
        validation_scores["candidate_aspect"].astype(str).isin(seen)
    ].copy()
    if set(calibration["candidate_aspect"].astype(str)) & heldout:
        raise AssertionError("Held-out candidates entered strict threshold calibration.")

    rows: list[dict[str, float]] = []
    for threshold in strict_threshold_candidates(calibration["score"]):
        _, gold, predicted, classes = _prediction_sets(
            calibration,
            threshold,
            allowed_aspects=set(seen),
        )
        metrics = evaluate_pair_and_aspect(gold, predicted, classes)
        rows.append(
            {
                "threshold": float(threshold),
                "pair_micro_f1": float(metrics["pair_micro_f1"]),
                "pair_samples_f1": float(metrics["pair_samples_f1"]),
                "pair_micro_precision": float(metrics["pair_micro_precision"]),
                "presence_false_positive_rows_per_100": float(
                    metrics["presence_false_positive_rows_per_100"]
                ),
            }
        )
    if not rows:
        raise ValueError("Strict threshold search produced no candidates.")
    ranked = sorted(
        rows,
        key=lambda row: (
            row["pair_micro_f1"],
            row["pair_samples_f1"],
            row["pair_micro_precision"],
            -row["presence_false_positive_rows_per_100"],
            row["threshold"],
        ),
        reverse=True,
    )
    threshold = float(ranked[0]["threshold"])
    metrics = evaluate_scored_grid(
        calibration,
        threshold,
        seen_aspects=seen,
        heldout_aspects=heldout,
    )
    return StrictThresholdSelection(
        threshold=threshold,
        metrics=metrics,
        sweep=pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True),
        calibration_aspects=seen,
    )


def fold_manifest(
    fold: TaxonomyFold,
    splits: Mapping[str, pd.DataFrame],
    condition: str,
    candidates: pd.DataFrame,
    training_manifest: pd.DataFrame,
    resource: Mapping[str, object],
) -> dict[str, object]:
    variants = candidate_representation_variants(fold, condition)
    return {
        "fold_id": fold.fold_id,
        "level": fold.level,
        "heldout_group": fold.heldout_group,
        "heldout_aspects": list(fold.heldout_aspects),
        "seen_aspects": list(fold.seen_aspects),
        "evaluation_aspects": list(fold.evaluation_aspects),
        "evaluation_label_scope": fold.evaluation_label_scope,
        "condition": condition,
        "candidate_representation_variants": variants,
        "description_resource_version": resource["version"],
        "description_resource_sha256": resource["content_sha256"],
        "candidate_representation_sha256": mixed_representation_sha256(
            fold.evaluation_aspects,
            CANDIDATE_SENTIMENTS,
            variants,
            resource,
        ),
        "row_uids": {
            name: sorted(frame["row_uid"].astype(str).tolist())
            for name, frame in splits.items()
        },
        "candidate_pairs": int(len(candidates)),
        "training_pairs": int(len(training_manifest)),
        "training_manifest_sha256": manifest_hash(training_manifest),
    }
