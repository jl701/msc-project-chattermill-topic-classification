"""Shared candidate-pair data construction for the isolated LOAO experiment.

The module deliberately keeps model-specific tokenisation and training out of the
data layer.  TF-IDF, DistilBERT, and Qwen therefore receive the same pair rows,
candidate wording, targets, and deterministic budget sample.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FROZEN_PROTOCOL_PATH = PROJECT_ROOT / "configs" / "experiments" / "loao_unified_candidate_pair_experimental_v1.json"
FROZEN_DESCRIPTION_PATH = PROJECT_ROOT / "configs" / "experiments" / "fabsa_aspect_descriptions_v1.json"
CANDIDATE_SENTIMENTS = ("negative", "neutral", "positive")
VARIANTS = ("control", "enhanced")
MANIFEST_KEY = ("row_uid", "candidate_aspect", "candidate_sentiment")

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen experiment resource: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Frozen experiment resource must contain a JSON object: {path}")
    return value


def load_experiment_config(path: Path | None = None) -> dict[str, object]:
    """Load and validate the preregistered unified candidate-pair protocol."""

    config = _read_json(path or FROZEN_PROTOCOL_PATH)
    if config.get("protocol_id") != "loao_unified_candidate_pair_experimental_v1":
        raise ValueError("Unexpected unified candidate-pair protocol_id.")
    if config.get("status") != "preregistered_before_model_execution":
        raise ValueError("The unified candidate-pair protocol is not marked preregistered.")

    task = config.get("task")
    manifest = config.get("training_manifest")
    outer = config.get("outer_protocol")
    if not isinstance(task, dict) or tuple(task.get("candidate_sentiments", ())) != CANDIDATE_SENTIMENTS:
        raise ValueError(f"Protocol candidate sentiments must be exactly {CANDIDATE_SENTIMENTS}.")
    if not isinstance(outer, dict) or outer.get("folds") != 12:
        raise ValueError("Unified candidate-pair protocol must retain all 12 LOAO folds.")
    if not isinstance(manifest, dict):
        raise ValueError("Protocol is missing training_manifest.")

    required_manifest_values = {
        "control_negatives_per_positive": 4,
        "model_budget_pairs_per_fold": 4096,
        "budget_positive_pairs": 2048,
        "budget_negative_pairs": 2048,
    }
    for key, expected in required_manifest_values.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Protocol training_manifest.{key} must be {expected!r}.")
    if tuple(manifest.get("deduplication_key", ())) != MANIFEST_KEY:
        raise ValueError(f"Protocol deduplication key must be exactly {MANIFEST_KEY}.")
    return config


def load_descriptions(path: Path | None = None) -> dict[str, object]:
    """Load the frozen taxonomy descriptions and validate all twelve labels."""

    descriptions = _read_json(path or FROZEN_DESCRIPTION_PATH)
    if descriptions.get("version") != "fabsa_aspect_descriptions_v1":
        raise ValueError("Unexpected frozen aspect-description version.")

    aspects = descriptions.get("aspects")
    sentiments = descriptions.get("sentiments")
    if not isinstance(aspects, dict) or len(aspects) != 12:
        raise ValueError("Frozen descriptions must contain exactly 12 aspects.")
    if not isinstance(sentiments, dict) or tuple(sentiments) != CANDIDATE_SENTIMENTS:
        raise ValueError(f"Frozen descriptions must contain sentiments in the order {CANDIDATE_SENTIMENTS}.")

    for aspect, payload in aspects.items():
        if not isinstance(aspect, str) or not aspect.strip() or ":" not in aspect:
            raise ValueError(f"Invalid canonical aspect name: {aspect!r}")
        if not isinstance(payload, dict):
            raise ValueError(f"Description for {aspect!r} must be an object.")
        if not all(payload.get(field) for field in ("definition", "cues", "boundary")):
            raise ValueError(f"Description for {aspect!r} is incomplete.")
        if not isinstance(payload["cues"], list) or not all(isinstance(cue, str) and cue for cue in payload["cues"]):
            raise ValueError(f"Description cues for {aspect!r} must be non-empty strings.")

    for sentiment, payload in sentiments.items():
        if not isinstance(payload, dict) or not payload.get("definition") or not payload.get("cues"):
            raise ValueError(f"Sentiment description for {sentiment!r} is incomplete.")
    return descriptions


def load_frozen_resources() -> tuple[dict[str, object], dict[str, object]]:
    """Load both frozen resources and cross-check their sentiment vocabulary."""

    config = load_experiment_config()
    descriptions = load_descriptions()
    configured = tuple(config["task"]["candidate_sentiments"])  # type: ignore[index]
    described = tuple(descriptions["sentiments"])  # type: ignore[arg-type]
    if configured != described:
        raise ValueError("Protocol and description sentiment vocabularies differ.")
    return config, descriptions


def _validated_descriptions(descriptions: Mapping[str, object] | None) -> Mapping[str, object]:
    if descriptions is None:
        _, loaded = load_frozen_resources()
        return loaded
    # Callers may reuse an already-loaded object, but it must still be structurally complete.
    aspects = descriptions.get("aspects")
    sentiments = descriptions.get("sentiments")
    if not isinstance(aspects, Mapping) or len(aspects) != 12:
        raise ValueError("Descriptions must contain exactly 12 aspects.")
    if not isinstance(sentiments, Mapping) or tuple(sentiments) != CANDIDATE_SENTIMENTS:
        raise ValueError("Descriptions have an invalid sentiment vocabulary.")
    return descriptions


def format_candidate_statement(
    aspect: str,
    sentiment: str,
    variant: str,
    descriptions: Mapping[str, object] | None = None,
) -> str:
    """Format the shared control or enhanced candidate statement."""

    if variant not in VARIANTS:
        raise ValueError(f"Unknown candidate-text variant: {variant!r}")
    if sentiment not in CANDIDATE_SENTIMENTS:
        raise ValueError(f"Unknown candidate sentiment: {sentiment!r}")
    resources = _validated_descriptions(descriptions)
    aspects = resources["aspects"]
    sentiments = resources["sentiments"]
    if aspect not in aspects:  # type: ignore[operator]
        raise ValueError(f"Aspect is not one of the 12 frozen labels: {aspect!r}")

    if variant == "control":
        return f"Aspect: {aspect}. Candidate sentiment: {sentiment}."

    aspect_payload = aspects[aspect]  # type: ignore[index]
    sentiment_payload = sentiments[sentiment]  # type: ignore[index]
    cues = ", ".join(aspect_payload["cues"])
    return (
        f"Aspect: {aspect}. Definition: {aspect_payload['definition']} "
        f"Lexical cues: {cues}. Boundary: {aspect_payload['boundary']} "
        f"Candidate sentiment: {sentiment}. "
        f"Sentiment definition: {sentiment_payload['definition']}"
    )


def _parent(aspect: str) -> str:
    return aspect.split(":", maxsplit=1)[0].strip()


def _normalised_tokens(text: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(text.casefold()))


def _aspect_tokens(aspect: str, descriptions: Mapping[str, object]) -> set[str]:
    payload = descriptions["aspects"][aspect]  # type: ignore[index]
    joined = " ".join(
        [aspect, str(payload["definition"]), str(payload["boundary"]), *[str(cue) for cue in payload["cues"]]]
    )
    return _normalised_tokens(joined)


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def _stable_digest(seed: int, *parts: object) -> str:
    encoded = json.dumps([int(seed), *[str(part) for part in parts]], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _ordered_random_candidates(
    values: Iterable[tuple[str, str]],
    seed: int,
    *context: object,
) -> list[tuple[str, str]]:
    return sorted(values, key=lambda value: (_stable_digest(seed, *context, *value), value))


def _row_labels(row: pd.Series) -> list[tuple[str, str]]:
    for column in ("supervision_labels", "labels"):
        if column in row.index:
            value = row[column]
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return []
            labels: list[tuple[str, str]] = []
            for item in value:
                if not isinstance(item, (tuple, list)) or len(item) != 2:
                    raise ValueError(f"{column} must contain (aspect, sentiment) pairs.")
                labels.append((str(item[0]), str(item[1])))
            return labels

    for column in ("supervision_pair_labels", "pair_labels"):
        if column in row.index:
            value = row[column]
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return []
            labels = []
            for item in value:
                parts = str(item).rsplit(" | ", maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(f"Malformed pair label: {item!r}")
                labels.append((parts[0], parts[1]))
            return labels
    raise ValueError("Frame must contain supervision_labels, labels, supervision_pair_labels, or pair_labels.")


def _validate_frame(frame: pd.DataFrame) -> None:
    missing = {"row_uid", "text"} - set(frame.columns)
    if missing:
        raise ValueError(f"Frame is missing required columns: {sorted(missing)}")
    if frame["row_uid"].isna().any() or (frame["row_uid"].astype(str).str.len() == 0).any():
        raise ValueError("row_uid values must be non-empty.")
    if frame["row_uid"].astype(str).duplicated().any():
        raise ValueError("row_uid values must be unique within a manifest input frame.")


def _validate_candidate_aspects(
    candidate_aspects: Sequence[str], descriptions: Mapping[str, object]
) -> tuple[str, ...]:
    candidates = tuple(dict.fromkeys(str(aspect) for aspect in candidate_aspects))
    if not candidates:
        raise ValueError("candidate_aspects must not be empty.")
    if len(candidates) != len(candidate_aspects):
        raise ValueError("candidate_aspects must not contain duplicates.")
    unknown = sorted(set(candidates) - set(descriptions["aspects"]))  # type: ignore[arg-type]
    if unknown:
        raise ValueError(f"candidate_aspects contains labels outside the frozen taxonomy: {unknown}")
    return candidates


def _hard_absent_aspect(
    source_aspect: str,
    absent_aspects: Sequence[str],
    descriptions: Mapping[str, object],
) -> tuple[str, str] | None:
    if not absent_aspects:
        return None
    siblings = [aspect for aspect in absent_aspects if _parent(aspect) == _parent(source_aspect)]
    pool = siblings or list(absent_aspects)
    source_tokens = _aspect_tokens(source_aspect, descriptions)
    selected = sorted(
        pool,
        key=lambda aspect: (-_jaccard(source_tokens, _aspect_tokens(aspect, descriptions)), aspect),
    )[0]
    negative_type = "same_parent_hard_absent" if siblings else "semantic_hard_absent"
    return selected, negative_type


def _manifest_record(
    *,
    row_uid: str,
    text: str,
    aspect: str,
    sentiment: str,
    target: int,
    negative_type: str,
    variant: str,
    descriptions: Mapping[str, object],
) -> dict[str, object]:
    return {
        "row_uid": row_uid,
        "text": text,
        "candidate_aspect": aspect,
        "candidate_sentiment": sentiment,
        "candidate_text": format_candidate_statement(aspect, sentiment, variant, descriptions),
        "parent_category": _parent(aspect),
        "target": int(target),
        "negative_type": negative_type,
        "variant": variant,
    }


def build_full_manifest(
    frame: pd.DataFrame,
    candidate_aspects: Sequence[str],
    variant: str,
    seed: int,
) -> pd.DataFrame:
    """Build all preregistered positive and negative training pairs.

    Only aspects in ``candidate_aspects`` can enter the result.  A LOAO caller
    therefore enforces the held-out boundary by omitting that label from this
    sequence.  Gold labels outside the permitted sequence are ignored rather
    than converted into negatives.
    """

    if variant not in VARIANTS:
        raise ValueError(f"Unknown candidate-text variant: {variant!r}")
    _validate_frame(frame)
    _, loaded = load_frozen_resources()
    descriptions: Mapping[str, object] = loaded
    candidates = _validate_candidate_aspects(candidate_aspects, descriptions)
    candidate_set = set(candidates)
    all_candidate_pairs = [(aspect, sentiment) for aspect in candidates for sentiment in CANDIDATE_SENTIMENTS]
    records: list[dict[str, object]] = []

    # Sorting makes the output invariant to input frame order.
    ordered_frame = frame.assign(_manifest_row_uid=frame["row_uid"].astype(str)).sort_values(
        "_manifest_row_uid", kind="stable"
    )
    for _, row in ordered_frame.iterrows():
        row_uid = str(row["row_uid"])
        text = "" if pd.isna(row["text"]) else str(row["text"])
        raw_gold = _row_labels(row)
        for _, sentiment in raw_gold:
            if sentiment not in CANDIDATE_SENTIMENTS:
                raise ValueError(f"Unknown gold sentiment {sentiment!r} in row {row_uid!r}.")
        gold = {(aspect, sentiment) for aspect, sentiment in raw_gold if aspect in candidate_set}
        gold_aspects = {aspect for aspect, _ in gold}

        for aspect, sentiment in sorted(gold):
            records.append(
                _manifest_record(
                    row_uid=row_uid,
                    text=text,
                    aspect=aspect,
                    sentiment=sentiment,
                    target=1,
                    negative_type="gold_positive",
                    variant=variant,
                    descriptions=descriptions,
                )
            )

            if variant == "control":
                non_gold = [pair for pair in all_candidate_pairs if pair not in gold]
                selected = _ordered_random_candidates(
                    non_gold, seed, "control", row_uid, aspect, sentiment
                )[:4]
                for negative_aspect, negative_sentiment in selected:
                    records.append(
                        _manifest_record(
                            row_uid=row_uid,
                            text=text,
                            aspect=negative_aspect,
                            sentiment=negative_sentiment,
                            target=0,
                            negative_type="random_non_gold_pair",
                            variant=variant,
                            descriptions=descriptions,
                        )
                    )
                continue

            # Multi-sentiment rows remain safe: a nominally "wrong" sentiment
            # that is another gold label is skipped and never assigned target 0.
            for wrong_sentiment in CANDIDATE_SENTIMENTS:
                pair = (aspect, wrong_sentiment)
                if wrong_sentiment == sentiment or pair in gold:
                    continue
                records.append(
                    _manifest_record(
                        row_uid=row_uid,
                        text=text,
                        aspect=aspect,
                        sentiment=wrong_sentiment,
                        target=0,
                        negative_type="wrong_sentiment_same_aspect",
                        variant=variant,
                        descriptions=descriptions,
                    )
                )

            absent_aspects = [candidate for candidate in candidates if candidate not in gold_aspects]
            hard = _hard_absent_aspect(aspect, absent_aspects, descriptions)
            hard_aspect: str | None = None
            if hard is not None:
                hard_aspect, hard_type = hard
                records.append(
                    _manifest_record(
                        row_uid=row_uid,
                        text=text,
                        aspect=hard_aspect,
                        sentiment=sentiment,
                        target=0,
                        negative_type=hard_type,
                        variant=variant,
                        descriptions=descriptions,
                    )
                )

            random_pool = [candidate for candidate in absent_aspects if candidate != hard_aspect]
            if not random_pool:
                random_pool = list(absent_aspects)
            if random_pool:
                random_aspect = _ordered_random_candidates(
                    [(candidate, sentiment) for candidate in random_pool],
                    seed,
                    "enhanced_random_absent",
                    row_uid,
                    aspect,
                    sentiment,
                )[0][0]
                records.append(
                    _manifest_record(
                        row_uid=row_uid,
                        text=text,
                        aspect=random_aspect,
                        sentiment=sentiment,
                        target=0,
                        negative_type="random_absent_same_sentiment",
                        variant=variant,
                        descriptions=descriptions,
                    )
                )

    columns = [
        "row_uid",
        "text",
        "candidate_aspect",
        "candidate_sentiment",
        "candidate_text",
        "parent_category",
        "target",
        "negative_type",
        "variant",
    ]
    if not records:
        return pd.DataFrame(columns=columns)

    manifest = pd.DataFrame.from_records(records, columns=columns)
    # Positives are emitted first, so keep='first' also makes any accidental
    # positive/negative collision resolve safely to positive supervision.
    manifest = manifest.drop_duplicates(list(MANIFEST_KEY), keep="first")
    conflicts = manifest.groupby(list(MANIFEST_KEY), sort=False)["target"].nunique()
    if (conflicts > 1).any():
        raise AssertionError("Conflicting targets survived manifest deduplication.")
    return manifest.sort_values(list(MANIFEST_KEY), kind="stable").reset_index(drop=True)


def build_eval_grid(
    frame: pd.DataFrame,
    target_aspect: str,
    variant: str,
) -> pd.DataFrame:
    """Expand every evaluation row to target-aspect x three sentiments."""

    if variant not in VARIANTS:
        raise ValueError(f"Unknown candidate-text variant: {variant!r}")
    _validate_frame(frame)
    _, loaded = load_frozen_resources()
    descriptions: Mapping[str, object] = loaded
    _validate_candidate_aspects([target_aspect], descriptions)
    records: list[dict[str, object]] = []

    ordered_frame = frame.assign(_manifest_row_uid=frame["row_uid"].astype(str)).sort_values(
        "_manifest_row_uid", kind="stable"
    )
    for row_index, (_, row) in enumerate(ordered_frame.iterrows()):
        row_uid = str(row["row_uid"])
        text = "" if pd.isna(row["text"]) else str(row["text"])
        gold = set(_row_labels(row))
        for _, sentiment in gold:
            if sentiment not in CANDIDATE_SENTIMENTS:
                raise ValueError(f"Unknown gold sentiment {sentiment!r} in row {row_uid!r}.")
        for sentiment in CANDIDATE_SENTIMENTS:
            target = int((target_aspect, sentiment) in gold)
            records.append(
                {
                    "row_index": int(row_index),
                    **_manifest_record(
                    row_uid=row_uid,
                    text=text,
                    aspect=target_aspect,
                    sentiment=sentiment,
                    target=target,
                    negative_type="gold_positive" if target else "evaluation_non_gold",
                    variant=variant,
                    descriptions=descriptions,
                    ),
                }
            )
    return pd.DataFrame.from_records(records).reset_index(drop=True)


def _round_robin_indices(frame: pd.DataFrame, seed: int, context: str) -> list[int]:
    groups: dict[str, list[int]] = {}
    for aspect, group in frame.groupby("candidate_aspect", sort=True):
        ordered = sorted(
            (int(index) for index in group.index),
            key=lambda index: (
                _stable_digest(
                    seed,
                    context,
                    frame.at[index, "row_uid"],
                    frame.at[index, "candidate_aspect"],
                    frame.at[index, "candidate_sentiment"],
                    frame.at[index, "target"],
                ),
                index,
            ),
        )
        groups[str(aspect)] = ordered

    selected: list[int] = []
    aspects = sorted(groups)
    depth = 0
    while True:
        added = False
        for aspect in aspects:
            values = groups[aspect]
            if depth < len(values):
                selected.append(values[depth])
                added = True
        if not added:
            return selected
        depth += 1


def budget_sample(
    manifest: pd.DataFrame,
    total_budget: int = 4096,
    positive_budget: int = 2048,
    seed: int = 13,
) -> pd.DataFrame:
    """Select a deterministic, aspect-balanced training budget.

    The positive and negative quotas are attempted first.  If either class has
    fewer available rows, the unused capacity is deterministically filled from
    the other class so that the total budget is used whenever possible.
    """

    if total_budget <= 0:
        raise ValueError("total_budget must be positive.")
    if positive_budget < 0 or positive_budget > total_budget:
        raise ValueError("positive_budget must be between zero and total_budget.")
    required = {*MANIFEST_KEY, "target"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    if not manifest.empty and not manifest["target"].isin([0, 1]).all():
        raise ValueError("Manifest targets must be binary 0/1 values.")
    if manifest.duplicated(list(MANIFEST_KEY)).any():
        raise ValueError(f"Manifest must be unique on {MANIFEST_KEY} before budget sampling.")

    frame = manifest.reset_index(drop=True).copy()
    positive = frame[frame["target"] == 1]
    negative = frame[frame["target"] == 0]
    positive_order = _round_robin_indices(positive, seed, "budget_positive")
    negative_order = _round_robin_indices(negative, seed, "budget_negative")

    negative_budget = total_budget - positive_budget
    selected = positive_order[:positive_budget] + negative_order[:negative_budget]
    selected_set = set(selected)
    remaining_capacity = min(total_budget, len(frame)) - len(selected)
    if remaining_capacity > 0:
        leftovers = frame.loc[[index for index in frame.index if index not in selected_set]]
        fill_order = _round_robin_indices(leftovers, seed, "budget_fill")
        selected.extend(fill_order[:remaining_capacity])

    result = frame.loc[selected].copy().reset_index(drop=True)
    result.insert(0, "budget_order", range(len(result)))
    return result


def _json_scalar(value: object) -> object:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if hasattr(value, "item"):
        return value.item()  # type: ignore[no-any-return, union-attr]
    if isinstance(value, Path):
        return str(value)
    return value


def manifest_hash(manifest: pd.DataFrame) -> str:
    """Return a canonical SHA-256 hex digest, independent of row order."""

    columns = sorted(str(column) for column in manifest.columns)
    records = [
        {column: _json_scalar(row[column]) for column in columns}
        for row in manifest.to_dict(orient="records")
    ]
    records.sort(
        key=lambda record: json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    )
    payload = {"columns": columns, "records": records}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
