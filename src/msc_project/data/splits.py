from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from msc_project.data.fabsa import SPLITS, aspect_labels, load_split, pair_labels


DEFAULT_HELDOUT_ASPECTS = [
    "Account management: Account access",
    "Company brand: Competitor",
    "Value: Discounts promotions",
]


@dataclass(frozen=True)
class SplitCandidate:
    validation_orgs: tuple[int, ...]
    test_orgs: tuple[int, ...]
    score: float
    validation_rows: int
    test_rows: int
    train_rows: int
    validation_aspects: int
    test_aspects: int
    train_aspects: int


def load_all_fabsa(data_dir: Path | None = None) -> pd.DataFrame:
    frames = []
    for split in SPLITS:
        frame = load_split(data_dir, split).copy()
        frame["original_split"] = split
        frame["row_uid"] = frame["original_split"].astype(str) + ":" + frame["id"].astype(str)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def flatten(rows: Iterable[Iterable[str]]) -> list[str]:
    values: list[str] = []
    for row in rows:
        values.extend(row)
    return values


def all_aspects(frame: pd.DataFrame) -> list[str]:
    return sorted(set(flatten(frame["aspect_labels"])))


def all_pairs(frame: pd.DataFrame) -> list[str]:
    return sorted(set(flatten(frame["pair_labels"])))


def has_any_aspect(labels: list[tuple[str, str]], aspects: set[str]) -> bool:
    return any(aspect in aspects for aspect, _ in labels)


def filter_labels(labels: list[tuple[str, str]], aspects: set[str], keep: str) -> list[tuple[str, str]]:
    if keep == "heldout":
        return [(aspect, sentiment) for aspect, sentiment in labels if aspect in aspects]
    if keep == "seen":
        return [(aspect, sentiment) for aspect, sentiment in labels if aspect not in aspects]
    raise ValueError(f"Unknown keep mode: {keep}")


def with_supervision_labels(frame: pd.DataFrame, labels_column: str = "labels") -> pd.DataFrame:
    frame = frame.copy()
    frame["supervision_labels"] = frame[labels_column]
    frame["supervision_pair_labels"] = frame["supervision_labels"].apply(pair_labels)
    frame["supervision_aspect_labels"] = frame["supervision_labels"].apply(aspect_labels)
    return frame


def summarise_frame(frame: pd.DataFrame, label_column: str = "pair_labels") -> dict[str, object]:
    aspect_column = "aspect_labels" if label_column == "pair_labels" else "supervision_aspect_labels"
    labels = flatten(frame[label_column]) if len(frame) else []
    aspects = flatten(frame[aspect_column]) if len(frame) else []
    return {
        "rows": int(len(frame)),
        "organisations": int(frame["org_index"].nunique()) if len(frame) else 0,
        "aspects": int(len(set(aspects))),
        "pair_labels": int(len(set(labels))),
        "labels_per_row": {str(k): int(v) for k, v in sorted(Counter(len(row) for row in frame[label_column]).items())},
        "top_pair_labels": dict(Counter(labels).most_common(20)),
        "top_aspects": dict(Counter(aspects).most_common(20)),
    }


def org_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for org, group in frame.groupby("org_index"):
        rows.append(
            {
                "org_index": int(org),
                "rows": int(len(group)),
                "aspects": len(all_aspects(group)),
                "pair_labels": len(all_pairs(group)),
                "industries": "; ".join(sorted(group["industry"].dropna().unique())),
                "data_sources": "; ".join(sorted(group["data_source"].dropna().unique())),
            }
        )
    return pd.DataFrame(rows).sort_values(["rows", "aspects"], ascending=[False, False])


def aspect_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for aspect in all_aspects(frame):
        mask = frame["labels"].apply(lambda labels: any(row_aspect == aspect for row_aspect, _ in labels))
        group = frame[mask]
        sentiments = Counter(sentiment for labels in group["labels"] for row_aspect, sentiment in labels if row_aspect == aspect)
        rows.append(
            {
                "aspect": aspect,
                "rows": int(len(group)),
                "organisations": int(group["org_index"].nunique()),
                "positive": int(sentiments.get("positive", 0)),
                "negative": int(sentiments.get("negative", 0)),
                "neutral": int(sentiments.get("neutral", 0)),
            }
        )
    return pd.DataFrame(rows).sort_values("rows", ascending=False)


def choose_org_split_candidates(
    frame: pd.DataFrame,
    target_validation_ratio: float = 0.15,
    target_test_ratio: float = 0.20,
    max_validation_orgs: int = 2,
    max_test_orgs: int = 2,
    min_org_aspects: int = 10,
    top_k: int = 20,
) -> list[SplitCandidate]:
    total_rows = len(frame)
    orgs = [
        int(row.org_index)
        for row in org_summary(frame).itertuples(index=False)
        if row.aspects >= min_org_aspects
    ]
    all_frame_aspects = set(all_aspects(frame))
    candidates: list[SplitCandidate] = []

    validation_sets = []
    test_sets = []
    for size in range(1, max_validation_orgs + 1):
        validation_sets.extend(itertools.combinations(orgs, size))
    for size in range(1, max_test_orgs + 1):
        test_sets.extend(itertools.combinations(orgs, size))

    for validation_orgs in validation_sets:
        validation_set = set(validation_orgs)
        validation_frame = frame[frame["org_index"].isin(validation_set)]

        for test_orgs in test_sets:
            test_set = set(test_orgs)
            if validation_set & test_set:
                continue

            test_frame = frame[frame["org_index"].isin(test_set)]
            train_frame = frame[~frame["org_index"].isin(validation_set | test_set)]

            validation_missing = len(all_frame_aspects - set(all_aspects(validation_frame)))
            test_missing = len(all_frame_aspects - set(all_aspects(test_frame)))
            train_missing = len(all_frame_aspects - set(all_aspects(train_frame)))

            validation_ratio = len(validation_frame) / total_rows
            test_ratio = len(test_frame) / total_rows
            score = (
                abs(validation_ratio - target_validation_ratio)
                + abs(test_ratio - target_test_ratio)
                + 0.08 * validation_missing
                + 0.08 * test_missing
                + 0.20 * train_missing
            )

            candidates.append(
                SplitCandidate(
                    validation_orgs=tuple(sorted(validation_orgs)),
                    test_orgs=tuple(sorted(test_orgs)),
                    score=float(score),
                    validation_rows=int(len(validation_frame)),
                    test_rows=int(len(test_frame)),
                    train_rows=int(len(train_frame)),
                    validation_aspects=len(all_aspects(validation_frame)),
                    test_aspects=len(all_aspects(test_frame)),
                    train_aspects=len(all_aspects(train_frame)),
                )
            )

    return sorted(candidates, key=lambda item: item.score)[:top_k]


def build_heldout_org_split(
    frame: pd.DataFrame,
    validation_orgs: Iterable[int],
    test_orgs: Iterable[int],
) -> dict[str, pd.DataFrame]:
    validation_set = set(int(org) for org in validation_orgs)
    test_set = set(int(org) for org in test_orgs)
    if validation_set & test_set:
        raise ValueError("Validation and test organisations overlap.")

    splits = {
        "train": frame[~frame["org_index"].isin(validation_set | test_set)].copy(),
        "validation": frame[frame["org_index"].isin(validation_set)].copy(),
        "test": frame[frame["org_index"].isin(test_set)].copy(),
    }
    return {name: with_supervision_labels(split) for name, split in splits.items()}


def build_heldout_aspect_split(
    frame: pd.DataFrame,
    heldout_aspects: Iterable[str],
    strategy: str,
    eval_label_scope: str = "heldout",
    eval_row_scope: str = "containing_heldout",
) -> dict[str, pd.DataFrame]:
    heldout = set(heldout_aspects)
    if strategy not in {"label_masked", "example_filtered"}:
        raise ValueError(f"Unknown held-out-aspect strategy: {strategy}")
    if eval_label_scope not in {"heldout", "full"}:
        raise ValueError(f"Unknown eval label scope: {eval_label_scope}")
    if eval_row_scope not in {"containing_heldout", "all"}:
        raise ValueError(f"Unknown eval row scope: {eval_row_scope}")

    official_train = frame[frame["original_split"] == "train"].copy()
    official_validation = frame[frame["original_split"] == "validation"].copy()
    official_test = frame[frame["original_split"] == "test"].copy()

    if strategy == "label_masked":
        official_train["supervision_labels"] = official_train["labels"].apply(lambda labels: filter_labels(labels, heldout, "seen"))
        train = official_train[official_train["supervision_labels"].apply(len) > 0].copy()
    else:
        train = official_train[~official_train["labels"].apply(lambda labels: has_any_aspect(labels, heldout))].copy()
        train["supervision_labels"] = train["labels"]

    eval_splits = {}
    for name, split in [("validation", official_validation), ("test", official_test)]:
        if eval_row_scope == "containing_heldout":
            selected = split[split["labels"].apply(lambda labels: has_any_aspect(labels, heldout))].copy()
        else:
            selected = split.copy()
        if eval_label_scope == "heldout":
            selected["supervision_labels"] = selected["labels"].apply(lambda labels: filter_labels(labels, heldout, "heldout"))
        else:
            selected["supervision_labels"] = selected["labels"]
        if eval_row_scope == "containing_heldout":
            selected = selected[selected["supervision_labels"].apply(len) > 0].copy()
        eval_splits[name] = selected.copy()

    splits = {"train": train, **eval_splits}
    for split in splits.values():
        split["supervision_pair_labels"] = split["supervision_labels"].apply(pair_labels)
        split["supervision_aspect_labels"] = split["supervision_labels"].apply(aspect_labels)
    return splits


def split_manifest(
    protocol: str,
    splits: dict[str, pd.DataFrame],
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    manifest = {
        "protocol": protocol,
        "splits": {
            name: {
                "summary": summarise_frame(frame, "supervision_pair_labels"),
                "organisations": sorted(int(org) for org in frame["org_index"].unique()),
                "row_uids": frame["row_uid"].tolist(),
            }
            for name, frame in splits.items()
        },
        "leakage_checks": leakage_checks(splits),
    }
    if extra:
        manifest.update(extra)
    return manifest


def leakage_checks(splits: dict[str, pd.DataFrame]) -> dict[str, object]:
    org_sets = {name: set(frame["org_index"].astype(int)) for name, frame in splits.items()}
    row_sets = {name: set(frame["row_uid"]) for name, frame in splits.items()}
    aspect_sets = {
        name: set(flatten(frame["supervision_aspect_labels"])) if "supervision_aspect_labels" in frame else set()
        for name, frame in splits.items()
    }
    pair_sets = {
        name: set(flatten(frame["supervision_pair_labels"])) if "supervision_pair_labels" in frame else set()
        for name, frame in splits.items()
    }
    return {
        "row_overlap": {
            f"{left}_{right}": len(row_sets[left] & row_sets[right])
            for left, right in itertools.combinations(row_sets, 2)
        },
        "organisation_overlap": {
            f"{left}_{right}": len(org_sets[left] & org_sets[right])
            for left, right in itertools.combinations(org_sets, 2)
        },
        "supervision_aspect_overlap": {
            f"{left}_{right}": {
                "count": len(aspect_sets[left] & aspect_sets[right]),
                "values": sorted(aspect_sets[left] & aspect_sets[right]),
            }
            for left, right in itertools.combinations(aspect_sets, 2)
        },
        "supervision_pair_label_overlap": {
            f"{left}_{right}": {
                "count": len(pair_sets[left] & pair_sets[right]),
                "values": sorted(pair_sets[left] & pair_sets[right]),
            }
            for left, right in itertools.combinations(pair_sets, 2)
        },
    }


def dataframe_for_summary(frame: pd.DataFrame, label_column: str = "supervision_pair_labels") -> pd.DataFrame:
    rows = []
    for split_name, group in frame.groupby("split_name"):
        summary = summarise_frame(group, label_column)
        rows.append({"split": split_name, **{key: value for key, value in summary.items() if not isinstance(value, dict)}})
    return pd.DataFrame(rows)
