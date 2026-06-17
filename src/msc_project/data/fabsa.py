from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd


SPLITS = ("train", "validation", "test")


def default_data_dir() -> Path:
    """Return the expected local FABSA export path."""
    env_path = os.environ.get("FABSA_DATA_DIR")
    if env_path:
        return Path(env_path)

    repo_root = Path(__file__).resolve().parents[3]
    return repo_root.parent / "Project_Preparation" / "Public_Datasets" / "FABSA"


def parse_labels_json(value: object) -> list[tuple[str, str]]:
    """Parse the human-readable FABSA labels."""
    if value is None or pd.isna(value):
        return []

    labels = json.loads(str(value))
    return [(str(aspect), str(sentiment)) for aspect, sentiment in labels]


def parse_label_codes(value: object) -> list[str]:
    """Parse machine-readable label codes stored as a Python-list string."""
    if value is None or pd.isna(value):
        return []

    codes = ast.literal_eval(str(value))
    return [str(code) for code in codes]


def format_pair_label(aspect: str, sentiment: str) -> str:
    return f"{aspect} | {sentiment}"


def pair_labels(labels: Iterable[tuple[str, str]]) -> list[str]:
    return [format_pair_label(aspect, sentiment) for aspect, sentiment in labels]


def aspect_labels(labels: Iterable[tuple[str, str]]) -> list[str]:
    return sorted({aspect for aspect, _ in labels})


def load_split(data_dir: Path | None, split: str) -> pd.DataFrame:
    """Load one FABSA split and add parsed label columns."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")

    root = data_dir or default_data_dir()
    path = root / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing FABSA split: {path}")

    df = pd.read_csv(path)
    df["labels"] = df["labels_json"].apply(parse_labels_json)
    df["label_codes_list"] = df["label_codes"].apply(parse_label_codes)
    df["pair_labels"] = df["labels"].apply(pair_labels)
    df["aspect_labels"] = df["labels"].apply(aspect_labels)
    df["text"] = df["text"].fillna("").astype(str)
    df["split"] = split
    return df


def load_splits(data_dir: Path | None = None, splits: Iterable[str] = SPLITS) -> dict[str, pd.DataFrame]:
    return {split: load_split(data_dir, split) for split in splits}


def unique_labels(frames: Iterable[pd.DataFrame], column: str = "pair_labels") -> list[str]:
    labels: set[str] = set()
    for frame in frames:
        for row_labels in frame[column]:
            labels.update(row_labels)
    return sorted(labels)

