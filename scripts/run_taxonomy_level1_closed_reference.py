from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MultiLabelBinarizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.data.fabsa import default_data_dir, format_pair_label, load_split
from msc_project.experiments.taxonomy_post_supervisor import score_frame_sha256
from msc_project.experiments.taxonomy_protocol import canonical_aspects
from msc_project.experiments.taxonomy_two_stage import (
    evaluate_prediction_mask,
    pair_prediction_mask,
    select_pair_threshold,
)
from msc_project.experiments.unified_candidate_pairs import CANDIDATE_SENTIMENTS


CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "taxonomy_level1_closed_reference_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _feature_union(config: dict[str, object]) -> FeatureUnion:
    features = config["model"]["features"]
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer=str(features["word_analyzer"]),
                    ngram_range=tuple(int(value) for value in features["word_ngram_range"]),
                    min_df=int(features["min_df"]),
                    max_features=int(features["word_max_features"]),
                    sublinear_tf=bool(features["sublinear_tf"]),
                ),
            ),
            (
                "character",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer=str(features["character_analyzer"]),
                    ngram_range=tuple(
                        int(value) for value in features["character_ngram_range"]
                    ),
                    min_df=int(features["min_df"]),
                    max_features=int(features["character_max_features"]),
                    sublinear_tf=bool(features["sublinear_tf"]),
                ),
            ),
        ]
    )


def build_score_grid(
    row_uids: pd.Series,
    classes: list[str],
    targets: np.ndarray,
    scores: np.ndarray,
) -> pd.DataFrame:
    if targets.shape != scores.shape or targets.shape != (len(row_uids), len(classes)):
        raise ValueError("Level 1 target and score matrices are not aligned.")
    aspects, sentiments = zip(
        *(label.split(" | ", maxsplit=1) for label in classes)
    )
    frame = pd.DataFrame(
        {
            "row_uid": np.repeat(row_uids.astype(str).to_numpy(), len(classes)),
            "candidate_aspect": np.tile(np.asarray(aspects, dtype=object), len(row_uids)),
            "candidate_sentiment": np.tile(
                np.asarray(sentiments, dtype=object), len(row_uids)
            ),
            "representation_variant": "closed_taxonomy_review_classifier",
            "pair_label": np.tile(np.asarray(classes, dtype=object), len(row_uids)),
            "target": targets.reshape(-1).astype(int),
            "score": scores.reshape(-1).astype(float),
            "aspect_score": scores.reshape(-1).astype(float),
            "sentiment_score": scores.reshape(-1).astype(float),
        }
    )
    if frame.duplicated(
        ["row_uid", "candidate_aspect", "candidate_sentiment"]
    ).any() or not np.isfinite(frame["score"]).all():
        raise ValueError("Level 1 score grid contains duplicate or non-finite values.")
    return frame


def run(args: argparse.Namespace) -> dict[str, object]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if (
        config.get("protocol_id") != "taxonomy_level1_closed_reference_v1"
        or config.get("status") != "preregistered_before_execution"
        or config["data_contract"].get("include_official_test") is not False
    ):
        raise ValueError("Level 1 protocol is not safely preregistered.")
    output_root = args.output_root.resolve()
    result_path = output_root / "result.json"
    protocol_sha256 = _sha256(CONFIG_PATH)
    data_dir = args.data_dir.resolve()
    input_hashes = {
        split: _sha256(data_dir / f"{split}.csv")
        for split in ("train", "validation")
    }
    expected_hashes = {
        "train": str(config["data_contract"]["train_csv_sha256"]),
        "validation": str(config["data_contract"]["validation_csv_sha256"]),
    }
    if input_hashes != expected_hashes:
        raise ValueError(
            f"Level 1 input hash mismatch: expected={expected_hashes}, observed={input_hashes}."
        )
    if result_path.is_file():
        if not args.resume:
            raise FileExistsError(f"Refusing to overwrite Level 1 result: {result_path}")
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            existing.get("protocol_sha256") != protocol_sha256
            or existing.get("input_hashes") != input_hashes
            or existing.get("test_contract_count") != 0
        ):
            raise ValueError("Level 1 resume contract conflict.")
        print(json.dumps(existing["selected"], indent=2), flush=True)
        return existing

    started = time.perf_counter()
    train = load_split(data_dir, "train")
    validation = load_split(data_dir, "validation")
    train["row_uid"] = "train:" + train["id"].astype(str)
    validation["row_uid"] = "validation:" + validation["id"].astype(str)
    aspects = canonical_aspects()
    classes = [
        format_pair_label(aspect, sentiment)
        for aspect in aspects
        for sentiment in CANDIDATE_SENTIMENTS
    ]
    if len(classes) != 36 or len(set(classes)) != 36:
        raise AssertionError("Level 1 requires exactly 36 canonical labels.")
    binarizer = MultiLabelBinarizer(classes=classes)
    binarizer.fit([classes])
    y_train = binarizer.transform(train["pair_labels"])
    y_validation = binarizer.transform(validation["pair_labels"])

    vectorizer = _feature_union(config)
    x_train = vectorizer.fit_transform(train["text"].astype(str))
    x_validation = vectorizer.transform(validation["text"].astype(str))
    sweep_rows: list[dict[str, object]] = []
    score_matrices: dict[float, np.ndarray] = {}
    score_grids: dict[float, pd.DataFrame] = {}
    for c_value in (float(value) for value in config["model"]["c_grid"]):
        model = OneVsRestClassifier(
            LogisticRegression(
                C=c_value,
                class_weight="balanced",
                max_iter=int(config["model"]["max_iter"]),
                solver=str(config["model"]["solver"]),
                random_state=int(config["model"]["seed"]),
            )
        )
        model.fit(x_train, y_train)
        probabilities = np.asarray(model.predict_proba(x_validation), dtype=float)
        if probabilities.shape != y_validation.shape or not np.isfinite(probabilities).all():
            raise ValueError("Level 1 produced invalid probabilities.")
        grid = build_score_grid(
            validation["row_uid"], classes, y_validation, probabilities
        )
        selection = select_pair_threshold(grid)
        mask = pair_prediction_mask(grid, selection.threshold)
        metrics = evaluate_prediction_mask(grid, mask, aspects=aspects)
        config_id = f"word_char_tfidf_balanced_logreg_c{c_value:g}"
        sweep_rows.append(
            {
                "config_id": config_id,
                "c": c_value,
                "threshold": float(selection.threshold),
                "threshold_candidates": int(len(selection.sweep)),
                "score_sha256": score_frame_sha256(grid),
                **metrics,
            }
        )
        score_matrices[c_value] = probabilities
        score_grids[c_value] = grid

    ranked = sorted(
        sweep_rows,
        key=lambda row: (
            float(row["pair_micro_f1"]),
            float(row["pair_samples_f1"]),
            float(row["pair_micro_precision"]),
            -float(row["pair_false_positive_rows_per_100"]),
            -float(row["c"]),
            float(row["threshold"]),
            str(row["config_id"]),
        ),
        reverse=True,
    )
    selected = dict(ranked[0])
    selected_c = float(selected["c"])
    selected_grid = score_grids[selected_c]
    selected_mask = pair_prediction_mask(selected_grid, float(selected["threshold"]))
    selected_predictions = selected_mask.to_numpy(dtype=int).reshape(
        len(validation), len(classes)
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        y_validation,
        selected_predictions,
        average=None,
        zero_division=0,
    )
    per_label = pd.DataFrame(
        {
            "pair_label": classes,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(int),
        }
    )

    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(sweep_rows).to_csv(
        output_root / "configuration_sweep.csv", index=False
    )
    per_label.to_csv(output_root / "selected_per_label.csv", index=False)
    np.savez_compressed(
        output_root / "selected_scores.npz",
        row_uid=validation["row_uid"].astype(str).to_numpy(),
        pair_label=np.asarray(classes, dtype=str),
        target=y_validation.astype(np.uint8),
        score=score_matrices[selected_c].astype(np.float32),
    )
    result: dict[str, object] = {
        "schema_version": "taxonomy_level1_closed_reference_result_v1",
        "protocol_id": str(config["protocol_id"]),
        "protocol_sha256": protocol_sha256,
        "input_hashes": input_hashes,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "aspect_count": len(aspects),
        "pair_label_count": len(classes),
        "feature_dimensions": int(x_train.shape[1]),
        "selection_partition": "validation_all_36_labels",
        "ensure_one": False,
        "top_k_fallback": False,
        "selected": selected,
        "configuration_sweep": sweep_rows,
        "failure_count": 0,
        "non_finite_value_count": 0,
        "test_contract_count": 0,
        "seconds": float(time.perf_counter() - started),
    }
    _write_json(result_path, result)
    print(json.dumps(_jsonable(selected), indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "experimental"
        / "taxonomy_level1_closed_reference_v1",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
