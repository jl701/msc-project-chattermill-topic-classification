from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from msc_project.experiments.taxonomy_protocol import PAIR_KEY
from msc_project.experiments.taxonomy_two_stage import crossfit_decoder_comparison


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "strict_train_only_tfidf",
    "e5_base_v2",
    "distilbert_review_candidate_cross_encoder",
    "frozen_qwen_candidate_pair",
    "qwen_candidate_pair_qlora",
)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _threshold_artifact(
    formal_root: Path, method_id: str, fold_id: str
) -> dict[str, object]:
    candidates = list(
        (
            formal_root
            / "selection"
            / method_id
            / fold_id
            / "seed-0013"
        ).glob("*/threshold_transfer.json")
    )
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one final threshold artifact for {method_id}/{fold_id}; "
            f"found {len(candidates)}."
        )
    value = _read_json(candidates[0])
    if value.get("level") != "L2" or value.get("method_id") != method_id:
        raise ValueError("Threshold artifact identity mismatch.")
    return value


def _load_bound_score_grid(
    formal_root: Path, method_id: str, fold_id: str
) -> tuple[pd.DataFrame, dict[str, object]]:
    threshold = _threshold_artifact(formal_root, method_id, fold_id)
    contracts = threshold.get("validation_score_contracts")
    if not isinstance(contracts, dict) or set(contracts) != {"D"}:
        raise ValueError("Level 2 threshold must bind one D score contract.")
    contract = str(contracts["D"])
    score_dir = (
        formal_root
        / "taxonomy_generalisation_precloud_v1"
        / method_id
        / "L2"
        / fold_id
        / "seen-calibration"
        / "validation"
        / contract[:16]
    )
    manifests = sorted(score_dir.glob("shard-*.manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"Missing score manifests: {score_dir}")
    shards: list[pd.DataFrame] = []
    observed_indices: set[int] = set()
    declared_count: int | None = None
    for manifest_path in manifests:
        manifest = _read_json(manifest_path)
        manifest_contract = manifest.get("contract")
        if not isinstance(manifest_contract, dict):
            raise ValueError("Score manifest lacks a contract.")
        if manifest_contract.get("contract_sha256") != contract:
            raise ValueError("Score manifest contract mismatch.")
        if manifest_contract.get("split") != "validation":
            raise ValueError("Study B may load validation score shards only.")
        index = int(manifest["shard_index"])
        count = int(manifest["shard_count"])
        declared_count = count if declared_count is None else declared_count
        if count != declared_count or index in observed_indices:
            raise ValueError("Score shard indices are inconsistent.")
        observed_indices.add(index)
        csv_path = manifest_path.with_name(
            manifest_path.name.replace(".manifest.json", ".csv")
        )
        if _sha256(csv_path) != manifest.get("csv_sha256"):
            raise ValueError(f"Score CSV hash mismatch: {csv_path}")
        shard = pd.read_csv(csv_path)
        if len(shard) != int(manifest["rows"]):
            raise ValueError("Score shard row count mismatch.")
        shards.append(shard)
    if declared_count is None or observed_indices != set(range(declared_count)):
        raise ValueError("Score shard set is incomplete.")
    frame = pd.concat(shards, ignore_index=True)
    if frame.duplicated(list(PAIR_KEY)).any():
        raise ValueError("Merged score shards contain duplicate identities.")
    if set(frame["split"].astype(str)) != {"validation"}:
        raise ValueError("Merged scores contain a non-validation split.")
    if not frame["is_seen"].astype(bool).all() or frame["is_heldout"].astype(bool).any():
        raise ValueError("Held-out candidates entered Study B calibration evidence.")
    scores = frame["score"].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Merged score grid contains non-finite scores.")
    return frame, threshold


def run(formal_root: Path, output_root: Path) -> dict[str, object]:
    if "test" in {part.casefold() for part in output_root.parts}:
        raise ValueError("Study B output path must not use a test namespace.")
    per_fold: list[dict[str, object]] = []
    for method_id in METHODS:
        for fold_index in range(1, 13):
            fold_id = f"l2-a{fold_index:02d}"
            frame, threshold = _load_bound_score_grid(
                formal_root, method_id, fold_id
            )
            aspects = tuple(str(value) for value in threshold["calibration_aspects"])
            result = crossfit_decoder_comparison(
                frame, aspects=aspects, folds=5
            )
            pair_metrics = result["pair_decoder"]
            hierarchy_metrics = result["hierarchical_decoder"]
            per_fold.append(
                {
                    "method_id": method_id,
                    "fold_id": fold_id,
                    "review_rows": int(frame["row_uid"].nunique()),
                    "candidate_pairs": int(len(frame)),
                    "pair_micro_f1": float(pair_metrics["pair_micro_f1"]),
                    "hierarchical_pair_micro_f1": float(
                        hierarchy_metrics["pair_micro_f1"]
                    ),
                    "pair_micro_f1_delta": float(
                        hierarchy_metrics["pair_micro_f1"]
                        - pair_metrics["pair_micro_f1"]
                    ),
                    "pair_micro_precision": float(
                        pair_metrics["pair_micro_precision"]
                    ),
                    "hierarchical_pair_micro_precision": float(
                        hierarchy_metrics["pair_micro_precision"]
                    ),
                    "pair_micro_recall": float(pair_metrics["pair_micro_recall"]),
                    "hierarchical_pair_micro_recall": float(
                        hierarchy_metrics["pair_micro_recall"]
                    ),
                    "aspect_micro_f1": float(pair_metrics["aspect_micro_f1"]),
                    "hierarchical_aspect_micro_f1": float(
                        hierarchy_metrics["aspect_micro_f1"]
                    ),
                    "conditional_sentiment_accuracy": float(
                        pair_metrics["sentiment_accuracy_when_gold_aspect_predicted"]
                    ),
                    "hierarchical_conditional_sentiment_accuracy": float(
                        hierarchy_metrics[
                            "sentiment_accuracy_when_gold_aspect_predicted"
                        ]
                    ),
                    "thresholds": result["thresholds"],
                }
            )

    frame = pd.DataFrame.from_records(per_fold)
    aggregate: list[dict[str, object]] = []
    for method_id, group in frame.groupby("method_id", sort=False):
        delta = group["pair_micro_f1_delta"].to_numpy(dtype=float)
        aggregate.append(
            {
                "method_id": str(method_id),
                "folds": int(len(group)),
                "mean_pair_micro_f1": float(group["pair_micro_f1"].mean()),
                "mean_hierarchical_pair_micro_f1": float(
                    group["hierarchical_pair_micro_f1"].mean()
                ),
                "mean_pair_micro_f1_delta": float(delta.mean()),
                "median_pair_micro_f1_delta": float(np.median(delta)),
                "hierarchical_wins": int((delta > 0).sum()),
                "ties": int((delta == 0).sum()),
                "hierarchical_losses": int((delta < 0).sum()),
                "mean_pair_precision_delta": float(
                    (
                        group["hierarchical_pair_micro_precision"]
                        - group["pair_micro_precision"]
                    ).mean()
                ),
                "mean_pair_recall_delta": float(
                    (
                        group["hierarchical_pair_micro_recall"]
                        - group["pair_micro_recall"]
                    ).mean()
                ),
                "mean_aspect_micro_f1_delta": float(
                    (
                        group["hierarchical_aspect_micro_f1"]
                        - group["aspect_micro_f1"]
                    ).mean()
                ),
            }
        )
    summary = {
        "schema_version": "taxonomy_hierarchical_decoder_study_b_v1",
        "protocol_id": "taxonomy_two_stage_validation_v1",
        "split": "validation",
        "official_test_accessed": False,
        "level": "L2",
        "crossfit_folds": 5,
        "aggregate": aggregate,
        "per_fold": per_fold,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"
    table_path = output_root / "per_fold.csv"
    if summary_path.exists() or table_path.exists():
        raise FileExistsError("Study B outputs already exist; use a new output root.")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    frame.drop(columns=["thresholds"]).to_csv(table_path, index=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "experimental"
            / "taxonomy_generalisation_formal_v1"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "experimental"
            / "taxonomy_two_stage_validation_v1"
            / "study_b"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args.formal_root, args.output_root)
    print(json.dumps(result["aggregate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
