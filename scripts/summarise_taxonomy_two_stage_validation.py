"""Combine the completed B/C summaries without crossing their evidence scopes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "strict_train_only_tfidf",
    "e5_base_v2",
    "frozen_qwen_candidate_pair",
)
VARIANT_COMPARISONS = (
    ("name_and_description", "name_only"),
    ("description_only", "name_and_description"),
    ("description_only", "name_only"),
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _paired_variant_deltas(records: pd.DataFrame) -> list[dict[str, Any]]:
    heldout = records[records["partition"].eq("heldout")].copy()
    output: list[dict[str, Any]] = []
    for left, right in VARIANT_COMPARISONS:
        left_rows = heldout[heldout["variant"].eq(left)].set_index("fold_id")
        right_rows = heldout[heldout["variant"].eq(right)].set_index("fold_id")
        if set(left_rows.index) != set(right_rows.index) or len(left_rows) != 12:
            raise ValueError("Variant comparison does not cover twelve matched folds.")
        delta = (
            left_rows.loc[sorted(left_rows.index), "pair_micro_f1"].to_numpy(float)
            - right_rows.loc[
                sorted(right_rows.index), "pair_micro_f1"
            ].to_numpy(float)
        )
        output.append(
            {
                "left_variant": left,
                "right_variant": right,
                "mean_pair_micro_f1_delta": float(delta.mean()),
                "median_pair_micro_f1_delta": float(np.median(delta)),
                "wins": int(np.sum(delta > 1e-12)),
                "ties": int(np.sum(np.abs(delta) <= 1e-12)),
                "losses": int(np.sum(delta < -1e-12)),
            }
        )
    return output


def summarise(output_root: Path) -> dict[str, Any]:
    b = _read(output_root / "study_b" / "summary.json")
    if b.get("official_test_accessed") is not False:
        raise ValueError("Study B test contract is invalid.")
    c: dict[str, Any] = {}
    for method in METHODS:
        summary = _read(output_root / "study_c" / method / "summary.json")
        if (
            int(summary.get("completed_folds", -1)) != 12
            or int(summary.get("failed_folds", -1)) != 0
            or int(summary.get("test_contract_count", -1)) != 0
        ):
            raise ValueError(f"Study C is incomplete for {method}.")
        aggregate = pd.DataFrame.from_records(summary["aggregate"])
        records = pd.DataFrame.from_records(summary["records"])
        heldout = aggregate[aggregate["partition"].eq("heldout")].copy()
        c[method] = {
            "heldout_fold_mean_metrics": heldout.to_dict(orient="records"),
            "paired_representation_deltas": _paired_variant_deltas(records),
        }
    return {
        "schema_version": "taxonomy_two_stage_combined_summary_v1",
        "protocol_id": "taxonomy_two_stage_validation_v1",
        "failure_count": 0,
        "official_test_accessed": False,
        "study_b_scope": "seen-validation decoder-only cross-fit diagnostic",
        "study_b": b["aggregate"],
        "study_c_scope": "held-out validation true two-stage evaluation",
        "study_c": c,
        "comparison_boundary": (
            "Study B and Study C metrics are not direct deltas: B evaluates reused "
            "seen-candidate score shards, whereas C evaluates held-out aspects."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "experimental"
        / "taxonomy_two_stage_validation_v1",
    )
    parser.add_argument("--write", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = summarise(args.output_root.resolve())
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
