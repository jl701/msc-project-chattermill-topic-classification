"""Replay the final decoder/analysis contract on frozen validation score shards.

This audit reads validation-only artifacts and their frozen thresholds.  It
does not load any FABSA split and cannot open the official test.  The expected
values were already published by the stage-hybrid study; exact reproduction is
therefore a code-path verification, not a new model-selection exercise.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msc_project.data.fabsa import format_pair_label
from msc_project.experiments.taxonomy_final_test import (
    FIXED_COMPOSITION_METHOD,
)
from msc_project.experiments.taxonomy_final_test_analysis import (
    synchronised_aspect_balanced_difference,
)
from msc_project.experiments.taxonomy_final_test_workflow import (
    atomic_json,
    decode_capped_two,
)

DEFAULT_CLOUD_ROOT = (
    PROJECT_ROOT.parent / "cloud_backups" / "taxonomy_two_stage_formal_v2_r2"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "docs"
    / "experiments"
    / "taxonomy_final_test_v1_validation_mirror_audit.json"
)
EXPECTED = {
    "H1": {
        "point_difference": 0.021048319336442134,
        "ci_lower": 0.0007745024408152671,
        "ci_upper": 0.04265633338746592,
    },
    "H2": {
        "point_difference": 0.04602771313351717,
        "ci_lower": 0.021611342430931577,
        "ci_upper": 0.06975257081336628,
    },
}


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _scope_id(fold_id: str) -> str:
    return "heldout-" + fold_id.removeprefix("l2-")


def _worker_for_qlora(cloud_root: Path, scope_id: str) -> Path:
    matches = [
        root
        for root in (cloud_root / "worker-qlora-a", cloud_root / "worker-qlora-b")
        if (root / "selections" / "qwen_candidate_pair_qlora" / f"{scope_id}.json").is_file()
    ]
    if len(matches) != 1:
        raise ValueError(f"QLoRA scope {scope_id} does not resolve to one worker.")
    return matches[0]


def _load_shards(worker: Path, method: str, fold_id: str) -> pd.DataFrame:
    root = worker / "scores" / method / "L2" / fold_id / "D"
    contracts = [value for value in root.iterdir() if value.is_dir()]
    if len(contracts) != 1:
        raise ValueError(f"Expected one validation score contract: {root}")
    paths = sorted(contracts[0].glob("shard-*-of-00008.csv"))
    if len(paths) != 8:
        raise ValueError(f"Expected eight validation score shards: {root}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if (
        set(frame["split"].astype(str)) != {"validation"}
        or set(frame["condition"].astype(str)) != {"D"}
        or set(frame["fold_id"].astype(str)) != {fold_id}
        or frame.duplicated(
            ["row_uid", "candidate_aspect", "candidate_sentiment"]
        ).any()
    ):
        raise ValueError("Frozen validation score shards violate their identity contract.")
    return frame.sort_values(
        ["row_uid", "candidate_aspect", "candidate_sentiment"], kind="stable"
    ).reset_index(drop=True)


def _thresholds(selection: Mapping[str, object], *, few_shot: bool) -> tuple[float, float]:
    value = selection.get("thresholds" if few_shot else "selected_thresholds")
    if not isinstance(value, Mapping):
        raise TypeError("Frozen selection lacks thresholds.")
    if int(selection.get("failure_count", -1)) or int(
        selection.get("test_contract_count", -1)
    ):
        raise ValueError("Frozen validation selection reports a failure/test contract.")
    return float(value["aspect"]), float(value["runner_up_sentiment"])


def _prediction_rows(frame: pd.DataFrame, method_id: str) -> pd.DataFrame:
    heldout = frame[frame["is_heldout"].astype(bool)].copy()
    grouped = heldout.groupby("row_uid", sort=False)
    return pd.DataFrame(
        {
            "method_id": method_id,
            "fold_id": str(frame["fold_id"].iloc[0]),
            "row_uid": list(grouped.groups),
            "gold_pairs": [
                sorted(
                    format_pair_label(str(row.candidate_aspect), str(row.candidate_sentiment))
                    for row in value.itertuples(index=False)
                    if int(row.target) == 1
                )
                for _, value in grouped
            ],
            "pred_pairs": [
                sorted(
                    format_pair_label(str(row.candidate_aspect), str(row.candidate_sentiment))
                    for row in value.itertuples(index=False)
                    if bool(row.prediction)
                )
                for _, value in grouped
            ],
        }
    )


def run(cloud_root: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    distil = cloud_root / "worker-distil-frozen"
    predictions: dict[str, list[pd.DataFrame]] = {
        FIXED_COMPOSITION_METHOD: [],
        "frozen_qwen_few_shot": [],
        "qwen_candidate_pair_qlora": [],
    }
    for index in range(1, 13):
        fold_id = f"l2-a{index:02d}"
        scope_id = _scope_id(fold_id)
        qworker = _worker_for_qlora(cloud_root, scope_id)
        few = _load_shards(distil, "frozen_qwen_few_shot", fold_id)
        qlora = _load_shards(qworker, "qwen_candidate_pair_qlora", fold_id)
        keys = ["row_uid", "candidate_aspect", "candidate_sentiment"]
        if not np.array_equal(few[keys].to_numpy(dtype=str), qlora[keys].to_numpy(dtype=str)):
            raise ValueError("Frozen few-shot and QLoRA validation identities differ.")
        if not np.array_equal(few["target"].to_numpy(), qlora["target"].to_numpy()):
            raise ValueError("Frozen source systems contain different validation targets.")
        few_selection = _read_object(
            distil / "selections" / "frozen_qwen_few_shot" / f"{scope_id}.json"
        )
        qlora_selection = _read_object(
            qworker / "selections" / "qwen_candidate_pair_qlora" / f"{scope_id}.json"
        )
        few_aspect, few_runner = _thresholds(few_selection, few_shot=True)
        qlora_aspect, qlora_runner = _thresholds(qlora_selection, few_shot=False)
        few["prediction"] = decode_capped_two(
            few, aspect_threshold=few_aspect, runner_up_threshold=few_runner
        )
        qlora["prediction"] = decode_capped_two(
            qlora, aspect_threshold=qlora_aspect, runner_up_threshold=qlora_runner
        )
        hybrid = few.copy()
        hybrid["sentiment_score"] = qlora["sentiment_score"].to_numpy(dtype=float)
        hybrid["prediction"] = decode_capped_two(
            hybrid, aspect_threshold=few_aspect, runner_up_threshold=qlora_runner
        )
        predictions["frozen_qwen_few_shot"].append(
            _prediction_rows(few, "frozen_qwen_few_shot")
        )
        predictions["qwen_candidate_pair_qlora"].append(
            _prediction_rows(qlora, "qwen_candidate_pair_qlora")
        )
        predictions[FIXED_COMPOSITION_METHOD].append(
            _prediction_rows(hybrid, FIXED_COMPOSITION_METHOD)
        )
    merged = {
        method: pd.concat(values, ignore_index=True)
        for method, values in predictions.items()
    }
    h1 = synchronised_aspect_balanced_difference(
        merged[FIXED_COMPOSITION_METHOD],
        merged["frozen_qwen_few_shot"],
        replicates=20_000,
        seed=13,
    )
    h2 = synchronised_aspect_balanced_difference(
        merged[FIXED_COMPOSITION_METHOD],
        merged["qwen_candidate_pair_qlora"],
        replicates=20_000,
        seed=13,
    )
    checks: dict[str, float] = {}
    for name, observed in (("H1", h1), ("H2", h2)):
        for key, expected in EXPECTED[name].items():
            delta = abs(float(observed[key]) - expected)
            checks[f"{name}.{key}.absolute_delta"] = delta
            if delta > 1e-12:
                raise AssertionError(
                    f"Validation mirror changed {name}.{key}: "
                    f"observed={observed[key]}, expected={expected}."
                )
    result: dict[str, object] = {
        "schema_version": "taxonomy_final_test_validation_mirror_audit_v1",
        "status": "pass",
        "evaluation_partition": "validation_only",
        "official_test_opened": False,
        "include_official_test": False,
        "test_contract_count": 0,
        "folds": 12,
        "review_clusters": 1057,
        "H1": h1,
        "H2": h2,
        "expected_value_checks": checks,
        "failure_count": 0,
    }
    atomic_json(output, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloud-root", type=Path, default=DEFAULT_CLOUD_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args.cloud_root.resolve(), args.output.resolve())
    print(
        json.dumps(
            {
                "status": result["status"],
                "folds": result["folds"],
                "review_clusters": result["review_clusters"],
                "official_test_opened": False,
                "test_contract_count": 0,
                "output": str(args.output.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
