from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import summarise_unified_candidate_pair_results as summary_script


def write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def local_payload(model: str, control: float, enhanced: float) -> dict[str, object]:
    rows = []
    for index, aspect in enumerate(summary_script.PILOT_FOLDS):
        rows.extend(
            [
                {
                    "model": model,
                    "variant": "control",
                    "heldout_aspect": aspect,
                    "split": "validation",
                    "pair_micro_f1": control + index * 0.01,
                },
                {
                    "model": model,
                    "variant": "enhanced",
                    "heldout_aspect": aspect,
                    "split": "validation",
                    "pair_micro_f1": enhanced + index * 0.01,
                },
            ]
        )
    return {"stage": "pilot", "results": rows}


def full_payload(mode: str, score: float) -> dict[str, object]:
    model = (
        "qwen_frozen_candidate_pair" if mode == "frozen" else "qwen_qlora_candidate_pair"
    )
    rows = []
    for index, aspect in enumerate(summary_script.FULL_FOLDS):
        for split in ("validation", "test"):
            rows.append(
                {
                    "model": model,
                    "variant": "enhanced",
                    "heldout_aspect": aspect,
                    "split": split,
                    "pair_micro_f1": score + index * 0.001,
                }
            )
    return {
        "protocol_id": summary_script.PROTOCOL_ID,
        "mode": mode,
        "expected_folds": list(summary_script.FULL_FOLDS),
        "qwen_runtime_contract": {"task_format": "pair", "max_length": 384},
        "results": rows,
    }


def historical_payload(score: float) -> dict[str, object]:
    return {
        "heldout_aspects": list(summary_script.FULL_FOLDS),
        "strategy": "label_masked",
        "eval_label_scope": "heldout",
        "eval_row_scope": "all",
        "results": [
            {
                "heldout_aspect": aspect,
                "split": "test",
                "pair_micro_f1": score + index * 0.001,
            }
            for index, aspect in enumerate(summary_script.FULL_FOLDS)
        ],
    }


def complete_input_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "tfidf_pilot_path": write_json(
            tmp_path / "tfidf.json", local_payload("tfidf", 0.20, 0.25)
        ),
        "distilbert_pilot_path": write_json(
            tmp_path / "distilbert.json", local_payload("distilbert", 0.30, 0.28)
        ),
        "frozen_full_path": write_json(
            tmp_path / "frozen.json", full_payload("frozen", 0.40)
        ),
        "historical_qwen_path": write_json(
            tmp_path / "historical.json", historical_payload(0.30)
        ),
        "qlora_full_path": write_json(
            tmp_path / "qlora.json", full_payload("qlora", 0.45)
        ),
    }


def test_complete_inputs_produce_four_comparisons_and_thirty_per_fold_rows(
    tmp_path: Path,
) -> None:
    summary, rows = summary_script.build_result_summary(**complete_input_paths(tmp_path))

    assert summary["status"] == "experimental_only_pending_user_approval"
    assert summary["comparison_count"] == 4
    assert summary["per_fold_row_count"] == 30
    assert len(rows) == 30
    comparisons = {row["comparison_id"]: row for row in summary["comparisons"]}
    assert comparisons["corrected_tfidf_pilot_enhanced_vs_control"][
        "mean_pair_micro_f1_delta"
    ] == pytest.approx(0.05)
    historical = comparisons["frozen_candidate_pair_qwen_full_vs_historical_json_qwen"]
    assert historical["matched_protocol"] is False
    assert "non-matched protocol" in historical["interpretation"]
    matched = comparisons["qlora_full_vs_frozen_candidate_pair_qwen"]
    assert matched["matched_protocol"] is True
    assert matched["mean_delta_percentage_points"] == pytest.approx(5.0)

    output_dir = tmp_path / "experimental" / "summary"
    resolved = summary_script.require_experimental_output_dir(
        output_dir, experimental_root=tmp_path / "experimental"
    )
    summary_path, csv_path = summary_script.write_outputs(summary, rows, resolved)
    assert json.loads(summary_path.read_text(encoding="utf-8"))["comparison_count"] == 4
    with csv_path.open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 30


def test_pilot_requires_exact_three_fold_control_and_enhanced_set(tmp_path: Path) -> None:
    paths = complete_input_paths(tmp_path)
    broken = local_payload("tfidf", 0.20, 0.25)
    broken["results"] = broken["results"][:-1]  # type: ignore[index]
    write_json(paths["tfidf_pilot_path"], broken)

    with pytest.raises(ValueError, match="exactly one control and one enhanced"):
        summary_script.build_result_summary(**paths)


def test_full_inputs_require_exact_complete_twelve_fold_validation_and_test_set(
    tmp_path: Path,
) -> None:
    paths = complete_input_paths(tmp_path)
    broken = full_payload("qlora", 0.45)
    broken["results"] = broken["results"][:-1]  # type: ignore[index]
    write_json(paths["qlora_full_path"], broken)

    with pytest.raises(ValueError, match="must be complete"):
        summary_script.build_result_summary(**paths)


def test_matched_qlora_comparison_rejects_runtime_contract_drift(tmp_path: Path) -> None:
    paths = complete_input_paths(tmp_path)
    qlora = full_payload("qlora", 0.45)
    qlora["qwen_runtime_contract"] = {"task_format": "different", "max_length": 384}
    write_json(paths["qlora_full_path"], qlora)

    with pytest.raises(ValueError, match="identical qwen_runtime_contract"):
        summary_script.build_result_summary(**paths)


def test_output_directory_must_be_a_dedicated_experimental_subdirectory(
    tmp_path: Path,
) -> None:
    experimental_root = tmp_path / "outputs" / "experimental"
    assert summary_script.require_experimental_output_dir(
        experimental_root / "isolated", experimental_root
    ) == (experimental_root / "isolated").resolve()

    with pytest.raises(ValueError, match="inside the experimental root"):
        summary_script.require_experimental_output_dir(tmp_path / "docs", experimental_root)
    with pytest.raises(ValueError, match="dedicated directory"):
        summary_script.require_experimental_output_dir(experimental_root, experimental_root)
