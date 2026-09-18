from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyse_taxonomy_final_test_results.py"
SPEC = importlib.util.spec_from_file_location("analyse_taxonomy_final_test_results", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _metric_row(condition: str, partition: str, f1: float) -> dict[str, object]:
    return {
        "method_id": "strict_train_only_tfidf",
        "level": "L2",
        "fold_id": "l2-a01",
        "condition": condition,
        "partition": partition,
        "pair_micro_precision": f1,
        "pair_micro_recall": f1,
        "pair_micro_f1": f1,
        "pair_tp": 2,
        "pair_fp": 1,
        "pair_fn": 1,
        "exact_set_match": 0.25,
        "mean_prediction_set_size": 1.5,
        "aspect_call_rate": 0.5,
    }


def test_l2_summary_keeps_fold_mean_and_pooled_sensitivity_separate() -> None:
    rows = [
        _metric_row(condition, partition, 0.4 if condition == "D" else 0.3)
        for condition in ("D", "N")
        for partition in ("heldout", "seen", "overall")
    ]
    summary = MODULE.build_l2_summary(pd.DataFrame.from_records(rows))
    d = summary[summary["condition"].eq("D")].iloc[0]
    assert d["heldout_pair_f1_fold_mean"] == pytest.approx(0.4)
    assert d["heldout_pair_f1_pooled_sensitivity"] == pytest.approx(2 / 3)
    assert d["fold_count"] == 1


def test_confirmatory_fold_table_computes_registered_differences() -> None:
    records = []
    values = {
        "frozen_qwen_few_shot_stage1__qlora_stage2": (0.6, 0.7),
        "frozen_qwen_few_shot": (0.5, 0.65),
        "qwen_candidate_pair_qlora": (0.55, 0.6),
    }
    for method, fold_values in values.items():
        for index, value in enumerate(fold_values, start=1):
            records.append(
                {
                    "method_id": method,
                    "level": "L2",
                    "fold_id": f"l2-a{index:02d}",
                    "condition": "D",
                    "partition": "heldout",
                    "pair_micro_f1": value,
                }
            )
    result = MODULE.build_confirmatory_folds(pd.DataFrame.from_records(records))
    first = result[result["fold_id"].eq("l2-a01")].iloc[0]
    assert first["fixed_minus_few_shot"] == pytest.approx(0.1)
    assert first["fixed_minus_qlora"] == pytest.approx(0.05)


def test_description_effects_report_d_minus_n_without_selection() -> None:
    frame = pd.DataFrame.from_records(
        [
            {
                "method_id": "strict_train_only_tfidf",
                "method": "TF-IDF",
                "condition": "D",
                "heldout_pair_f1_fold_mean": 0.4,
                "overall_pair_f1_fold_mean": 0.5,
            },
            {
                "method_id": "strict_train_only_tfidf",
                "method": "TF-IDF",
                "condition": "N",
                "heldout_pair_f1_fold_mean": 0.3,
                "overall_pair_f1_fold_mean": 0.45,
            },
        ]
    )
    result = MODULE.build_description_effects(frame).iloc[0]
    assert result["heldout_D_minus_N"] == pytest.approx(0.1)
    assert result["overall_D_minus_N"] == pytest.approx(0.05)
