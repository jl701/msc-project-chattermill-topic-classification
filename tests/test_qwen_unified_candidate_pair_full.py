from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_qwen_unified_candidate_pair_full as full


def scientific_arguments(mode: str = "frozen") -> dict[str, object]:
    values = full.preregistered_scientific_parameters(full.load_experiment_config())
    return {**values, "mode": mode, "eval_batch_size": 6}


def result_row(
    aspect: str,
    split: str,
    *,
    model: str = full.FROZEN_RESULT_MODEL,
    examples: int = 2,
    f1: float = 0.5,
) -> dict[str, object]:
    return {
        "model": model,
        "heldout_aspect": aspect,
        "split": split,
        "variant": full.ENHANCED_VARIANT,
        "selected_threshold": 0.5,
        "threshold": 0.5,
        "pair_micro_f1": f1,
        "pair_micro_precision": 0.5,
        "pair_micro_recall": 0.5,
        "pair_samples_f1": 0.5,
        "presence_f1": 0.5,
        "presence_false_positive_rows_per_100": 1.0,
        "presence_false_negative_rows_per_100": 1.0,
        "conditional_sentiment_macro_f1": 0.5,
        "examples": examples,
    }


def write_scoring_artifacts(root: Path, split: str, examples: int, aspect: str) -> None:
    pd.DataFrame(
        [
            {
                "candidate_aspect": aspect,
                "candidate_sentiment": sentiment,
                "score": 0.5,
            }
            for _ in range(examples)
            for sentiment in full.CANDIDATE_SENTIMENTS
        ]
    ).to_csv(root / f"{split}_pair_scores.csv", index=False)
    (root / f"{split}_predictions.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "row_uid": f"uid-{index}",
                    "text": f"text-{index}",
                    "gold_pair_labels": [],
                }
            )
            + "\n"
            for index in range(examples)
        ),
        encoding="utf-8",
    )


def expected_frame(examples: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_uid": [f"uid-{index}" for index in range(examples)],
            "text": [f"text-{index}" for index in range(examples)],
            "supervision_pair_labels": [[] for _ in range(examples)],
        }
    )


def test_formal_full_fold_set_is_exactly_twelve_unique_dataset_folds() -> None:
    canonical = [f"fold-{index}" for index in range(12)]
    assert full.validate_full_fold_set([], canonical) == canonical
    assert full.validate_full_fold_set(list(reversed(canonical)), canonical) == list(
        reversed(canonical)
    )

    with pytest.raises(ValueError, match="exactly 12 unique"):
        full.validate_full_fold_set(canonical[:8], canonical)
    with pytest.raises(ValueError, match="exactly 12 unique"):
        full.validate_full_fold_set(canonical[:-1] + [canonical[0]], canonical)
    with pytest.raises(ValueError, match="differs from the dataset"):
        full.validate_full_fold_set(canonical[:-1] + ["other"], canonical)


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        ("model_name", "another/model"),
        ("weight_decay", 0.02),
        ("warmup_ratio", 0.2),
        ("max_grad_norm", 0.5),
        ("learning_rate", 1e-5),
        ("train_budget", 2048),
    ],
)
def test_all_scientific_qwen_parameters_are_frozen(key: str, bad_value: object) -> None:
    config = full.load_experiment_config()
    values = scientific_arguments()
    full.validate_scientific_parameters(values, config)

    values[key] = bad_value
    with pytest.raises(ValueError, match=key):
        full.validate_scientific_parameters(values, config)


def test_quantisation_and_lora_runtime_contract_is_frozen() -> None:
    config = full.load_experiment_config()
    full.validate_scientific_parameters(scientific_arguments(), config)
    changed = json.loads(json.dumps(config))
    changed["models"]["qwen"]["load_in_4bit"] = False

    with pytest.raises(ValueError, match="load_in_4bit"):
        full.validate_scientific_parameters(scientific_arguments(), changed)


def test_legacy_resume_is_narrowly_limited_to_frozen_identity(tmp_path: Path) -> None:
    config = full.load_experiment_config()
    folds = [f"fold-{index}" for index in range(12)]
    payload = {
        "protocol_id": full.PROTOCOL_ID,
        "mode": "frozen",
        "stage": "full_12_fold_confirmation",
        "folds": folds,
        "arguments": scientific_arguments("frozen"),
    }
    path = tmp_path / "run_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    _, legacy = full.validate_existing_run_manifest(
        path,
        mode="frozen",
        folds=folds,
        preregistration=config,
        expected_dataset_identity={"sha256": "new"},
    )
    assert legacy is True

    payload["mode"] = "qlora"
    payload["arguments"]["mode"] = "qlora"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity schema"):
        full.validate_existing_run_manifest(
            path,
            mode="qlora",
            folds=folds,
            preregistration=config,
            expected_dataset_identity={"sha256": "new"},
        )


def test_completed_fold_requires_exact_identity_rows_examples_and_artifacts(
    tmp_path: Path,
) -> None:
    aspect = "Account management: Account access"
    write_scoring_artifacts(tmp_path, "validation", 2, aspect)
    write_scoring_artifacts(tmp_path, "test", 3, aspect)
    rows = [
        result_row(aspect, "validation", examples=2),
        result_row(aspect, "test", examples=3),
    ]
    payload = {
        "integrity_schema_version": full.INTEGRITY_SCHEMA_VERSION,
        "protocol_id": full.PROTOCOL_ID,
        "mode": "frozen",
        "heldout_aspect": aspect,
        "variant": full.ENHANCED_VARIANT,
        "base_model_name_or_path": full.QWEN_MODEL_NAME,
        "qwen_runtime_contract": full.preregistered_qwen_runtime_contract(
            full.load_experiment_config()
        ),
        "results": rows,
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    completed = full.completed_fold_results(
        path,
        heldout_aspect=aspect,
        mode="frozen",
        validation_examples=2,
        test_examples=3,
        validation_frame=expected_frame(2),
        test_frame=expected_frame(3),
    )
    assert [row["split"] for row in completed] == ["validation", "test"]

    wrong_test_frame = expected_frame(3)
    wrong_test_frame.loc[0, "row_uid"] = "different"
    with pytest.raises(ValueError, match="row_uid sequence"):
        full.completed_fold_results(
            path,
            heldout_aspect=aspect,
            mode="frozen",
            validation_examples=2,
            test_examples=3,
            validation_frame=expected_frame(2),
            test_frame=wrong_test_frame,
        )

    payload["results"] = [*rows, dict(rows[-1])]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly two"):
        full.completed_fold_results(
            path,
            heldout_aspect=aspect,
            mode="frozen",
            validation_examples=2,
            test_examples=3,
        )

    payload.pop("integrity_schema_version")
    payload.pop("protocol_id")
    payload.pop("mode")
    payload.pop("heldout_aspect")
    payload.pop("variant")
    payload.pop("base_model_name_or_path")
    payload.pop("qwen_runtime_contract")
    payload["results"] = rows
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert full.completed_fold_results(
        path,
        heldout_aspect=aspect,
        mode="frozen",
        validation_examples=2,
        test_examples=3,
        allow_legacy_frozen=True,
        validation_frame=expected_frame(2),
        test_frame=expected_frame(3),
    )
    with pytest.raises(ValueError, match="Legacy fold summaries"):
        full.completed_fold_results(
            path,
            heldout_aspect=aspect,
            mode="qlora",
            validation_examples=2,
            test_examples=3,
        )


def test_adapter_validation_checks_base_lora_training_summary_and_hash(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    adapter_config = {
        **full._expected_adapter_parameters(),
        "target_modules": list(full.QLORA_TARGET_MODULES),
        "inference_mode": True,
    }
    config_path = adapter / "adapter_config.json"
    config_path.write_text(json.dumps(adapter_config), encoding="utf-8")
    history = [{"epoch": 1, "global_step": 512, "train_loss": 1.0}]
    digest = "a" * 64
    result = {
        "training_manifest_hash": digest,
        "training_pairs": 4096,
        "training_positive_pairs": 2048,
    }
    summary = full._training_summary_payload(
        heldout_aspect="Staff support: Phone",
        manifest_digest=digest,
        train_seconds=1.0,
        peak_memory=1,
        history=history,
    )
    summary_path = tmp_path / "training_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    result["training_summary"] = summary

    full.validate_adapter_artifacts(
        adapter,
        expected_heldout_aspect="Staff support: Phone",
        expected_manifest_hash=digest,
        expected_result=result,
        training_summary_path=summary_path,
        allow_legacy_pilot_summary=False,
    )

    adapter_config["base_model_name_or_path"] = "wrong/model"
    config_path.write_text(json.dumps(adapter_config), encoding="utf-8")
    with pytest.raises(ValueError, match="base_model_name_or_path"):
        full.validate_adapter_artifacts(
            adapter,
            expected_heldout_aspect="Staff support: Phone",
            expected_manifest_hash=digest,
            expected_result=result,
            training_summary_path=summary_path,
            allow_legacy_pilot_summary=False,
        )


def test_stored_pilot_gate_is_mode_specific_and_recomputed() -> None:
    rows = [
        {
            "heldout_aspect": aspect,
            "pair_micro_f1": 0.8,
            "pair_micro_recall": 0.8,
            "presence_false_positive_rows_per_100": 0.0,
        }
        for aspect in full.PILOT_FOLDS
    ]
    gate = full.historical_frozen_gate(rows)
    full._validate_pilot_gate(
        {"frozen_gate": gate},
        mode="frozen",
        results=rows,
        frozen_results=None,
    )

    with pytest.raises(ValueError, match="unexpectedly contains a QLoRA gate"):
        full._validate_pilot_gate(
            {"qlora_gate": {"passed": True}},
            mode="frozen",
            results=rows,
            frozen_results=None,
        )
    forged = {**gate, "passed": False}
    with pytest.raises(ValueError, match="did not pass"):
        full._validate_pilot_gate(
            {"frozen_gate": forged},
            mode="frozen",
            results=rows,
            frozen_results=None,
        )


def aggregate_row(aspect: str, f1: float) -> dict[str, object]:
    return {
        **result_row(aspect, "test", model=full.QLORA_RESULT_MODEL, examples=3, f1=f1),
        "pair_micro_precision": f1,
        "pair_micro_recall": f1,
    }


def test_confirmation_is_only_computed_for_matching_complete_twelve_fold_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folds = [f"fold-{index}" for index in range(12)]
    frozen_rows = [
        result_row(aspect, split, examples=3, f1=0.4)
        for aspect in folds
        for split in ("validation", "test")
    ]
    frozen_path = tmp_path / "frozen.json"
    frozen_metadata = {
        "integrity_schema_version": full.INTEGRITY_SCHEMA_VERSION,
        "protocol_id": full.PROTOCOL_ID,
        "mode": "frozen",
        "expected_folds": folds,
        "qwen_runtime_contract": full.preregistered_qwen_runtime_contract(
            full.load_experiment_config()
        ),
    }
    frozen_path.write_text(
        json.dumps({**frozen_metadata, "results": frozen_rows}), encoding="utf-8"
    )
    monkeypatch.setattr(
        full,
        "paired_aspect_statistics",
        lambda *_: {"mean_delta": 0.03, "wins": 12, "ties": 0, "losses": 0},
    )

    partial = [aggregate_row(aspect, 0.43) for aspect in folds[:8]]
    summary = full.aggregate(partial, tmp_path / "partial", "qlora", frozen_path, folds)
    assert summary["full_confirmation"] is None

    complete = [
        {
            **aggregate_row(aspect, 0.43),
            "split": split,
        }
        for aspect in folds
        for split in ("validation", "test")
    ]
    summary = full.aggregate(complete, tmp_path / "complete", "qlora", frozen_path, folds)
    assert summary["full_confirmation"]["passed"] is True
    assert len(summary["full_confirmation"]["per_fold"]) == 12

    frozen_path.write_text(
        json.dumps({**frozen_metadata, "results": frozen_rows[:-2]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="complete frozen reference"):
        full.aggregate(complete, tmp_path / "bad", "qlora", frozen_path, folds)
