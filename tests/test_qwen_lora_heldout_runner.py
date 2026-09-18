from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_qwen_lora_heldout_aspect import (
    ChatSftDataset,
    apply_existing_adapter_resume,
    build_manifest,
    complete_existing_predictions,
    encode_sft_example,
    existing_predictions_are_usable,
    load_existing_summary_state,
    load_sft_split,
    normalise_prediction_output,
    prediction_path,
    resolve_sft_data_dir,
    run_manifest_path_for,
    should_skip_training,
    write_json,
    write_jsonl,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    pad_token = "<pad>"
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False):
        if tokenize:
            raise NotImplementedError
        text = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
        if add_generation_prompt:
            text += "\nassistant:"
        return text

    def __call__(self, text, truncation=True, max_length=None):
        tokens = list(range(1, len(text.split()) + 1))
        if truncation and max_length is not None:
            tokens = tokens[:max_length]
        return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}


def sft_row(row_id: str = "r1") -> dict[str, object]:
    return {
        "id": row_id,
        "row_uid": f"validation:{row_id}",
        "original_split": "validation",
        "candidate_aspects": ["Account management: Account access"],
        "messages": [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": "Review: login failed"},
            {"role": "assistant", "content": '[{"aspect_id": "A1", "sentiment": "negative"}]'},
        ],
        "pair_labels": ["Account management: Account access | negative"],
    }


def manifest_args(**overrides):
    values = {
        "model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "load_in_4bit": True,
        "gradient_checkpointing": True,
        "resume_from_adapter": None,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_target_modules": "q_proj,v_proj",
        "save_adapter": True,
        "adapter_output_name": "adapter_final",
        "save_epoch_adapters": False,
        "epochs": 1,
        "max_train_steps": 1,
        "batch_size": 1,
        "grad_accumulation_steps": 1,
        "learning_rate": 0.0001,
        "weight_decay": 0.0,
        "warmup_ratio": 0.0,
        "max_grad_norm": 1.0,
        "log_every_steps": 50,
        "prompt_variant": "indexed",
        "strategy": "example_filtered",
        "train_limit": 4,
        "validation_limit": 2,
        "test_limit": 2,
        "eval_split": ["validation", "test"],
        "max_length": 256,
        "max_input_tokens": 256,
        "max_new_tokens": 64,
        "resume_predictions": True,
        "skip_existing_predictions": True,
        "skip_training_if_adapter_exists": False,
        "seed": 13,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class QwenLoraHeldoutRunnerTests(unittest.TestCase):
    def test_load_sft_split_validates_rows_and_applies_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "example_filtered"
            write_json({"strategy": "example_filtered", "prompt_variant": "indexed"}, data_dir / "metadata.json")
            for split in ("train", "validation", "test"):
                write_jsonl([sft_row("r1"), sft_row("r2")], data_dir / f"{split}.jsonl")

            resolved = resolve_sft_data_dir(Path(tmp), "example_filtered")
            rows = load_sft_split(resolved, "validation", limit=1)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "r1")

    def test_encode_sft_example_masks_prompt_tokens_only(self) -> None:
        encoded = encode_sft_example(FakeTokenizer(), sft_row(), max_length=128)
        first_unmasked = next(index for index, label in enumerate(encoded["labels"]) if label != -100)

        self.assertIn(-100, encoded["labels"])
        self.assertTrue(all(label == -100 for label in encoded["labels"][:first_unmasked]))
        self.assertTrue(any(label != -100 for label in encoded["labels"]))
        self.assertEqual(len(encoded["input_ids"]), len(encoded["labels"]))

    def test_chat_sft_dataset_skips_rows_with_fully_truncated_answer(self) -> None:
        dataset = ChatSftDataset([sft_row()], FakeTokenizer(), max_length=2)

        self.assertEqual(len(dataset), 0)
        self.assertEqual(dataset.skipped_no_label_count, 1)

    def test_prediction_resume_requires_prefix_row_ids(self) -> None:
        eval_rows = [sft_row("r1"), sft_row("r2")]
        rows = [{"row_index": 0, "id": "r1", "pred_pair_labels": []}]

        self.assertTrue(existing_predictions_are_usable(rows, eval_rows))
        self.assertFalse(existing_predictions_are_usable([{**rows[0], "id": "wrong"}], eval_rows))
        self.assertFalse(existing_predictions_are_usable([{**rows[0], "row_index": 1}], eval_rows))

    def test_complete_existing_predictions_checks_path_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            path = prediction_path(output_dir, "validation")
            eval_rows = [sft_row("r1")]
            write_jsonl([{"row_index": 0, "id": "r1", "pred_pair_labels": []}], path)

            self.assertTrue(complete_existing_predictions(path, eval_rows))

    def test_prediction_parser_maps_indexed_candidate_ids(self) -> None:
        parsed = normalise_prediction_output(
            '[{"aspect_id": "A1", "sentiment": "positive"}]',
            ["Staff support: Email"],
            require_aspect_id=True,
        )

        self.assertEqual(parsed["pred_pair_labels"], ["Staff support: Email | positive"])
        self.assertTrue(parsed["valid_json"])
        self.assertTrue(parsed["schema_valid"])

    def test_build_manifest_records_reproducibility_fields(self) -> None:
        args = manifest_args()
        metadata = {
            "strategy": "example_filtered",
            "prompt_variant": "indexed",
            "eval_label_scope": "heldout",
            "heldout_aspects": ["Staff support: Email"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            manifest = build_manifest(
                args=args,
                data_dir=Path(tmp) / "sft",
                metadata=metadata,
                row_counts={"train": 4, "validation": 2, "test": 2},
                output_dir=Path(tmp) / "out",
                started_at="2026-07-02T12:00:00",
                finished_at="2026-07-02T12:00:01",
                runtime_seconds=1.0,
            )

        self.assertEqual(manifest["model_name"], "Qwen/Qwen3-4B-Instruct-2507")
        self.assertEqual(manifest["loading"]["quantisation"], "bitsandbytes 4-bit NF4 double quantisation")
        self.assertEqual(manifest["lora"]["target_modules"], ["q_proj", "v_proj"])
        self.assertIn("training_resume_limitation", manifest["resume_and_recovery"])
        self.assertEqual(manifest["row_scope"]["row_counts_after_limits"]["train"], 4)

    def test_should_skip_training_when_adapter_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "adapter_final").mkdir()

            self.assertTrue(should_skip_training(output_dir, manifest_args(skip_training_if_adapter_exists=True)))
            self.assertFalse(should_skip_training(output_dir, manifest_args(skip_training_if_adapter_exists=False)))

    def test_existing_adapter_skip_sets_resume_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            adapter_dir = output_dir / "adapter_final"
            adapter_dir.mkdir()
            args = manifest_args(skip_training_if_adapter_exists=True, resume_from_adapter=None)

            apply_existing_adapter_resume(output_dir, args)

            self.assertEqual(args.resume_from_adapter, adapter_dir)

    def test_existing_summary_state_replaces_only_requested_eval_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            write_json(
                {
                    "train_history": [{"epoch": 1, "loss": 0.5}],
                    "eval_results": [
                        {"split": "validation", "pair_micro_f1": 0.4},
                        {"split": "test", "pair_micro_f1": 0.3},
                    ],
                },
                output_dir / "summary.json",
            )

            train_history, eval_results = load_existing_summary_state(output_dir, ["test"])

        self.assertEqual(train_history, [{"epoch": 1, "loss": 0.5}])
        self.assertEqual(eval_results, [{"split": "validation", "pair_micro_f1": 0.4}])

    def test_run_manifest_path_is_windows_safe_and_split_specific(self) -> None:
        path = run_manifest_path_for(Path("outputs") / "run", "2026-07-02T05:11:32", ["validation", "test"])

        self.assertEqual(path.name, "20260702_051132_validation_test_manifest.json")
        self.assertNotIn(":", str(path))


if __name__ == "__main__":
    unittest.main()
