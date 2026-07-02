from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.baselines.candidate_label import candidate_pair_labels
from msc_project.evaluation.metrics import evaluate_pair_and_aspect
from msc_project.llm.candidate_label import (
    PROMPT_VARIANTS,
    aggregate_llm_diagnostics,
    diagnostic_dict,
    parse_candidate_output,
)


DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
SPLIT_NAMES = ("train", "validation", "test")
PACKAGE_VERSION_NAMES = ("torch", "transformers", "peft", "bitsandbytes", "accelerate")


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(row: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def limit_rows(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return list(rows)
    return list(rows[:limit])


def resolve_sft_data_dir(sft_data_dir: Path, strategy: str) -> Path:
    direct = sft_data_dir
    nested = sft_data_dir / strategy
    if (direct / "metadata.json").exists():
        data_dir = direct
    elif (nested / "metadata.json").exists():
        data_dir = nested
    else:
        raise FileNotFoundError(
            f"Could not find metadata.json in {direct} or {nested}. "
            "Run scripts/prepare_qwen_heldout_aspect_sft_data.py first."
        )

    missing = [split for split in SPLIT_NAMES if not (data_dir / f"{split}.jsonl").exists()]
    if missing:
        raise FileNotFoundError(f"SFT data directory {data_dir} is missing split files: {missing}")
    return data_dir


def validate_sft_row(row: dict[str, Any], split_name: str, index: int) -> dict[str, Any]:
    for key in ("id", "candidate_aspects", "messages", "pair_labels"):
        if key not in row:
            raise ValueError(f"{split_name} row {index} is missing required key {key!r}.")
    if not isinstance(row["candidate_aspects"], list) or not all(isinstance(item, str) for item in row["candidate_aspects"]):
        raise ValueError(f"{split_name} row {index} has invalid candidate_aspects.")
    if not isinstance(row["messages"], list) or not row["messages"]:
        raise ValueError(f"{split_name} row {index} has invalid messages.")
    if row["messages"][-1].get("role") != "assistant":
        raise ValueError(f"{split_name} row {index} must include the assistant answer as the final message.")
    if not isinstance(row["pair_labels"], list):
        raise ValueError(f"{split_name} row {index} has invalid pair_labels.")
    return row


def load_sft_split(data_dir: Path, split_name: str, limit: int | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(data_dir / f"{split_name}.jsonl")
    rows = [validate_sft_row(row, split_name, index) for index, row in enumerate(rows)]
    return limit_rows(rows, limit)


def prompt_messages_from_sft_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("SFT messages must end with an assistant answer.")
    return copy.deepcopy(messages[:-1])


def chat_text(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def encode_sft_example(tokenizer: Any, row: dict[str, Any], max_length: int) -> dict[str, list[int]]:
    prompt_text = chat_text(tokenizer, prompt_messages_from_sft_messages(row["messages"]), add_generation_prompt=True)
    full_text = chat_text(tokenizer, row["messages"], add_generation_prompt=False)

    prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_length)["input_ids"]
    encoded = tokenizer(full_text, truncation=True, max_length=max_length)
    input_ids = list(encoded["input_ids"])
    attention_mask = list(encoded["attention_mask"])
    labels = list(input_ids)
    mask_count = min(len(prompt_ids), len(labels))
    labels[:mask_count] = [-100] * mask_count
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class ChatSftDataset:
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        self.items = [encode_sft_example(tokenizer, row, max_length=max_length) for row in rows]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.items[index]


def collate_batch(batch: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, Any]:
    import torch

    max_len = max(len(item["input_ids"]) for item in batch)
    rows = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        rows["input_ids"].append(torch.tensor(item["input_ids"] + [pad_token_id] * pad_len, dtype=torch.long))
        rows["attention_mask"].append(torch.tensor(item["attention_mask"] + [0] * pad_len, dtype=torch.long))
        rows["labels"].append(torch.tensor(item["labels"] + [-100] * pad_len, dtype=torch.long))
    return {key: torch.stack(values) for key, values in rows.items()}


def split_limit(args: argparse.Namespace, split_name: str) -> int | None:
    return getattr(args, f"{split_name}_limit")


def prediction_path(output_dir: Path, split_name: str) -> Path:
    return output_dir / "predictions" / f"{split_name}_predictions.jsonl"


def request_path(output_dir: Path, split_name: str) -> Path:
    return output_dir / "requests" / f"{split_name}_requests.jsonl"


def existing_predictions_are_usable(rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> bool:
    if len(rows) > len(eval_rows):
        return False
    for expected_index, row in enumerate(rows):
        if int(row.get("row_index", -1)) != expected_index:
            return False
        if str(row.get("id", "")) != str(eval_rows[expected_index]["id"]):
            return False
        if "pred_pair_labels" not in row:
            return False
    return True


def complete_existing_predictions(path: Path, eval_rows: list[dict[str, Any]]) -> bool:
    if not path.exists():
        return False
    try:
        rows = read_jsonl(path)
    except (OSError, ValueError):
        return False
    return len(rows) == len(eval_rows) and existing_predictions_are_usable(rows, eval_rows)


def candidate_aspects_for_rows(rows: list[dict[str, Any]]) -> list[str]:
    aspects: list[str] = []
    for row in rows:
        for aspect in row["candidate_aspects"]:
            if aspect not in aspects:
                aspects.append(aspect)
    return aspects


def normalise_prediction_output(raw_output: str, candidate_aspects: list[str], require_aspect_id: bool) -> dict[str, Any]:
    parsed = parse_candidate_output(
        raw_output,
        candidate_aspects,
        require_aspect_id=require_aspect_id,
        allow_object_wrapper=True,
    )
    return {"pred_pair_labels": parsed.pair_labels, **diagnostic_dict(parsed)}


def metrics_from_rows(rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    aligned_eval = eval_rows[: len(rows)]
    candidate_aspects = candidate_aspects_for_rows(aligned_eval)
    pair_classes = candidate_pair_labels(candidate_aspects)
    true_labels = [row["pair_labels"] for row in aligned_eval]
    pred_labels = [row["pred_pair_labels"] for row in rows]
    metrics = evaluate_pair_and_aspect(true_labels, pred_labels, pair_classes)
    diagnostics = aggregate_llm_diagnostics(rows)
    total = max(1, len(rows))
    seconds = sum(float(row.get("seconds") or 0.0) for row in rows)
    metrics.update(
        {
            "examples": int(len(rows)),
            "positive_gold_rows": int(sum(len(row["pair_labels"]) > 0 for row in aligned_eval)),
            "predicted_labels_per_example": float(sum(len(row) for row in pred_labels) / total),
            "gold_labels_per_example": float(sum(len(row) for row in true_labels) / total),
            "seconds": float(seconds),
            "seconds_per_example": float(seconds / total),
            **diagnostics,
        }
    )
    return metrics


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def hardware_summary() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"torch_available": False, "error": str(exc)}

    info: dict[str, Any] = {
        "torch_available": True,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        info.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(0),
                "cuda_device_count": int(torch.cuda.device_count()),
                "cuda_memory_total_bytes": int(torch.cuda.get_device_properties(0).total_memory),
            }
        )
    return info


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package_name in PACKAGE_VERSION_NAMES:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def parse_target_modules(value: str) -> list[str]:
    modules = [item.strip() for item in value.split(",") if item.strip()]
    if not modules:
        raise ValueError("At least one LoRA target module is required.")
    return modules


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_qwen_lora_model(args: argparse.Namespace):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    if args.resume_from_adapter:
        model = PeftModel.from_pretrained(model, args.resume_from_adapter, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=parse_target_modules(args.lora_target_modules),
        )
        model = get_peft_model(model, lora_config)

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return tokenizer, model


def model_device(model: Any):
    return next(model.parameters()).device


def prepare_model_for_inference(model: Any) -> None:
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    base_model = getattr(model, "base_model", None)
    if base_model is not None and hasattr(base_model, "gradient_checkpointing_disable"):
        base_model.gradient_checkpointing_disable()
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        config.use_cache = True


def train_model(
    model: Any,
    tokenizer: Any,
    train_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
) -> list[dict[str, Any]]:
    import math
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    dataset = ChatSftDataset(train_rows, tokenizer, max_length=args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
    )
    if len(loader) == 0:
        raise ValueError("Training split is empty after limits were applied.")

    update_steps_per_epoch = math.ceil(len(loader) / args.grad_accumulation_steps)
    planned_steps = max(1, update_steps_per_epoch * args.epochs)
    if args.max_train_steps is not None:
        planned_steps = max(1, min(planned_steps, args.max_train_steps))
    warmup_steps = int(planned_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, planned_steps)
    device = model_device(model)
    history: list[dict[str, Any]] = []
    global_step = 0
    stop_training = False
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0
        epoch_start = time.time()

        for batch_step, batch in enumerate(loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accumulation_steps
            loss.backward()
            epoch_loss += float(loss.detach().cpu()) * args.grad_accumulation_steps
            batches += 1

            should_update = batch_step % args.grad_accumulation_steps == 0 or batch_step == len(loader)
            if should_update:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if args.max_train_steps is not None and global_step >= args.max_train_steps:
                    stop_training = True
                    break

        history.append(
            {
                "epoch": int(epoch),
                "global_step": int(global_step),
                "train_loss": float(epoch_loss / max(1, batches)),
                "seconds": float(time.time() - epoch_start),
            }
        )
        print(json.dumps(history[-1], indent=2), flush=True)

        if args.save_epoch_adapters:
            model.save_pretrained(output_dir / "checkpoints" / f"adapter_epoch_{epoch:02d}")

        if stop_training:
            break

    return history


def generate_one(
    tokenizer: Any,
    model: Any,
    messages: list[dict[str, str]],
    max_input_tokens: int,
    max_new_tokens: int,
) -> tuple[str, dict[str, int | None]]:
    import torch

    prompt = chat_text(tokenizer, messages, add_generation_prompt=True)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    ).to(model_device(model))
    input_tokens = int(inputs["input_ids"].shape[1])

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0, inputs["input_ids"].shape[1] :]
    raw_output = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return raw_output, {"input_tokens": input_tokens, "output_tokens": int(len(generated))}


def write_request_rows(eval_rows: list[dict[str, Any]], split_name: str, output_dir: Path) -> Path:
    rows = []
    for row_index, row in enumerate(eval_rows):
        rows.append(
            {
                "row_index": int(row_index),
                "id": str(row["id"]),
                "row_uid": str(row.get("row_uid", "")),
                "original_split": str(row.get("original_split", "")),
                "split": split_name,
                "messages": prompt_messages_from_sft_messages(row["messages"]),
                "candidate_aspects": row["candidate_aspects"],
            }
        )
    path = request_path(output_dir, split_name)
    write_jsonl(rows, path)
    return path


def load_existing_prediction_rows(
    path: Path,
    eval_rows: list[dict[str, Any]],
    resume_predictions: bool,
) -> list[dict[str, Any]]:
    if not resume_predictions or not path.exists():
        if path.exists():
            path.unlink()
        return []

    try:
        rows = read_jsonl(path)
    except (OSError, ValueError):
        path.unlink()
        return []

    if not existing_predictions_are_usable(rows, eval_rows):
        path.unlink()
        return []
    return rows


def run_prediction_split(
    tokenizer: Any,
    model: Any,
    eval_rows: list[dict[str, Any]],
    split_name: str,
    metadata: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    prepare_model_for_inference(model)
    requests_file = write_request_rows(eval_rows, split_name, output_dir)
    predictions_file = prediction_path(output_dir, split_name)

    if args.skip_existing_predictions and complete_existing_predictions(predictions_file, eval_rows):
        rows = read_jsonl(predictions_file)
        metrics = metrics_from_rows(rows, eval_rows)
        return {
            "split": split_name,
            "skipped_existing_predictions": True,
            "generated_examples_this_run": 0,
            "requests_file": str(requests_file),
            "predictions_file": str(predictions_file),
            **metrics,
        }

    rows = load_existing_prediction_rows(predictions_file, eval_rows, args.resume_predictions)
    generated_this_run = 0
    split_start = time.time()
    require_aspect_id = str(metadata.get("prompt_variant", args.prompt_variant)).startswith("indexed")

    for row_index, row in enumerate(eval_rows):
        if row_index < len(rows):
            continue

        example_start = time.time()
        raw_output, token_counts = generate_one(
            tokenizer=tokenizer,
            model=model,
            messages=prompt_messages_from_sft_messages(row["messages"]),
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
        )
        seconds = time.time() - example_start
        parsed_payload = normalise_prediction_output(raw_output, row["candidate_aspects"], require_aspect_id=require_aspect_id)
        prediction_row = {
            "row_index": int(row_index),
            "id": str(row["id"]),
            "row_uid": str(row.get("row_uid", "")),
            "original_split": str(row.get("original_split", "")),
            "split": split_name,
            "candidate_aspects": row["candidate_aspects"],
            "gold_pair_labels": row["pair_labels"],
            "raw_output": raw_output,
            "seconds": float(seconds),
            **token_counts,
            **parsed_payload,
        }
        rows.append(prediction_row)
        append_jsonl(prediction_row, predictions_file)
        generated_this_run += 1
        print(
            f"[{split_name}] {row_index + 1}/{len(eval_rows)} "
            f"valid_json={prediction_row['valid_json']} "
            f"schema_valid={prediction_row['schema_valid']} "
            f"predicted={len(prediction_row['pred_pair_labels'])} "
            f"seconds={seconds:.2f}",
            flush=True,
        )

    metrics = metrics_from_rows(rows, eval_rows)
    return {
        "split": split_name,
        "skipped_existing_predictions": False,
        "generated_examples_this_run": int(generated_this_run),
        "wall_seconds_this_run": float(time.time() - split_start),
        "requests_file": str(requests_file),
        "predictions_file": str(predictions_file),
        **metrics,
    }


def build_manifest(
    args: argparse.Namespace,
    data_dir: Path,
    metadata: dict[str, Any],
    row_counts: dict[str, int],
    output_dir: Path,
    started_at: str,
    finished_at: str | None = None,
    runtime_seconds: float | None = None,
) -> dict[str, Any]:
    return {
        "task": "Qwen held-out-aspect LoRA SFT runner",
        "command": " ".join(sys.argv),
        "cwd": str(Path.cwd()),
        "project_root": str(PROJECT_ROOT),
        "git_commit": git_commit(),
        "model_name": args.model_name,
        "loading": {
            "load_in_4bit": bool(args.load_in_4bit),
            "quantisation": "bitsandbytes 4-bit NF4 double quantisation" if args.load_in_4bit else "none",
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "resume_from_adapter": str(args.resume_from_adapter) if args.resume_from_adapter else None,
        },
        "lora": {
            "r": int(args.lora_r),
            "alpha": int(args.lora_alpha),
            "dropout": float(args.lora_dropout),
            "target_modules": parse_target_modules(args.lora_target_modules),
            "save_adapter": bool(args.save_adapter),
            "adapter_output_dir": str(output_dir / args.adapter_output_name),
            "save_epoch_adapters": bool(args.save_epoch_adapters),
        },
        "optimisation": {
            "epochs": int(args.epochs),
            "max_train_steps": args.max_train_steps,
            "batch_size": int(args.batch_size),
            "grad_accumulation_steps": int(args.grad_accumulation_steps),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "warmup_ratio": float(args.warmup_ratio),
            "max_grad_norm": float(args.max_grad_norm),
        },
        "prompt_variant": metadata.get("prompt_variant", args.prompt_variant),
        "split_protocol": {
            "source": "indexed held-out-aspect SFT JSONL",
            "strategy": metadata.get("strategy", args.strategy),
            "eval_label_scope": metadata.get("eval_label_scope", "heldout"),
            "protocol": "fixed held-out-aspect train/validation/test evaluation; reusable for one-aspect LOAO fold directories",
            "heldout_aspects": metadata.get("heldout_aspects"),
            "train_candidate_aspects": metadata.get("train_candidate_aspects"),
            "validation_candidate_aspects": metadata.get("validation_candidate_aspects"),
            "test_candidate_aspects": metadata.get("test_candidate_aspects"),
        },
        "row_scope": {
            "sft_data_dir": str(data_dir),
            "row_counts_after_limits": row_counts,
            "limits": {
                "train": args.train_limit,
                "validation": args.validation_limit,
                "test": args.test_limit,
            },
            "eval_splits": args.eval_split,
        },
        "generation": {
            "max_length": int(args.max_length),
            "max_input_tokens": int(args.max_input_tokens),
            "max_new_tokens": int(args.max_new_tokens),
        },
        "resume_and_recovery": {
            "resume_predictions": bool(args.resume_predictions),
            "skip_existing_predictions": bool(args.skip_existing_predictions),
            "skip_training_if_adapter_exists": bool(args.skip_training_if_adapter_exists),
            "training_resume_limitation": (
                "Adapter weights can be loaded with --resume-from-adapter, but optimiser and scheduler state "
                "are not restored by this runner."
            ),
            "partial_prediction_recovery": (
                "Prediction JSONL files are validated by row_index and id prefix before appending missing rows."
            ),
        },
        "seed": int(args.seed),
        "runtime": {
            "started_at": started_at,
            "finished_at": finished_at,
            "seconds": runtime_seconds,
        },
        "hardware": hardware_summary(),
        "packages": package_versions(),
        "output_dir": str(output_dir),
    }


def build_summary(
    args: argparse.Namespace,
    output_dir: Path,
    data_dir: Path,
    metadata: dict[str, Any],
    row_counts: dict[str, int],
    train_history: list[dict[str, Any]],
    eval_results: list[dict[str, Any]],
    manifest_path: Path,
    adapter_dir: Path | None,
    runtime_seconds: float,
) -> dict[str, Any]:
    return {
        "task": "Qwen held-out-aspect LoRA SFT",
        "model_name": args.model_name,
        "strategy": metadata.get("strategy", args.strategy),
        "prompt_variant": metadata.get("prompt_variant", args.prompt_variant),
        "sft_data_dir": str(data_dir),
        "row_counts_after_limits": row_counts,
        "adapter_dir": str(adapter_dir) if adapter_dir else None,
        "manifest_path": str(manifest_path),
        "runtime_seconds": float(runtime_seconds),
        "train_history": train_history,
        "eval_results": eval_results,
    }


def adapter_dir_for_args(output_dir: Path, args: argparse.Namespace) -> Path:
    return output_dir / args.adapter_output_name


def should_skip_training(output_dir: Path, args: argparse.Namespace) -> bool:
    adapter_dir = adapter_dir_for_args(output_dir, args)
    return bool(args.skip_training_if_adapter_exists and adapter_dir.exists())


def apply_existing_adapter_resume(output_dir: Path, args: argparse.Namespace) -> None:
    adapter_dir = adapter_dir_for_args(output_dir, args)
    if args.skip_training_if_adapter_exists and adapter_dir.exists() and args.resume_from_adapter is None:
        args.resume_from_adapter = adapter_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen held-out-aspect candidate-label LoRA/QLoRA SFT.")
    parser.add_argument("--sft-data-dir", type=Path, default=PROJECT_ROOT / "outputs" / "qwen_heldout_aspect_sft")
    parser.add_argument("--strategy", choices=["label_masked", "example_filtered"], default="example_filtered")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--prompt-variant", choices=PROMPT_VARIANTS, default="indexed")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default=",".join(DEFAULT_TARGET_MODULES))
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--eval-split", action="append", choices=["validation", "test"], default=[])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--save-adapter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adapter-output-name", default="adapter_final")
    parser.add_argument("--save-epoch-adapters", action="store_true")
    parser.add_argument("--resume-from-adapter", type=Path, default=None)
    parser.add_argument("--skip-training-if-adapter-exists", action="store_true")
    parser.add_argument("--resume-predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing-predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate data and write manifest without loading the model.")
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.grad_accumulation_steps < 1:
        raise ValueError("--grad-accumulation-steps must be at least 1.")

    set_seed(args.seed)
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "outputs" / "llm" / f"qwen_lora_heldout_aspect_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_existing_adapter_resume(output_dir, args)

    started_at = datetime.now().isoformat(timespec="seconds")
    wall_start = time.time()
    data_dir = resolve_sft_data_dir(args.sft_data_dir, args.strategy)
    metadata = read_json(data_dir / "metadata.json")
    if metadata.get("prompt_variant") and metadata["prompt_variant"] != args.prompt_variant:
        print(
            f"Using prompt_variant={metadata['prompt_variant']} from metadata instead of CLI default {args.prompt_variant}.",
            flush=True,
        )

    split_rows = {
        split_name: load_sft_split(data_dir, split_name, limit=split_limit(args, split_name))
        for split_name in SPLIT_NAMES
    }
    row_counts = {split_name: int(len(rows)) for split_name, rows in split_rows.items()}
    eval_splits = args.eval_split or ["validation", "test"]
    args.eval_split = eval_splits

    manifest_path = output_dir / "manifest.json"
    manifest = build_manifest(
        args=args,
        data_dir=data_dir,
        metadata=metadata,
        row_counts=row_counts,
        output_dir=output_dir,
        started_at=started_at,
    )
    write_json(manifest, manifest_path)

    if args.dry_run:
        summary = build_summary(
            args=args,
            output_dir=output_dir,
            data_dir=data_dir,
            metadata=metadata,
            row_counts=row_counts,
            train_history=[],
            eval_results=[],
            manifest_path=manifest_path,
            adapter_dir=None,
            runtime_seconds=time.time() - wall_start,
        )
        write_json(summary, output_dir / "summary.json")
        print(json.dumps(summary, indent=2), flush=True)
        return

    tokenizer, model = load_qwen_lora_model(args)
    train_history: list[dict[str, Any]] = []
    adapter_dir = adapter_dir_for_args(output_dir, args)
    if should_skip_training(output_dir, args):
        print(f"Skipping training because adapter directory already exists: {adapter_dir}", flush=True)
    else:
        train_history = train_model(model, tokenizer, split_rows["train"], args, output_dir)
        if args.save_adapter:
            model.save_pretrained(adapter_dir)
            tokenizer.save_pretrained(adapter_dir)

    eval_results: list[dict[str, Any]] = []
    for split_name in eval_splits:
        result = run_prediction_split(
            tokenizer=tokenizer,
            model=model,
            eval_rows=split_rows[split_name],
            split_name=split_name,
            metadata=metadata,
            args=args,
            output_dir=output_dir,
        )
        eval_results.append(result)
        interim_summary = build_summary(
            args=args,
            output_dir=output_dir,
            data_dir=data_dir,
            metadata=metadata,
            row_counts=row_counts,
            train_history=train_history,
            eval_results=eval_results,
            manifest_path=manifest_path,
            adapter_dir=adapter_dir if args.save_adapter else None,
            runtime_seconds=time.time() - wall_start,
        )
        write_json(interim_summary, output_dir / "summary.json")

    finished_at = datetime.now().isoformat(timespec="seconds")
    runtime_seconds = time.time() - wall_start
    final_manifest = build_manifest(
        args=args,
        data_dir=data_dir,
        metadata=metadata,
        row_counts=row_counts,
        output_dir=output_dir,
        started_at=started_at,
        finished_at=finished_at,
        runtime_seconds=runtime_seconds,
    )
    write_json(final_manifest, manifest_path)
    summary = build_summary(
        args=args,
        output_dir=output_dir,
        data_dir=data_dir,
        metadata=metadata,
        row_counts=row_counts,
        train_history=train_history,
        eval_results=eval_results,
        manifest_path=manifest_path,
        adapter_dir=adapter_dir if args.save_adapter else None,
        runtime_seconds=runtime_seconds,
    )
    write_json(summary, output_dir / "summary.json")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved Qwen LoRA held-out-aspect outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
