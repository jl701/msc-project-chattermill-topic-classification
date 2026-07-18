from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.data.fabsa import default_data_dir
from msc_project.data.splits import build_heldout_aspect_split, load_all_fabsa
from msc_project.experiments.unified_candidate_pairs import (
    CANDIDATE_SENTIMENTS,
    build_eval_grid,
    build_full_manifest,
    budget_sample,
    load_experiment_config,
    manifest_hash,
)
from msc_project.experiments.unified_pair_evaluation import (
    evaluate_score_matrix,
    pilot_gate,
    score_matrix_from_grid,
    select_threshold,
)
from msc_project.llm.qwen_pair_classifier import (
    CandidatePairBatchCollator,
    QwenPairQLoRAConfig,
    VerbalizerTokenIds,
    load_qwen_pair_qlora,
    score_candidate_pairs,
    training_items_from_pairs,
    validate_verbalizer_token_ids,
)


PILOT_FOLDS = (
    "Company brand: Competitor",
    "Company brand: General satisfaction",
    "Staff support: Email",
)
HISTORICAL_QWEN_VALIDATION = {
    "Company brand: Competitor": {
        "pair_micro_f1": 0.23974763406940064,
        "pair_micro_recall": 0.4418604651162791,
        "false_positive_rows_per_100": 17.786187322611163,
    },
    "Company brand: General satisfaction": {
        "pair_micro_f1": 0.535593220338983,
        "pair_micro_recall": 0.7880299251870324,
        "false_positive_rows_per_100": 42.762535477767265,
    },
    "Staff support: Email": {
        "pair_micro_f1": 0.08627450980392157,
        "pair_micro_recall": 0.8461538461538461,
        "false_positive_rows_per_100": 21.759697256385998,
    },
}


def json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON serialise {type(value).__name__}.")


def write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


def write_jsonl(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def package_versions() -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for name in ("torch", "transformers", "peft", "bitsandbytes", "accelerate"):
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = None
    return values


def ordered_eval_frame(frame: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    ordered = frame.assign(_uid=frame["row_uid"].astype(str)).sort_values("_uid", kind="stable")
    if limit is not None:
        ordered = ordered.head(limit)
    return ordered.drop(columns="_uid").reset_index(drop=True)


def seen_aspects(frame: pd.DataFrame) -> list[str]:
    return sorted(
        {
            str(aspect)
            for labels in frame["supervision_labels"]
            for aspect, _ in labels
        }
    )


def load_frozen_qwen(model_name: str, load_in_4bit: bool = True):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantisation = None
    if load_in_4bit:
        quantisation = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quantisation,
        trust_remote_code=True,
    )
    model.eval()
    ids = validate_verbalizer_token_ids(tokenizer)
    return tokenizer, model, ids


def load_saved_qwen_pair_adapter(model_name: str, adapter_dir: Path):
    """Load a completed adapter for scoring-only crash recovery."""

    from peft import PeftModel

    tokenizer, base_model, ids = load_frozen_qwen(model_name, load_in_4bit=True)
    model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    return tokenizer, model, ids


def prediction_rows(
    frame: pd.DataFrame,
    scores: np.ndarray,
    predictions: list[list[str]],
    threshold: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row_index, row in frame.iterrows():
        rows.append(
            {
                "row_index": int(row_index),
                "id": str(row["id"]),
                "row_uid": str(row["row_uid"]),
                "text": str(row["text"]),
                "gold_pair_labels": list(row["supervision_pair_labels"]),
                "pred_pair_labels": predictions[row_index],
                "candidate_sentiment_scores": {
                    sentiment: float(scores[row_index, index])
                    for index, sentiment in enumerate(CANDIDATE_SENTIMENTS)
                },
                "threshold": float(threshold),
            }
        )
    return rows


def score_fold(
    model: Any,
    tokenizer: Any,
    verbalizer_ids: VerbalizerTokenIds,
    frame: pd.DataFrame,
    heldout_aspect: str,
    variant: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    grid = build_eval_grid(frame, heldout_aspect, variant)
    started = time.time()
    flat_scores = score_candidate_pairs(
        model,
        tokenizer,
        [str(value) for value in grid["text"]],
        [str(value) for value in grid["candidate_text"]],
        max_length=args.max_length,
        batch_size=args.eval_batch_size,
        verbalizer_ids=verbalizer_ids,
        padding_side="right",
    )
    score_seconds = time.time() - started
    scores = score_matrix_from_grid(grid, flat_scores, row_count=len(frame))
    true_pairs = [list(labels) for labels in frame["supervision_pair_labels"]]
    selected = select_threshold(true_pairs, scores, heldout_aspect)
    metrics, predictions = evaluate_score_matrix(
        true_pairs,
        scores,
        heldout_aspect,
        selected.threshold,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    score_rows = grid.copy()
    score_rows["score"] = flat_scores
    score_rows.to_csv(output_dir / "validation_pair_scores.csv", index=False)
    selected.sweep.to_csv(output_dir / "validation_threshold_sweep.csv", index=False)
    write_jsonl(
        prediction_rows(frame, scores, predictions, selected.threshold),
        output_dir / "validation_predictions.jsonl",
    )
    return {
        "heldout_aspect": heldout_aspect,
        "split": "validation",
        "variant": variant,
        "selected_threshold": float(selected.threshold),
        "score_seconds": float(score_seconds),
        "seconds_per_pair": float(score_seconds / len(grid)),
        **metrics,
    }


class EncodedPairDataset(Dataset):
    def __init__(self, items: list[dict[str, list[int]]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.items[index]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_qwen_adapter(
    model: Any,
    tokenizer: Any,
    verbalizer_ids: VerbalizerTokenIds,
    manifest: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], int]:
    pairs = [
        (str(row.text), str(row.candidate_text), int(row.target))
        for row in manifest.itertuples(index=False)
    ]
    items = training_items_from_pairs(
        tokenizer,
        pairs,
        max_length=args.max_length,
        verbalizer_ids=verbalizer_ids,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        EncodedPairDataset(items),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=CandidatePairBatchCollator(tokenizer, padding_side="right"),
    )
    updates_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
    planned_steps = updates_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(planned_steps * args.warmup_ratio),
        num_training_steps=planned_steps,
    )
    device = next(model.parameters()).device
    history: list[dict[str, object]] = []
    global_step = 0
    peak_memory = 0
    optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_started = time.time()
        loss_sum = 0.0
        batches = 0
        for batch_number, batch in enumerate(loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            if not torch.isfinite(output.loss.detach()).item():
                raise RuntimeError(f"Non-finite QLoRA loss at batch {batch_number}.")
            loss = output.loss / args.gradient_accumulation_steps
            loss.backward()
            loss_sum += float(output.loss.detach().cpu())
            batches += 1
            update = (
                batch_number % args.gradient_accumulation_steps == 0
                or batch_number == len(loader)
            )
            if update:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step == 1 or global_step % args.log_every_steps == 0:
                    print(
                        json.dumps(
                            {
                                "event": "qwen_train_progress",
                                "epoch": epoch,
                                "global_step": global_step,
                                "planned_steps": planned_steps,
                                "mean_loss": loss_sum / batches,
                            }
                        ),
                        flush=True,
                    )
        history.append(
            {
                "epoch": int(epoch),
                "global_step": int(global_step),
                "train_loss": float(loss_sum / max(1, batches)),
                "seconds": float(time.time() - epoch_started),
            }
        )
        if torch.cuda.is_available():
            peak_memory = max(peak_memory, int(torch.cuda.max_memory_allocated()))
    return history, peak_memory


def historical_frozen_gate(rows: list[dict[str, object]]) -> dict[str, object]:
    historical = [HISTORICAL_QWEN_VALIDATION[str(row["heldout_aspect"])] for row in rows]
    candidate_mean = float(np.mean([float(row["pair_micro_f1"]) for row in rows]))
    historical_mean = float(np.mean([row["pair_micro_f1"] for row in historical]))
    candidate_fp = float(
        np.mean([float(row["presence_false_positive_rows_per_100"]) for row in rows])
    )
    historical_fp = float(np.mean([row["false_positive_rows_per_100"] for row in historical]))
    reduction = (historical_fp - candidate_fp) / historical_fp if historical_fp else 0.0
    passed = candidate_mean - historical_mean >= -0.01 and reduction >= 0.15
    return {
        "passed": bool(passed),
        "candidate_pair_mean_pair_micro_f1": candidate_mean,
        "historical_json_mean_pair_micro_f1": historical_mean,
        "pair_micro_f1_delta": candidate_mean - historical_mean,
        "candidate_pair_mean_false_positive_rows_per_100": candidate_fp,
        "historical_json_mean_false_positive_rows_per_100": historical_fp,
        "false_positive_reduction_fraction": float(reduction),
        "required_minimum_f1_delta": -0.01,
        "required_false_positive_reduction_fraction": 0.15,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen or QLoRA Qwen on the preregistered unified candidate-pair pilot."
    )
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["frozen", "qlora"], required=True)
    parser.add_argument("--stage", choices=["smoke", "pilot"], default="pilot")
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--variant", choices=["control", "enhanced"], default="enhanced")
    parser.add_argument("--heldout-aspect", action="append", default=[])
    parser.add_argument("--frozen-summary", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every-steps", type=int, default=64)
    parser.add_argument("--train-budget", type=int, default=4096)
    parser.add_argument("--positive-budget", type=int, default=2048)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--train-row-limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_experiment_config()
    if args.stage == "pilot":
        frozen_values = (
            args.seed == 13
            and args.max_length == 384
            and args.batch_size == 1
            and args.gradient_accumulation_steps == 8
            and args.epochs == 1
            and args.learning_rate == 5e-6
            and args.train_budget == 4096
            and args.positive_budget == 2048
            and args.eval_limit is None
            and args.train_row_limit is None
        )
        if not frozen_values:
            raise ValueError("Formal Qwen pilot parameters differ from the preregistration.")
    if args.mode == "qlora":
        if args.variant != "enhanced":
            raise ValueError("The preregistered QLoRA pilot uses only the enhanced variant.")
        if args.frozen_summary is None or not args.frozen_summary.exists():
            raise ValueError("QLoRA requires the completed frozen candidate-pair summary.")
        frozen_payload = json.loads(args.frozen_summary.read_text(encoding="utf-8"))
        if not frozen_payload.get("frozen_gate", {}).get("passed", False):
            raise ValueError("Frozen candidate-pair Qwen did not pass its preregistered gate.")

    folds = args.heldout_aspect or (
        [PILOT_FOLDS[0]] if args.stage == "smoke" else list(PILOT_FOLDS)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "protocol_id": config["protocol_id"],
        "mode": args.mode,
        "stage": args.stage,
        "variant": args.variant,
        "folds": folds,
        "command": " ".join(sys.argv),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "packages": package_versions(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "arguments": vars(args),
    }
    write_json(run_manifest, args.output_dir / "run_manifest.json")
    frame = load_all_fabsa(args.data_dir)
    all_results: list[dict[str, object]] = []

    if args.mode == "frozen":
        tokenizer, model, verbalizer_ids = load_frozen_qwen(args.model_name, load_in_4bit=True)
        for fold_number, heldout_aspect in enumerate(folds, start=1):
            output = args.output_dir / slugify(heldout_aspect)
            summary_path = output / "summary.json"
            if args.resume and summary_path.exists():
                result = json.loads(summary_path.read_text(encoding="utf-8"))["result"]
                all_results.append(result)
                continue
            splits = build_heldout_aspect_split(
                frame,
                [heldout_aspect],
                strategy="example_filtered",
                eval_label_scope="heldout",
                eval_row_scope="all",
            )
            validation = ordered_eval_frame(splits["validation"], args.eval_limit)
            print(
                json.dumps(
                    {
                        "event": "qwen_frozen_fold",
                        "fold": fold_number,
                        "folds": len(folds),
                        "heldout_aspect": heldout_aspect,
                        "pairs": len(validation) * 3,
                    }
                ),
                flush=True,
            )
            result = {
                "model": "qwen_frozen_candidate_pair",
                **score_fold(
                    model,
                    tokenizer,
                    verbalizer_ids,
                    validation,
                    heldout_aspect,
                    args.variant,
                    output,
                    args,
                ),
            }
            all_results.append(result)
            write_json({"result": result}, summary_path)
            interim = {
                "results": all_results,
                "frozen_gate": historical_frozen_gate(all_results)
                if set(row["heldout_aspect"] for row in all_results) == set(PILOT_FOLDS)
                else None,
            }
            write_json(interim, args.output_dir / "summary.json")
        del model, tokenizer
        torch.cuda.empty_cache()
        frozen_gate = (
            historical_frozen_gate(all_results)
            if set(row["heldout_aspect"] for row in all_results) == set(PILOT_FOLDS)
            else None
        )
        final = {"results": all_results, "frozen_gate": frozen_gate}
    else:
        frozen_payload = json.loads(args.frozen_summary.read_text(encoding="utf-8"))
        frozen_rows = pd.DataFrame(frozen_payload["results"])
        for fold_number, heldout_aspect in enumerate(folds, start=1):
            output = args.output_dir / slugify(heldout_aspect)
            summary_path = output / "summary.json"
            if args.resume and summary_path.exists():
                all_results.append(json.loads(summary_path.read_text(encoding="utf-8"))["result"])
                continue
            splits = build_heldout_aspect_split(
                frame,
                [heldout_aspect],
                strategy="example_filtered",
                eval_label_scope="heldout",
                eval_row_scope="all",
            )
            train = splits["train"]
            if args.train_row_limit is not None:
                train = train.head(args.train_row_limit).copy()
            validation = ordered_eval_frame(splits["validation"], args.eval_limit)
            candidates = seen_aspects(train)
            full_manifest = build_full_manifest(train, candidates, args.variant, seed=args.seed)
            manifest = budget_sample(
                full_manifest,
                total_budget=args.train_budget,
                positive_budget=args.positive_budget,
                seed=args.seed,
            )
            output.mkdir(parents=True, exist_ok=True)
            training_manifest_path = output / "training_manifest.jsonl"
            if not (args.resume and training_manifest_path.exists()):
                write_jsonl(manifest.to_dict(orient="records"), training_manifest_path)
            manifest_digest = manifest_hash(manifest)
            set_seed(args.seed)
            qlora_config = QwenPairQLoRAConfig(
                load_in_4bit=True,
                gradient_checkpointing=True,
                lora_r=4,
                lora_alpha=8,
                lora_dropout=0.05,
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            )
            adapter_dir = output / "adapter"
            training_summary_path = output / "training_summary.json"
            adapter_complete = (adapter_dir / "adapter_model.safetensors").exists()
            if args.resume and adapter_complete:
                tokenizer, model, verbalizer_ids = load_saved_qwen_pair_adapter(
                    args.model_name,
                    adapter_dir,
                )
                if training_summary_path.exists():
                    recovered_training = json.loads(training_summary_path.read_text(encoding="utf-8"))
                    history = recovered_training.get("history", [])
                    peak_memory = int(recovered_training.get("peak_cuda_memory_bytes", 0))
                    train_seconds = float(recovered_training.get("train_seconds", 0.0))
                else:
                    train_seconds = max(
                        0.0,
                        (adapter_dir / "adapter_model.safetensors").stat().st_mtime
                        - training_manifest_path.stat().st_mtime,
                    )
                    history = [
                        {
                            "recovered_from_completed_adapter_after_scoring_oom": True,
                            "global_step": 512,
                            "train_loss_from_captured_stdout": 1.4367384985943363
                            if heldout_aspect == "Company brand: General satisfaction"
                            else None,
                        }
                    ]
                    peak_memory = 0
                print(
                    json.dumps(
                        {
                            "event": "qwen_resume_saved_adapter",
                            "fold": fold_number,
                            "heldout_aspect": heldout_aspect,
                            "adapter_dir": str(adapter_dir),
                        }
                    ),
                    flush=True,
                )
            else:
                tokenizer, model, verbalizer_ids = load_qwen_pair_qlora(args.model_name, qlora_config)
                print(
                    json.dumps(
                        {
                            "event": "qwen_qlora_fold",
                            "fold": fold_number,
                            "folds": len(folds),
                            "heldout_aspect": heldout_aspect,
                            "training_pairs": len(manifest),
                            "manifest_hash": manifest_digest,
                        }
                    ),
                    flush=True,
                )
                train_started = time.time()
                history, peak_memory = train_qwen_adapter(
                    model,
                    tokenizer,
                    verbalizer_ids,
                    manifest,
                    args,
                )
                train_seconds = time.time() - train_started
                model.save_pretrained(adapter_dir)
                write_json(
                    {
                        "training_manifest_hash": manifest_digest,
                        "train_seconds": float(train_seconds),
                        "peak_cuda_memory_bytes": int(peak_memory),
                        "history": history,
                    },
                    training_summary_path,
                )
            model.eval()
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = True
            scored = score_fold(
                model,
                tokenizer,
                verbalizer_ids,
                validation,
                heldout_aspect,
                args.variant,
                output,
                args,
            )
            result = {
                "model": "qwen_qlora_candidate_pair",
                "training_manifest_hash": manifest_digest,
                "training_pairs": int(len(manifest)),
                "training_positive_pairs": int(manifest["target"].sum()),
                "train_seconds": float(train_seconds),
                "peak_cuda_memory_bytes": int(peak_memory),
                "history": history,
                **scored,
            }
            all_results.append(result)
            write_json({"result": result}, summary_path)
            del model, tokenizer
            torch.cuda.empty_cache()
            write_json({"results": all_results}, args.output_dir / "summary.json")

        qlora_rows = pd.DataFrame(all_results)
        frozen_subset = frozen_rows[frozen_rows["heldout_aspect"].isin(folds)]
        qlora_gate = (
            pilot_gate(qlora_rows, frozen_subset)
            if set(qlora_rows["heldout_aspect"]) == set(PILOT_FOLDS)
            else None
        )
        final = {
            "results": all_results,
            "frozen_reference": frozen_subset.to_dict(orient="records"),
            "qlora_gate": qlora_gate,
        }

    pd.DataFrame(all_results).to_csv(args.output_dir / "all_results.csv", index=False)
    run_manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(run_manifest, args.output_dir / "run_manifest.json")
    write_json(final, args.output_dir / "summary.json")
    print(json.dumps({"event": "qwen_run_complete", **final}, default=json_default), flush=True)


if __name__ == "__main__":
    main()
