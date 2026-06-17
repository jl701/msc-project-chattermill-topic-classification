from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.data.fabsa import default_data_dir, load_split, unique_labels
from msc_project.evaluation.metrics import evaluate_pair_and_aspect
from msc_project.llm.qwen_format import aspect_taxonomy, build_messages, parse_model_output


def chat_text(tokenizer, messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str:
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


class ChatSftDataset(Dataset):
    def __init__(self, frame, tokenizer, aspects: list[str], max_length: int) -> None:
        self.items = []
        for _, row in frame.iterrows():
            prompt_messages = build_messages(row["text"], aspects)
            full_messages = build_messages(row["text"], aspects, row["pair_labels"])

            prompt_text = chat_text(tokenizer, prompt_messages, add_generation_prompt=True)
            full_text = chat_text(tokenizer, full_messages, add_generation_prompt=False)

            prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_length)["input_ids"]
            encoded = tokenizer(full_text, truncation=True, max_length=max_length)
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            labels = copy.deepcopy(input_ids)
            labels[: min(len(prompt_ids), len(labels))] = [-100] * min(len(prompt_ids), len(labels))

            self.items.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        return self.items[index]


def collate_batch(batch: list[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(item["input_ids"]) for item in batch)
    rows = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        rows["input_ids"].append(torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id)]))
        rows["attention_mask"].append(torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
        rows["labels"].append(torch.cat([item["labels"], torch.full((pad_len,), -100)]))
    return {key: torch.stack(value) for key, value in rows.items()}


def load_qwen_model(model_name: str, lora_r: int, lora_alpha: int, lora_dropout: float):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return tokenizer, model


def train_epoch(model, loader, optimizer, scheduler, device, grad_accumulation_steps: int) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss / grad_accumulation_steps
        loss.backward()
        total_loss += float(loss.detach().cpu()) * grad_accumulation_steps

        if step % grad_accumulation_steps == 0 or step == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    return total_loss / max(1, len(loader))


@torch.no_grad()
def generate_predictions(tokenizer, model, frame, aspects: list[str], max_input_tokens: int, max_new_tokens: int):
    model.eval()
    predictions = []
    device = next(model.parameters()).device
    for _, row in frame.iterrows():
        messages = build_messages(row["text"], aspects)
        prompt = chat_text(tokenizer, messages, add_generation_prompt=True)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        ).to(device)

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
        parsed = parse_model_output(raw_output, aspects)
        predictions.append(
            {
                "id": str(row["id"]),
                "gold_pair_labels": row["pair_labels"],
                "raw_output": raw_output,
                "pred_pair_labels": parsed.pair_labels,
                "valid_json": parsed.valid_json,
                "parse_error": parsed.error,
            }
        )
    return predictions


def evaluate_predictions(frame, predictions, pair_classes: list[str]) -> dict[str, float]:
    pred_labels = [row["pred_pair_labels"] for row in predictions]
    metrics = evaluate_pair_and_aspect(frame["pair_labels"].tolist(), pred_labels, pair_classes)
    metrics["valid_json_rate"] = sum(row["valid_json"] for row in predictions) / max(1, len(predictions))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny Qwen3-4B LoRA/QLoRA pilot on FABSA.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "qwen_lora_pilot")
    parser.add_argument("--train-limit", type=int, default=500)
    parser.add_argument("--eval-limit", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--save-adapter", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(13)
    train_df_full = load_split(args.data_dir, "train")
    validation_df = load_split(args.data_dir, "validation").head(args.eval_limit).copy()
    train_df = train_df_full.head(args.train_limit).copy()
    aspects = aspect_taxonomy(train_df_full)
    pair_classes = unique_labels([train_df_full], "pair_labels")

    tokenizer, model = load_qwen_model(args.model_name, args.lora_r, args.lora_alpha, args.lora_dropout)
    device = next(model.parameters()).device
    print(f"Training on device: {device}")

    dataset = ChatSftDataset(train_df, tokenizer, aspects, args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
    )
    update_steps = max(1, (len(loader) // args.grad_accumulation_steps) * args.epochs)
    warmup_steps = int(update_steps * args.warmup_ratio)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, update_steps)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, optimizer, scheduler, device, args.grad_accumulation_steps)
        predictions = generate_predictions(
            tokenizer,
            model,
            validation_df,
            aspects,
            args.max_input_tokens,
            args.max_new_tokens,
        )
        metrics = evaluate_predictions(validation_df, predictions, pair_classes)
        metrics.update({"epoch": epoch, "train_loss": float(loss)})
        history.append(metrics)
        print(json.dumps(metrics, indent=2))

        with (args.output_dir / f"predictions_epoch_{epoch}.jsonl").open("w", encoding="utf-8") as handle:
            for row in predictions:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    elapsed = time.time() - start
    summary = {
        "model_name": args.model_name,
        "train_examples": int(len(train_df)),
        "eval_examples": int(len(validation_df)),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "grad_accumulation_steps": int(args.grad_accumulation_steps),
        "learning_rate": float(args.learning_rate),
        "lora_r": int(args.lora_r),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "seconds": elapsed,
        "history": history,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.save_adapter:
        model.save_pretrained(args.output_dir / "adapter")
        tokenizer.save_pretrained(args.output_dir / "adapter")

    print(json.dumps(summary, indent=2))
    print(f"Saved pilot outputs to {args.output_dir}")


if __name__ == "__main__":
    main()

