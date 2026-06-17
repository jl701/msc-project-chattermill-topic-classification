from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.data.fabsa import default_data_dir, load_split, unique_labels
from msc_project.evaluation.metrics import evaluate_pair_and_aspect
from msc_project.llm.qwen_format import aspect_taxonomy, build_messages, parse_model_output


def chat_text(tokenizer, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_model(model_name: str, load_in_4bit: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    quantization_config = None
    if load_in_4bit:
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
    model.eval()
    return tokenizer, model


def generate_one(tokenizer, model, messages, max_input_tokens: int, max_new_tokens: int) -> str:
    prompt = chat_text(tokenizer, messages)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    ).to(model.device)

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
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small zero-shot Qwen FABSA smoke test.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "qwen_zero_shot")
    args = parser.parse_args()

    train_df = load_split(args.data_dir, "train")
    eval_df = load_split(args.data_dir, args.split).head(args.limit).copy()
    aspects = aspect_taxonomy(train_df)
    pair_classes = unique_labels([train_df], "pair_labels")

    print(f"Loading {args.model_name}")
    tokenizer, model = load_model(args.model_name, args.load_in_4bit)

    predictions = []
    start = time.time()
    for index, row in eval_df.iterrows():
        messages = build_messages(row["text"], aspects)
        raw_output = generate_one(tokenizer, model, messages, args.max_input_tokens, args.max_new_tokens)
        parsed = parse_model_output(raw_output, aspects)
        predictions.append(
            {
                "id": str(row["id"]),
                "text": row["text"],
                "gold_pair_labels": row["pair_labels"],
                "raw_output": raw_output,
                "pred_pair_labels": parsed.pair_labels,
                "valid_json": parsed.valid_json,
                "parse_error": parsed.error,
            }
        )
        print(f"{len(predictions)}/{len(eval_df)} valid_json={parsed.valid_json} predicted={len(parsed.pair_labels)}")

    elapsed = time.time() - start
    pred_labels = [row["pred_pair_labels"] for row in predictions]
    metrics = evaluate_pair_and_aspect(eval_df["pair_labels"].tolist(), pred_labels, pair_classes)
    metrics.update(
        {
            "model_name": args.model_name,
            "split": args.split,
            "examples": int(len(eval_df)),
            "valid_json_rate": sum(row["valid_json"] for row in predictions) / max(1, len(predictions)),
            "seconds": elapsed,
            "seconds_per_example": elapsed / max(1, len(predictions)),
            "load_in_4bit": bool(args.load_in_4bit),
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()

