from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.baselines.candidate_label import candidate_pair_labels
from msc_project.data.fabsa import default_data_dir
from msc_project.data.splits import DEFAULT_HELDOUT_ASPECTS, build_heldout_aspect_split, load_all_fabsa
from msc_project.evaluation.metrics import evaluate_pair_and_aspect
from msc_project.llm.qwen_format import build_candidate_messages, parse_model_output


PROMPT_VARIANTS = (
    "standard",
    "conservative",
    "descriptive",
    "conservative_descriptive",
    "indexed",
    "indexed_conservative",
    "indexed_descriptive",
    "indexed_conservative_descriptive",
)


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


def select_eval_rows(frame, limit: int | None, sample: bool, seed: int):
    if limit is None or limit >= len(frame):
        return frame.copy()
    if not sample:
        return frame.head(limit).copy()
    indices = list(range(len(frame)))
    rng = random.Random(seed)
    selected = sorted(rng.sample(indices, limit))
    return frame.iloc[selected].copy()


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


def run_variant(tokenizer, model, eval_df, aspects: list[str], pair_classes: list[str], args, variant: str):
    predictions = []
    start = time.time()
    for row_number, (_, row) in enumerate(eval_df.iterrows(), start=1):
        messages = build_candidate_messages(row["text"], aspects, prompt_variant=variant)
        example_start = time.time()
        raw_output = generate_one(tokenizer, model, messages, args.max_input_tokens, args.max_new_tokens)
        parsed = parse_model_output(raw_output, aspects)
        predictions.append(
            {
                "row_index": row_number - 1,
                "id": str(row["id"]),
                "row_uid": str(row.get("row_uid", "")),
                "original_split": str(row.get("original_split", "")),
                "org_index": int(row["org_index"]),
                "text": row["text"],
                "gold_pair_labels": row["supervision_pair_labels"],
                "raw_output": raw_output,
                "pred_pair_labels": parsed.pair_labels,
                "valid_json": parsed.valid_json,
                "parse_error": parsed.error,
                "seconds": time.time() - example_start,
            }
        )
        print(
            f"[{variant}] {row_number}/{len(eval_df)} "
            f"valid_json={parsed.valid_json} predicted={len(parsed.pair_labels)}",
            flush=True,
        )

    pred_labels = [row["pred_pair_labels"] for row in predictions]
    metrics = evaluate_pair_and_aspect(eval_df["supervision_pair_labels"].tolist(), pred_labels, pair_classes)
    elapsed = time.time() - start
    metrics.update(
        {
            "prompt_variant": variant,
            "examples": int(len(eval_df)),
            "valid_json_rate": sum(row["valid_json"] for row in predictions) / max(1, len(predictions)),
            "predicted_labels_per_example": sum(len(row["pred_pair_labels"]) for row in predictions) / max(1, len(predictions)),
            "seconds": elapsed,
            "seconds_per_example": elapsed / max(1, len(predictions)),
        }
    )
    return metrics, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen held-out-aspect candidate-label smoke tests.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--strategy", choices=["label_masked", "example_filtered"], default="label_masked")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--prompt-variant", action="append", choices=PROMPT_VARIANTS, default=[])
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "qwen_heldout_aspect_smoke")
    args = parser.parse_args()

    variants = args.prompt_variant or list(PROMPT_VARIANTS)
    frame = load_all_fabsa(args.data_dir)
    splits = build_heldout_aspect_split(
        frame,
        DEFAULT_HELDOUT_ASPECTS,
        strategy=args.strategy,
        eval_label_scope="heldout",
    )
    eval_df = select_eval_rows(splits[args.split], args.limit, args.sample, args.seed)
    pair_classes = candidate_pair_labels(DEFAULT_HELDOUT_ASPECTS)

    print(f"Loading {args.model_name}", flush=True)
    tokenizer, model = load_model(args.model_name, args.load_in_4bit)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model_name": args.model_name,
        "strategy": args.strategy,
        "split": args.split,
        "limit": args.limit,
        "sample": bool(args.sample),
        "seed": int(args.seed),
        "heldout_aspects": DEFAULT_HELDOUT_ASPECTS,
        "load_in_4bit": bool(args.load_in_4bit),
        "prompt_variants": {},
    }

    for variant in variants:
        metrics, predictions = run_variant(tokenizer, model, eval_df, DEFAULT_HELDOUT_ASPECTS, pair_classes, args, variant)
        summary["prompt_variants"][variant] = metrics
        with (args.output_dir / f"predictions_{variant}.jsonl").open("w", encoding="utf-8") as handle:
            for row in predictions:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(metrics, indent=2), flush=True)

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved Qwen held-out-aspect smoke outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
