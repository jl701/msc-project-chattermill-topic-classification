from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.baselines.candidate_label import candidate_pair_labels
from msc_project.data.fabsa import default_data_dir
from msc_project.data.splits import all_aspects, build_heldout_aspect_split, load_all_fabsa
from msc_project.evaluation.metrics import evaluate_pair_and_aspect
from msc_project.llm.candidate_label import (
    PROMPT_VARIANTS,
    aggregate_llm_diagnostics,
    build_candidate_messages,
    diagnostic_dict,
    parse_candidate_output,
)
from msc_project.llm.qwen_local import generate_one, load_qwen_causal_lm


SPREAD_METRICS = [
    "pair_samples_f1",
    "pair_micro_f1",
    "pair_micro_precision",
    "pair_micro_recall",
    "pair_macro_f1",
    "pair_false_positive_rows_per_100",
    "pair_false_positive_labels_per_100",
    "pair_false_negative_rows_per_100",
    "pair_exact_match_rate",
    "aspect_samples_f1",
    "aspect_micro_f1",
    "aspect_micro_precision",
    "aspect_micro_recall",
    "aspect_macro_f1",
    "aspect_false_positive_rows_per_100",
    "aspect_false_positive_labels_per_100",
    "aspect_false_negative_rows_per_100",
    "aspect_exact_match_rate",
    "presence_precision",
    "presence_recall",
    "presence_f1",
    "presence_prevalence",
    "presence_false_positive_rows_per_100",
    "presence_false_negative_rows_per_100",
    "sentiment_accuracy_when_gold_aspect_predicted",
    "valid_json_rate",
    "schema_valid_rate",
    "predicted_labels_per_example",
    "gold_labels_per_example",
    "seconds_per_example",
    "mean_latency_seconds",
    "median_latency_seconds",
]


def aspect_slug(aspect: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", aspect).strip("_").lower()
    return slug or "aspect"


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
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_eval_rows(frame: pd.DataFrame, limit: int | None, sample: bool, seed: int) -> pd.DataFrame:
    if limit is None or limit >= len(frame):
        return frame.copy()
    if not sample:
        return frame.head(limit).copy()
    rng = random.Random(seed)
    selected = sorted(rng.sample(list(range(len(frame))), limit))
    return frame.iloc[selected].copy()


def selected_aspects(frame: pd.DataFrame, requested_aspects: list[str]) -> list[str]:
    aspects = all_aspects(frame)
    if not requested_aspects:
        return aspects
    unknown = sorted(set(requested_aspects) - set(aspects))
    if unknown:
        raise ValueError(f"Unknown held-out aspects: {unknown}")
    return requested_aspects


def split_counts(splits: dict[str, pd.DataFrame]) -> dict[str, int]:
    return {
        "train_rows": int(len(splits["train"])),
        "validation_rows": int(len(splits["validation"])),
        "validation_positive_rows": int(splits["validation"]["supervision_pair_labels"].apply(len).gt(0).sum()),
        "test_rows": int(len(splits["test"])),
        "test_positive_rows": int(splits["test"]["supervision_pair_labels"].apply(len).gt(0).sum()),
    }


def prediction_path(output_dir: Path, aspect: str, split_name: str, variant: str) -> Path:
    return output_dir / "predictions" / aspect_slug(aspect) / f"predictions_{split_name}_{variant}.jsonl"


def request_path(output_dir: Path, aspect: str, split_name: str, variant: str) -> Path:
    return output_dir / "requests" / aspect_slug(aspect) / f"requests_{split_name}_{variant}.jsonl"


def existing_predictions_are_usable(rows: list[dict[str, Any]], eval_df: pd.DataFrame) -> bool:
    if len(rows) > len(eval_df):
        return False
    for expected_index, row in enumerate(rows):
        if int(row.get("row_index", -1)) != expected_index:
            return False
        if str(row.get("id", "")) != str(eval_df.iloc[expected_index]["id"]):
            return False
        if "pred_pair_labels" not in row:
            return False
    return True


def complete_existing_predictions(path: Path, eval_df: pd.DataFrame) -> bool:
    try:
        rows = read_jsonl(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    return len(rows) == len(eval_df) and existing_predictions_are_usable(rows, eval_df)


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
            }
        )
    return info


def build_request_rows(eval_df: pd.DataFrame, aspect: str, variant: str) -> list[dict[str, Any]]:
    request_rows = []
    for row_number, (_, row) in enumerate(eval_df.iterrows(), start=1):
        messages = build_candidate_messages(
            text=str(row["text"]),
            aspects=[aspect],
            prompt_variant=variant,
            output_container="array",
        )
        request_rows.append(
            {
                "row_index": row_number - 1,
                "id": str(row["id"]),
                "messages": messages,
                "prompt_variant": variant,
                "candidate_aspects": [aspect],
                "output_container": "array",
            }
        )
    return request_rows


def metrics_from_rows(
    rows: list[dict[str, Any]],
    eval_df: pd.DataFrame,
    pair_classes: list[str],
) -> dict[str, Any]:
    pred_labels = [row["pred_pair_labels"] for row in rows]
    true_labels = eval_df["supervision_pair_labels"].tolist()
    metrics = evaluate_pair_and_aspect(true_labels, pred_labels, pair_classes)
    diagnostics = aggregate_llm_diagnostics(rows)
    recorded_seconds = sum(float(row.get("seconds") or 0.0) for row in rows)
    examples = max(1, len(rows))
    metrics.update(
        {
            "examples": int(len(rows)),
            "positive_gold_rows": int(sum(len(row) > 0 for row in true_labels)),
            "predicted_labels_per_example": float(sum(len(row) for row in pred_labels) / examples),
            "gold_labels_per_example": float(sum(len(row) for row in true_labels) / examples),
            "seconds": float(recorded_seconds),
            "seconds_per_example": float(recorded_seconds / examples),
            **diagnostics,
        }
    )
    return metrics


def run_fold(
    eval_df: pd.DataFrame,
    split_name: str,
    aspect: str,
    variant: str,
    pair_classes: list[str],
    counts: dict[str, int],
    tokenizer: Any,
    model: Any,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    requests_file = request_path(output_dir, aspect, split_name, variant)
    predictions_file = prediction_path(output_dir, aspect, split_name, variant)
    write_jsonl(build_request_rows(eval_df, aspect, variant), requests_file)

    existing_rows: list[dict[str, Any]] = []
    if args.resume:
        try:
            existing_rows = read_jsonl(predictions_file)
        except (OSError, json.JSONDecodeError, ValueError):
            existing_rows = []
        if not existing_predictions_are_usable(existing_rows, eval_df):
            existing_rows = []
            if predictions_file.exists():
                predictions_file.unlink()

    if not args.resume and predictions_file.exists():
        predictions_file.unlink()

    if args.dry_run:
        return {
            "heldout_aspect": aspect,
            "split": split_name,
            "prompt_variant": variant,
            "dry_run": True,
            "examples": int(len(eval_df)),
            **counts,
            "requests_file": str(requests_file),
            "predictions_file": str(predictions_file),
        }

    rows = list(existing_rows)
    generated_this_run = 0
    fold_wall_start = time.time()

    if len(rows) < len(eval_df) and (tokenizer is None or model is None):
        raise RuntimeError("Model is required because this fold has missing predictions.")

    for row_number, (_, row) in enumerate(eval_df.iterrows(), start=1):
        row_index = row_number - 1
        if row_index < len(rows):
            continue

        messages = build_candidate_messages(
            text=str(row["text"]),
            aspects=[aspect],
            prompt_variant=variant,
            output_container="array",
        )
        example_start = time.time()
        raw_output = generate_one(
            tokenizer,
            model,
            messages,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
        )
        seconds = time.time() - example_start
        parsed = parse_candidate_output(
            raw_output,
            [aspect],
            require_aspect_id=variant.startswith("indexed"),
            allow_object_wrapper=True,
        )
        prediction_row = {
            "row_index": row_index,
            "id": str(row["id"]),
            "row_uid": str(row.get("row_uid", "")),
            "original_split": str(row.get("original_split", "")),
            "org_index": int(row["org_index"]),
            "text": str(row["text"]),
            "gold_pair_labels": row["supervision_pair_labels"],
            "raw_output": raw_output,
            "pred_pair_labels": parsed.pair_labels,
            **diagnostic_dict(parsed),
            "seconds": seconds,
            "prompt_variant": variant,
            "candidate_aspects": [aspect],
        }
        rows.append(prediction_row)
        append_jsonl(prediction_row, predictions_file)
        generated_this_run += 1
        print(
            f"[{split_name} {aspect} {variant}] {row_number}/{len(eval_df)} "
            f"valid_json={parsed.valid_json} schema_valid={parsed.schema_valid} "
            f"predicted={len(parsed.pair_labels)} seconds={seconds:.2f}",
            flush=True,
        )

    metrics = metrics_from_rows(rows, eval_df, pair_classes)
    result = {
        "heldout_aspect": aspect,
        "split": split_name,
        "prompt_variant": variant,
        "dry_run": False,
        "model_name": args.model_name,
        "load_in_4bit": bool(args.load_in_4bit),
        "prompt_output_container": "array",
        "eval_label_scope": "heldout",
        "eval_row_scope": args.eval_row_scope,
        "limit": args.limit,
        "sample": bool(args.sample),
        "seed": int(args.seed),
        **counts,
        **metrics,
        "generated_examples_this_run": int(generated_this_run),
        "wall_seconds_this_run": float(time.time() - fold_wall_start),
        "requests_file": str(requests_file),
        "predictions_file": str(predictions_file),
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def aggregate_spread(results: list[dict[str, Any]]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    frame = pd.DataFrame(results)
    rows = []
    for (split_name, variant), group in frame.groupby(["split", "prompt_variant"]):
        row: dict[str, Any] = {
            "split": split_name,
            "prompt_variant": variant,
            "aspects": int(group["heldout_aspect"].nunique()),
        }
        for metric in SPREAD_METRICS:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["split", "prompt_variant"]).reset_index(drop=True)


def write_result_tables(results: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_frame = pd.DataFrame(results)
    if not results_frame.empty:
        results_frame.to_csv(output_dir / "all_results.csv", index=False)
        for split_name, split_group in results_frame.groupby("split"):
            split_group.to_csv(output_dir / f"{split_name}_results.csv", index=False)
    spread = aggregate_spread(results)
    if not spread.empty:
        spread.to_csv(output_dir / "spread.csv", index=False)
    return spread.to_dict(orient="records")


def build_summary(
    args: argparse.Namespace,
    output_dir: Path,
    aspects: list[str],
    variants: list[str],
    split_names: list[str],
    started_at: str,
    results: list[dict[str, Any]],
    finished_at: str | None = None,
) -> dict[str, Any]:
    spread = write_result_tables(results, output_dir)
    return {
        "task": "Qwen open-weight zero-shot leave-one-aspect-out held-out-aspect evaluation",
        "model_name": args.model_name,
        "loading": {
            "load_in_4bit": bool(args.load_in_4bit),
            "quantisation": "bitsandbytes 4-bit NF4 double quantisation" if args.load_in_4bit else "none",
        },
        "prompt_variants": variants,
        "prompt_output_container": "array",
        "protocol": "LOAO over FABSA aspects with indexed candidate-label prompts",
        "strategy": args.strategy,
        "eval_label_scope": "heldout",
        "eval_row_scope": args.eval_row_scope,
        "splits": split_names,
        "heldout_aspects": aspects,
        "limit": args.limit,
        "sample": bool(args.sample),
        "seed": int(args.seed),
        "max_input_tokens": int(args.max_input_tokens),
        "max_new_tokens": int(args.max_new_tokens),
        "resume": bool(args.resume),
        "dry_run": bool(args.dry_run),
        "output_dir": str(output_dir),
        "command": " ".join(sys.argv),
        "cwd": str(PROJECT_ROOT),
        "git_commit": git_commit(),
        "hardware": hardware_summary(),
        "started_at": started_at,
        "finished_at": finished_at,
        "results": results,
        "spread": spread,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Qwen zero-shot LOAO held-out-aspect evaluation.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--strategy", choices=["label_masked", "example_filtered"], default="label_masked")
    parser.add_argument("--split", choices=["validation", "test", "both"], default="both")
    parser.add_argument("--heldout-aspect", action="append", default=[])
    parser.add_argument("--eval-row-scope", choices=["all", "containing_heldout"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--prompt-variant", action="append", choices=PROMPT_VARIANTS, default=[])
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true", help="Reuse complete or partial prediction JSONL files.")
    parser.add_argument("--dry-run", action="store_true", help="Write request JSONL files without loading the model.")
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "outputs" / "llm" / f"qwen_loao_heldout_aspect_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now().isoformat(timespec="seconds")
    frame = load_all_fabsa(args.data_dir)
    aspects = selected_aspects(frame, args.heldout_aspect)
    variants = args.prompt_variant or ["indexed"]
    split_names = ["validation", "test"] if args.split == "both" else [args.split]

    work_items = []
    for aspect in aspects:
        splits = build_heldout_aspect_split(
            frame,
            [aspect],
            strategy=args.strategy,
            eval_label_scope="heldout",
            eval_row_scope=args.eval_row_scope,
        )
        counts = split_counts(splits)
        pair_classes = candidate_pair_labels([aspect])
        for split_name in split_names:
            eval_df = select_eval_rows(splits[split_name], args.limit, args.sample, args.seed)
            for variant in variants:
                work_items.append(
                    {
                        "aspect": aspect,
                        "split_name": split_name,
                        "variant": variant,
                        "eval_df": eval_df,
                        "pair_classes": pair_classes,
                        "counts": counts,
                    }
                )

    needs_model = False
    if not args.dry_run:
        for item in work_items:
            path = prediction_path(output_dir, item["aspect"], item["split_name"], item["variant"])
            if not (args.resume and complete_existing_predictions(path, item["eval_df"])):
                needs_model = True
                break

    tokenizer = None
    model = None
    if needs_model:
        print(f"Loading {args.model_name} load_in_4bit={args.load_in_4bit}", flush=True)
        tokenizer, model = load_qwen_causal_lm(args.model_name, load_in_4bit=args.load_in_4bit)
    elif not args.dry_run:
        print("All requested prediction files are complete; recomputing metrics without loading the model.", flush=True)

    all_results: list[dict[str, Any]] = []
    for step, item in enumerate(work_items, start=1):
        print(
            json.dumps(
                {
                    "step": step,
                    "total": len(work_items),
                    "heldout_aspect": item["aspect"],
                    "split": item["split_name"],
                    "prompt_variant": item["variant"],
                }
            ),
            flush=True,
        )
        result = run_fold(
            eval_df=item["eval_df"],
            split_name=item["split_name"],
            aspect=item["aspect"],
            variant=item["variant"],
            pair_classes=item["pair_classes"],
            counts=item["counts"],
            tokenizer=tokenizer,
            model=model,
            args=args,
            output_dir=output_dir,
        )
        all_results.append(result)
        interim_summary = build_summary(
            args=args,
            output_dir=output_dir,
            aspects=aspects,
            variants=variants,
            split_names=split_names,
            started_at=started_at,
            results=all_results,
        )
        write_json(interim_summary, output_dir / "summary.json")

    finished_at = datetime.now().isoformat(timespec="seconds")
    summary = build_summary(
        args=args,
        output_dir=output_dir,
        aspects=aspects,
        variants=variants,
        split_names=split_names,
        started_at=started_at,
        finished_at=finished_at,
        results=all_results,
    )
    write_json(summary, output_dir / "summary.json")
    print(json.dumps(summary["spread"], indent=2), flush=True)
    print(f"Saved Qwen LOAO outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
