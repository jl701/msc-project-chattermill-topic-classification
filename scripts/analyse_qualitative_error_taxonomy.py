from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.evaluation.metrics import PAIR_SEPARATOR


DEFAULT_LOCAL = (
    PROJECT_ROOT
    / "outputs"
    / "baselines"
    / "aspect_label_aware_transformer_sentiment_lr3e-5_ep3_neg3_example_filtered"
    / "example_filtered"
    / "best_test_predictions.jsonl"
)
DEFAULT_QWEN = PROJECT_ROOT / "outputs" / "qwen_heldout_aspect_smoke" / "test_indexed_full" / "predictions_indexed.jsonl"
DEFAULT_FLASH_LITE = PROJECT_ROOT / "outputs" / "llm" / "gemini_candidate_label_20260701_034545_flash_lite_fixed_full" / "predictions_test_indexed.jsonl"
DEFAULT_FLASH = PROJECT_ROOT / "outputs" / "llm" / "gemini_candidate_label_20260701_0145_fixed_full" / "predictions_test_indexed.jsonl"
DEFAULT_PRO = PROJECT_ROOT / "outputs" / "llm" / "gemini_candidate_label_20260701_031040_pro_fixed_full" / "predictions_test_indexed.jsonl"
DEFAULT_FLASH_LITE_DESC = (
    PROJECT_ROOT
    / "outputs"
    / "llm"
    / "gemini_candidate_label_20260701_desc_flash_lite_test_full"
    / "predictions_test_indexed_generated_descriptions.jsonl"
)
DEFAULT_FLASH_LITE_BOUNDARY = (
    PROJECT_ROOT
    / "outputs"
    / "llm"
    / "gemini_candidate_label_20260701_desc_boundary_flash_lite_test_full"
    / "predictions_test_indexed_generated_descriptions.jsonl"
)
DEFAULT_FLASH_DESC = (
    PROJECT_ROOT
    / "outputs"
    / "llm"
    / "gemini_candidate_label_20260701_desc_flash_test_full"
    / "predictions_test_indexed_generated_descriptions.jsonl"
)
DEFAULT_PRO_CASCADE = (
    PROJECT_ROOT / "outputs" / "analysis" / "local_gemini_cascade_pro_grid1" / "selected_test_predictions.jsonl"
)
DEFAULT_CASCADE_DEEP_DIVE = PROJECT_ROOT / "outputs" / "analysis" / "local_gemini_cascade_pro_deep_dive" / "summary.json"
DEFAULT_QWEN_LOAO_SUMMARY = PROJECT_ROOT / "outputs" / "analysis" / "qwen_loao_full_interpretation_20260701" / "summary.json"
DEFAULT_QWEN_VS_DISTILBERT_ASPECTS = (
    PROJECT_ROOT / "outputs" / "analysis" / "qwen_loao_full_interpretation_20260701" / "qwen_vs_distilbert_per_aspect_test.csv"
)
DEFAULT_QWEN_POSITIVE_GAP = (
    PROJECT_ROOT / "outputs" / "analysis" / "qwen_loao_full_interpretation_20260701" / "qwen_all_row_vs_positive_gold_test.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis" / "qualitative_error_taxonomy_20260702"


FIXED_MODEL_ORDER = [
    "local",
    "qwen",
    "flash_lite",
    "flash",
    "pro",
    "flash_lite_desc",
    "flash_lite_boundary",
    "flash_desc",
    "pro_cascade",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def row_key(row: dict[str, Any]) -> str:
    row_uid = row.get("row_uid")
    if row_uid:
        return str(row_uid)
    split = row.get("original_split") or row.get("split") or "test"
    return f"{split}:{row.get('id')}"


def aspect(label: str) -> str:
    return label.rsplit(PAIR_SEPARATOR, maxsplit=1)[0]


def sentiment(label: str) -> str:
    return label.rsplit(PAIR_SEPARATOR, maxsplit=1)[1]


def sample_f1(gold: set[str], pred: set[str]) -> float:
    denominator = len(gold) + len(pred)
    return 2 * len(gold & pred) / denominator if denominator else 0.0


def row_counts(gold: set[str], pred: set[str]) -> dict[str, int]:
    return {
        "tp": len(gold & pred),
        "fp": len(pred - gold),
        "fn": len(gold - pred),
        "pred_count": len(pred),
    }


def model_error_tags(gold: set[str], pred: set[str]) -> list[str]:
    gold_aspects = {aspect(label) for label in gold}
    pred_aspects = {aspect(label) for label in pred}
    tags: list[str] = []
    if gold == pred:
        tags.append("exact")
    if gold and not pred:
        tags.append("empty_abstention")
    if gold_aspects - pred_aspects:
        tags.append("aspect_miss")
    if pred_aspects - gold_aspects:
        tags.append("aspect_overprediction")
    if any(aspect(label) in pred_aspects and label not in pred for label in gold):
        tags.append("sentiment_error_when_aspect_predicted")
    if not tags:
        tags.append("partial_pair_mismatch")
    return tags


def load_model_rows(sources: dict[str, Path]) -> dict[str, dict[str, dict[str, Any]]]:
    loaded: dict[str, dict[str, dict[str, Any]]] = {}
    missing = [f"{name}: {path}" for name, path in sources.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing source prediction files:\n" + "\n".join(missing))
    for name, path in sources.items():
        loaded[name] = {row_key(row): row for row in read_jsonl(path)}
    return loaded


def align_fixed_rows(model_rows: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    base_keys = list(model_rows["local"])
    missing = {
        model: sorted(set(base_keys) - set(rows))
        for model, rows in model_rows.items()
        if set(base_keys) - set(rows)
    }
    if missing:
        raise ValueError(f"Prediction files do not align: {missing}")

    aligned = []
    for key in base_keys:
        base = model_rows["local"][key]
        gold = set(base["gold_pair_labels"])
        for model, rows in model_rows.items():
            model_gold = set(rows[key]["gold_pair_labels"])
            if model_gold != gold:
                raise ValueError(f"Gold labels differ for {key} in {model}.")
        aligned.append(
            {
                "row_uid": key,
                "id": str(base.get("id", "")),
                "original_split": str(base.get("original_split", "test")),
                "text": str(base.get("text", "")),
                "gold_pair_labels": sorted(gold),
                "predictions": {
                    model: sorted(set(rows[key]["pred_pair_labels"]))
                    for model, rows in model_rows.items()
                },
            }
        )
    return aligned


def fixed_row_analysis(row: dict[str, Any]) -> dict[str, Any]:
    gold = set(row["gold_pair_labels"])
    predictions = {model: set(labels) for model, labels in row["predictions"].items()}
    model_stats = {
        model: {
            **row_counts(gold, pred),
            "sample_f1": sample_f1(gold, pred),
            "tags": model_error_tags(gold, pred),
        }
        for model, pred in predictions.items()
    }
    row_tags = infer_row_tags(gold, predictions, model_stats)
    return {
        "model_stats": model_stats,
        "row_tags": row_tags,
        "disagreement_span": max(stats["sample_f1"] for stats in model_stats.values())
        - min(stats["sample_f1"] for stats in model_stats.values()),
    }


def infer_row_tags(
    gold: set[str],
    predictions: dict[str, set[str]],
    model_stats: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    tags: dict[str, list[str]] = defaultdict(list)
    gold_aspects = {aspect(label) for label in gold}
    gold_sentiments = {sentiment(label) for label in gold}
    competitor_positive = "Company brand: Competitor | positive" in gold
    has_discounts_gold_or_pred = "Value: Discounts promotions" in gold_aspects or any(
        "Value: Discounts promotions" in {aspect(label) for label in pred}
        for pred in predictions.values()
    )

    for model, pred in predictions.items():
        pred_aspects = {aspect(label) for label in pred}
        pred_sentiments = {sentiment(label) for label in pred}
        if competitor_positive and "Company brand: Competitor" not in pred_aspects:
            tags["competitor_positive_miss"].append(model)
        if "Account management: Account access" not in gold_aspects and "Account management: Account access" in pred_aspects:
            tags["account_access_overprediction"].append(model)
        if has_discounts_gold_or_pred and (set(model_error_tags(gold, pred)) - {"exact"}):
            tags["discounts_value_boundary"].append(model)
        if "neutral" in gold_sentiments and "neutral" not in pred_sentiments:
            tags["neutral_under_recall"].append(model)
        if not pred and gold:
            tags["empty_abstention"].append(model)

    if model_stats["local"]["fp"] > 0 and model_stats["local"]["pred_count"] > len(gold):
        tags["local_overprediction"].append("local")
    if model_stats["qwen"]["fp"] > 0 and model_stats["qwen"]["pred_count"] > len(gold):
        tags["qwen_overprediction"].append("qwen")
    if not predictions["pro"] and predictions["pro_cascade"] and model_stats["pro_cascade"]["sample_f1"] > model_stats["pro"]["sample_f1"]:
        tags["pro_empty_cascade_recovery"].extend(["pro", "pro_cascade"])

    for base, desc in [("flash_lite", "flash_lite_desc"), ("flash_lite", "flash_lite_boundary"), ("flash", "flash_desc")]:
        base_stats = model_stats[base]
        desc_stats = model_stats[desc]
        if desc_stats["fp"] < base_stats["fp"] and desc_stats["sample_f1"] >= base_stats["sample_f1"]:
            tags["description_precision_shift"].extend([base, desc])
        if desc_stats["fn"] > base_stats["fn"] and desc_stats["sample_f1"] < base_stats["sample_f1"]:
            tags["description_recall_loss"].extend([base, desc])
        if desc_stats["sample_f1"] > base_stats["sample_f1"]:
            tags["description_row_gain"].extend([base, desc])

    return {tag: sorted(set(models)) for tag, models in sorted(tags.items())}


def row_without_text(row: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_uid": row["row_uid"],
        "id": row["id"],
        "gold_pair_labels": row["gold_pair_labels"],
        "predictions": row["predictions"],
        "row_tags": analysis["row_tags"],
        "model_stats": analysis["model_stats"],
        "disagreement_span": analysis["disagreement_span"],
    }


def row_with_text(row: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    output = row_without_text(row, analysis)
    output["text"] = row["text"]
    return output


def category_summary(no_text_rows: list[dict[str, Any]], max_examples: int) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    involved_models: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sorted_rows = sorted(
        no_text_rows,
        key=lambda row: (len(row["row_tags"]), row["disagreement_span"], row["row_uid"]),
        reverse=True,
    )
    for row in sorted_rows:
        for tag, models in row["row_tags"].items():
            counts[tag] += 1
            involved_models[tag].update(models)
            if len(examples[tag]) < max_examples:
                examples[tag].append(
                    {
                        "row_uid": row["row_uid"],
                        "gold_pair_labels": row["gold_pair_labels"],
                        "row_tags": {tag: models},
                        "model_sample_f1": {
                            model: round(stats["sample_f1"], 4)
                            for model, stats in row["model_stats"].items()
                        },
                        "predictions": row["predictions"],
                    }
                )
    return {
        "counts": dict(sorted(counts.items())),
        "involved_models": {
            tag: dict(sorted(counter.items()))
            for tag, counter in sorted(involved_models.items())
        },
        "examples": dict(sorted(examples.items())),
    }


def model_summary(no_text_rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for model in FIXED_MODEL_ORDER:
        f1s = [row["model_stats"][model]["sample_f1"] for row in no_text_rows]
        tag_counts = Counter(tag for row in no_text_rows for tag in row["model_stats"][model]["tags"])
        output[model] = {
            "rows": len(no_text_rows),
            "mean_row_sample_f1": mean(f1s),
            "exact_rows": tag_counts.get("exact", 0),
            "empty_abstention_rows": tag_counts.get("empty_abstention", 0),
            "aspect_miss_rows": tag_counts.get("aspect_miss", 0),
            "aspect_overprediction_rows": tag_counts.get("aspect_overprediction", 0),
            "sentiment_error_rows": tag_counts.get("sentiment_error_when_aspect_predicted", 0),
        }
    return output


def read_loao_evidence(summary_path: Path, aspects_path: Path, positive_gap_path: Path) -> dict[str, Any]:
    summary = read_json(summary_path)
    aspects = read_csv(aspects_path)
    positive_gap = read_csv(positive_gap_path)

    def top_rows(rows: list[dict[str, Any]], key: str, reverse: bool, limit: int = 4) -> list[dict[str, Any]]:
        selected = sorted(rows, key=lambda row: float(row[key]), reverse=reverse)[:limit]
        keep = [
            "heldout_aspect",
            "qwen_pair_micro_f1",
            "distilbert_pair_micro_f1",
            "delta_pair_micro_f1",
            "qwen_pair_micro_precision",
            "qwen_pair_micro_recall",
            "qwen_pair_false_positive_rows_per_100",
            "distilbert_pair_false_positive_rows_per_100",
        ]
        return [{name: row.get(name) for name in keep} for row in selected]

    gap_keep = [
        "heldout_aspect",
        "all_row_pair_micro_f1",
        "positive_gold_pair_micro_f1",
        "gap_pair_micro_f1",
        "all_row_fp_rows_per_100",
    ]
    largest_positive_gap = sorted(
        positive_gap,
        key=lambda row: float(row.get("gap_pair_micro_f1") or 0.0),
        reverse=True,
    )[:4]
    return {
        "summary": summary,
        "largest_qwen_pair_micro_gains": top_rows(aspects, "delta_pair_micro_f1", True),
        "largest_qwen_pair_micro_losses": top_rows(aspects, "delta_pair_micro_f1", False),
        "largest_positive_gold_gaps": [
            {name: row.get(name) for name in gap_keep}
            for row in largest_positive_gap
        ],
    }


def build_gemini_prompt(
    category_report: dict[str, Any],
    with_text_rows: list[dict[str, Any]],
    loao_evidence: dict[str, Any],
    max_examples_per_category: int,
    snippet_chars: int,
) -> str:
    selected_examples: list[dict[str, Any]] = []
    rows_by_uid = {row["row_uid"]: row for row in with_text_rows}
    for tag, examples in category_report["examples"].items():
        for example in examples[:max_examples_per_category]:
            row = rows_by_uid[example["row_uid"]]
            selected_examples.append(
                {
                    "category_tag": tag,
                    "row_uid": row["row_uid"],
                    "text_snippet": row["text"][:snippet_chars],
                    "gold_pair_labels": row["gold_pair_labels"],
                    "predictions": row["predictions"],
                    "row_tags": row["row_tags"],
                }
            )

    payload = {
        "category_counts": category_report["counts"],
        "category_involved_models": category_report["involved_models"],
        "selected_fixed_split_examples": selected_examples,
        "loao_evidence": loao_evidence,
    }
    return (
        "You are helping draft a qualitative error taxonomy for an MSc dissertation on "
        "open-topic aspect-based sentiment classification. Use the evidence below to propose "
        "a concise taxonomy of recurring error mechanisms across local DistilBERT, Qwen zero-shot, "
        "Gemini hosted models, aspect-description prompts, and local-to-Gemini cascades.\n\n"
        "Important constraints:\n"
        "- Do not quote review text in the final taxonomy.\n"
        "- Treat these as candidate categories for manual review, not final truth.\n"
        "- Prefer mechanisms that explain multiple systems or motivate later Qwen fine-tuning.\n"
        "- Distinguish observed evidence from interpretation.\n"
        "- Return JSON only.\n\n"
        "Evidence JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def taxonomy_json_schema() -> dict[str, Any]:
    category_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string"},
            "definition": {"type": "string"},
            "diagnostic_signals": {"type": "array", "items": {"type": "string"}},
            "supporting_row_ids": {"type": "array", "items": {"type": "string"}},
            "main_models_involved": {"type": "array", "items": {"type": "string"}},
            "thesis_use": {"type": "string"},
            "qwen_fine_tuning_implication": {"type": "string"},
        },
        "required": [
            "name",
            "definition",
            "diagnostic_signals",
            "supporting_row_ids",
            "main_models_involved",
            "thesis_use",
            "qwen_fine_tuning_implication",
        ],
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "qualitative_error_taxonomy",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "taxonomy": {"type": "array", "items": category_schema},
                    "model_comparison_notes": {"type": "array", "items": {"type": "string"}},
                    "fine_tuning_priorities": {"type": "array", "items": {"type": "string"}},
                    "caveats": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["taxonomy", "model_comparison_notes", "fine_tuning_priorities", "caveats"],
            },
        },
    }


def call_gemini(prompt: str, model: str, max_tokens: int, temperature: float, timeout: float) -> dict[str, Any]:
    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not base_url or not api_key:
        raise RuntimeError("OPENAI_BASE_URL and OPENAI_API_KEY are required for --gemini-draft.")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": taxonomy_json_schema(),
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini draft request failed with HTTP {exc.code}: {body}") from exc


def completion_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Gemini-assisted qualitative error taxonomy packet.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-examples-per-category", type=int, default=8)
    parser.add_argument("--max-gemini-examples-per-category", type=int, default=3)
    parser.add_argument("--snippet-chars", type=int, default=320)
    parser.add_argument("--gemini-draft", action="store_true")
    parser.add_argument("--gemini-model", default="vertex_ai/gemini-2.5-pro")
    parser.add_argument("--gemini-max-tokens", type=int, default=5000)
    parser.add_argument("--gemini-temperature", type=float, default=0.0)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    args = parser.parse_args()

    sources = {
        "local": DEFAULT_LOCAL,
        "qwen": DEFAULT_QWEN,
        "flash_lite": DEFAULT_FLASH_LITE,
        "flash": DEFAULT_FLASH,
        "pro": DEFAULT_PRO,
        "flash_lite_desc": DEFAULT_FLASH_LITE_DESC,
        "flash_lite_boundary": DEFAULT_FLASH_LITE_BOUNDARY,
        "flash_desc": DEFAULT_FLASH_DESC,
        "pro_cascade": DEFAULT_PRO_CASCADE,
    }
    model_rows = load_model_rows(sources)
    aligned_rows = align_fixed_rows(model_rows)

    with_text_rows = []
    no_text_rows = []
    for row in aligned_rows:
        analysis = fixed_row_analysis(row)
        with_text_rows.append(row_with_text(row, analysis))
        no_text_rows.append(row_without_text(row, analysis))

    category_report = category_summary(no_text_rows, args.max_examples_per_category)
    loao_evidence = read_loao_evidence(DEFAULT_QWEN_LOAO_SUMMARY, DEFAULT_QWEN_VS_DISTILBERT_ASPECTS, DEFAULT_QWEN_POSITIVE_GAP)
    summary = {
        "task": "Gemini-assisted qualitative error taxonomy",
        "source_paths": {name: str(path) for name, path in sources.items()},
        "loao_source_paths": {
            "summary": str(DEFAULT_QWEN_LOAO_SUMMARY),
            "qwen_vs_distilbert_per_aspect": str(DEFAULT_QWEN_VS_DISTILBERT_ASPECTS),
            "qwen_positive_gap": str(DEFAULT_QWEN_POSITIVE_GAP),
            "cascade_deep_dive": str(DEFAULT_CASCADE_DEEP_DIVE),
        },
        "parameters": {
            "split": "test",
            "rows": len(no_text_rows),
            "max_examples_per_category": args.max_examples_per_category,
            "max_gemini_examples_per_category": args.max_gemini_examples_per_category,
            "snippet_chars": args.snippet_chars,
            "gemini_draft": bool(args.gemini_draft),
            "gemini_model": args.gemini_model if args.gemini_draft else None,
        },
        "model_summary": model_summary(no_text_rows),
        "category_report": category_report,
        "loao_evidence": loao_evidence,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(summary, args.output_dir / "summary.json")
    write_jsonl(no_text_rows, args.output_dir / "fixed_split_cases_no_text.jsonl")
    write_jsonl(with_text_rows, args.output_dir / "fixed_split_cases_with_text.jsonl")
    write_csv(
        [
            {
                "tag": tag,
                "rows": count,
                "involved_models": json.dumps(category_report["involved_models"].get(tag, {}), ensure_ascii=False),
            }
            for tag, count in category_report["counts"].items()
        ],
        args.output_dir / "category_counts.csv",
    )

    prompt = build_gemini_prompt(
        category_report=category_report,
        with_text_rows=with_text_rows,
        loao_evidence=loao_evidence,
        max_examples_per_category=args.max_gemini_examples_per_category,
        snippet_chars=args.snippet_chars,
    )
    (args.output_dir / "gemini_taxonomy_prompt.md").write_text(prompt, encoding="utf-8")

    if args.gemini_draft:
        response = call_gemini(
            prompt=prompt,
            model=args.gemini_model,
            max_tokens=args.gemini_max_tokens,
            temperature=args.gemini_temperature,
            timeout=args.request_timeout,
        )
        content = completion_text(response)
        try:
            parsed_content: Any = json.loads(content) if content.strip().startswith("{") else content
            parse_error = None
        except json.JSONDecodeError as exc:
            parsed_content = content
            parse_error = str(exc)
        draft = {
            "model": args.gemini_model,
            "raw_response": response,
            "content_parse_error": parse_error,
            "content": parsed_content,
        }
        write_json(draft, args.output_dir / "gemini_taxonomy_draft.json")

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "rows": len(no_text_rows),
                "category_counts": category_report["counts"],
                "gemini_draft": bool(args.gemini_draft),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
