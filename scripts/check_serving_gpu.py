"""GPU correctness gates against the historical scorer, without ground truth.

Only label equality determines a decoding gate. Every floating-point score
difference is reported separately, including differences that do not cross a
threshold. Outputs are private and contain predictions, never review text.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from msc_project.experiments.taxonomy_two_stage import capped_two_sentiment_prediction_mask
from msc_project.llm.qwen_pair_classifier import load_frozen_qwen_pair
from msc_project.llm.qwen_two_stage_classifier import score_two_stage_prompts
from msc_project.serving.inference import QwenServingEngine, SENTIMENTS, VARIANTS


def select_by_length(path: Path, count: int) -> list[dict[str, str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or count < 1 or count > len(rows):
        raise ValueError("Sample count must fit the development replay.")
    rows.sort(key=lambda row: (len(row["text"]), row["request_id"]))
    positions = np.rint(np.linspace(0, len(rows) - 1, count)).astype(int)
    return [{"request_id": str(rows[i]["request_id"]), "text": rows[i]["text"]} for i in positions]


def historical_reference(engine, requests, *, plain_base: bool = False):
    """Original full-grid scorer and original dataframe decoder, without labels."""
    variant = engine.variant
    rows = [request["text"] for request in requests for _ in engine.candidates]
    cards = [candidate["text"] for _ in requests for candidate in engine.candidates]
    scores = {}
    for mode in ["aspect", "sentiment"]:
        adapted = (variant == "qlora") if mode == "aspect" else (variant != "few_shot")
        context = nullcontext() if adapted or plain_base else engine.model.disable_adapter()
        with context:
            scores[mode] = score_two_stage_prompts(
                engine.model, engine.tokenizer, rows, cards, mode=mode,
                max_length=int(engine.artifacts["qlora" if adapted else "few_shot"]["max_length"]),
                batch_size=engine.pair_batch_size,
                demonstrations=None if adapted else engine.demonstrations[mode],
            )
    records = []
    for ri, request in enumerate(requests):
        for ai, candidate in enumerate(engine.candidates):
            position = ri * len(engine.candidates) + ai
            for si, sentiment in enumerate(SENTIMENTS):
                records.append({
                    "row_uid": request["request_id"], "candidate_aspect": candidate["aspect"],
                    "candidate_sentiment": sentiment, "pair_label": candidate["aspect"] + " / " + sentiment,
                    "target": 0,  # Decoder's structural field only; no true label is read.
                    "aspect_score": float(scores["aspect"][position, 0]),
                    "sentiment_score": float(scores["sentiment"][position, si]),
                })
    grid = pd.DataFrame(records)
    thresholds = engine.artifacts["variants"][variant]
    grid["selected"] = capped_two_sentiment_prediction_mask(
        grid, aspect_threshold=float(thresholds["aspect_threshold"]),
        second_sentiment_threshold=float(thresholds["runner_up_threshold"]),
    )
    responses = []
    for ri, request in enumerate(requests):
        predictions = []
        aspect_scores = []
        for ai, candidate in enumerate(engine.candidates):
            position = ri * len(engine.candidates) + ai
            p_aspect = float(scores["aspect"][position, 0])
            aspect_scores.append({"aspect": candidate["aspect"], "score": p_aspect})
            group = grid[(grid.row_uid == request["request_id"]) & (grid.candidate_aspect == candidate["aspect"])]
            selected = group[group.selected]
            if not selected.empty:
                predictions.append({
                    "aspect": candidate["aspect"],
                    "sentiments": selected["candidate_sentiment"].tolist(),
                    "aspect_score": p_aspect,
                    "sentiment_scores": dict(zip(SENTIMENTS, map(float, scores["sentiment"][position]))),
                })
        responses.append({"request_id": request["request_id"], "variant": variant,
                          "predictions": predictions, "aspect_scores": aspect_scores,
                          "diagnostics": {"candidate_count": len(engine.candidates),
                                          "sentiment_prompt_count": len(engine.candidates),
                                          "full_grid": True, "historical_reference": True}})
    return responses


def compare_outputs(reference, candidate, thresholds) -> dict[str, Any]:
    refs = {row["request_id"]: row for row in reference}
    candidates = {row["request_id"]: row for row in candidate}
    if refs.keys() != candidates.keys():
        raise ValueError("Comparison request identities differ.")
    aspect_differences = []
    sentiment_differences = []
    margins = []
    runner_margins = []
    changes = []
    for uid, ref in refs.items():
        value = candidates[uid]
        ref_aspects = {item["aspect"]: float(item["score"]) for item in ref["aspect_scores"]}
        new_aspects = {item["aspect"]: float(item["score"]) for item in value["aspect_scores"]}
        assert ref_aspects.keys() == new_aspects.keys()
        aspect_differences.extend(abs(score - new_aspects[name]) for name, score in ref_aspects.items())
        margins.extend(abs(score - thresholds["aspect_threshold"]) for score in [*ref_aspects.values(), *new_aspects.values()])
        ref_predictions = {item["aspect"]: item for item in ref["predictions"]}
        new_predictions = {item["aspect"]: item for item in value["predictions"]}
        ref_labels = {(item["aspect"], s) for item in ref["predictions"] for s in item["sentiments"]}
        new_labels = {(item["aspect"], s) for item in value["predictions"] for s in item["sentiments"]}
        if ref_labels != new_labels:
            changes.append({"request_id": uid, "removed": sorted(ref_labels - new_labels),
                            "added": sorted(new_labels - ref_labels)})
        for aspect in ref_predictions.keys() & new_predictions.keys():
            left = ref_predictions[aspect]["sentiment_scores"]
            right = new_predictions[aspect]["sentiment_scores"]
            sentiment_differences.extend(abs(left[s] - right[s]) for s in SENTIMENTS)
        for prediction in [*ref_predictions.values(), *new_predictions.values()]:
            runner = sorted(prediction["sentiment_scores"].values(), reverse=True)[1]
            runner_margins.append(abs(runner - thresholds["runner_up_threshold"]))
    return {
        "request_count": len(refs), "label_gate_pass": not changes,
        "label_changed_requests": len(changes), "label_changes": changes,
        "aspect_max_abs_difference": max(aspect_differences, default=0.0),
        "aspect_nonidentical_score_count": sum(value != 0 for value in aspect_differences),
        "common_selected_sentiment_max_abs_difference": max(sentiment_differences, default=0.0),
        "common_selected_sentiment_nonidentical_score_count": sum(value != 0 for value in sentiment_differences),
        "closest_aspect_threshold_margin": min(margins, default=None),
        "closest_runner_up_threshold_margin": min(runner_margins, default=None),
        "scope": "Sentiment differences compare commonly selected aspects only. Label differences are never ignored.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "outputs/serving_20261109/artifacts.json")
    parser.add_argument("--replay", type=Path, default=ROOT / "outputs/serving_20261109/replay_development.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/serving_20261109/correctness")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--pair-batch-size", type=int, default=4)
    parser.add_argument("--max-batch-tokens", type=int, default=4096)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--check-frozen-base", action="store_true")
    args = parser.parse_args()
    variants = args.variants.split(",")
    if any(variant not in VARIANTS for variant in variants):
        parser.error("Unknown variant.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    journal = args.output_dir / "steps.jsonl"
    if journal.exists():
        parser.error("Output journal already exists; choose a fresh output directory.")
    requests = select_by_length(args.replay, args.samples)
    summary = {
        "status": "running", "selection": "equally spaced ranks after sorting development reviews by character length and request ID",
        "samples": [{"request_id": row["request_id"], "characters": len(row["text"])} for row in requests],
        "variants": variants, "comparisons": [], "steps": [], "label_gate_pass": False,
        "uses_ground_truth": False, "official_test_accessed": False,
    }

    def persist():
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def measure(name, function):
        print(f"START {name}", flush=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        outputs = function()
        torch.cuda.synchronize()
        evidence = {"step": name, "seconds": time.perf_counter() - started,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "outputs": outputs}
        with journal.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(evidence) + "\n")
        summary["steps"].append({k: v for k, v in evidence.items() if k != "outputs"})
        persist()
        print(f"DONE {name}: {evidence['seconds']:.2f}s peak {evidence['peak_allocated_bytes']/2**30:.2f}GiB", flush=True)
        return outputs

    def compare(name, left, right, thresholds):
        record = {"comparison": name, **compare_outputs(left, right, thresholds)}
        summary["comparisons"].append(record)
        persist()
        print(f"COMPARE {name}: labels={'PASS' if record['label_gate_pass'] else 'FAIL'}, "
              f"aspect_maxdiff={record['aspect_max_abs_difference']:.9g}, "
              f"sentiment_maxdiff={record['common_selected_sentiment_max_abs_difference']:.9g}", flush=True)

    persist()
    engine = None
    try:
        print("Loading one resident quantized model and selected adapter", flush=True)
        engine = QwenServingEngine.from_artifacts(
            args.artifacts, pair_batch_size=args.pair_batch_size,
            max_batch_tokens=args.max_batch_tokens, logits_mode="full",
        )
        summary["engine"] = engine.metadata()
        frozen_expected = None
        for variant in variants:
            engine.set_variant(variant)
            thresholds = engine.artifacts["variants"][variant]
            engine.logits_mode = "full"
            reference = measure(variant + "/historical_full_grid", lambda: historical_reference(engine, requests))
            full_grid = measure(variant + "/engine_full_grid", lambda: engine.predict_batch(requests, full_grid=True))
            conditional = measure(variant + "/engine_conditional_full_logits", lambda: engine.predict_batch(requests))
            compare(variant + "/historical_vs_engine_full_grid", reference, full_grid, thresholds)
            compare(variant + "/full_grid_vs_conditional", full_grid, conditional, thresholds)
            engine.logits_mode = "last_token"
            reduced = measure(variant + "/engine_conditional_last_token", lambda: engine.predict_batch(requests))
            compare(variant + "/full_logits_vs_last_token", conditional, reduced, thresholds)
            individual = measure(variant + "/individual_last_token", lambda: [engine.predict_batch([request])[0] for request in requests])
            compare(variant + "/batch_vs_individual", reduced, individual, thresholds)
            if variant == "few_shot":
                frozen_expected = reference
        if args.check_frozen_base:
            if frozen_expected is None:
                engine.set_variant("few_shot")
                frozen_expected = measure("few_shot/historical_disabled_adapter", lambda: historical_reference(engine, requests))
            artifacts = engine.artifacts
            thresholds = artifacts["variants"]["few_shot"]
            del engine
            engine = None
            gc.collect()
            torch.cuda.empty_cache()
            print("Loading independent frozen base after releasing PEFT instance", flush=True)
            tokenizer, model, _ = load_frozen_qwen_pair(
                artifacts["model"]["local_snapshot_path"], load_in_4bit=True, local_files_only=True,
            )
            engine = QwenServingEngine(artifacts, tokenizer, model, variant="few_shot",
                                       pair_batch_size=args.pair_batch_size, max_batch_tokens=args.max_batch_tokens)
            independent = measure("few_shot/independent_base", lambda: historical_reference(engine, requests, plain_base=True))
            compare("few_shot/disabled_adapter_vs_independent_base", frozen_expected, independent, thresholds)
        summary["label_gate_pass"] = all(item["label_gate_pass"] for item in summary["comparisons"])
        summary["status"] = "completed"
    except Exception as error:
        summary["status"] = "failed"
        summary["exception"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        print(summary["exception"]["traceback"], flush=True)
    finally:
        persist()
        if engine is not None:
            del engine
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps({"status": summary["status"], "label_gate_pass": summary["label_gate_pass"],
                      "summary": str(args.output_dir / 'summary.json')}), flush=True)
    return 0 if summary["label_gate_pass"] and summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
