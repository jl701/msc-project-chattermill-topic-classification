"""Inspect saturated aspect probabilities without changing model scoring.

Run only after reserving the GPU. Twenty deterministic development length ranks
are probed with fixed and E5 demonstrations; no labels are read. A read-only
outer-model forward hook captures the same Y/N positions used by the serving
engine. Its CPU transfers make these runs unsuitable for latency claims.
"""
from __future__ import annotations

from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from check_rag_faults_gpu import compare_sentiment_anchors, read_json, sha_file, write_json


def length_rank_inputs(path: Path, count: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if (count < 2 or len(rows) < count or any(set(r) != {"request_id", "text"} or
            not isinstance(r["request_id"], str) or not r["request_id"] or
            not isinstance(r["text"], str) or not r["text"].strip() for r in rows) or
            len({r["request_id"] for r in rows}) != len(rows)):
        raise ValueError("Need enough unique input-only development records for deterministic ranks")
    ordered = sorted(rows, key=lambda r: (len(r["text"]), r["request_id"]))
    positions = [round(i * (len(ordered)-1)/(count-1)) for i in range(count)]
    return [ordered[i] for i in positions]


def extract_pair_statistics(logits, attention_mask, token_ids, kept_positions=None) -> list[dict]:
    """Mirror _score_encoded's last-token mapping, using only detached copies."""
    import torch
    from msc_project.llm.qwen_pair_classifier import last_nonpad_indices

    if len(token_ids) != 2 or logits.ndim != 3 or logits.shape[0] != attention_mask.shape[0]:
        raise ValueError("Expected a two-verbalizer batch with matching logits and masks")
    final_positions = last_nonpad_indices(attention_mask)
    if kept_positions is not None:
        if kept_positions.ndim != 1 or logits.shape[1] != len(kept_positions):
            raise ValueError("Kept-position shape differs from the last-token output")
        if not torch.equal(kept_positions, torch.unique(kept_positions, sorted=True)):
            raise ValueError("Kept positions must be sorted and unique")
        positions = torch.searchsorted(kept_positions, final_positions)
        if torch.any(positions >= len(kept_positions)) or not torch.equal(kept_positions[positions], final_positions):
            raise ValueError("A last non-padding token is missing from retained positions")
    else:
        positions = final_positions
    final = logits.detach()[torch.arange(len(attention_mask), device=logits.device), positions]
    pair = final[:, list(token_ids)]
    source64 = pair.to(dtype=torch.float64)
    if not torch.isfinite(source64).all():
        raise ValueError("Non-finite raw verbalizer logits")
    probability32 = torch.softmax(pair.float(), dim=-1)[:, 0]
    probability64 = torch.softmax(source64, dim=-1)[:, 0]
    margins64 = source64[:, 0] - source64[:, 1]
    values = torch.stack([source64[:,0], source64[:,1], margins64,
                          probability32.to(torch.float64), probability64], dim=1).cpu().tolist()
    return [dict(zip(("logit_y", "logit_n", "margin_y_minus_n", "probability_float32",
                      "probability_float64"), values_row), source_logits_dtype=str(logits.dtype))
            for values_row in values]


def summarize_rows(rows: list[dict]) -> dict:
    def one(values):
        margins = [r["margin_y_minus_n"] for r in values]
        saturated = [r for r in values if r["probability_float32"] == 1.0]
        def bounds(numbers):
            return {"min": min(numbers), "max": max(numbers)} if numbers else {"min":None,"max":None}
        return {"pair_count":len(values),
            "float32_probability_one":sum(r["probability_float32"] == 1.0 for r in values),
            "float32_probability_zero":sum(r["probability_float32"] == 0.0 for r in values),
            "float64_probability_one":sum(r["probability_float64"] == 1.0 for r in values),
            "float64_probability_zero":sum(r["probability_float64"] == 0.0 for r in values),
            "raw_margin_range":bounds(margins),
            "float32_one_bucket_raw_margin_range":bounds([r["margin_y_minus_n"] for r in saturated]),
            "distinct_margins_in_float32_one_bucket":len({r["margin_y_minus_n"] for r in saturated}),
            "distinct_float64_probabilities_in_float32_one_bucket":len({r["probability_float64"] for r in saturated}),
            "distinct_float32_probabilities":len({r["probability_float32"] for r in values}),
            "distinct_float64_probabilities":len({r["probability_float64"] for r in values}),
            "source_logits_dtypes":dict(Counter(r["source_logits_dtype"] for r in values)),
            "max_abs_float32_vs_engine_difference":max(
                (abs(r["probability_float32"]-r["original_engine_probability"]) for r in values), default=0.0)}
    return {"all_candidate_pairs":one(rows),
            "seen_candidate_pairs":one([r for r in rows if r["seen_candidate"]]),
            "heldout_candidate_pairs":one([r for r in rows if not r["seen_candidate"]])}


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--encoder-path", type=Path, required=True)
    parser.add_argument("--development-replay", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args(argv)
    requests = length_rank_inputs(args.development_replay, args.samples)
    gate_requests = length_rank_inputs(args.development_replay, 3)
    artifacts, manifest = read_json(args.artifacts), read_json(args.index / "manifest.json")
    if sha_file(args.development_replay) not in manifest.get("exclusion_source_hashes", {}).values():
        raise ValueError("Development replay lacks a registered index exclusion certificate")
    candidates = artifacts["taxonomy"]["candidates"]
    if len(candidates) != 12:
        raise ValueError("This registered probe expects the frozen twelve-aspect taxonomy")
    if not args.encoder_path.is_dir():
        raise ValueError("Pinned local E5 encoder directory is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources = ["scripts/probe_rag_logit_margins_gpu.py", "scripts/check_rag_faults_gpu.py",
               "src/msc_project/serving/inference.py", "src/msc_project/serving/retrieval.py",
               "src/msc_project/serving/fallback_policy.py", "src/msc_project/llm/qwen_pair_classifier.py"]
    summary = {"status":"running", "scope":"numeric_diagnostic_not_timing_benchmark",
        "uses_gold_labels":False, "model_tensors_modified":False, "core_scoring_modified":False,
        "prefix_cache":False, "retrieval_fallback":False, "methods":["fixed","e5"],
        "sampling":"evenly spaced character-length ranks with request-ID tie breaking; Python round",
        "samples":[{"request_id":r["request_id"],"characters":len(r["text"])} for r in requests],
        "sentiment_gate_ids":[r["request_id"] for r in gate_requests],
        "input_sha256":{"artifacts":sha_file(args.artifacts),"development_replay":sha_file(args.development_replay),
            "index_files":{p.name:sha_file(p) for p in args.index.iterdir() if p.is_file()}},
        "source_sha256":{p:sha_file(ROOT/p) for p in sources}, "results":{}, "gates_pass":False}
    write_json(args.output_dir/"preregistration.json",summary)
    import torch
    from msc_project.serving.inference import QwenServingEngine
    from msc_project.serving.retrieval import DemonstrationRetriever

    engine = None
    hook_handle = None
    active = {"capture":False, "stage":None, "adapter":None, "token_ids":None, "item_hashes":None}
    encoded_trace, captured = [], []

    def persist():
        write_json(args.output_dir/"summary.json",summary)

    def hooked(module, positional, kwargs, output):
        if not active["capture"] or active["stage"] != "aspect":
            return None
        if active["adapter"]:
            raise AssertionError("The registered aspect probe must use the frozen base")
        logits = output["logits"] if isinstance(output, dict) else output.logits
        values = extract_pair_statistics(logits, kwargs["attention_mask"], active["token_ids"], kwargs.get("logits_to_keep"))
        if len(values) != len(active["item_hashes"]):
            raise AssertionError("Hook batch order/count differs from encoded model inputs")
        captured.extend(dict(value, encoded_input_sha256=sha) for value,sha in zip(values,active["item_hashes"]))
        return None  # A forward hook returning None cannot replace the model output.

    def gate_step(name, function):
        encoded_trace.clear()
        responses = function()
        entry = {"step":name,"responses":responses,"encoded_inputs":list(encoded_trace)}
        with (args.output_dir/"sentiment_gate_steps.jsonl").open("a",encoding="utf-8") as handle:
            handle.write(json.dumps(entry,ensure_ascii=False,allow_nan=False)+"\n")
        return entry

    try:
        engine = QwenServingEngine.from_artifacts(args.artifacts, variant="hybrid", pair_batch_size=4,
            max_batch_tokens=4096, logits_mode="last_token", enable_prefix_cache=False,
            device="cuda", local_files_only=True)
        summary["engine"] = engine.metadata()
        summary["device"] = torch.cuda.get_device_name()
        original_score = engine._score_encoded
        def traced(items, token_ids, **kwargs):
            encoded_trace.append({"mode":kwargs.get("mode"),"adapter":kwargs.get("use_adapter",True),
                "encoded_inputs_sha256":hashlib.sha256(json.dumps(items,sort_keys=True).encode()).hexdigest(),
                "prompt_count":len(items)})
            active.update(stage=kwargs.get("mode"),adapter=kwargs.get("use_adapter",True),token_ids=tuple(token_ids),
                item_hashes=[hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest() for item in items])
            try:
                return original_score(items,token_ids,**kwargs)
            finally:
                active.update(stage=None,adapter=None,token_ids=None,item_hashes=None)
        engine._score_encoded = traced
        hook_handle = engine.model.register_forward_hook(hooked, with_kwargs=True)
        legacy = gate_step("legacy_full_grid",lambda:engine.predict_batch(gate_requests,full_grid=True))
        sentiment_only = gate_step("sentiment_only",lambda:engine.score_full_sentiment(gate_requests))
        gate = compare_sentiment_anchors(legacy,sentiment_only,
            [float(artifacts["variants"]["hybrid"]["runner_up_threshold"])])
        summary["sentiment_anchor_gate"] = gate
        persist()
        print(json.dumps({"sentiment_anchor_gate":gate}),flush=True)
        if (not gate["encoded_sentiment_inputs_equal"] or not gate["sentiment_truncation_equal"]
                or gate["label_changed_aspects"]):
            raise AssertionError("Sentiment-only correctness gate failed")

        heldout = {manifest["heldout_aspect"]}
        active["capture"] = True
        for method in ("fixed","e5"):
            engine.set_retriever(DemonstrationRetriever(args.index,method=method,seed=13,
                fixed_demonstrations=engine.demonstrations["aspect"],encoder_path=args.encoder_path),
                fallback_on_error=False)
            rows = []
            for position,request in enumerate(requests,1):
                captured.clear()
                encoded_trace.clear()
                response = engine.score_presence([request])[0]
                scores = response["aspect_scores"]
                if len(captured) != len(candidates) or [r["aspect"] for r in scores] != [c["aspect"] for c in candidates]:
                    raise AssertionError("Captured logits do not align with the twelve candidate outputs")
                selections = response["diagnostics"]["batch_aspect"]["retrieval"]
                record_rows = []
                for statistics,score,selection in zip(captured,scores,selections):
                    if statistics["probability_float32"] != score["score"]:
                        raise AssertionError("Hook probability does not exactly reproduce core scoring")
                    row = {"request_id":request["request_id"],"aspect":score["aspect"],
                        "seen_candidate":score["aspect"] not in heldout,"method":method,**statistics,
                        "original_engine_probability":score["score"],"pool_sha256":engine.retriever.fingerprint,
                        "demonstration_sha256":selection.get("demonstration_sha256")}
                    record_rows.append(row)
                with (args.output_dir/f"margins_{method}.jsonl").open("a",encoding="utf-8") as handle:
                    for row in record_rows:
                        handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
                rows.extend(record_rows)
                summary["progress"] = {"method":method,"completed_reviews":position,"total_reviews":len(requests)}
                persist()
            summary["results"][method] = summarize_rows(rows)
            persist()
            print(json.dumps({"method":method,"seen_candidate_pairs":summary["results"][method]["seen_candidate_pairs"]}),flush=True)
        summary.update(status="completed",gates_pass=True)
    except Exception as exc:
        summary.update(status="failed",exception={"type":type(exc).__name__,"message":str(exc),
                                                    "traceback":traceback.format_exc()})
    finally:
        if hook_handle is not None:
            hook_handle.remove()
        persist()
        if engine is not None:
            del engine
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps({"status":summary["status"],"gates_pass":summary["gates_pass"],
                      "summary":str(args.output_dir/"summary.json")}),flush=True)
    return 0 if summary["gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
