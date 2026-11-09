"""Bounded diagnosis of prefix-cache score differences on two development rows."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM

from msc_project.llm.qwen_pair_classifier import last_nonpad_indices
from msc_project.llm.qwen_two_stage_classifier import _encode_segment_aware_few_shot_prompt
from msc_project.serving.inference import QwenServingEngine

def select_diagnostic_targets(gate_log, threshold):
    """Resolve private identities from local gate results, never tracked code."""
    records = [json.loads(line) for line in gate_log.read_text(encoding="utf-8").splitlines()]
    comparisons = [row for row in records if row.get("phase") == "comparison" and row.get("variant") == "few_shot"]
    scored = [(row["request_id"], aspect) for row in comparisons for aspect in row["all_aspect_scores"]]
    if not scored:
        raise ValueError("No few-shot aspect comparisons found in the local gate log.")
    worst = max(scored, key=lambda item: abs(item[1]["difference"]))
    remaining = [item for item in scored if item[0] != worst[0]]
    if not remaining:
        raise ValueError("Two distinct development reviews are required for diagnosis.")
    nearest = min(remaining, key=lambda item: min(abs(item[1][side]-threshold) for side in ["cache_off", "cache_on"]))
    return [{"request_id": uid, "aspect": aspect["aspect"], "reason": reason}
            for (uid, aspect), reason in [(worst, "largest aspect difference"), (nearest, "nearest aspect threshold")]]


def cpu_tiny_check():
    """Exercise actual Qwen3/DynamicCache absolute positions and unequal padding."""
    torch.manual_seed(13)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=8, max_position_embeddings=128,
                         attention_dropout=0.0)
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).to(device="cpu", dtype=torch.float32).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 10, 0]])
    mask = (ids != 0).long()
    prefix_length = 7
    with torch.inference_mode():
        full = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        prepared = model(input_ids=ids[:1, :prefix_length], attention_mask=mask[:1, :prefix_length], use_cache=True)
        cache = DynamicCache(ddp_cache_data=((layer.keys.repeat(2, 1, 1, 1), layer.values.repeat(2, 1, 1, 1)) for layer in prepared.past_key_values.layers), config=config)
        positions = torch.arange(prefix_length, ids.shape[1])
        split = model(input_ids=ids[:, prefix_length:], attention_mask=mask,
                      cache_position=positions, position_ids=positions.unsqueeze(0),
                      past_key_values=cache, use_cache=True).logits
        index = last_nonpad_indices(mask)
        expected = full[torch.arange(2), index]
        observed = split[torch.arange(2), index-prefix_length]
        error = (expected-observed).abs().max().item()
    return {"model": "random two-layer Qwen3, CPU FP32 eager attention", "max_abs_logits_difference": error,
            "passed_at_1e_5": error < 1e-5, "scope": "Checks cache positions, full attention mask and suffix-relative final logits with real Qwen code; does not prove quantized 4B equivalence."}


def score_summary(logits, relative_positions, token_ids):
    raw = logits[torch.arange(logits.shape[0], device=logits.device), relative_positions][:, list(token_ids)].float()
    scores = torch.softmax(raw, dim=-1)
    return {"restricted_logits": raw.detach().cpu().tolist(),
            "yn_logit_margin": (raw[:, 0]-raw[:, 1]).detach().cpu().tolist(),
            "presence_scores": scores[:, 0].detach().cpu().tolist()}


def cache_pairs(cache, prefix_length, *, first_only=False):
    return tuple((layer.keys[:1 if first_only else layer.keys.shape[0], :, :prefix_length].detach().clone(),
                  layer.values[:1 if first_only else layer.values.shape[0], :, :prefix_length].detach().clone())
                 for layer in cache.layers)


def kv_difference(left, right):
    result = []
    for layer, ((ak, av), (bk, bv)) in enumerate(zip(left, right)):
        errors = [(ak[:1].float()-bk[:1].float()).abs(), (av[:1].float()-bv[:1].float()).abs()]
        result.append({"layer": layer, "key_max": errors[0].max().item(), "key_mean": errors[0].mean().item(),
                       "value_max": errors[1].max().item(), "value_mean": errors[1].mean().item()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "outputs/serving_20261109/artifacts.json")
    parser.add_argument("--replay", type=Path, default=ROOT / "outputs/serving_20261109/replay_development.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/serving_20261109/prefix_diagnosis")
    parser.add_argument("--gate-log", type=Path, default=ROOT / "outputs/serving_20261109/prefix_gate_100/samples.jsonl")
    parser.add_argument("--include-math", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    if args.cpu_only:
        result = cpu_tiny_check()
        print(json.dumps(result), flush=True)
        return 0 if result["passed_at_1e_5"] else 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    journal = args.output_dir / "steps.jsonl"
    if journal.exists():
        parser.error("Output exists; choose a fresh output directory.")
    rows = {row["request_id"]: row for row in map(json.loads, args.replay.read_text().splitlines())}
    artifacts = json.loads(args.artifacts.read_text(encoding="utf-8"))
    targets = select_diagnostic_targets(args.gate_log, artifacts["variants"]["few_shot"]["aspect_threshold"])
    result = {"status": "running", "targets": targets, "cpu_tiny": cpu_tiny_check(), "reviews": []}

    def persist():
        (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    def measured(name, fn):
        print("START " + name, flush=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        value, extra = fn()
        torch.cuda.synchronize()
        record = {"step": name, "seconds": time.perf_counter()-start,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(), **value}
        with journal.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record)+"\n")
        print(f"DONE {name}: {record['seconds']:.2f}s", flush=True)
        return record, extra

    engine = None
    try:
        engine = QwenServingEngine.from_artifacts(args.artifacts, variant="few_shot", pair_batch_size=4,
                                                 max_batch_tokens=4096, logits_mode="last_token")
        result["engine"] = engine.metadata()
        result["attention_implementation"] = engine.model.config._attn_implementation
        result["compute_dtype"] = str(next(engine.model.parameters()).dtype)
        device = next(engine.model.parameters()).device
        token_ids = (engine.aspect_ids.yes, engine.aspect_ids.no)
        with engine.model.disable_adapter(), torch.inference_mode():
            stored = engine.prefix_cache._prepare("aspect")
            length = len(stored.ids)
            result["prefix_length"] = length
            result["threshold"] = engine.artifacts["variants"]["few_shot"]["aspect_threshold"]
            for target in targets:
                uid = target["request_id"]
                request = rows[uid]
                target_index = next(i for i, candidate in enumerate(engine.candidates) if candidate["aspect"] == target["aspect"])
                group_start = target_index // 4 * 4
                candidates = engine.candidates[group_start:group_start+4]
                if len(candidates) != 4:
                    raise ValueError("The selected diagnostic microbatch must contain four candidates.")
                items = [_encode_segment_aware_few_shot_prompt(engine.tokenizer, request["text"], candidate["text"], mode="aspect",
                         demonstrations=engine.demonstrations["aspect"], max_length=1024) for candidate in candidates]
                assert all(tuple(item["input_ids"][:length]) == stored.ids for item in items)
                batch = {k:v.to(device) for k,v in engine.collator(items).items()}
                last = last_nonpad_indices(batch["attention_mask"])
                keep = torch.unique(last, sorted=True)
                indices = torch.searchsorted(keep, last)
                suffix = batch["input_ids"][:, length:]
                suffix_positions = torch.arange(length, batch["input_ids"].shape[1], device=device)
                suffix_last = last-length
                suffix_keep = torch.unique(suffix_last, sorted=True)
                suffix_indices = torch.searchsorted(suffix_keep, suffix_last)
                observations = {"request_id": uid, "characters": len(request["text"]),
                                "candidates": [candidate["aspect"] for candidate in candidates],
                                "lengths": [len(item["input_ids"]) for item in items], "cases": {}}

                def full(use_cache=False):
                    output = engine.model(**batch, use_cache=use_cache, logits_to_keep=keep)
                    pairs = cache_pairs(output.past_key_values, length) if use_cache else None
                    return score_summary(output.logits, indices, token_ids), pairs

                def prepare_prefix(count):
                    prefix_ids = torch.tensor([stored.ids]*count, dtype=torch.long, device=device)
                    output = engine.model(input_ids=prefix_ids, attention_mask=torch.ones_like(prefix_ids), use_cache=True, logits_to_keep=1)
                    return cache_pairs(output.past_key_values, length, first_only=True)

                def split(pairs):
                    copies = ((key.repeat(4, 1, 1, 1) if key.shape[0] == 1 else key.clone(),
                               value.repeat(4, 1, 1, 1) if value.shape[0] == 1 else value.clone()) for key,value in pairs)
                    cache = DynamicCache(ddp_cache_data=copies, config=engine.model.config)
                    output = engine.model(input_ids=suffix, attention_mask=batch["attention_mask"],
                                          past_key_values=cache, cache_position=suffix_positions,
                                          position_ids=suffix_positions.unsqueeze(0), use_cache=True,
                                          logits_to_keep=suffix_keep)
                    return score_summary(output.logits, suffix_indices, token_ids), None

                for name, function in [("A_full_repeat1", full), ("A_full_repeat2", full),
                                       ("B_cached_single_prefix", lambda: split(stored.key_values))]:
                    record, _ = measured(uid+"/"+name, function)
                    observations["cases"][name] = record
                actual = engine.prefix_cache.score_if_compatible(items, token_ids, mode="aspect",
                                                                fingerprint=engine.artifacts_fingerprint, adapter_enabled=False)
                observations["existing_cache_module_presence_scores"] = actual[:, 0].tolist()
                prefix4 = prepare_prefix(4)
                record, _ = measured(uid+"/C_prefix_built_batch4", lambda pairs=prefix4: split(pairs))
                observations["cases"]["C_prefix_built_batch4"] = record
                observations["kv_single_vs_batch4"] = kv_difference(stored.key_values, prefix4)
                record, from_full = measured(uid+"/D_full_with_cache", lambda: full(True))
                observations["cases"]["D_full_with_cache"] = record
                observations["kv_single_vs_full_prefill"] = kv_difference(stored.key_values, from_full)
                record, _ = measured(uid+"/D_suffix_from_full_prefill_KV", lambda pairs=from_full: split(pairs))
                observations["cases"]["D_suffix_from_full_prefill_KV"] = record
                if args.include_math:
                    from torch.nn.attention import SDPBackend, sdpa_kernel
                    with sdpa_kernel(SDPBackend.MATH):
                        record, _ = measured(uid+"/E_math_full", full)
                        observations["cases"]["E_math_full"] = record
                        math_prefix = prepare_prefix(1)
                        record, _ = measured(uid+"/E_math_split", lambda pairs=math_prefix: split(pairs))
                        observations["cases"]["E_math_split"] = record
                    del math_prefix
                baseline = observations["cases"]["A_full_repeat1"]["presence_scores"]
                for name, record in observations["cases"].items():
                    record["difference_from_A"] = [b-a for a,b in zip(baseline, record["presence_scores"])]
                result["reviews"].append(observations)
                persist()
                del prefix4, from_full
                gc.collect()
                torch.cuda.empty_cache()
        result["status"] = "completed"
    except Exception as error:
        result["status"] = "failed"
        result["exception"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        print(result["exception"]["traceback"], flush=True)
    finally:
        persist()
        if engine is not None:
            engine.prefix_cache.clear()
            del engine
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps({"status": result["status"], "summary": str(args.output_dir/'summary.json')}), flush=True)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
