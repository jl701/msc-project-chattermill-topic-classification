"""Compare native auxiliary margins with immutable outer-hook probe evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from check_rag_faults_gpu import read_json,sha_file,write_json


def compare_native_row(actual: dict, reference: dict, encoded_sha: str) -> dict:
    return {"probability_equal":float(actual["score"]) == float(reference["original_engine_probability"]),
        "margin_equal":float(actual["presence_logit_margin"]) == float(reference["margin_y_minus_n"]),
        "encoded_input_equal":encoded_sha == reference["encoded_input_sha256"],
        "probability_abs_delta":abs(float(actual["score"])-float(reference["original_engine_probability"])),
        "margin_abs_delta":abs(float(actual["presence_logit_margin"])-float(reference["margin_y_minus_n"]))}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("artifacts","index","encoder-path","development-replay","reference-probe","output-dir"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args(argv)
    reference=read_json(args.reference_probe/"summary.json")
    if reference.get("status") != "completed" or not reference.get("gates_pass"):
        raise ValueError("Completed immutable hook probe is required")
    inputs=reference["input_sha256"]
    if inputs["artifacts"] != sha_file(args.artifacts) or inputs["development_replay"] != sha_file(args.development_replay):
        raise ValueError("Native gate source artifacts/replay differ from hook probe")
    if inputs["index_files"] != {p.name:sha_file(p) for p in args.index.iterdir() if p.is_file()}:
        raise ValueError("Native gate index differs from hook probe")
    source_rows=[json.loads(line) for line in args.development_replay.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if any(set(r) != {"request_id","text"} for r in source_rows):
        raise ValueError("Native gate accepts input-only records")
    by_id={r["request_id"]:r for r in source_rows}
    requests=[by_id[r["request_id"]] for r in reference["samples"]]
    args.output_dir.mkdir(parents=True,exist_ok=False)
    sources=["scripts/check_rag_native_margin_gpu.py","src/msc_project/serving/inference.py",
        "src/msc_project/serving/prefix_cache.py","src/msc_project/serving/presence_policy.py",
        "src/msc_project/serving/fallback_policy.py","src/msc_project/serving/retrieval.py"]
    summary={"status":"running","all_gates_pass":False,"scope":"native_capture_parity_no_gold",
        "source_sha256":{p:sha_file(ROOT/p) for p in sources},"input_sha256":inputs,
        "reference_summary_sha256":sha_file(args.reference_probe/"summary.json"),
        "reference_pair_file_sha256":{m:sha_file(args.reference_probe/f"margins_{m}.jsonl") for m in ("fixed","e5")},
        "reviews":len(requests),"methods":{}}
    write_json(args.output_dir/"preregistration.json",summary)
    import torch
    from msc_project.serving.inference import QwenServingEngine
    from msc_project.serving.retrieval import DemonstrationRetriever
    engine=None
    try:
        engine=QwenServingEngine.from_artifacts(args.artifacts,variant="hybrid",pair_batch_size=4,
            max_batch_tokens=4096,logits_mode="last_token",enable_prefix_cache=False,device="cuda",local_files_only=True)
        original=engine._score_encoded
        input_hashes=[]
        def traced(items,token_ids,**kwargs):
            if kwargs.get("mode") != "aspect":
                raise AssertionError("Presence-only parity gate unexpectedly entered Stage2")
            input_hashes.extend(hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest() for item in items)
            return original(items,token_ids,**kwargs)
        engine._score_encoded=traced
        for method in ("fixed","e5"):
            rows=[json.loads(line) for line in (args.reference_probe/f"margins_{method}.jsonl").read_text().splitlines() if line.strip()]
            refs={(r["request_id"],r["aspect"]):r for r in rows}
            if len(refs) != len(rows):
                raise ValueError("Reference probe has duplicate pairs")
            engine.set_retriever(DemonstrationRetriever(args.index,method=method,seed=13,
                fixed_demonstrations=engine.demonstrations["aspect"],encoder_path=args.encoder_path),fallback_on_error=False)
            checks=[]
            for request in requests:
                input_hashes.clear()
                response=engine.score_presence([request])[0]
                if len(input_hashes) != len(response["aspect_scores"]):
                    raise AssertionError("Native encoded-input count differs from candidates")
                for actual,encoded_sha in zip(response["aspect_scores"],input_hashes):
                    old=refs[(request["request_id"],actual["aspect"])]
                    check=compare_native_row(actual,old,encoded_sha)
                    checks.append(check)
                    with (args.output_dir/f"native_{method}.jsonl").open("a",encoding="utf-8") as handle:
                        handle.write(json.dumps({"request_id":request["request_id"],"aspect":actual["aspect"],
                            "score":actual["score"],"presence_logit_margin":actual["presence_logit_margin"],
                            "encoded_input_sha256":encoded_sha,**check})+"\n")
            result={"pairs":len(checks),"probability_max_abs_delta":max(r["probability_abs_delta"] for r in checks),
                "margin_max_abs_delta":max(r["margin_abs_delta"] for r in checks),
                "encoded_inputs_equal":all(r["encoded_input_equal"] for r in checks),
                "pass":all(r["probability_equal"] and r["margin_equal"] and r["encoded_input_equal"] for r in checks)}
            summary["methods"][method]=result
            write_json(args.output_dir/"summary.json",summary)
            print(json.dumps({"method":method,"parity_gate":result}),flush=True)
            if not result["pass"]:
                raise AssertionError("Native margin capture changed probabilities, margins or model inputs")
        summary.update(status="completed",all_gates_pass=True)
    except Exception as exc:
        summary.update(status="failed",exception={"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()})
    finally:
        write_json(args.output_dir/"summary.json",summary)
        if engine is not None:
            del engine
        torch.cuda.empty_cache()
    print(json.dumps({"status":summary["status"],"all_gates_pass":summary["all_gates_pass"],
                      "summary":str(args.output_dir/"summary.json")}),flush=True)
    return 0 if summary["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
