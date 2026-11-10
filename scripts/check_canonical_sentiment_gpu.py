"""Reserved-GPU correctness gate: canonical groups vs full grid and frozen anchor.

Use the new immutable service plan. This reads four development inputs and
existing model scores, never gold labels or the 957-review evaluation data.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from run_rag_quality import digest,read_lines,select_requests,verified_anchor_reuse,write_json
from check_rag_faults_gpu import compare_canonical_conditional
from msc_project.serving.retrieval_policy import resolve_retrieval_method


def selected_anchor_check(response,anchor,runner_threshold):
    from msc_project.serving.inference import decode_sentiments,SENTIMENTS
    grid={row["aspect"]:row["sentiment_scores"] for row in anchor["full_grid_scores"]}
    deltas=[]
    labels=True
    for item in response["predictions"]:
        expected=grid[item["aspect"]]
        labels &= item["sentiments"]==decode_sentiments([expected[k] for k in SENTIMENTS],runner_threshold)
        deltas.extend(abs(item["sentiment_scores"][k]-expected[k]) for k in SENTIMENTS)
    return {"selected_vectors_exact":all(d==0 for d in deltas),
            "selected_labels_equal":bool(labels),"selected_max_abs_delta":max(deltas,default=0)}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("plan","development-replay","reference-anchor","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--boundary-request-id",required=True,
        help="Private development boundary input ID supplied at invocation, never embedded in source")
    args=parser.parse_args(argv)
    plan=json.loads(args.plan.read_text(encoding="utf-8-sig"))
    settings=plan["server"]
    if settings.get("sentiment_execution")!="canonical_full_grid_groups":
        raise ValueError("A new canonical service plan is required")
    for source,expected in plan["source_hashes"].items():
        if digest(ROOT/source)!=expected: raise ValueError("Service source changed after registration: "+source)
    rows=read_lines(args.development_replay)
    select_requests(rows,None)
    by_id={r["request_id"]:r for r in rows}
    if args.boundary_request_id not in by_id: raise ValueError("Boundary input is not in development replay")
    sampled=select_requests(rows,3)
    requests=list({r["request_id"]:r for r in [*sampled,by_id[args.boundary_request_id]]}.values())
    first=json.loads(Path(plan["methods"]["fixed"]["artifacts"]).read_text(encoding="utf-8"))
    anchors,receipt=verified_anchor_reuse(args.reference_anchor,
        artifacts_sha=plan["input_hashes"]["artifacts"],replay_sha=digest(args.development_replay),
        expected={r["request_id"] for r in requests},aspects=[c["aspect"] for c in first["taxonomy"]["candidates"]],
        config={"inference":{k:settings[k] for k in ("pair_batch_size","max_batch_tokens","logits_mode")}})
    args.output.mkdir(parents=True,exist_ok=False)
    summary={"status":"running","passed":False,"plan_sha256":digest(args.plan),
        "source_sha256":digest(Path(__file__)),"anchor_receipt":receipt,
        "request_ids":[r["request_id"] for r in requests],"no_gold_read":True,"methods":{}}
    write_json(args.output/"registration.json",summary)
    import torch
    from msc_project.serving.inference import QwenServingEngine
    from msc_project.serving.retrieval import DemonstrationRetriever
    from msc_project.llm.qwen_pair_classifier import last_nonpad_indices
    engine=None
    try:
        for method in ("fixed",plan["dynamic_candidate"]):
            config=plan["methods"][method]
            artifacts_path=Path(config["artifacts"])
            if digest(artifacts_path)!=config["artifacts_sha256"]: raise ValueError("Service artifacts changed")
            engine=QwenServingEngine.from_artifacts(artifacts_path,variant="hybrid",
                pair_batch_size=settings["pair_batch_size"],max_batch_tokens=settings["max_batch_tokens"],
                logits_mode=settings["logits_mode"],enable_prefix_cache=config["prefix_cache"]=="on",
                sentiment_execution="canonical_full_grid_groups")
            engine.set_retriever(DemonstrationRetriever(Path(plan["index"]),**resolve_retrieval_method(method),
                encoder_path=plan["encoder_path"],fixed_demonstrations=engine.demonstrations["aspect"]))
            trace=[]
            original=engine._score_encoded
            def traced(items,token_ids,*,_original_score=original,**kwargs):
                if kwargs.get("mode")=="sentiment":
                    batch=engine.collator(items)
                    positions=last_nonpad_indices(batch["attention_mask"])
                    tensors={k:v.tolist() for k,v in batch.items()}
                    tensors["logits_to_keep"]=torch.unique(positions,sorted=True).tolist() if engine.logits_mode=="last_token" else None
                    value=json.dumps(tensors,sort_keys=True).encode()
                    trace.append({"collated_tensor_sha256":hashlib.sha256(value).hexdigest(),
                        "shape":list(batch["input_ids"].shape),"logits_to_keep":tensors["logits_to_keep"]})
                return _original_score(items,token_ids,**kwargs)
            engine._score_encoded=traced
            checks=[]
            for request in requests:
                trace.clear()
                full=engine.predict_batch([request],full_grid=True)[0]
                full_trace=deepcopy(trace)
                trace.clear()
                conditional=engine.predict_batch([request])[0]
                check=compare_canonical_conditional({"responses":[full]},{"responses":[conditional]})
                groups=conditional["diagnostics"]["batch_sentiment"]["executed_canonical_group_indices"]
                check["collated_groups_exact"]=trace==[full_trace[i] for i in groups]
                anchor_check=selected_anchor_check(conditional,anchors[request["request_id"]]["response"],
                    conditional["diagnostics"]["effective_thresholds"]["runner_up_threshold"])
                check["anchor"]=anchor_check
                check["passed"] &= check["collated_groups_exact"] and anchor_check["selected_vectors_exact"] and anchor_check["selected_labels_equal"]
                check.update(request_id=request["request_id"],full_group_trace=full_trace,
                    conditional_group_trace=deepcopy(trace),selected_group_indices=groups,
                    executed_prompts=conditional["diagnostics"]["sentiment_prompt_count"],
                    selected_aspects=conditional["diagnostics"]["selected_aspect_count"])
                checks.append(check)
                with (args.output/f"{method}_responses.jsonl").open("a",encoding="utf-8") as handle:
                    handle.write(json.dumps({"request_id":request["request_id"],"full":full,"conditional":conditional})+"\n")
            summary["methods"][method]={"checks":checks,"passed":all(r["passed"] for r in checks)}
            write_json(args.output/"summary.json",summary)
            print(json.dumps({"method":method,"reviews":len(checks),"passed":summary["methods"][method]["passed"]}),flush=True)
            engine._score_encoded=original
            del original,traced,engine
            engine=None
            gc.collect()
            torch.cuda.empty_cache()
        summary.update(status="complete",passed=all(r["passed"] for r in summary["methods"].values()))
    except Exception as error:
        summary.update(status="failed",error_type=type(error).__name__,error=str(error))
        raise
    finally:
        write_json(args.output/"summary.json",summary)
    return 0 if summary["passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
