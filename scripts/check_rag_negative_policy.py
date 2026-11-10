"""Input-only selection/encoding audit and optional three-review native GPU gate.

The CPU mode audits all development inputs without opening labels. GPU mode
reuses that immutable audit, checks the old E5 comparator against its frozen
seen-only scores, and records new-policy inputs without evaluating quality.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from run_rag_quality import digest, read_lines, select_requests, write_json
from msc_project.serving.retrieval_policy import GLOBAL, SAME_ASPECT_NEGATIVE_SEEN


def item_sha(item):
    return hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest()


def compare_selection(before, after, *, aspect, heldout):
    if [d.answer for d in after] != list("YYNN") or before[:2] != after[:2]:
        raise AssertionError("Policy changed the positive examples or 2Y2N ordering")
    if len({d.row_uid for d in after[2:]}) != 2:
        raise AssertionError("Policy repeats a negative review")
    old_aligned = sum(d.candidate_aspect == aspect for d in before[2:])
    if aspect == heldout or old_aligned == 2:
        if before != after:
            raise AssertionError("Policy changed held-out or already-aligned selections")
    elif any(d.candidate_aspect != aspect for d in after[2:]):
        raise AssertionError("Seen negatives are not both same-aspect")
    return {"old_negative_alignment":old_aligned, "selection_changed":before != after,
            "positive_examples_equal":True}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("artifacts","index","encoder-path","development-replay","reference-e5","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--cpu-audit",type=Path,help="Reuse completed CPU audit and run the 3-review GPU gate")
    args=parser.parse_args(argv)
    artifacts=json.loads(args.artifacts.read_text(encoding="utf-8-sig"))
    requests=read_lines(args.development_replay)
    select_requests(requests,None)  # Input-only schema and unique IDs.
    candidates=artifacts["taxonomy"]["candidates"]
    manifest=json.loads((args.index/"manifest.json").read_text(encoding="utf-8-sig"))
    heldout=manifest["heldout_aspect"]
    source_identity=json.loads((args.reference_e5.parent/"identity.json").read_text(encoding="utf-8-sig"))
    input_hashes={"artifacts":digest(args.artifacts),"development_replay":digest(args.development_replay),
                  "reference_e5":digest(args.reference_e5),"index_manifest":digest(args.index/"manifest.json")}
    if source_identity["artifacts"] != input_hashes["artifacts"] or source_identity["replay"] != input_hashes["development_replay"]:
        raise ValueError("Reference E5 artifacts/replay differ")
    if set(source_identity["request_ids"]) != {r["request_id"] for r in requests}:
        raise ValueError("Reference E5 population differs")
    references={r["request_id"]:r["response"] for r in read_lines(args.reference_e5)}
    if len(references) != len(requests):
        raise ValueError("Incomplete reference E5 outputs")
    args.output.mkdir(parents=True,exist_ok=False)
    sources=["scripts/check_rag_negative_policy.py","src/msc_project/serving/retrieval.py",
        "src/msc_project/serving/retrieval_policy.py","src/msc_project/serving/inference.py"]
    summary={"status":"running","input_sha256":input_hashes,
        "source_sha256":{p:digest(ROOT/p) for p in sources},"no_gold_read":True,
        "scope":"selection_and_encoding_only" if args.cpu_audit is None else "native_seen_numeric_parity_only"}
    write_json(args.output/"registration.json",summary)
    from msc_project.serving.retrieval import DemonstrationRetriever, encode_budgeted_prompt, normalise_text
    try:
        if args.cpu_audit is None:
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(artifacts["model"]["local_snapshot_path"],local_files_only=True)
            retriever=DemonstrationRetriever(args.index,method="e5",encoder_path=args.encoder_path)
            counts=Counter()
            for position,request in enumerate(requests,1):
                retriever.prepare([request["text"]],[c["text"] for c in candidates])
                old_by_aspect={r["candidate_aspect"]:r for r in references[request["request_id"]]["diagnostics"]["batch_aspect"]["retrieval"]}
                for candidate in candidates:
                    aspect=candidate["aspect"]
                    retriever.selection_policy=GLOBAL
                    before,info=retriever.select(request["text"],aspect,candidate["text"])
                    if info["selected_ids"] != old_by_aspect[aspect]["selected_ids"]:
                        raise AssertionError("Old global E5 selections changed from frozen reference")
                    retriever.selection_policy=SAME_ASPECT_NEGATIVE_SEEN
                    after,new_info=retriever.select(request["text"],aspect,candidate["text"])
                    check=compare_selection(before,after,aspect=aspect,heldout=heldout)
                    if any(normalise_text(d.review_text)==normalise_text(request["text"]) for d in after):
                        raise AssertionError("Current query entered its own demonstrations")
                    enc,old_trim=encode_budgeted_prompt(tokenizer,request["text"],candidate["text"],before,max_length=artifacts["few_shot"]["max_length"])
                    new_enc,new_trim=encode_budgeted_prompt(tokenizer,request["text"],candidate["text"],after,max_length=artifacts["few_shot"]["max_length"])
                    if not check["selection_changed"] and enc != new_enc:
                        raise AssertionError("Identical selections changed encoded inputs")
                    prefix="heldout" if aspect==heldout else "seen"
                    counts[prefix+"_prompts"]+=1
                    counts[prefix+"_changed"]+=int(check["selection_changed"])
                    counts["positive_retained_length_changed"]+=int(old_trim["demo_characters_removed"][:2] != new_trim["demo_characters_removed"][:2])
                    row={"request_id":request["request_id"],"aspect":aspect,**check,
                        "old_selected_ids":info["selected_ids"],"new_selected_ids":new_info["selected_ids"],
                        "old_encoded_sha256":item_sha(enc),"new_encoded_sha256":item_sha(new_enc),
                        "old_truncation":old_trim,"new_truncation":new_trim}
                    with (args.output/"selection_pairs.jsonl").open("a",encoding="utf-8") as handle:
                        handle.write(json.dumps(row)+"\n")
                if position%10==0:
                    print(json.dumps({"phase":"cpu_selection_audit","completed":position,"total":len(requests)}),flush=True)
            summary.update(status="complete",gates_pass=True,counts=dict(counts),reviews=len(requests),
                selection_pairs_sha256=digest(args.output/"selection_pairs.jsonl"),
                budget_note="Positive example identities remain fixed; replacing N can redistribute demo-first truncation across retained Y text.")
        else:
            audit=json.loads((args.cpu_audit/"summary.json").read_text(encoding="utf-8"))
            if audit.get("status") != "complete" or audit.get("input_sha256") != input_hashes or audit.get("source_sha256") != summary["source_sha256"]:
                raise ValueError("CPU audit is incomplete or no longer matches inputs/code")
            if digest(args.cpu_audit/"selection_pairs.jsonl") != audit["selection_pairs_sha256"]:
                raise ValueError("CPU selection evidence hash changed")
            pairs={(r["request_id"],r["aspect"]):r for r in read_lines(args.cpu_audit/"selection_pairs.jsonl")}
            from msc_project.serving.inference import QwenServingEngine
            engine=QwenServingEngine.from_artifacts(args.artifacts,variant="hybrid",pair_batch_size=4,
                max_batch_tokens=4096,logits_mode="last_token",enable_prefix_cache=False,device="cuda",local_files_only=True)
            retriever=DemonstrationRetriever(args.index,method="e5",encoder_path=args.encoder_path,
                fixed_demonstrations=engine.demonstrations["aspect"])
            captured=[]
            original=engine._score_encoded
            def traced(items,token_ids,**kwargs):
                captured.extend(item_sha(item) for item in items)
                return original(items,token_ids,**kwargs)
            engine._score_encoded=traced
            sampled=select_requests(requests,3)
            checks=[]
            for policy in (GLOBAL,SAME_ASPECT_NEGATIVE_SEEN):
                retriever.selection_policy=policy
                engine.set_retriever(retriever)
                for request in sampled:
                    captured.clear()
                    response=engine.score_presence([request])[0]
                    if len(captured)!=len(candidates): raise AssertionError("Native input count differs")
                    ref_scores={r["aspect"]:r for r in references[request["request_id"]]["aspect_scores"] if r["aspect"]!=heldout}
                    for actual,encoded in zip(response["aspect_scores"],captured):
                        aspect=actual["aspect"]
                        reference=pairs[(request["request_id"],aspect)]
                        if encoded != reference[("old" if policy==GLOBAL else "new")+"_encoded_sha256"]:
                            raise AssertionError("Native encoder differs from CPU selection audit")
                        if policy==GLOBAL and aspect!=heldout:
                            old=ref_scores[aspect]
                            if actual["score"]!=old["score"] or actual["presence_logit_margin"]!=old["presence_logit_margin"]:
                                raise AssertionError("Global E5 native seen scores changed from frozen reference")
                        checks.append({"policy":policy,"request_id":request["request_id"],"aspect":aspect,"encoded_input_equal":True})
            summary.update(status="complete",gates_pass=True,reviews=3,encoded_pairs=len(checks),
                legacy_seen_probability_and_margin_exact=True,policy_checks=checks,
                cpu_audit_summary_sha256=digest(args.cpu_audit/"summary.json"))
    except Exception as exc:
        summary.update(status="failed",gates_pass=False,error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        write_json(args.output/"summary.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k in ("status","gates_pass","counts","reviews","encoded_pairs","legacy_seen_probability_and_margin_exact")}),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
