"""Batch-shape-sensitive CPU model proves canonical groups preserve full-grid inputs."""
from copy import deepcopy

import numpy as np
import pytest

from test_serving_inference import artifacts, Tokenizer, Model, predictions_only
from msc_project.serving.inference import QwenServingEngine, required_canonical_groups


class ShapeSensitiveModel(Model):
    def forward(self,input_ids,attention_mask,**kwargs):
        output=super().forward(input_ids,attention_mask,**kwargs)
        # A deterministic stand-in for quantized batch-shape rounding drift.
        output.logits[:,:,10] += input_ids.shape[0] * .37 + input_ids.shape[1] * .013
        return output


def configured_engine(artifacts, selected, *, execution, logits_mode="last_token", max_tokens=4096):
    config=deepcopy(artifacts)
    config["taxonomy"]["candidates"]=[{"aspect":str(i),"text":"Topic " + "word "*(i%5+1)} for i in range(12)]
    engine=QwenServingEngine(config,Tokenizer(),ShapeSensitiveModel(),variant="hybrid",pair_batch_size=4,
        max_batch_tokens=max_tokens,logits_mode=logits_mode,sentiment_execution=execution)
    original_stage=engine._score_stage
    def stage(pairs,*,mode,use_adapter,**kwargs):
        if mode=="aspect":
            yes=[i in selected for i in range(len(pairs))]
            scores=np.array([[.9,.1] if hit else [.1,.9] for hit in yes])
            return scores,{"presence_logit_margins":[2. if hit else -2. for hit in yes],
                "retrieval":[],"retrieval_fallback":False},[1]*len(pairs)
        return original_stage(pairs,mode=mode,use_adapter=use_adapter,**kwargs)
    engine._score_stage=stage
    trace=[]
    original_encoded=engine._score_encoded
    def encoded(items,*args,**kwargs):
        if kwargs.get("mode")=="sentiment": trace.append(deepcopy(items))
        return original_encoded(items,*args,**kwargs)
    engine._score_encoded=encoded
    return engine,trace


@pytest.mark.parametrize("selected", [set(),{1},{1,9},set(range(12))])
@pytest.mark.parametrize("logits_mode", ["full","last_token"])
def test_conditionally_execute_complete_original_groups(artifacts,selected,logits_mode):
    engine,trace=configured_engine(artifacts,selected,execution="canonical_full_grid_groups",logits_mode=logits_mode)
    request=[{"request_id":"one","text":"A synthetic customer review"}]
    full=engine.predict_batch(request,full_grid=True)
    full_trace=deepcopy(trace)
    trace.clear()
    conditional=engine.predict_batch(request)
    assert predictions_only(conditional)==predictions_only(full)
    groups=sorted({i//4 for i in selected})
    assert trace==[full_trace[i] for i in groups]
    diagnostic=conditional[0]["diagnostics"]
    assert diagnostic["sentiment_prompt_count"]==4*len(groups)
    assert diagnostic["sentiment_companion_prompt_count"]==4*len(groups)-len(selected)
    assert diagnostic["sentiment_full_grid_prompt_count"]==12
    assert diagnostic["batch_sentiment"]["encoded_prompt_count"]==12
    assert diagnostic["batch_sentiment"]["executed_canonical_group_indices"]==groups
    assert len(conditional[0]["predictions"])==len(selected)
    assert "full_grid_scores" not in conditional[0]
    assert engine.metadata()["sentiment_execution"]=="canonical_full_grid_groups"
    contract=engine.sentiment_group_contract()
    assert engine.model_version().endswith("sentiment-canonicalgroups-"+contract["contract_sha256"][:12])
    assert diagnostic["sentiment_group_contract"]==contract


def test_selected_only_can_drift_while_full_grid_and_sentiment_only_remain_identical(artifacts):
    selected={1,6}
    request=[{"request_id":"one","text":"A synthetic customer review"}]
    old,_=configured_engine(artifacts,selected,execution="selected_only")
    old_full=old.predict_batch(request,full_grid=True)
    assert predictions_only(old.predict_batch(request)) != predictions_only(old_full)
    canonical,_=configured_engine(artifacts,selected,execution="canonical_full_grid_groups")
    canonical_full=canonical.predict_batch(request,full_grid=True)
    assert canonical_full[0]["full_grid_scores"]==old_full[0]["full_grid_scores"]
    assert canonical.score_full_sentiment(request)[0]["full_grid_scores"]==old_full[0]["full_grid_scores"]
    assert predictions_only(canonical.predict_batch(request))==predictions_only(old_full)


def test_canonical_groups_follow_token_budget_and_same_multi_request_batch(artifacts):
    engine,trace=configured_engine(artifacts,{1,13,22},execution="canonical_full_grid_groups",max_tokens=220)
    requests=[{"request_id":"one","text":"short"},{"request_id":"two","text":"long "*45}]
    full=engine.predict_batch(requests,full_grid=True)
    full_trace=deepcopy(trace)
    trace.clear()
    conditional=engine.predict_batch(requests)
    assert predictions_only(conditional)==predictions_only(full)
    assert sum(row["diagnostics"]["sentiment_prompt_count"] for row in conditional)==conditional[0]["diagnostics"]["batch_sentiment"]["prompt_count"]
    assert sum(row["diagnostics"]["sentiment_companion_prompt_count"] for row in conditional)==conditional[0]["diagnostics"]["batch_sentiment"]["companion_prompt_count"]
    selected_groups=conditional[0]["diagnostics"]["batch_sentiment"]["executed_canonical_group_indices"]
    assert trace==[full_trace[i] for i in selected_groups]
    assert all(len(items)*max(len(item["input_ids"]) for item in items)<=220 for items in trace)


def test_group_selector_rejects_invalid_requests_and_never_repacks():
    batches=[[0,1],[2],[3,4]]
    assert required_canonical_groups(batches,[0,4])==([[0,1],[3,4]],[0,2])
    for invalid in ([0,0],[5],[-1],[True],[1.5]):
        with pytest.raises(ValueError,match="required positions"):
            required_canonical_groups(batches,invalid)


def test_group_contract_changes_with_order_limits_or_execution_and_rejects_artifact_conflict(artifacts):
    from msc_project.serving.sentiment_execution import build_sentiment_group_contract
    args=dict(execution="canonical_full_grid_groups",pair_batch_size=4,max_batch_tokens=4096,logits_mode="last_token")
    candidates=artifacts["taxonomy"]["candidates"]
    base=build_sentiment_group_contract(candidates,**args)
    assert build_sentiment_group_contract(list(reversed(candidates)),**args)!=base
    assert build_sentiment_group_contract(candidates,**{**args,"pair_batch_size":2})!=base
    assert build_sentiment_group_contract(candidates,**{**args,"execution":"selected_only"})!=base
    artifacts["service_execution"]={"sentiment_execution":"canonical_full_grid_groups"}
    with pytest.raises(ValueError,match="registered service"):
        QwenServingEngine(artifacts,Tokenizer(),Model())


def test_native_gate_requires_exact_anchor_vector_and_keeps_runner_boundary():
    from check_canonical_sentiment_gpu import selected_anchor_check
    anchor={"full_grid_scores":[{"aspect":"A","sentiment_scores":{"negative":.1,"neutral":.39995,"positive":.50005}}]}
    response={"predictions":[{"aspect":"A","sentiments":["positive"],"sentiment_scores":deepcopy(anchor["full_grid_scores"][0]["sentiment_scores"])}]}
    result=selected_anchor_check(response,anchor,.4)
    assert result["selected_vectors_exact"] and result["selected_labels_equal"]
    response["predictions"][0].update(sentiment_scores={"negative":.1,"neutral":.405,"positive":.495},sentiments=["positive","neutral"])
    result=selected_anchor_check(response,anchor,.4)
    assert not result["selected_vectors_exact"] and not result["selected_labels_equal"]
