import json

import pytest
import torch

from probe_rag_logit_margins_gpu import extract_pair_statistics, length_rank_inputs, summarize_rows


def test_last_nonpad_mapping_matches_reduced_logits_and_does_not_change_tensors():
    masks = torch.tensor([[1,1,0,0],[1,1,1,1],[1,1,1,0]])
    logits = torch.zeros(3,4,6,dtype=torch.float16)
    logits[0,1,2],logits[0,1,3] = 20.,0.
    logits[1,3,2],logits[1,3,3] = 21.,1.
    logits[2,2,2],logits[2,2,3] = 40.,0.
    original_logits,original_masks = logits.clone(),masks.clone()
    full = extract_pair_statistics(logits,masks,(2,3))
    kept = torch.tensor([1,2,3])
    reduced = extract_pair_statistics(logits[:,kept,:],masks,(2,3),kept)
    assert full == reduced
    assert all(row["probability_float32"] == 1.0 for row in full)
    assert full[0]["probability_float64"] < 1.0
    assert full[2]["probability_float64"] == 1.0
    assert [row["margin_y_minus_n"] for row in full] == [20.,20.,40.]
    assert torch.equal(logits,original_logits) and torch.equal(masks,original_masks)
    with pytest.raises(ValueError,match="missing"):
        extract_pair_statistics(logits[:,[1,3],:],masks,(2,3),torch.tensor([1,3]))


def test_raw_margin_subtraction_does_not_round_back_to_half_precision():
    logits = torch.tensor([[[1000.,-.5]]],dtype=torch.float16)
    value = extract_pair_statistics(logits,torch.ones(1,1,dtype=torch.long),(0,1))[0]
    assert value["margin_y_minus_n"] == 1000.5
    assert value["logit_y"] == 1000. and value["logit_n"] == -.5


def test_saturation_summary_counts_candidate_scope_and_lost_ranking():
    base = {"probability_float32":1.,"source_logits_dtype":"torch.float16", "original_engine_probability":1.}
    rows = [dict(base,margin_y_minus_n=20.,probability_float64=.999999998,seen_candidate=True),
            dict(base,margin_y_minus_n=30.,probability_float64=.9999999999999,seen_candidate=True),
            dict(base,margin_y_minus_n=40.,probability_float64=1.,seen_candidate=False)]
    result = summarize_rows(rows)
    assert result["all_candidate_pairs"]["float32_probability_one"] == 3
    assert result["all_candidate_pairs"]["float64_probability_one"] == 1
    assert result["seen_candidate_pairs"]["distinct_margins_in_float32_one_bucket"] == 2
    assert result["seen_candidate_pairs"]["float32_one_bucket_raw_margin_range"] == {"min":20.,"max":30.}
    assert result["heldout_candidate_pairs"]["pair_count"] == 1


def test_probe_selection_is_deterministic_and_rejects_label_inputs(tmp_path):
    path = tmp_path/"development.jsonl"
    rows = [{"request_id":str(i),"text":"x"*(i+1)} for i in range(100)]
    path.write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf-8")
    selected = length_rank_inputs(path,20)
    assert len(selected) == 20 and selected[0]["request_id"] == "0" and selected[-1]["request_id"] == "99"
    assert selected == length_rank_inputs(path,20)
    assert [r["request_id"] for r in length_rank_inputs(path,3)] == ["0","50","99"]
    rows[0]["labels"] = []
    path.write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf-8")
    with pytest.raises(ValueError,match="input-only"):
        length_rank_inputs(path,20)
