from dataclasses import asdict
import json

import numpy as np
import pytest

from msc_project.llm.qwen_two_stage_classifier import TwoStageDemonstration
from msc_project.serving.retrieval import (
    DemonstrationRetriever, build_pool, digest, encode_budgeted_prompt, exclude_duplicates,
)


def make_index(tmp_path):
    pool = []
    for i in range(8):
        pool.append({"row_uid": f"train:{i}", "candidate_aspect": "A" if i%2 else "B",
                     "review_text": f"Different example number {i} with words {'helpful' if i<4 else 'slow'}",
                     "aspect_candidate": "A meaning" if i%2 else "B meaning",
                     "answer": "Y" if i<4 else "N"})
    (tmp_path/"pool.jsonl").write_text("".join(json.dumps(r)+"\n" for r in pool), encoding="utf-8")
    (tmp_path/"manifest.json").write_text(json.dumps({"schema_version":"test", "pool_sha256":digest(pool),
                                                    "heldout_aspect":"H"}), encoding="utf-8")
    return pool


def test_only_train_and_no_heldout_reviews_enter_pool():
    rows = [{"row_uid":"train:1", "original_split":"train", "text":"normal example", "labels":[("A","positive")]},
            {"row_uid":"train:2", "original_split":"train", "text":"heldout and seen", "labels":[("H","negative"),("A","positive")]},
            {"row_uid":"train:3", "original_split":"train", "text":"Evaluation MATCH!", "labels":[("A","negative")]}]
    pool, audit = build_pool(rows, [{"aspect":a,"text":a} for a in ("A","B","H")], "H", ["evaluation match"])
    assert {r["row_uid"] for r in pool} == {"train:1"}
    assert {r["candidate_aspect"] for r in pool} == {"A","B"}
    assert audit["heldout_reviews_excluded"] == 1
    rows[0]["original_split"] = "validation"
    with pytest.raises(ValueError, match="official train"):
        build_pool(rows, [{"aspect":"A","text":"A"}], "H", [])


def test_near_duplicate_removed_from_training_not_evaluation():
    original = "A long customer comment with many words and an unchanged account issue repeated for comparison"
    rows = [{"row_uid":"a", "text":original+"!"}, {"row_uid":"b", "text":original+" now"},
            {"row_uid":"c", "text":"Totally separate short message"}]
    accepted, counts = exclude_duplicates(rows, [original])
    assert [r["row_uid"] for r in accepted] == ["c"]
    assert counts == {"evaluation_exact":1, "evaluation_near":1}


@pytest.mark.parametrize("method", ["random", "bm25"])
def test_balanced_deterministic_diverse_and_query_exclusion(tmp_path, method):
    pool = make_index(tmp_path)
    retriever = DemonstrationRetriever(tmp_path, method=method, seed=13)
    query = pool[0]["review_text"]
    one, meta = retriever.select(query, "H", "H definition")
    two, _ = retriever.select(query, "H", "H definition")
    assert one == two
    assert [r.answer for r in one] == list("YYNN")
    assert all(r.row_uid != "train:0" for r in one)
    assert len({r.row_uid for r in one[:2]}) == 2
    assert len({r.row_uid for r in one[2:]}) == 2
    assert meta["fallback"] is False


def test_unknown_method_tampered_index_and_illegal_fixed_rejected(tmp_path):
    pool = make_index(tmp_path)
    with pytest.raises(ValueError, match="Unknown"):
        DemonstrationRetriever(tmp_path, method="invalid")
    fixed = [TwoStageDemonstration(**pool[i]) for i in (0,1,4,5)]
    r = DemonstrationRetriever(tmp_path, method="fixed", fixed_demonstrations=fixed)
    assert r.select("new query", "A", "A")[0] == fixed
    with pytest.raises(ValueError, match="duplicates"):
        r.select(pool[0]["review_text"], "A", "A")
    altered = TwoStageDemonstration(**{**asdict(fixed[0]), "answer":"N"})
    with pytest.raises(ValueError, match="legal cleaned"):
        DemonstrationRetriever(tmp_path, method="random", fixed_demonstrations=[altered])
    (tmp_path/"pool.jsonl").write_text(json.dumps(pool[0])+"\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        DemonstrationRetriever(tmp_path)


def test_rrf_rank_fusion_keeps_both_classes(tmp_path, monkeypatch):
    from msc_project.serving import retrieval
    pool = make_index(tmp_path)
    vectors = np.array([[1.,float(i+1)] for i in range(len(pool))],dtype=np.float32)
    vectors /= np.linalg.norm(vectors,axis=1,keepdims=True)
    np.save(tmp_path/"embeddings.npy", vectors)
    import hashlib
    manifest = json.loads((tmp_path/"manifest.json").read_text())
    manifest.update(encoder_path="test", embeddings_sha256=hashlib.sha256((tmp_path/"embeddings.npy").read_bytes()).hexdigest())
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    class Encoder:
        def __init__(self, path):
            self.calls = []
        def encode(self, texts):
            self.calls.append(list(texts))
            return np.tile(np.array([[.6,.8]], dtype=np.float32), (len(texts),1))
    monkeypatch.setattr(retrieval,"E5Encoder",Encoder)
    r = DemonstrationRetriever(tmp_path, method="rrf")
    r.prepare(["first", "first"], ["A","A"])
    assert len(r.encoder.calls[0]) == 1
    result,_ = r.select("first","A","A")
    assert [d.answer for d in result] == list("YYNN")
    r.prepare(["second"],["A"])
    assert "first" not in r._query_vectors
    assert len(r.encoder.calls) == 3  # review1, staticcard, review2


def test_budget_encoder_trims_demos_before_query():
    class Tokenizer:
        def apply_chat_template(self,messages,**kw):
            return " ".join(m["content"] for m in messages)
        def __call__(self,text,**kw):
            return {"input_ids":list(range(len(text.split())))}
    demos = [TwoStageDemonstration(str(i),"A","extra "*100,"A",answer) for i,answer in enumerate("YYNN")]
    _, report = encode_budgeted_prompt(Tokenizer(),"short current review","A",demos,max_length=210)
    assert sum(report["demo_characters_removed"]) > 0
    assert report["query_characters_removed"] == 0
    assert not report["severe_query_truncation"]
