import hashlib
import json

import numpy as np
import pytest

from msc_project.serving import retrieval
from msc_project.serving.retrieval_policy import SAME_ASPECT_NEGATIVE_SEEN


def make_retriever(tmp_path, monkeypatch, *, omit_b_negatives=False):
    specs = [("shared","A","Y"),("y2","B","Y"),("y3","A","Y"),
             ("n1","A","N"),("shared","B","N"),("n2","A","N"),
             ("n3","A","N"),("n4","B","N")]
    if omit_b_negatives:
        specs = [s for s in specs if s[1:] != ("B","N")]
    pool = [{"row_uid":uid,"candidate_aspect":a,"review_text":f"Train review {uid}",
             "aspect_candidate":f"Card {a}","answer":label} for uid,a,label in specs]
    (tmp_path/"pool.jsonl").write_text("".join(json.dumps(r)+"\n" for r in pool),encoding="utf-8")
    np.save(tmp_path/"embeddings.npy",np.ones((len(pool),2),dtype=np.float32))
    (tmp_path/"manifest.json").write_text(json.dumps({"schema_version":"test", "pool_sha256":retrieval.digest(pool),
        "heldout_aspect":"H","encoder_path":"synthetic",
        "embeddings_sha256":hashlib.sha256((tmp_path/"embeddings.npy").read_bytes()).hexdigest()}),encoding="utf-8")
    class Encoder:
        def __init__(self,path): pass
        def encode(self,texts): return np.ones((len(texts),2),dtype=np.float32)
    monkeypatch.setattr(retrieval,"E5Encoder",Encoder)
    r = retrieval.DemonstrationRetriever(tmp_path,method="e5")
    r._scores = lambda text,card: np.arange(len(pool),0,-1,dtype=np.float64)
    return r


def test_only_seen_negative_universe_changes_positive_identity_and_heldout_exact(tmp_path,monkeypatch):
    r = make_retriever(tmp_path,monkeypatch)
    before = {a:r.select("new text",a,f"Card {a}")[0] for a in ("A","B","H")}
    r.selection_policy = SAME_ASPECT_NEGATIVE_SEEN
    for aspect in ("A","B","H"):
        after,meta = r.select("new text",aspect,f"Card {aspect}")
        assert after[:2] == before[aspect][:2]
        assert [d.answer for d in after] == list("YYNN")
        if aspect == "H":
            assert after == before[aspect]
            assert meta["effective_selection_policy"] == "global"
        else:
            assert all(d.candidate_aspect == aspect for d in after[2:])
            assert len({d.row_uid for d in after[2:]}) == 2
            assert meta["effective_selection_policy"] == SAME_ASPECT_NEGATIVE_SEEN
    # Do not bundle a new cross-class dedup policy: a preserved Y for A can
    # share its review with a same-candidate N for B, as under the old rules.
    after,_ = r.select("new text","B","Card B")
    assert after[0].row_uid == after[2].row_uid == "shared"


def test_self_exclusion_before_shortlist_and_no_silent_global_fallback(tmp_path,monkeypatch):
    r = make_retriever(tmp_path,monkeypatch)
    r.selection_policy = SAME_ASPECT_NEGATIVE_SEEN
    after,_ = r.select("TRAIN review n1!","A","Card A")
    assert [d.row_uid for d in after[2:]] == ["n2","n3"]
    with pytest.raises(ValueError,match="Insufficient"):
        r.select("Train review shared","B","Card B")
    with pytest.raises(ValueError,match="Insufficient"):
        r.select("new text","unknown","unknown card")


def test_already_two_same_negative_examples_remain_exact(tmp_path,monkeypatch):
    r = make_retriever(tmp_path,monkeypatch,omit_b_negatives=True)
    before,_ = r.select("new text","A","Card A")
    r.selection_policy = SAME_ASPECT_NEGATIVE_SEEN
    after,_ = r.select("new text","A","Card A")
    assert after == before
