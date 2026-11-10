"""Train-only demonstration retrieval; raw examples live only in private indexes.

No query labels are accepted. E5 query embeddings are prepared once per request
batch and discarded on the next batch, not cached across HTTP requests.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Sequence
import unicodedata

import numpy as np

from msc_project.llm.qwen_two_stage_classifier import TwoStageDemonstration

METHODS = ("fixed", "random", "bm25", "e5", "rrf")
E5_REVISION = "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"


def encode_budgeted_prompt(tokenizer, text, candidate, demonstrations, *, max_length):
    """Preserve query first; retain structured instructions/cards/labels exactly."""
    from msc_project.llm.qwen_two_stage_classifier import (
        render_two_stage_few_shot_prompt, _tokenize_without_special_tokens,
    )
    reviews = [d.review_text for d in demonstrations] + [text]
    original_lengths = list(map(len, reviews))
    original_tokens = None
    while True:
        adjusted = [TwoStageDemonstration(**{**asdict(d), "review_text": reviews[i]})
                    for i, d in enumerate(demonstrations)]
        prompt = render_two_stage_few_shot_prompt(tokenizer, reviews[-1], candidate,
                                                mode="aspect", demonstrations=adjusted)
        ids = _tokenize_without_special_tokens(tokenizer, prompt)
        if original_tokens is None:
            original_tokens = len(ids)
        if len(ids) <= max_length:
            if not ids:
                raise ValueError("Empty budgeted prompt")
            removed = [n-len(t) for n,t in zip(original_lengths, reviews)]
            return {"input_ids": ids, "attention_mask": [1]*len(ids)}, {
                "policy": "demonstration_review_first_left_character_trim_v1",
                "original_prompt_tokens": original_tokens, "final_prompt_tokens": len(ids),
                "demo_characters_removed": removed[:-1], "query_characters_removed": removed[-1],
                "query_characters_original": original_lengths[-1],
                "severe_query_truncation": removed[-1] > .2 * max(1, original_lengths[-1]),
            }
        nonempty_demos = [i for i in range(len(demonstrations)) if reviews[i]]
        target = max(nonempty_demos, key=lambda i:(len(reviews[i]), -i)) if nonempty_demos else len(reviews)-1
        if not reviews[target]:
            raise ValueError("Structured non-review prompt exceeds budget")
        reviews[target] = reviews[target][max(1, len(reviews[target])//8):]


def normalise_text(text: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def character_shingles(text: str, n: int = 5) -> set[str]:
    value = normalise_text(text)
    return {value[i:i+n] for i in range(max(1, len(value)-n+1))}


def exclude_duplicates(rows: list[dict], excluded_texts: Sequence[str]) -> tuple[list[dict], dict]:
    """Exclude canonical matches or char-5-gram Jaccard >= .90 (length >=20).

    The inverted shingle lookup avoids all-pairs string comparisons. Within
    training, only canonical duplicates are collapsed; evaluation near matches
    are removed from training, never removed from the evaluation population.
    """
    excluded = {normalise_text(t) for t in excluded_texts if normalise_text(t)}
    shingles = [character_shingles(t) for t in sorted(excluded) if len(t) >= 20]
    postings: dict[str, list[int]] = {}
    for i, values in enumerate(shingles):
        for value in values:
            postings.setdefault(value, []).append(i)
    accepted, seen = [], set()
    counts = Counter()
    for row in sorted(rows, key=lambda r: str(r["row_uid"])):
        text = normalise_text(row["text"])
        if not text:
            counts["empty"] += 1
            continue
        if text in excluded:
            counts["evaluation_exact"] += 1
            continue
        values = character_shingles(text)
        overlap: Counter = Counter()
        if len(text) >= 20:
            for value in values:
                overlap.update(postings.get(value, ()))
        if any(n / (len(values) + len(shingles[i]) - n) >= .90
               for i, n in overlap.items()):
            counts["evaluation_near"] += 1
            continue
        if text in seen:
            counts["train_exact"] += 1
            continue
        seen.add(text)
        accepted.append(row)
    return accepted, dict(counts)


def build_pool(rows: Sequence[dict], candidates: Sequence[dict], heldout: str,
               excluded_texts: Sequence[str]) -> tuple[list[dict], dict]:
    legal = []
    rejected_heldout = 0
    for row in rows:
        if row.get("original_split", row.get("split")) != "train":
            raise ValueError("Only explicitly declared official train rows may enter the index")
        aspects = {str(a) for a, _ in row["labels"]}
        if heldout in aspects:
            rejected_heldout += 1
            continue
        legal.append(dict(row))
    legal, exclusions = exclude_duplicates(legal, excluded_texts)
    pool = []
    for row in legal:
        aspects = {str(a) for a, _ in row["labels"]}
        for candidate in candidates:
            if candidate["aspect"] == heldout:
                continue
            pool.append({"row_uid": str(row["row_uid"]),
                         "candidate_aspect": candidate["aspect"],
                         "review_text": row["text"], "aspect_candidate": candidate["text"],
                         "answer": "Y" if candidate["aspect"] in aspects else "N"})
    return pool, {"train_input_reviews": len(rows), "heldout_reviews_excluded": rejected_heldout,
                  "dedup_exclusions": exclusions, "legal_reviews": len(legal),
                  "pool_pairs": len(pool), "heldout_aspect": heldout,
                  "duplicate_rule": "NFKC casefold word normalisation exact; char5 Jaccard>=0.90 for length>=20"}


class E5Encoder:
    def __init__(self, path: str | Path, *, batch_size: int = 32, device: str = "cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.path = str(path)
        if device not in {"cpu", "cuda"}:
            raise ValueError("E5 device must be cpu or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for E5 but unavailable")
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(path, local_files_only=True).to(device).eval()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        torch = self.torch
        outputs = []
        for offset in range(0, len(texts), self.batch_size):
            values = self.tokenizer(list(texts[offset:offset+self.batch_size]),
                                    max_length=256, truncation=True, padding=True,
                                    return_tensors="pt")
            values = {name: value.to(self.device) for name, value in values.items()}
            with torch.inference_mode():
                hidden = self.model(**values).last_hidden_state
                mask = values["attention_mask"].unsqueeze(-1)
                mean = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                outputs.append(torch.nn.functional.normalize(mean, p=2, dim=1).cpu().numpy())
            if len(texts) >= 320 and ((offset // self.batch_size + 1) % 10 == 0 or offset + self.batch_size >= len(texts)):
                print(json.dumps({"stage":"e5_encoding", "completed":min(offset+self.batch_size,len(texts)),
                                  "total":len(texts), "device":self.device}), file=sys.stderr, flush=True)
        return np.concatenate(outputs).astype(np.float32)


def encode_pool(pool: Sequence[dict], encoder: E5Encoder) -> np.ndarray:
    texts = sorted({r["review_text"] for r in pool})
    cards = sorted({r["aspect_candidate"] for r in pool})
    a = encoder.encode(["passage: " + t for t in texts])
    b = encoder.encode(["query: " + t for t in cards])
    text_pos, card_pos = {t:i for i,t in enumerate(texts)}, {t:i for i,t in enumerate(cards)}
    values = np.stack([a[text_pos[r["review_text"]]] + b[card_pos[r["aspect_candidate"]]]
                       for r in pool])
    return values / np.linalg.norm(values, axis=1, keepdims=True)


class DemonstrationRetriever:
    def __init__(self, index_dir: str | Path, *, method: str = "bm25", seed: int = 13,
                 fixed_demonstrations: Sequence[TwoStageDemonstration] = (),
                 encoder_path: str | Path | None = None):
        if method not in METHODS:
            raise ValueError(f"Unknown retrieval method {method}")
        self.index_dir, self.method, self.seed = Path(index_dir), method, int(seed)
        self.manifest = json.loads((self.index_dir / "manifest.json").read_text(encoding="utf-8"))
        self.pool = [json.loads(line) for line in (self.index_dir / "pool.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()]
        self.fingerprint = digest(self.pool)
        if self.fingerprint != self.manifest["pool_sha256"]:
            raise ValueError("Retrieval index fingerprint mismatch")
        if not self.pool or any(r["answer"] not in ("Y", "N") for r in self.pool):
            raise ValueError("Invalid retrieval examples")
        if any(r["candidate_aspect"] == self.manifest["heldout_aspect"] for r in self.pool):
            raise ValueError("Held-out aspect entered retrieval pool")
        self.fixed = list(fixed_demonstrations)
        legal = {digest(r) for r in self.pool}
        if self.fixed and any(digest(asdict(d)) not in legal for d in self.fixed):
            raise ValueError("Fixed demonstration not present in legal cleaned index")
        if method == "fixed" and tuple(d.answer for d in self.fixed) != ("Y", "Y", "N", "N"):
            raise ValueError("Fixed method requires exactly Y,Y,N,N demonstrations")
        self._query_vectors: dict[str, np.ndarray] = {}
        self._card_vectors: dict[str, np.ndarray] = {}
        self.prepare_ms = 0.0
        self.encoder = None
        self.embeddings = None
        self._bm25 = None
        self._normal_texts = [normalise_text(r["review_text"]) for r in self.pool]
        self._class = {label: np.array([i for i,r in enumerate(self.pool) if r["answer"] == label])
                       for label in ("Y", "N")}
        if any(len(i) < 2 for i in self._class.values()):
            raise ValueError("Index needs two examples of each answer class")
        if method in ("bm25", "rrf"):
            from sklearn.feature_extraction.text import CountVectorizer
            self.vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", dtype=np.float32)
            x = self.vectorizer.fit_transform([r["review_text"] + "\n" + r["aspect_candidate"]
                                               for r in self.pool]).tocsr()
            lengths = np.asarray(x.sum(1)).ravel()
            df = np.bincount(x.indices, minlength=x.shape[1])
            idf = np.log(1 + (x.shape[0] - df + .5) / (df + .5))
            normaliser = 1.2 * (.25 + .75 * lengths / max(lengths.mean(), 1))
            x.data = (x.data * 2.2 / (x.data + np.repeat(normaliser, np.diff(x.indptr)))
                      * idf[x.indices]).astype(np.float32)
            self._bm25 = x
        if method in ("e5", "rrf"):
            path = encoder_path or self.manifest.get("encoder_path")
            if not path:
                raise ValueError("E5 encoder path is required")
            self.embeddings = np.load(self.index_dir / "embeddings.npy", mmap_mode="r")
            if self.embeddings.shape[0] != len(self.pool) or not np.isfinite(self.embeddings).all():
                raise ValueError("Index embeddings do not match pool")
            expected = self.manifest.get("embeddings_sha256")
            if not expected or hashlib.sha256((self.index_dir / "embeddings.npy").read_bytes()).hexdigest() != expected:
                raise ValueError("Embedding file fingerprint mismatch")
            self.encoder = E5Encoder(path)

    def metadata(self) -> dict:
        return {"method": self.method, "seed": self.seed, "pool_sha256": self.fingerprint,
                "pool_pairs": len(self.pool), "index_version": self.manifest["schema_version"],
                "e5_revision": self.manifest.get("e5_revision"),
                "query_embedding_cache": "within_current_request_batch_only"}

    def prepare(self, texts: Sequence[str], candidate_texts: Sequence[str]) -> None:
        started = time.perf_counter()
        self.prepare_ms = 0.0
        self._query_vectors = {}
        if self.encoder:
            unique = list(dict.fromkeys(texts))
            values = self.encoder.encode(["passage: " + t for t in unique])
            self._query_vectors = dict(zip(unique, values))
            missing = [t for t in dict.fromkeys(candidate_texts) if t not in self._card_vectors]
            if missing:
                self._card_vectors.update(zip(missing, self.encoder.encode(["query: " + t for t in missing])))
        self.prepare_ms = (time.perf_counter() - started) * 1000

    def _scores(self, text: str, candidate_text: str) -> np.ndarray:
        if self.method == "random":
            seed = int(hashlib.sha256(f"{self.seed}|{text}|{candidate_text}".encode()).hexdigest()[:16], 16)
            return np.random.default_rng(seed).random(len(self.pool))
        sparse = dense = None
        if self._bm25 is not None:
            query = self.vectorizer.transform([text + "\n" + candidate_text])
            sparse = (self._bm25 @ query.T).toarray().ravel()
        if self.encoder:
            if text not in self._query_vectors or candidate_text not in self._card_vectors:
                self.prepare([text], [candidate_text])
            vector = self._query_vectors[text] + self._card_vectors[candidate_text]
            vector = vector / np.linalg.norm(vector)
            dense = np.asarray(self.embeddings @ vector)
        if self.method == "bm25":
            return sparse
        if self.method == "e5":
            return dense
        result = np.zeros(len(self.pool), dtype=np.float64)
        for scores in (sparse, dense):
            for indices in self._class.values():
                order = indices[np.argsort(-scores[indices], kind="stable")]
                result[order] += 1.0 / (60 + np.arange(1, len(order)+1))
        return result

    def select(self, text: str, candidate_aspect: str, candidate_text: str):
        started = time.perf_counter()
        if self.method == "fixed":
            if any(normalise_text(d.review_text) == normalise_text(text) for d in self.fixed):
                raise ValueError("Current query duplicates a fixed demonstration")
            selected = list(self.fixed)
        else:
            scores = self._scores(text, candidate_text)
            if not np.isfinite(scores).all():
                raise ValueError("Non-finite retrieval scores")
            selected = []
            query_normal = normalise_text(text)
            for answer in ("Y", "N"):
                indices = self._class[answer]
                order = indices[np.argsort(-scores[indices], kind="stable")]
                # Full ranking allows runtime query-self exclusion before shortlist.
                order = [int(i) for i in order if self._normal_texts[i] != query_normal][:256]
                chosen, rows, aspects = [], set(), set()
                for diverse in (True, False):
                    for i in order:
                        r = self.pool[i]
                        if r["row_uid"] in rows or (diverse and r["candidate_aspect"] in aspects):
                            continue
                        chosen.append(TwoStageDemonstration(**r))
                        rows.add(r["row_uid"])
                        aspects.add(r["candidate_aspect"])
                        if len(chosen) == 2:
                            break
                    if len(chosen) == 2:
                        break
                if len(chosen) != 2:
                    raise ValueError("Insufficient diverse query-safe retrieval examples")
                selected.extend(chosen)
        return selected, {**self.metadata(), "candidate_aspect": candidate_aspect,
                          "selected_ids": [d.row_uid + "|" + d.candidate_aspect for d in selected],
                          "selected_answers": [d.answer for d in selected],
                          "demonstration_sha256": digest([asdict(d) for d in selected]),
                          "retrieval_ms": (time.perf_counter()-started)*1000,
                          "fallback": False}
