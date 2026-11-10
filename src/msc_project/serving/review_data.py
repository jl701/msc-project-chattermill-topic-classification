"""Public-review snapshots and duplicate-safe, label-free annotation splits."""
from __future__ import annotations

import hashlib
import html
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def canonical_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", html.unescape(text)).casefold().split())


def private_safe_text(text: str) -> str:
    text = html.unescape(text).replace("\x00", "")
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[EMAIL]", text)
    text = re.sub(r"(?<!\w)(?:\+\d[\d ()-]{8,}\d|0\d[\d ()-]{8,}\d)(?!\w)", "[PHONE]", text)
    return text.strip()


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\w+", canonical_text(text)))


def shingles(text: str) -> frozenset[tuple[str, ...]]:
    words = tokens(text)
    return frozenset(tuple(words[i:i + 3]) for i in range(len(words) - 2)) if len(words) >= 8 else frozenset()


def parse_feed(payload: dict, *, app: str, app_id: str, country: str, source_url: str) -> list[dict]:
    rows = []
    for item in payload.get("feed", {}).get("entry", []):
        if "im:rating" not in item:
            continue
        body = str(item.get("content", {}).get("label", ""))
        title = str(item.get("title", {}).get("label", ""))
        text = private_safe_text(body)
        if not text:
            continue
        source_id = str(item["id"]["label"])
        rows.append({
            "request_id": "app-" + digest(f"{country}:{app_id}:{source_id}")[:20],
            "source_review_id": source_id, "app": app, "app_id": app_id,
            "storefront": country, "source_url": source_url,
            "published_at": item.get("updated", {}).get("label"),
            "text": text, "title": private_safe_text(title),
            "raw_body_sha256": digest(body), "text_sha256": digest(text),
            "rating_metadata_not_gold": item["im:rating"]["label"],
            "language_status": "requires_human_confirmation",
        })
    return rows


def deduplicate(rows: list[dict], historical_texts: Iterable[str]) -> tuple[list[dict], dict]:
    """Exclude exact/near groups touching historical text; retain one per new group.

    Union all matches, not only the first, so transitive duplicate groups cannot
    straddle partitions. For fewer than eight tokens only exact matches count.
    """
    documents = list(dict.fromkeys(canonical_text(t) for t in historical_texts if t.strip()))
    history_count = len(documents)
    documents += [canonical_text(row["text"]) for row in rows]
    parent = list(range(len(documents)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    exact: dict[str, int] = {}
    postings: dict[tuple[str, ...], list[int]] = defaultdict(list)
    signatures = []
    lengths = []
    near_edges = 0
    for i, text in enumerate(documents):
        signature = shingles(text)
        signatures.append(signature)
        lengths.append(len(tokens(text)))
        if text in exact:
            union(i, exact[text])
        exact.setdefault(text, i)
        if signature:
            candidates: Counter[int] = Counter()
            for shingle in signature:
                candidates.update(postings[shingle])
            for j, overlap in candidates.items():
                if min(lengths[i], lengths[j]) / max(lengths[i], lengths[j]) < .9:
                    continue
                if overlap / (len(signature) + len(signatures[j]) - overlap) >= .9:
                    union(i, j)
                    near_edges += 1
            for shingle in signature:
                postings[shingle].append(i)
    historical_groups = {find(i) for i in range(history_count)}
    seen = set()
    kept, excluded_historical, excluded_new = [], 0, 0
    for offset, row in enumerate(rows):
        group = find(history_count + offset)
        if group in historical_groups:
            excluded_historical += 1
        elif group in seen:
            excluded_new += 1
        else:
            seen.add(group)
            kept.append(dict(row, duplicate_group_id=digest(documents[group])[:20]))
    return kept, {
        "input_count": len(rows), "historical_unique_normalized_texts": history_count,
        "historical_overlap_removed": excluded_historical,
        "new_duplicate_group_members_removed": excluded_new,
        "near_duplicate_edges": near_edges, "unique_candidates": len(kept),
    }


def assign_splits(rows: list[dict], *, target: int = 800, seed: int = 20261111) -> tuple[list[dict], dict]:
    """Proportional allocation by source with deterministic shuffles; no labels."""
    if len({r["request_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate request IDs")
    if len({r["duplicate_group_id"] for r in rows}) != len(rows):
        raise ValueError("Retain one representative per duplicate group before splitting")
    rng = random.Random(seed)
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["request_id"]):
        by_source[row["app"]].append(row)
    for source in sorted(by_source):
        rng.shuffle(by_source[source])
    count = min(target, len(rows))
    # Stratified interleaving preserves source proportions in any prefix.
    chosen = []
    used = Counter()
    for step in range(count):
        eligible = [s for s in by_source if used[s] < len(by_source[s])]
        source = max(sorted(eligible), key=lambda s: (step + 1) * len(by_source[s]) / len(rows) - used[s])
        chosen.append(by_source[source][used[source]])
        used[source] += 1
    bounds = [count // 4, count // 2, count]
    output = []
    for i, row in enumerate(chosen):
        split = "adaptation" if i < bounds[0] else "calibration" if i < bounds[1] else "locked"
        output.append(dict(row, split=split, pilot=split == "adaptation" and i < 50))
    return output, {"seed": seed, "target": target, "actual": count,
                    "shortfall": target - count,
                    "counts": dict(Counter(r["split"] for r in output)),
                    "source_by_split": {s: dict(Counter(r["app"] for r in output if r["split"] == s))
                                        for s in ("adaptation", "calibration", "locked")}}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
