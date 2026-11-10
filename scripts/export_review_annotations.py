"""Export explicitly confirmed human labels, preserving the frozen split."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from msc_project.serving.annotation import AnnotationStore
from msc_project.serving.review_data import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "outputs/rag_20261111/new_reviews")
    parser.add_argument("--output", type=Path, required=True, help="A new export directory; old exports are immutable")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Export directory already exists; choose a new version")
    store = AnnotationStore(args.data / "annotation_reviews.jsonl", args.data / "annotation_taxonomy.json", args.data / "human_labels")
    args.output.mkdir(parents=True)
    summary = {"annotation_sha256": hashlib.sha256(store.path.read_bytes()).hexdigest() if store.path.exists() else None,
               "scope": "Only manually confirmed, English-confirmed labels; not model predictions", "splits": {}}
    for split in ("adaptation", "calibration", "locked"):
        ids = store.group_ids(split)
        accepted = [i for i in ids if store.annotations.get(i, {}).get("status") == "confirmed"
                    and store.annotations[i].get("english_confirmed") and not store.annotations[i].get("ambiguous")]
        labels = [{"request_id": i,
                   "pairs": [{"aspect": pair[0], "sentiment": pair[1]} for pair in store.annotations[i]["labels"]],
                   "aspects": sorted({pair[0] for pair in store.annotations[i]["labels"]}),
                   "source": "human_annotation", "revision": store.annotations[i]["revision"]} for i in accepted]
        replay = [{"request_id": i, "text": store.rows[i]["text"]} for i in accepted]
        write_jsonl(args.output / f"labels_{split}.jsonl", labels)
        write_jsonl(args.output / f"replay_{split}.jsonl", replay)
        summary["splits"][split] = {"total": len(ids), "confirmed": len(accepted),
                                    "complete": len(accepted) == len(ids), "pending": len(ids) - len(accepted)}
    (args.output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
