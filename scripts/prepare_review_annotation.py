"""Freeze duplicate-screened public-review annotation splits; read no test file."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from msc_project.serving.annotation import ZH
from msc_project.serving.review_data import assign_splits, deduplicate, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/rag_20261111/new_reviews")
    parser.add_argument("--data-dir", type=Path, default=ROOT.parent / "Project_Preparation/Public_Datasets/FABSA")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "outputs/serving_20261109/artifacts.json")
    args = parser.parse_args()
    if (args.input / "annotation_manifest.json").exists():
        raise SystemExit("Annotation split exists; preserve it and use a new snapshot")
    raw = args.input / "collected_reviews.jsonl"
    rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line.strip()]
    historical = []
    sources = []
    for split in ("train", "validation"):
        path = args.data_dir / f"{split}.csv"
        with path.open(encoding="utf-8-sig", newline="") as f:
            historical.extend(row["text"] for row in csv.DictReader(f))
        sources.append({"split": split, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    kept, dedup = deduplicate(rows, historical)
    selected, assignment = assign_splits(kept)
    write_jsonl(args.input / "annotation_reviews.jsonl", selected)
    write_jsonl(args.input / "replay_new_inputs.jsonl", [{"request_id": r["request_id"], "text": r["text"]} for r in selected])
    selected_ids = {r["request_id"] for r in selected}
    write_jsonl(args.input / "reserve_reviews.jsonl", [r for r in kept if r["request_id"] not in selected_ids])
    artifacts = json.loads(args.artifacts.read_text(encoding="utf-8"))
    taxonomy = [{"aspect": t["aspect"], "definition": t["text"],
                 "zh_name": ZH[t["aspect"]][0], "zh_description": ZH[t["aspect"]][1]}
                for t in artifacts["taxonomy"]["candidates"]]
    (args.input / "annotation_taxonomy.json").write_text(json.dumps(taxonomy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {"collection_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(), "historical_sources": sources,
                "official_test_opened": False, "deduplication": dedup, "assignment": assignment,
                "annotation_reviews_sha256": hashlib.sha256((args.input / "annotation_reviews.jsonl").read_bytes()).hexdigest(),
                "replay_new_inputs_sha256": hashlib.sha256((args.input / "replay_new_inputs.jsonl").read_bytes()).hexdigest(),
                "language_screening": "UK storefront; human English confirmation required before confirmed gold",
                "model_predictions_exposed": False, "labels_present": False,
                "input_representation": "Review body only; title and rating hidden from annotation to match inference input"}
    (args.input / "annotation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"deduplication": dedup, "assignment": assignment}, ensure_ascii=False))


if __name__ == "__main__":
    main()
