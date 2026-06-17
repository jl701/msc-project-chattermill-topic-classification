from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.data.fabsa import SPLITS, default_data_dir, load_splits, unique_labels


def count_nested(rows: list[list[str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for labels in rows:
        counts.update(labels)
    return counts


def split_summary(frame) -> dict[str, object]:
    pair_counts = count_nested(frame["pair_labels"].tolist())
    aspect_counts = count_nested(frame["aspect_labels"].tolist())
    labels_per_row = Counter(len(labels) for labels in frame["pair_labels"])

    return {
        "rows": int(len(frame)),
        "organisations": int(frame["org_index"].nunique()),
        "aspects": int(len(aspect_counts)),
        "pair_labels": int(len(pair_counts)),
        "labels_per_row": {str(k): int(v) for k, v in sorted(labels_per_row.items())},
        "top_pair_labels": dict(pair_counts.most_common(20)),
        "top_aspects": dict(aspect_counts.most_common(20)),
        "data_sources": frame["data_source"].value_counts().astype(int).to_dict(),
        "industries": frame["industry"].value_counts().astype(int).to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore the local FABSA export.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "fabsa_exploration")
    args = parser.parse_args()

    frames = load_splits(args.data_dir, SPLITS)
    summary = {
        "data_dir": str(args.data_dir),
        "splits": {split: split_summary(frame) for split, frame in frames.items()},
        "all_pair_labels": unique_labels(frames.values(), "pair_labels"),
        "all_aspects": unique_labels(frames.values(), "aspect_labels"),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary["splits"], indent=2))
    print(f"Saved exploration summary to {output_path}")


if __name__ == "__main__":
    main()

