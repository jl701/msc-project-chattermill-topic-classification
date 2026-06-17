from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.data.fabsa import SPLITS, default_data_dir, load_split
from msc_project.llm.qwen_format import aspect_taxonomy, build_messages


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Qwen SFT chat-format JSONL files from FABSA.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "qwen_sft")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    train_df = load_split(args.data_dir, "train")
    aspects = aspect_taxonomy(train_df)

    metadata = {
        "task": "FABSA closed-topic aspect-sentiment extraction",
        "aspects": aspects,
        "output_format": "JSON array with aspect and sentiment keys",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for split in SPLITS:
        frame = load_split(args.data_dir, split)
        if args.limit is not None:
            frame = frame.head(args.limit).copy()

        rows = []
        for _, row in frame.iterrows():
            rows.append(
                {
                    "id": str(row["id"]),
                    "messages": build_messages(
                        text=row["text"],
                        aspects=aspects,
                        gold_pair_labels=row["pair_labels"],
                    ),
                    "pair_labels": row["pair_labels"],
                }
            )

        write_jsonl(rows, args.output_dir / f"{split}.jsonl")
        print(f"Wrote {len(rows)} rows to {args.output_dir / f'{split}.jsonl'}")

    print(f"Wrote metadata to {args.output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()

