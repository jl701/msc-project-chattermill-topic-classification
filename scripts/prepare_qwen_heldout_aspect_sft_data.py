from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.data.fabsa import default_data_dir
from msc_project.data.splits import DEFAULT_HELDOUT_ASPECTS, build_heldout_aspect_split, load_all_fabsa
from msc_project.llm.qwen_format import build_candidate_messages


def write_jsonl(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def limit_rows(frame, limit: int | None):
    if limit is None:
        return frame
    return frame.head(limit).copy()


def flatten(rows):
    values = []
    for row in rows:
        values.extend(row)
    return values


def build_rows(frame, candidate_aspects: list[str], prompt_variant: str, limit: int | None) -> list[dict[str, object]]:
    rows = []
    frame = limit_rows(frame, limit)
    for _, row in frame.iterrows():
        rows.append(
            {
                "id": str(row["id"]),
                "row_uid": str(row.get("row_uid", "")),
                "original_split": str(row.get("original_split", "")),
                "candidate_aspects": candidate_aspects,
                "messages": build_candidate_messages(
                    text=row["text"],
                    aspects=candidate_aspects,
                    prompt_variant=prompt_variant,
                    gold_pair_labels=row["supervision_pair_labels"],
                ),
                "pair_labels": row["supervision_pair_labels"],
            }
        )
    return rows


def write_strategy_data(frame, strategy: str, args) -> None:
    splits = build_heldout_aspect_split(
        frame,
        DEFAULT_HELDOUT_ASPECTS,
        strategy=strategy,
        eval_label_scope="heldout",
    )
    train_aspects = sorted(set(flatten(splits["train"]["supervision_aspect_labels"])))
    output_dir = args.output_dir / strategy
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "task": "FABSA held-out-aspect candidate-label SFT",
        "strategy": strategy,
        "eval_label_scope": "heldout",
        "prompt_variant": args.prompt_variant,
        "heldout_aspects": DEFAULT_HELDOUT_ASPECTS,
        "train_candidate_aspects": train_aspects,
        "validation_candidate_aspects": DEFAULT_HELDOUT_ASPECTS,
        "test_candidate_aspects": DEFAULT_HELDOUT_ASPECTS,
        "notes": [
            "Training rows use seen-aspect supervision only.",
            "Validation and test rows use held-out aspect labels only.",
            "Candidate labels are provided in the prompt and outputs should use canonical IDs/labels.",
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    split_candidates = {
        "train": train_aspects,
        "validation": DEFAULT_HELDOUT_ASPECTS,
        "test": DEFAULT_HELDOUT_ASPECTS,
    }
    for split_name, split_frame in splits.items():
        rows = build_rows(split_frame, split_candidates[split_name], args.prompt_variant, args.limit)
        write_jsonl(rows, output_dir / f"{split_name}.jsonl")
        print(f"[{strategy}] wrote {len(rows)} rows to {output_dir / f'{split_name}.jsonl'}")

    print(f"[{strategy}] wrote metadata to {output_dir / 'metadata.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Qwen SFT JSONL files for held-out-aspect FABSA.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "qwen_heldout_aspect_sft")
    parser.add_argument("--strategy", choices=["label_masked", "example_filtered", "both"], default="both")
    parser.add_argument("--prompt-variant", default="indexed")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    frame = load_all_fabsa(args.data_dir)
    strategies = ["label_masked", "example_filtered"] if args.strategy == "both" else [args.strategy]
    for strategy in strategies:
        write_strategy_data(frame, strategy, args)


if __name__ == "__main__":
    main()

