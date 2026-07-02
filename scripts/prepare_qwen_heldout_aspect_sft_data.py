from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.data.fabsa import default_data_dir
from msc_project.data.splits import DEFAULT_HELDOUT_ASPECTS, all_aspects, build_heldout_aspect_split, load_all_fabsa
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


def sft_row(row, candidate_aspects: list[str], prompt_variant: str, pair_labels: list[str], id_suffix: str = "") -> dict[str, object]:
    return {
        "id": f"{row['id']}{id_suffix}",
        "row_uid": f"{row.get('row_uid', '')}{id_suffix}",
        "original_split": str(row.get("original_split", "")),
        "candidate_aspects": candidate_aspects,
        "messages": build_candidate_messages(
            text=row["text"],
            aspects=candidate_aspects,
            prompt_variant=prompt_variant,
            gold_pair_labels=pair_labels,
        ),
        "pair_labels": pair_labels,
    }


def build_rows(frame, candidate_aspects: list[str], prompt_variant: str, limit: int | None) -> list[dict[str, object]]:
    rows = []
    frame = limit_rows(frame, limit)
    for _, row in frame.iterrows():
        rows.append(sft_row(row, candidate_aspects, prompt_variant, list(row["supervision_pair_labels"])))
    return rows


def aspect_from_pair_label(pair_label: str) -> str:
    return pair_label.rsplit(" | ", maxsplit=1)[0]


def slug(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")


def build_singleton_train_rows(
    frame,
    candidate_aspects: list[str],
    prompt_variant: str,
    limit: int | None,
    negative_ratio: int,
    seed: int,
) -> list[dict[str, object]]:
    rows = []
    rng = random.Random(seed)
    frame = limit_rows(frame, limit)
    for _, row in frame.iterrows():
        pair_labels = list(row["supervision_pair_labels"])
        labels_by_aspect = {
            aspect: [label for label in pair_labels if aspect_from_pair_label(label) == aspect]
            for aspect in candidate_aspects
        }
        positive_aspects = [aspect for aspect, labels in labels_by_aspect.items() if labels]
        for aspect in positive_aspects:
            rows.append(
                sft_row(
                    row,
                    [aspect],
                    prompt_variant,
                    labels_by_aspect[aspect],
                    id_suffix=f"::pos::{slug(aspect)}",
                )
            )

        negative_pool = [aspect for aspect in candidate_aspects if aspect not in set(positive_aspects)]
        negative_count = min(len(negative_pool), max(1, len(positive_aspects)) * negative_ratio)
        for aspect in rng.sample(negative_pool, negative_count):
            rows.append(
                sft_row(
                    row,
                    [aspect],
                    prompt_variant,
                    [],
                    id_suffix=f"::neg::{slug(aspect)}",
                )
            )

    rng.shuffle(rows)
    return rows


def selected_heldout_aspects(frame, requested_aspects: list[str]) -> list[str]:
    if not requested_aspects:
        return list(DEFAULT_HELDOUT_ASPECTS)
    known_aspects = set(all_aspects(frame))
    unknown = sorted(set(requested_aspects) - known_aspects)
    if unknown:
        raise ValueError(f"Unknown held-out aspects: {unknown}")
    return requested_aspects


def write_strategy_data(frame, strategy: str, heldout_aspects: list[str], args) -> None:
    splits = build_heldout_aspect_split(
        frame,
        heldout_aspects,
        strategy=strategy,
        eval_label_scope="heldout",
        eval_row_scope=args.eval_row_scope,
    )
    train_aspects = sorted(set(flatten(splits["train"]["supervision_aspect_labels"])))
    output_dir = args.output_dir / strategy
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "task": "FABSA held-out-aspect candidate-label SFT",
        "strategy": strategy,
        "eval_label_scope": "heldout",
        "eval_row_scope": args.eval_row_scope,
        "train_candidate_mode": args.train_candidate_mode,
        "singleton_negative_ratio": args.singleton_negative_ratio,
        "seed": args.seed,
        "prompt_variant": args.prompt_variant,
        "heldout_aspects": heldout_aspects,
        "train_candidate_aspects": train_aspects,
        "validation_candidate_aspects": heldout_aspects,
        "test_candidate_aspects": heldout_aspects,
        "notes": [
            "Training rows use seen-aspect supervision only.",
            "Validation and test rows use held-out aspect labels only.",
            "Validation and test row scope is controlled by eval_row_scope.",
            "Candidate labels are provided in the prompt and outputs should use canonical IDs/labels.",
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for split_name, split_frame in splits.items():
        if split_name == "train" and args.train_candidate_mode == "singleton":
            rows = build_singleton_train_rows(
                split_frame,
                train_aspects,
                args.prompt_variant,
                args.limit,
                args.singleton_negative_ratio,
                args.seed,
            )
        else:
            split_candidates = train_aspects if split_name == "train" else heldout_aspects
            rows = build_rows(split_frame, split_candidates, args.prompt_variant, args.limit)
        write_jsonl(rows, output_dir / f"{split_name}.jsonl")
        print(f"[{strategy}] wrote {len(rows)} rows to {output_dir / f'{split_name}.jsonl'}")

    print(f"[{strategy}] wrote metadata to {output_dir / 'metadata.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Qwen SFT JSONL files for held-out-aspect FABSA.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "qwen_heldout_aspect_sft")
    parser.add_argument("--strategy", choices=["label_masked", "example_filtered", "both"], default="both")
    parser.add_argument("--prompt-variant", default="indexed")
    parser.add_argument("--heldout-aspect", action="append", default=[])
    parser.add_argument("--eval-row-scope", choices=["containing_heldout", "all"], default="containing_heldout")
    parser.add_argument("--train-candidate-mode", choices=["grouped", "singleton"], default="grouped")
    parser.add_argument("--singleton-negative-ratio", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.singleton_negative_ratio < 0:
        raise ValueError("--singleton-negative-ratio must be non-negative.")

    frame = load_all_fabsa(args.data_dir)
    heldout_aspects = selected_heldout_aspects(frame, args.heldout_aspect)
    strategies = ["label_masked", "example_filtered"] if args.strategy == "both" else [args.strategy]
    for strategy in strategies:
        write_strategy_data(frame, strategy, heldout_aspects, args)


if __name__ == "__main__":
    main()
