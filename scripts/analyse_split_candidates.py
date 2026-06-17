from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.data.fabsa import default_data_dir
from msc_project.data.splits import (
    build_heldout_aspect_split,
    choose_org_split_candidates,
    load_all_fabsa,
    org_summary,
    aspect_summary,
    split_manifest,
)


def aspect_combo_rows(frame: pd.DataFrame, combo: tuple[str, ...]) -> dict[str, object]:
    heldout = set(combo)
    splits = build_heldout_aspect_split(frame, heldout, strategy="label_masked")
    validation = splits["validation"]
    test = splits["test"]
    train = splits["train"]

    return {
        "heldout_aspects": " || ".join(combo),
        "train_rows_label_masked": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "validation_min_aspect_rows": int(
            min(
                validation["supervision_aspect_labels"].apply(lambda labels: aspect in labels).sum()
                for aspect in combo
            )
        ),
        "test_min_aspect_rows": int(
            min(
                test["supervision_aspect_labels"].apply(lambda labels: aspect in labels).sum()
                for aspect in combo
            )
        ),
        "validation_orgs": int(validation["org_index"].nunique()),
        "test_orgs": int(test["org_index"].nunique()),
    }


def rank_aspect_combinations(frame: pd.DataFrame, combo_size: int, top_k: int) -> pd.DataFrame:
    rows = []
    aspects = aspect_summary(frame)["aspect"].tolist()
    for combo in combinations(aspects, combo_size):
        row = aspect_combo_rows(frame, combo)
        target_validation = 0.20 * len(frame[frame["original_split"] == "validation"])
        target_test = 0.20 * len(frame[frame["original_split"] == "test"])
        row["score"] = (
            abs(row["validation_rows"] - target_validation) / max(1, target_validation)
            + abs(row["test_rows"] - target_test) / max(1, target_test)
            - 0.03 * row["validation_orgs"]
            - 0.03 * row["test_orgs"]
            - 0.01 * min(row["validation_min_aspect_rows"], 50)
            - 0.01 * min(row["test_min_aspect_rows"], 50)
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("score").head(top_k)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse FABSA candidates for held-out organisation/aspect splits.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "split_analysis")
    parser.add_argument("--aspect-combo-size", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    frame = load_all_fabsa(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    orgs = org_summary(frame)
    aspects = aspect_summary(frame)
    org_candidates = choose_org_split_candidates(frame, top_k=args.top_k)
    org_candidate_rows = [candidate.__dict__ for candidate in org_candidates]
    aspect_candidates = rank_aspect_combinations(frame, args.aspect_combo_size, args.top_k)

    orgs.to_csv(args.output_dir / "organisation_summary.csv", index=False)
    aspects.to_csv(args.output_dir / "aspect_summary.csv", index=False)
    pd.DataFrame(org_candidate_rows).to_csv(args.output_dir / "heldout_org_candidates.csv", index=False)
    aspect_candidates.to_csv(args.output_dir / "heldout_aspect_candidates.csv", index=False)

    summary = {
        "rows": int(len(frame)),
        "organisations": int(frame["org_index"].nunique()),
        "aspects": int(len(aspects)),
        "top_heldout_org_candidates": org_candidate_rows[:5],
        "top_heldout_aspect_candidates": aspect_candidates.head(5).to_dict(orient="records"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved split analysis to {args.output_dir}")


if __name__ == "__main__":
    main()

