from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.experiments.taxonomy_methods import METHOD_IDS
from msc_project.experiments.taxonomy_protocol import (
    registered_folds,
    training_scope_id,
)
from msc_project.experiments.taxonomy_tuning import (
    fixed_registered_selection,
    select_registered_tuning_candidate,
)


def load_observations(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        raise ValueError("At least one validation summary is required.")
    rows = []
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or value.get("event") != "threshold_selected"
            or "tuning_observation" not in value
        ):
            raise ValueError(f"Not a tuning validation summary: {path}")
        observation = value["tuning_observation"]
        if not isinstance(observation, dict):
            raise ValueError(f"Malformed tuning observation: {path}")
        rows.append({**observation, "source_summary": str(path)})
    return pd.DataFrame.from_records(rows)


def write_selection(path: Path, selection: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite tuning selection: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Stale tuning temporary file: {temporary}")
    temporary.write_text(
        json.dumps(selection, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze one registered seen-validation tuning grid."
    )
    parser.add_argument("--method", choices=METHOD_IDS, required=True)
    parser.add_argument("--summary", type=Path, action="append", default=[])
    parser.add_argument(
        "--fixed",
        action="store_true",
        help="Freeze the sole registered recipe for a method with no tuning grid.",
    )
    parser.add_argument("--fold-id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.fixed:
        if args.summary:
            raise ValueError("--fixed cannot be combined with --summary.")
        if args.fold_id is not None:
            raise ValueError("--fixed cannot be combined with --fold-id.")
        selection = fixed_registered_selection(args.method)
        selection["training_scope_id"] = "global_fixed_recipe"
        selection["source_summaries"] = []
    else:
        if args.fold_id is None:
            raise ValueError("Nested tuning selection requires --fold-id.")
        matching_folds = [
            fold
            for level in ("L1", "L2", "L3", "L4")
            for fold in registered_folds(level)
            if fold.fold_id == args.fold_id
        ]
        if len(matching_folds) != 1:
            raise ValueError(f"Unknown registered fold: {args.fold_id!r}")
        fold = matching_folds[0]
        observations = load_observations(args.summary)
        selection = select_registered_tuning_candidate(
            observations,
            args.method,
            required_folds=(fold.fold_id,),
        )
        selection["training_scope_id"] = training_scope_id(fold)
        selection["source_summaries"] = observations["source_summary"].tolist()
    write_selection(args.output, selection)
    print(json.dumps(selection, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
