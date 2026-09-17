"""Render reported aggregate results without opening labels or running models."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

OFFICIAL = Path("docs/thesis_figure_data/taxonomy_final_test_v1/official_confirmatory_results.csv")
POLICY = Path("docs/results/training_policy_completed/aggregate_metrics.csv")
METHODS = {"fewshot": "Frozen Qwen few-shot", "qlora": "QLoRA", "composition": "Fixed composition"}


def _records(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path.name}: empty result file")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{path.name}: CSV row width differs from its header")
    return rows


def _number(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Result table contains a non-finite value")
    return number


def render_results(root: Path) -> str:
    """Validate the expected result scopes and render two separately labelled tables."""
    official = _records(
        root / OFFICIAL,
        {
            "hypothesis",
            "challenger_point",
            "reference_point",
            "point_difference",
            "ci_lower",
            "ci_upper",
            "endpoint",
            "folds",
            "comparator",
            "candidate",
        },
    )
    comparisons = {row["hypothesis"]: row for row in official}
    if len(official) != 2 or set(comparisons) != {"H1", "H2"}:
        raise ValueError("Expected exactly the two registered official comparisons")
    expected = {"H1": "frozen_qwen_few_shot", "H2": "qwen_candidate_pair_qlora"}
    for hypothesis, row in comparisons.items():
        if (
            row["endpoint"] != "L2_D_aspect_balanced_mean_heldout_pair_micro_f1"
            or int(row["folds"]) != 12
            or row["comparator"] != expected[hypothesis]
            or row["candidate"] != "frozen_qwen_few_shot_stage1__qlora_stage2"
        ):
            raise ValueError("Official comparison scope differs from the registered endpoint")
        delta = _number(row["challenger_point"]) - _number(row["reference_point"])
        if any(
            not 0 <= _number(row[field]) <= 1 for field in ("challenger_point", "reference_point")
        ):
            raise ValueError("Official F1 must be between 0 and 1")
        if not math.isclose(delta, _number(row["point_difference"]), abs_tol=1e-12):
            raise ValueError("Reported contrast disagrees with its reported point estimates")
        if _number(row["ci_lower"]) > _number(row["ci_upper"]):
            raise ValueError("Reversed confidence interval")
    if comparisons["H1"]["challenger_point"] != comparisons["H2"]["challenger_point"]:
        raise ValueError("The two comparisons must use the same primary system score")
    lines = [
        "# Reported results",
        "",
        "## Locked official test: review filtering, names plus definitions",
        "",
        "Endpoint: mean held-out aspect-sentiment pair micro-F1 across 12 aspects.",
        "",
        "| System | F1 |",
        "|---|---:|",
        f"| Fixed composition | {_number(comparisons['H1']['challenger_point']):.4f} |",
        f"| Frozen Qwen few-shot | {_number(comparisons['H1']['reference_point']):.4f} |",
        f"| QLoRA | {_number(comparisons['H2']['reference_point']):.4f} |",
        "",
        "| Comparison | Difference | Paired 95% interval |",
        "|---|---:|---|",
    ]
    for hypothesis in ("H1", "H2"):
        row = comparisons[hypothesis]
        name = "few-shot" if hypothesis == "H1" else "QLoRA (gated after H1)"
        lines.append(
            f"| {hypothesis}: composition minus {name} | "
            f"{_number(row['point_difference']):+.4f} | "
            f"[{_number(row['ci_lower']):+.4f}, {_number(row['ci_upper']):+.4f}] |"
        )
    policy = _records(root / POLICY, {"method", "policy", "condition", "folds", "heldout_pair_f1"})
    scopes = [(row["method"], row["policy"], row["condition"]) for row in policy]
    policies = ("review_filtered", "label_masked_all_reviews")
    expected_scopes = {(m, p, c) for m in METHODS for p in policies for c in ("N", "D")}
    if len(scopes) != len(expected_scopes) or set(scopes) != expected_scopes:
        raise ValueError("Expected all 12 method/policy/condition aggregate rows")
    values = {}
    for row in policy:
        value = _number(row["heldout_pair_f1"])
        if int(row["folds"]) != 12 or not 0 <= value <= 1:
            raise ValueError("Invalid fold count or F1 in policy result")
        values[row["method"], row["policy"], row["condition"]] = value
    lines += [
        "",
        "## Later validation-only robustness study: names plus definitions",
        "",
        "These are validation estimates, not a second official-test confirmation.",
        "",
        "| System | Reviews removed | Reviews retained, target supervision masked |",
        "|---|---:|---:|",
    ]
    for method, name in METHODS.items():
        lines.append(
            f"| {name} | {values[method, policies[0], 'D']:.4f} | "
            f"{values[method, policies[1], 'D']:.4f} |"
        )
    lines += [
        "",
        "Sources: `" + OFFICIAL.as_posix() + "` and `" + POLICY.as_posix() + "`.",
        "This command formats saved aggregate evidence. It does not recompute inference or intervals.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path.cwd(), help="Checkout containing docs/"
    )
    parser.add_argument(
        "--output", type=Path, help="New Markdown file; existing files are protected"
    )
    args = parser.parse_args(argv)
    try:
        text = render_results(args.repo_root)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(text)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
