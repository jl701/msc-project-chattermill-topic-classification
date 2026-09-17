"""User-facing examples and reporting fail closed on malformed evidence."""

import csv
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from msc_project.demo import main as demo_main
from msc_project.demo import run_demo
from msc_project.reporting import OFFICIAL, POLICY, main, render_results

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def evidence(tmp_path):
    for relative in (OFFICIAL, POLICY):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return tmp_path


def rewrite(path, transform):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns, rows = reader.fieldnames, list(reader)
    transform(rows)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_demo_uses_real_policy_and_rejects_absent_candidate():
    output = run_demo()
    assert output["model_loaded"] is False
    assert output["example_type"] == "synthetic_mechanics_only"
    assert output["policies"]["review_filtered"]["retained_reviews"] == 2
    assert output["policies"]["label_masked_all_reviews"]["retained_reviews"] == 3
    assert all(v["heldout_training_targets"] == 0 for v in output["policies"].values())
    assert len(output["examples"][0]["predicted_pairs"]) == 2
    assert output["examples"][1]["predicted_pairs"] == ["Staff support | positive"]


def test_aggregate_tables_have_distinct_evidence_roles(evidence):
    text = render_results(evidence)
    assert "Locked official test" in text
    assert "Later validation-only robustness" in text
    for number in ("0.5075", "0.4954", "0.4864", "0.5261", "0.5004"):
        assert number in text


@pytest.mark.parametrize(
    "column,value",
    [
        ("folds", "11"),
        ("candidate", "another winner"),
        ("endpoint", "overall F1"),
        ("reference_point", "nan"),
        ("point_difference", "0.5"),
        ("ci_lower", "1"),
    ],
)
def test_report_rejects_wrong_official_scope_or_numbers(evidence, column, value):
    rewrite(evidence / OFFICIAL, lambda rows: rows[0].update({column: value}))
    with pytest.raises(ValueError):
        render_results(evidence)


@pytest.mark.parametrize(
    "column,value",
    [("condition", "R"), ("folds", "11"), ("heldout_pair_f1", "inf"), ("heldout_pair_f1", "2")],
)
def test_report_rejects_wrong_policy_scope_or_numbers(evidence, column, value):
    rewrite(evidence / POLICY, lambda rows: rows[0].update({column: value}))
    with pytest.raises(ValueError):
        render_results(evidence)


@pytest.mark.parametrize("relative", [OFFICIAL, POLICY])
def test_report_rejects_duplicate_scope(evidence, relative):
    rewrite(evidence / relative, lambda rows: rows.append(rows[0].copy()))
    with pytest.raises(ValueError):
        render_results(evidence)


def test_output_is_not_overwritten(evidence, capsys):
    target = evidence / "report.md"
    assert main(["--repo-root", str(evidence), "--output", str(target)]) == 0
    original = target.read_bytes()
    with pytest.raises(SystemExit) as exc:
        main(["--repo-root", str(evidence), "--output", str(target)])
    assert exc.value.code == 2
    assert target.read_bytes() == original
    demo = evidence / "demo.json"
    assert demo_main(["--output", str(demo)]) == 0
    original = demo.read_bytes()
    with pytest.raises(SystemExit):
        demo_main(["--output", str(demo)])
    assert demo.read_bytes() == original


def test_missing_evidence_returns_cli_error(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["--repo-root", str(tmp_path)])
    assert exc.value.code == 2


def test_malformed_csv_row_fails_cleanly(evidence):
    with (evidence / POLICY).open("a", encoding="utf-8") as stream:
        stream.write("incomplete,row\n")
    with pytest.raises(ValueError, match="row width"):
        render_results(evidence)


def test_module_commands_work_outside_checkout(tmp_path):
    # sys.path is supplied explicitly here so source-only pytest also works.
    for module, arguments in [("demo", []), ("reporting", ["--repo-root", str(ROOT)])]:
        code = (
            f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r}); "
            f"from msc_project.{module} import main; main({arguments!r})"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=tmp_path, check=True, capture_output=True, text=True
        )
        assert result.stdout
