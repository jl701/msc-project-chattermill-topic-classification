"""A partial long experiment must not be accepted as complete evidence."""
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    'serving_study_runner', Path(__file__).resolve().parents[1] / 'scripts/run_serving_study.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_reject_interrupted_and_incomplete_closed_runs():
    run = {'mode': 'closed', 'requests': 957}
    with pytest.raises(RuntimeError, match='interrupted'):
        runner.validate_completed_run({'interrupted': True, 'attempts': 957}, run)
    with pytest.raises(RuntimeError, match='request count'):
        runner.validate_completed_run({'interrupted': False, 'attempts': 956}, run)
    runner.validate_completed_run({'interrupted': False, 'attempts': 957}, run)


def test_reject_short_soak_observation():
    run = {'mode': 'open', 'duration': 3600}
    with pytest.raises(RuntimeError, match='observation period'):
        runner.validate_completed_run({'interrupted': False, 'wall_seconds_including_drain': 3599}, run)
    runner.validate_completed_run({'interrupted': False, 'wall_seconds_including_drain': 3601}, run)
