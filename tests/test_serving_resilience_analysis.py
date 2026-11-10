"""Synthetic CPU-only checks for preregistered recovery and drift interpretation."""
import json
from pathlib import Path

import pytest

from analyse_serving_resilience import (
    analyse, recover_from_burst, time_windows, LATENCY_MARGIN_MULTIPLIER,
    BASELINE_MIN_SUCCESS, RECOVERY_CONSECUTIVE_REQUESTS, QUEUE_MIN_BAND_MS,
)


def row(index, scheduled, phase='normal', success=True, latency=1000, queue=1, length_bin=0, sample='sample_secret'):
    return {'request_id': f'secret_attempt_{index}', 'sample_id': sample, 'attempt_index': index,
            'scheduled_s': scheduled, 'started_s': scheduled, 'finished_s': scheduled + latency / 1000,
            'latency_ms': latency, 'scheduled_latency_ms': latency, 'schedule_lag_ms': 0,
            'success': success, 'status_code': 200 if success else 429,
            'error': None if success else 'http_429', 'phase': phase, 'length_bin': length_bin,
            'input_chars': 100 if length_bin == 0 else 1000,
            'response': {'variant': 'hybrid', 'fold_id': 'a01', 'timings_ms': {'queue_wait': queue},
                         'predictions': [{'aspect': 'held', 'sentiments': ['negative']}],
                         'text': 'DO NOT EXPORT REVIEW'}}


PHASES = [
    {'name': 'normal', 'scheduled_start_s': 0, 'scheduled_end_s': 60, 'rate': 0.5},
    {'name': 'burst', 'scheduled_start_s': 60, 'scheduled_end_s': 90, 'rate': 2},
    {'name': 'recovery', 'scheduled_start_s': 90, 'scheduled_end_s': 270, 'rate': 0.5},
]


def test_three_request_recovery_with_failure_reset_and_low_sample_limits():
    normal = [row(i, i * 2) for i in range(10)]
    recovery = [row(20, 90, 'recovery', latency=5000, queue=4000),
                row(21, 95, 'recovery'), row(22, 100, 'recovery', success=False),
                row(23, 105, 'recovery'), row(24, 110, 'recovery'), row(25, 115, 'recovery')]
    got = recover_from_burst(normal + recovery, PHASES, [])
    assert got['status'] == 'confirmed_under_declared_request_rule'
    assert got['latency_band_ms'] == pytest.approx(1100)
    assert got['queue_band_ms'] == 100
    assert got['first_request_in_band']['scheduled_arrival_after_burst_s'] == 5
    assert got['confirmation']['first_request_scheduled_after_burst_s'] == 15
    assert got['confirmation']['confirmed_by_completion_after_burst_s'] == 26
    short = recover_from_burst(normal + recovery[-2:], PHASES, [])
    assert short['status'] == 'not_confirmed_insufficient_recovery_requests'
    fallback = recover_from_burst(normal[:2] + recovery, PHASES, normal)
    assert fallback['baseline_source'].endswith('fallback')
    insufficient = recover_from_burst(normal[:2] + recovery, PHASES, normal[:2])
    assert insufficient['confirmation'] is None
    assert 'insufficient_baseline' in insufficient['status']


def test_drift_signal_can_coexist_with_perfect_unique_quality_and_failed_attempts():
    reference = [row(i, i) for i in range(10)]
    current = [row(i, i, length_bin=9) for i in range(10)]
    current.append(row(10, 10, success=False, length_bin=9))
    labels = [{'request_id': 'sample_secret', 'pairs': [{'aspect': 'held', 'sentiment': 'negative'}]}]
    taxonomy = {'canonical_order': ['held', 'seen'], 'heldout_aspects': ['held']}
    windows = time_windows(current, 60, 60, labels=labels, taxonomy=taxonomy, reference=reference)
    window = windows[0]
    assert window['input_length_js_bits'] == 1
    assert window['input_length_shift_signal']
    assert window['success_rate'] == pytest.approx(10 / 11)
    assert window['unique_attempted_review_count'] == 1
    assert window['delayed_label_diagnostic']['successful_unique_quality']['overall']['pair']['micro_f1'] == 1
    assert window['minimum_sample_warning']


def test_planned_baseline_does_not_substitute_for_actual_successes_or_invent_fallback():
    normal = [row(i, i * 2, success=i < 9) for i in range(20)]
    recovery = [row(20 + i, 90 + i * 2, 'recovery') for i in range(3)]
    result = recover_from_burst(normal + recovery, PHASES, [])
    assert result['status'] == 'not_evaluable_insufficient_baseline_requests'
    assert result['baseline_source'] == 'no_baseline'
    assert result['normal_phase_successful_requests'] == 9
    assert result['baseline_successful_requests'] == 0
    assert result['confirmation'] is None
    assert result['first_request_in_band'] is None
    assert result['latency_band_ms'] is None
    assert result['phases'][0]['attempts'] == 20
    assert result['phases'][0]['failed'] == 11


def test_no_normal_phase_and_no_external_reference_has_no_baseline():
    result = recover_from_burst([], PHASES, [])
    assert result['baseline_source'] == 'no_baseline'
    assert result['status'].startswith('not_evaluable_')
    assert result['normal_phase_successful_requests'] == 0


def test_rules_are_frozen_in_tracked_configuration():
    path = Path(__file__).resolve().parents[1] / 'configs/experiments/serving_resilience_analysis_20261109.json'
    rules = json.loads(path.read_text(encoding='utf-8'))['recovery']
    assert rules['baseline_min_successful_requests'] == BASELINE_MIN_SUCCESS
    assert rules['latency_margin_multiplier'] == LATENCY_MARGIN_MULTIPLIER
    assert rules['queue_band_floor_ms'] == QUEUE_MIN_BAND_MS
    assert rules['consecutive_successful_requests_required'] == RECOVERY_CONSECUTIVE_REQUESTS


def test_completed_shift_export_has_no_private_identifiers(tmp_path):
    def put(path, value, lines=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(json.dumps(v) for v in value) if lines else json.dumps(value), encoding='utf-8')
    primary, campaign, output = tmp_path / 'primary', tmp_path / 'resilience', tmp_path / 'public'
    put(primary / 'plan.json', {'sessions': [{'name': 'hybrid', 'server': {'variant': 'hybrid'}}]})
    for repeat in range(1, 4):
        directory = primary / f'runs/hybrid__closed_c1_r{repeat}'
        records = [row(i, i) for i in range(10)]
        put(directory / 'requests.jsonl', records, True)
        put(directory / 'summary.json', {'attempts': len(records), 'interrupted': False})
    put(campaign / 'plan.json', {'sessions': [{'name': 'h', 'server': {'variant': 'hybrid'},
        'runs': [{'name': 'input_length_shift'}]}]})
    put(campaign / 'state.json', {'status': 'completed'})
    directory = campaign / 'runs/h__input_length_shift'
    phases = [{'name': 'normal_before', 'scheduled_start_s': 0, 'scheduled_end_s': 120, 'rate': 0.5},
              {'name': 'long_input', 'scheduled_start_s': 120, 'scheduled_end_s': 240, 'rate': 0.5},
              {'name': 'normal_after', 'scheduled_start_s': 240, 'scheduled_end_s': 360, 'rate': 0.5}]
    records = [row(i, phase['scheduled_start_s'] + i, phase=phase['name'],
                   length_bin=9 if phase['name'] == 'long_input' else 0)
               for phase in phases for i in range(5)]
    put(directory / 'requests.jsonl', records, True)
    put(directory / 'summary.json', {'attempts': 15, 'interrupted': False, 'wall_seconds_including_drain': 360})
    put(directory / 'config.json', {'phases': phases})
    labels = [{'request_id': 'sample_secret', 'pairs': [{'aspect': 'held', 'sentiment': 'negative'}]}]
    taxonomy = {'canonical_order': ['held', 'seen'], 'heldout_aspects': ['held']}
    result = analyse(campaign, primary, output, labels, taxonomy)
    assert len(result['input_shift'][0]['windows']) == 6
    assert result['input_shift'][0]['windows'][2]['input_length_shift_signal']
    for path in output.iterdir():
        text = path.read_text(encoding='utf-8-sig')
        for forbidden in ['sample_secret', 'secret_attempt_', 'DO NOT EXPORT REVIEW', str(tmp_path)]:
            assert forbidden not in text
