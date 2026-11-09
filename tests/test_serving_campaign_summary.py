"""CPU-only tests for aggregate privacy, timing scope and quality gating."""
import json

import pytest

from summarise_serving_campaign import summarise, telemetry_summary


def put(path, value, jsonl=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(json.dumps(v) for v in value) if jsonl else json.dumps(value), encoding='utf-8')


def fixture_campaign(root, quality=True, utc=True):
    put(root / 'plan.json', {'artifacts': 'C:/secret_machine/private.json', 'sessions': [
        {'name': 'hybrid', 'server': {'variant': 'hybrid', 'request_batch_size': 2, 'pair_batch_size': 8},
         'runs': [{'name': 'evaluation', 'quality_evaluation': quality}]}]})
    put(root / 'state.json', {'status': 'completed', 'current_run': 'hybrid__evaluation',
                              'server_dir': 'C:/secret_machine/private'})
    session = root / 'server_sessions/hybrid_20261109T000000Z'
    put(session / 'server_metadata.json', {'arguments': {'artifacts': 'C:/secret_machine/private'},
        'engine': {'model_id': 'Qwen/example', 'model_revision': 'abcdef', 'fold_id': 'a01', 'variant': 'hybrid',
                   'adapter_path': 'C:/secret_machine/private'}, 'source_sha256': {'server.py': '123'}})
    telem = [{'event': 'configuration', 'utc': '2026-11-09T00:00:00Z'}]
    for seconds, rss in [(1, 100), (6, 500), (20, 900)]:
        telem.append({'event': 'measurement', 'utc': f'2026-11-09T00:00:{seconds:02}Z',
                      'process': {'rss_bytes': rss, 'members': [{'pid': 999}]},
                      'gpu': {'temperature.gpu': 60 + seconds, 'memory.used': 7000}})
    put(session / 'telemetry.jsonl', telem, True)
    run = root / 'runs/hybrid__evaluation'
    put(run / 'config.json', {'mode': 'closed', 'concurrency': 2,
        **({'utc_start': '2026-11-09T00:00:05Z'} if utc else {})})
    row = {'request_id': 'secret_attempt', 'sample_id': 'secret_sample', 'success': True,
           'started_s': 0, 'scheduled_s': 0, 'finished_s': 2, 'latency_ms': 2000,
           'scheduled_latency_ms': 2000, 'schedule_lag_ms': 0, 'status_code': 200, 'length_bin': 1,
           'response': {'request_id': 'secret_attempt', 'text': 'SECRET REVIEW CONTENT', 'variant': 'hybrid',
                        'fold_id': 'a01', 'predictions': [{'aspect': 'held', 'sentiments': ['negative']}],
                        'request_batch_size': 2, 'timings_ms': {'queue_wait': 10, 'worker_batch': 1900}}}
    put(run / 'requests.jsonl', [row], True)
    put(run / 'summary.json', {'attempts': 1, 'wall_seconds_including_drain': 2})
    labels = [{'request_id': 'secret_sample', 'pairs': [{'aspect': 'held', 'sentiment': 'negative'}]}]
    taxonomy = {'canonical_order': ['held', 'seen'], 'heldout_aspects': ['held']}
    return labels, taxonomy


def test_only_aggregates_exported_and_telemetry_matches_run_window(tmp_path):
    labels, taxonomy = fixture_campaign(tmp_path / 'raw')
    output = tmp_path / 'public'
    result = summarise(tmp_path / 'raw', output, labels, taxonomy)
    row = result['runs'][0]
    assert row['queue_wait_successful_requests']['p50_ms'] == 10
    assert row['independent_attempted_review_count'] == 1
    assert row['telemetry']['measurement_count'] == 1
    assert row['telemetry']['metrics']['server_process_tree_rss_bytes']['max'] == 500
    assert row['quality']['successful_unique_quality']['heldout']['gold_pair_support'] == 1
    assert row['quality']['successful_unique_quality']['heldout']['pair']['micro_f1'] == 1
    for path in output.iterdir():
        text = path.read_text(encoding='utf-8-sig')
        for forbidden in ['secret_machine', 'secret_attempt', 'secret_sample', 'SECRET REVIEW CONTENT', '999']:
            assert forbidden not in text
    assert (output / 'quality.csv').exists()


def test_nonquality_run_never_joins_labels_and_unknown_timing_is_session_level(tmp_path):
    fixture_campaign(tmp_path / 'raw', quality=False, utc=False)
    result = summarise(tmp_path / 'raw', tmp_path / 'public')
    row = result['runs'][0]
    assert 'quality' not in row
    assert row['telemetry_scope'].startswith('Session-level only')
    assert row['telemetry']['measurement_count'] == 3


def test_explicit_quality_requires_labels_and_complete_count_matches(tmp_path):
    labels, taxonomy = fixture_campaign(tmp_path / 'raw')
    with pytest.raises(ValueError, match='requires'):
        summarise(tmp_path / 'raw', tmp_path / 'public')
    put(tmp_path / 'raw/runs/hybrid__evaluation/summary.json', {'attempts': 2})
    with pytest.raises(ValueError, match='attempt count'):
        summarise(tmp_path / 'raw', tmp_path / 'public', labels, taxonomy)


def test_counter_microseconds_are_distinct_from_sampled_instantaneous_flags():
    rows = [{'event': 'measurement', 'gpu': {
        'clocks_event_reasons.sw_thermal_slowdown': False,
        'clocks_event_reasons_counters.sw_thermal_slowdown': count,
    }} for count in [20_000_000, 21_500_000]]
    state = telemetry_summary(rows)['clock_limit_states']['sw_thermal_slowdown']
    assert state['active_sample_fraction'] == 0
    assert state['counter_unit'] == 'microseconds'
    assert state['cumulative_counter_delta_raw'] == 1_500_000
    assert state['cumulative_counter_delta_seconds'] == 1.5
