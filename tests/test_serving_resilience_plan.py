"""CPU-only synthetic pre-registration checks; no network or GPU execution."""
import json

import pytest

from create_serving_resilience_plan import create_plan


def put(path, value, jsonl=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(json.dumps(v) for v in value) if jsonl else json.dumps(value), encoding='utf-8')


def fixture(root, completed=True):
    put(root / 'primary/state.json', {'status': 'completed' if completed else 'running'})
    server = {'variant': 'hybrid', 'prefix_cache': 'on', 'pair_batch_size': 4, 'request_batch_size': 1}
    put(root / 'primary/plan.json', {'artifacts': 'artifacts.json', 'sessions': [
        {'name': 'h', 'server': server, 'runs': [
            {'name': f'closed_c1_r{i}', 'mode': 'closed', 'concurrency': 1} for i in range(1, 4)]}]})
    for i, capacity in enumerate([0.4, 0.7, 0.5], 1):
        put(root / f'primary/runs/h__closed_c1_r{i}/summary.json', {
            'successful_reviews_per_second': capacity, 'success_rate': 1, 'attempts': 30})
    replay = [{'request_id': str(i), 'text': 'x' * i} for i in range(1, 21)]
    metadata = [{'request_id': str(i), 'character_count': i, 'length_decile': (i - 1) // 2}
                for i in range(1, 21)]
    put(root / 'replay.jsonl', replay, True)
    put(root / 'metadata.jsonl', metadata, True)
    put(root / 'artifacts.json', {'placeholder': True})
    return server


def test_capacity_phase_durations_fixed_hybrid_and_label_free_long_selection(tmp_path):
    server = fixture(tmp_path)
    plan = create_plan('primary', 'plans', 'replay.jsonl', 'metadata.jsonl', root=tmp_path)
    assert plan['reference_capacity_reviews_per_second'] == 0.5
    assert plan['sessions'][0]['server'] == server
    assert plan['planned_arrival_seconds_excluding_warmup_and_drain'] == 4770
    runs = plan['sessions'][0]['runs']
    assert len(runs) == 6
    assert runs[1]['duration'] == 3600
    assert runs[1]['rate'] == pytest.approx(0.35)
    assert all(not run['quality_evaluation'] for run in runs)
    burst = json.loads((tmp_path / 'plans/burst_recovery_r1.json').read_text())
    assert [p['duration'] for p in burst['phases']] == [60, 30, 180]
    assert [p['rate'] for p in burst['phases']] == [0.25, 1, 0.25]
    shift = json.loads((tmp_path / 'plans/input_length_shift.json').read_text())
    assert [p['rate'] for p in shift['phases']] == [0.25] * 3
    assert not any(p.get('drain') for p in shift['phases'])
    selected = [json.loads(line) for line in (tmp_path / 'plans/replay_long_development.jsonl').read_text().splitlines()]
    assert [len(row['text']) for row in selected] == [20, 19]
    assert plan['long_input_selection']['evaluation_reviews_used'] is False
    assert not any(str(tmp_path) in p.read_text() for p in (tmp_path / 'plans').iterdir())


def test_not_completed_or_existing_output_cannot_rewrite_history(tmp_path):
    fixture(tmp_path, completed=False)
    with pytest.raises(ValueError, match='completed'):
        create_plan('primary', 'plans', 'replay.jsonl', 'metadata.jsonl', root=tmp_path)
    assert not (tmp_path / 'plans').exists()
    put(tmp_path / 'primary/state.json', {'status': 'completed'})
    create_plan('primary', 'plans', 'replay.jsonl', 'metadata.jsonl', root=tmp_path)
    old = (tmp_path / 'plans/plan.json').read_bytes()
    with pytest.raises(FileExistsError, match='overwrite'):
        create_plan('primary', 'plans', 'replay.jsonl', 'metadata.jsonl', root=tmp_path)
    assert (tmp_path / 'plans/plan.json').read_bytes() == old


def test_failed_capacity_or_label_contaminated_metadata_rejected(tmp_path):
    fixture(tmp_path)
    put(tmp_path / 'primary/runs/h__closed_c1_r1/summary.json', {
        'successful_reviews_per_second': 10, 'success_rate': 0.5, 'attempts': 30})
    with pytest.raises(ValueError, match='failed'):
        create_plan('primary', 'plans', 'replay.jsonl', 'metadata.jsonl', root=tmp_path)
    fixture(tmp_path)
    rows = [json.loads(line) for line in (tmp_path / 'metadata.jsonl').read_text().splitlines()]
    rows[0]['labels'] = ['do not select on labels']
    put(tmp_path / 'metadata.jsonl', rows, True)
    with pytest.raises(ValueError, match='label-free'):
        create_plan('primary', 'plans', 'replay.jsonl', 'metadata.jsonl', root=tmp_path)
