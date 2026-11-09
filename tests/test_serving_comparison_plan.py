"""Synthetic CPU checks for preregistered post-primary comparison plans."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from create_serving_comparison_plan import CAPACITY_NAMES, REGISTRATION, create_plan, sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def fixture_campaign(root):
    registry = Path(__file__).resolve().parents[1] / REGISTRATION
    write(root / REGISTRATION, json.loads(registry.read_text(encoding='utf-8')))
    parts = {}
    for split, count in [('development', 100), ('evaluation', 957)]:
        parts[split] = {'count': count, 'files': {}}
        for kind in ['replay', 'metadata']:
            path = root / 'outputs' / f'{kind}_{split}.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('synthetic fixture content\n', encoding='utf-8')
            parts[split]['files'][kind] = {'path': str(path), 'sha256': sha(path)}
    artifacts = root / 'outputs/artifacts.json'
    write(artifacts, {'fold_id': 'l2-a01', 'replay': {'splits': parts}})
    primary = root / 'outputs/primary'
    plan = {'artifacts': artifacts.relative_to(root).as_posix(), 'artifacts_sha256': sha(artifacts),
            'prefix_cache': 'on', 'sessions': []}
    defaults = {'port': 8765, 'prefix_cache': 'on', 'pair_batch_size': 4,
                'max_batch_tokens': 4096, 'request_batch_size': 1, 'batch_wait_ms': 0,
                'queue_size': 8, 'timeout_s': 120, 'logits_mode': 'last_token'}
    for variant, capacity in [('few_shot', 2), ('qlora', 8), ('hybrid', 4)]:
        server = dict(defaults, variant=variant)
        runs = [{'name': name, 'mode': 'closed', 'concurrency': 1, 'requests': 30} for name in CAPACITY_NAMES]
        runs.append({'name': 'quality_evaluation', 'mode': 'closed', 'concurrency': 1,
                     'requests': 957, 'quality_evaluation': True})
        plan['sessions'].append({'name': variant, 'server': server, 'runs': runs})
        for index, name in enumerate(CAPACITY_NAMES):
            write(primary / 'runs' / f'{variant}__{name}' / 'summary.json',
                  {'attempts': 30, 'success_rate': 1, 'interrupted': False,
                   'successful_reviews_per_second': capacity * [.9, 1, 1.1][index]})
        directory = primary / 'runs' / (variant + '__quality_evaluation')
        write(directory / 'summary.json', {'attempts': 957, 'interrupted': False})
        write(directory / 'config.json', {'replay_sha256': parts['evaluation']['files']['replay']['sha256']})
    write(primary / 'plan.json', plan)
    write(primary / 'state.json', {'status': 'completed'})
    return primary, plan


def test_shared_rate_and_prefix_off_references_preserve_registered_settings(tmp_path):
    primary, original = fixture_campaign(tmp_path)
    plan = create_plan(primary, tmp_path / 'outputs/extension', root=tmp_path)
    assert plan['common_absolute_rate_reviews_per_second'] == 1
    assert plan['per_variant_median_capacity'] == {'few_shot': 2, 'qlora': 8, 'hybrid': 4}
    assert len(plan['sessions']) == 5
    common = [s for s in plan['sessions'] if s['name'].endswith('_common_rate')]
    for session in common:
        assert session['server']['prefix_cache'] == 'on'
        measured = session['runs'][1:]
        assert len(measured) == 3
        assert {r['rate'] for r in measured} == {1}
        assert {r['duration'] for r in measured} == {90}
        assert [r['seed'] for r in measured] == [20261109, 20261110, 20261111]
    off = [s for s in plan['sessions'] if s['name'].endswith('_prefix_off')]
    assert {s['server']['variant'] for s in off} == {'few_shot', 'hybrid'}
    for session in off:
        expected = deepcopy(next(s['server'] for s in original['sessions']
                                 if s['server']['variant'] == session['server']['variant']))
        expected['prefix_cache'] = 'off'
        assert session['server'] == expected
        assert session['runs'][0]['requests'] == 3
        assert session['runs'][1]['requests'] == 957
        assert session['runs'][1]['quality_evaluation'] is True
        assert 'shuffle' not in session['runs'][1]
    assert plan['selection_boundaries']['labels_read'] is False
    assert not any(str(tmp_path) in str(value) for value in plan.values())


@pytest.mark.parametrize('fault', ['incomplete', 'failed_capacity', 'changed_input',
                                  'wrong_prefix', 'partial_quality', 'unequal_settings'])
def test_rejects_incomplete_or_changed_primary_evidence_without_creating_output(tmp_path, fault):
    primary, plan = fixture_campaign(tmp_path)
    if fault == 'incomplete':
        write(primary / 'state.json', {'status': 'running'})
    elif fault == 'failed_capacity':
        write(primary / 'runs/few_shot__closed_c1_r1/summary.json',
              {'attempts': 30, 'success_rate': .9, 'successful_reviews_per_second': 2})
    elif fault == 'changed_input':
        (tmp_path / 'outputs/replay_evaluation.jsonl').write_text('different', encoding='utf-8')
    elif fault == 'wrong_prefix':
        plan['prefix_cache'] = 'off'
        write(primary / 'plan.json', plan)
    elif fault == 'partial_quality':
        write(primary / 'runs/hybrid__quality_evaluation/summary.json', {'attempts': 956})
    elif fault == 'unequal_settings':
        plan['sessions'][1]['server']['request_batch_size'] = 2
        write(primary / 'plan.json', plan)
    output = tmp_path / 'outputs/extension'
    with pytest.raises(ValueError):
        create_plan(primary, output, root=tmp_path)
    assert not output.exists()


def test_existing_extension_is_never_overwritten(tmp_path):
    primary, _ = fixture_campaign(tmp_path)
    output = tmp_path / 'outputs/extension'
    create_plan(primary, output, root=tmp_path)
    before = sha(output / 'plan.json')
    with pytest.raises(FileExistsError):
        create_plan(primary, output, root=tmp_path)
    assert sha(output / 'plan.json') == before
