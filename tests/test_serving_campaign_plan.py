"""CPU-only admission checks for the expanded frozen-prefix gate."""

import json

import pytest

from create_serving_campaign_plan import canonical_sha256, sha256, verify_prefix_gate


def write(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def setup_gate(tmp_path):
    replay_path, artifacts_path = tmp_path / 'replay.jsonl', tmp_path / 'artifacts.json'
    rows = [{'request_id': f'dev-{i}', 'text': 'synthetic ' + str(i)} for i in range(100)]
    replay_path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    artifacts = {'fold_id': 'l2-a01', 'model': {'model_id': 'qwen', 'revision': 'fixed'},
                 'replay': {'splits': {'development': {'count': 100,
                    'request_id_sha256': canonical_sha256([row['request_id'] for row in rows]),
                    'files': {'replay': {'sha256': sha256(replay_path)}}}}}}
    write(artifacts_path, artifacts)
    gate = {'status': 'completed', 'label_gate_pass': True, 'official_test_accessed': False,
            'samples': [{'request_id': row['request_id'], 'characters': len(row['text'])} for row in rows],
            'engine': {'artifacts_sha256': sha256(artifacts_path), 'fold_id': 'l2-a01',
                       'model_id': 'qwen', 'model_revision': 'fixed', 'logits_mode': 'last_token',
                       'pair_batch_size': 4, 'max_batch_tokens': 4096},
            'parameters': {'samples': 100, 'pair_batch_size': 4, 'max_batch_tokens': 4096,
                           'replay': str(replay_path), 'artifacts': str(artifacts_path),
                           'variants': 'few_shot,hybrid'},
            'variants': {}, 'combined_batch_checks': []}
    journal = []
    for variant in ['few_shot', 'hybrid']:
        gate['variants'][variant] = {'request_count': 100, 'label_gate_pass': True,
            'label_changed_ids': [], 'aspect_max_abs_difference': .12, 'sentiment_max_abs_difference': .04}
        gate['combined_batch_checks'].append({'variant': variant, 'label_gate_pass': True,
            'stored_prefix_immutable': True, 'request_count': 3, 'label_changed_requests': 0})
        for row in rows:
            uid = row['request_id']
            for phase in ['cache_off', 'cache_on']:
                journal.append({'variant': variant, 'phase': phase, 'request_ids': [uid],
                                'outputs': [{'request_id': uid}]})
            journal.append({'variant': variant, 'phase': 'comparison', 'request_id': uid,
                            'label_gate_pass': True})
    gate_path = tmp_path / 'summary.json'
    write(gate_path, gate)
    journal_path = tmp_path / 'samples.jsonl'
    journal_path.write_text(''.join(json.dumps(row) + '\n' for row in journal), encoding='utf-8')
    return gate_path, artifacts_path, replay_path, gate


def test_complete_matching_gate_records_provenance_without_claiming_score_identity(tmp_path):
    gate_path, artifacts_path, replay_path, _ = setup_gate(tmp_path)
    result = verify_prefix_gate(gate_path, artifacts_path, replay_path)
    assert result['status'] == 'passed'
    assert result['unique_development_reviews'] == 100
    assert result['gate_summary_sha256'] == sha256(gate_path)
    assert result['verified_variants'] == ['few_shot', 'hybrid']
    assert result['probability_differences']['few_shot']['aspect_max_abs_difference'] == .12
    assert 'does not assert numerical' in result['gate_meaning']


@pytest.mark.parametrize('failure', ['running', 'wrong_artifacts', 'wrong_fold', 'wrong_batch',
                                    'duplicate_samples', 'missing_variant', 'wrong_ids',
                                    'mutable_prefix', 'missing_journal_record', 'changed_replay'])
def test_admission_rejects_stale_incomplete_and_mismatched_evidence(tmp_path, failure):
    gate_path, artifacts_path, replay_path, gate = setup_gate(tmp_path)
    if failure == 'running':
        gate['status'] = 'running'
    elif failure == 'wrong_artifacts':
        gate['engine']['artifacts_sha256'] = 'wrong'
    elif failure == 'wrong_fold':
        gate['engine']['fold_id'] = 'l2-a02'
    elif failure == 'wrong_batch':
        gate['engine']['pair_batch_size'] = 8
    elif failure == 'duplicate_samples':
        gate['samples'][1] = gate['samples'][0]
    elif failure == 'missing_variant':
        del gate['variants']['hybrid']
    elif failure == 'wrong_ids':
        gate['samples'][0]['request_id'] = 'evaluation-row'
    elif failure == 'mutable_prefix':
        gate['combined_batch_checks'][0]['stored_prefix_immutable'] = False
    elif failure == 'missing_journal_record':
        journal = tmp_path / 'samples.jsonl'
        journal.write_text('\n'.join(journal.read_text().splitlines()[1:]), encoding='utf-8')
    elif failure == 'changed_replay':
        replay_path.write_text(replay_path.read_text().replace('synthetic 0', 'new text 00'), encoding='utf-8')
    write(gate_path, gate)
    with pytest.raises((ValueError, OSError)):
        verify_prefix_gate(gate_path, artifacts_path, replay_path)
