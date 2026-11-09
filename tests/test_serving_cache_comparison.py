"""CPU-only counterfactual cache comparison, coverage and provenance checks."""

import json

import pytest

from compare_serving_cache_modes import analyse_campaigns, compare_records, sha
from serving_quality_intervals import canonical_sha256


def fixture_data():
    labels = [{'request_id': 'sample-a', 'pairs': [{'aspect': 'held', 'sentiment': 'negative'}]},
              {'request_id': 'sample-b', 'pairs': [{'aspect': 'seen', 'sentiment': 'positive'}]},
              {'request_id': 'sample-c', 'pairs': [{'aspect': 'held', 'sentiment': 'neutral'}]}]
    artifact = {'fold_id': 'l2-a01', 'model': {'model_id': 'qwen', 'revision': 'fixed'},
                'taxonomy': {'canonical_order': ['held', 'seen'], 'heldout_aspects': ['held']},
                'replay': {'splits': {'evaluation': {'count': 3,
                    'request_id_sha256': canonical_sha256([r['request_id'] for r in labels])}}}}
    return labels, artifact


def make_rows(labels, *, mode, variant='few_shot', artifact_hash='hash'):
    rows = []
    for i, label in enumerate(labels):
        attempt = f'{mode}-attempt-{i}'
        predictions = [{'aspect': pair['aspect'], 'sentiments': [pair['sentiment']],
                        'sentiment_scores': {s: .8 if s == pair['sentiment'] else .1
                                             for s in ['negative', 'neutral', 'positive']}}
                       for pair in label['pairs']]
        rows.append({'request_id': attempt, 'sample_id': label['request_id'], 'success': True,
                     'started_s': i, 'finished_s': i + 1, 'scheduled_s': i,
                     'latency_ms': 1000 if mode == 'on' else 2000,
                     'scheduled_latency_ms': 1000 if mode == 'on' else 2000,
                     'schedule_lag_ms': 0, 'status_code': 200, 'error': None,
                     'response': {'request_id': attempt, 'variant': variant, 'fold_id': 'l2-a01',
                        'model_version': f'fixed-l2-a01-{artifact_hash[:12]}-{variant}-prefix{mode}',
                        'predictions': predictions,
                        'aspect_scores': [{'aspect': 'held', 'score': .9}, {'aspect': 'seen', 'score': .9}]}})
    return rows


def test_label_change_counts_score_scope_and_descriptive_timing_ratio():
    labels, artifact = fixture_data()
    off, on = make_rows(labels, mode='off'), make_rows(labels, mode='on')
    on[0]['response']['predictions'][0]['sentiments'] = ['positive']
    on[0]['response']['predictions'][0]['sentiment_scores'] = {'negative': .1, 'neutral': .1, 'positive': .8}
    on[1]['response']['predictions'] = []
    on[0]['response']['aspect_scores'][0]['score'] = .7
    output = compare_records(on, off, labels, artifact, on_wall=3, off_wall=6, replicates=100)
    changes = output['model_output_changes_common_success']
    assert changes['pair_label_changed_reviews'] == 2
    assert changes['presence_changed_reviews'] == 1
    assert changes['sentiment_only_changed_reviews'] == 1
    assert output['all_evaluation_final_label_sets_equal'] is False
    scores = output['score_differences_common_success']
    assert scores['aspect_scores_compared'] == 6
    assert scores['aspect_max_absolute_difference'] == pytest.approx(.2)
    assert scores['common_selected_aspect_instances'] == 2
    assert scores['sentiment_max_absolute_difference'] == pytest.approx(.7)
    assert output['performance']['ratios']['on_over_off_p50_latency'] == .5
    assert output['performance']['ratios']['on_over_off_successful_throughput'] == 2
    assert 'causal' in output['performance']['ratios']['interpretation']


def test_failures_are_not_counted_as_observed_model_label_changes():
    labels, artifact = fixture_data()
    off, on = make_rows(labels, mode='off'), make_rows(labels, mode='on')
    on[1]['success'] = False
    on[1]['response'] = {}
    output = compare_records(on, off, labels, artifact, on_wall=3, off_wall=6, replicates=100)
    assert output['coverage']['both_successful_reviews'] == 2
    assert output['model_output_changes_common_success']['pair_label_changed_reviews'] == 0
    assert output['delivered_changes_failure_as_empty']['pair_label_changed_reviews'] == 1
    assert output['all_evaluation_final_label_sets_equal'] is None


def test_missing_or_repeated_evaluation_identity_is_rejected():
    labels, artifact = fixture_data()
    off, on = make_rows(labels, mode='off'), make_rows(labels, mode='on')
    on[1]['sample_id'] = on[0]['sample_id']
    with pytest.raises(ValueError, match='exact frozen evaluation set'):
        compare_records(on, off, labels, artifact, on_wall=3, off_wall=6, replicates=10)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def setup_campaigns(tmp_path):
    labels, artifact = fixture_data()
    labels_path, artifacts_path = tmp_path / 'labels.jsonl', tmp_path / 'artifacts.json'
    labels_path.write_text(''.join(json.dumps(row) + '\n' for row in labels), encoding='utf-8')
    artifact['replay']['splits']['evaluation']['files'] = {
        'labels': {'sha256': sha(labels_path)}, 'replay': {'sha256': 'frozen-replay-hash'}}
    write(artifacts_path, artifact)
    digest = sha(artifacts_path)
    for campaign_name, mode in [('primary', 'on'), ('comparison', 'off')]:
        campaign = tmp_path / campaign_name
        sessions = []
        for variant in ['few_shot', 'hybrid']:
            name = variant + '_prefix_' + mode
            server = {'variant': variant, 'prefix_cache': mode, 'pair_batch_size': 4,
                      'max_batch_tokens': 4096, 'request_batch_size': 1, 'batch_wait_ms': 0,
                      'queue_size': 8, 'timeout_s': 120, 'logits_mode': 'last_token'}
            run = {'name': 'quality_evaluation', 'quality_evaluation': True, 'requests': 3,
                   'mode': 'closed', 'concurrency': 1}
            sessions.append({'name': name, 'server': server, 'runs': [run]})
            run_dir = campaign / 'runs' / (name + '__quality_evaluation')
            write(run_dir / 'config.json', {'replay_sha256': 'frozen-replay-hash',
                  'expected_variant': variant, 'mode': 'closed', 'concurrency': 1})
            write(run_dir / 'summary.json', {'attempts': 3, 'wall_seconds_including_drain': 3,
                                             'interrupted': False})
            rows = make_rows(labels, mode=mode, variant=variant, artifact_hash=digest)
            (run_dir / 'requests.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
            write(campaign / 'server_sessions' / (name + '_timestamp') / 'server_metadata.json', {
                'arguments': server, 'device': 'synthetic-device', 'versions': {'torch': 'fixed'},
                'engine': {'artifacts_sha256': digest, 'fold_id': 'l2-a01', 'quantization': 'fixed-nf4',
                    'model_id': 'qwen', 'model_revision': 'fixed', 'prefix_cache_enabled': mode == 'on'},
                'source_sha256': {'inference.py': 'frozen-inference', 'prefix_cache.py': 'frozen-prefix'}})
        write(campaign / 'plan.json', {'sessions': sessions, 'artifacts_sha256': digest})
        write(campaign / 'state.json', {'status': 'completed'})
    return labels_path, artifacts_path


def test_campaign_comparison_is_aggregate_only_and_matches_all_identical_outputs(tmp_path):
    labels_path, artifacts_path = setup_campaigns(tmp_path)
    output = analyse_campaigns(tmp_path / 'primary', tmp_path / 'comparison', labels_path,
                               artifacts_path, replicates=100)
    for variant in output['variants'].values():
        assert variant['all_evaluation_final_label_sets_equal'] is True
        assert variant['score_differences_common_success']['aspect_max_absolute_difference'] == 0
    text = json.dumps(output)
    assert 'sample-a' not in text and 'attempt-0' not in text and str(tmp_path) not in text


@pytest.mark.parametrize('fault', ['incomplete', 'input_hash', 'fold', 'source', 'cache_identity'])
def test_campaign_identity_and_provenance_fail_closed(tmp_path, fault):
    labels_path, artifacts_path = setup_campaigns(tmp_path)
    comparison = tmp_path / 'comparison'
    run = comparison / 'runs/few_shot_prefix_off__quality_evaluation'
    if fault == 'incomplete':
        write(comparison / 'state.json', {'status': 'running'})
    elif fault == 'input_hash':
        value = json.loads((run / 'config.json').read_text())
        value['replay_sha256'] = 'wrong'
        write(run / 'config.json', value)
    elif fault in ['fold', 'cache_identity']:
        rows = [json.loads(line) for line in (run / 'requests.jsonl').read_text().splitlines()]
        if fault == 'fold':
            rows[0]['response']['fold_id'] = 'l2-a02'
        else:
            rows[0]['response']['model_version'] = 'wrong-cache-identity'
        (run / 'requests.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    else:
        metadata = comparison / 'server_sessions/few_shot_prefix_off_timestamp/server_metadata.json'
        value = json.loads(metadata.read_text())
        value['source_sha256']['inference.py'] = 'different'
        write(metadata, value)
    with pytest.raises(ValueError):
        analyse_campaigns(tmp_path / 'primary', comparison, labels_path, artifacts_path, replicates=10)
