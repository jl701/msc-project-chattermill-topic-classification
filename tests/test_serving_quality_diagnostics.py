"""Synthetic exact error partitions, conditional support and delivery semantics."""

import json

import pytest

from diagnose_serving_quality import diagnose_campaigns, diagnose_records


def sample(key, gold, predicted=(), *, success=True, started=0):
    label = {'request_id': key, 'pairs': [{'aspect': a, 'sentiment': s} for a, s in gold]}
    request_id = f'attempt-{key}-{started}'
    record = {'request_id': request_id, 'sample_id': key, 'started_s': started, 'success': success,
              'response': {'request_id': request_id, 'variant': 'hybrid', 'fold_id': 'l2-a01',
                           'model_version': 'fixed-hybrid-prefixon',
                           'predictions': [{'aspect': a, 'sentiments': [s]} for a, s in predicted]}}
    return label, record


def example():
    cases = [
        sample('wrong-aspect', [('held', 'positive')], [('seen', 'negative')], started=0),
        sample('wrong-sentiment', [('held', 'positive')], [('held', 'negative')], started=1),
        sample('multi-sentiment', [('held', 'negative'), ('held', 'positive')],
               [('held', 'positive'), ('held', 'neutral')], started=2),
        sample('no-gold', [], [('held', 'positive')], started=3),
        sample('missed', [('held', 'positive')], [], started=4),
        sample('delivery-failed', [('seen', 'positive')], [], success=False, started=5),
    ]
    return [x[0] for x in cases], [x[1] for x in cases]


TAXONOMY = {'canonical_order': ['held', 'seen'], 'heldout_aspects': ['held']}


def test_exact_fp_fn_partition_matches_monitoring_in_every_scope_and_view():
    labels, records = example()
    output = diagnose_records(records, labels, TAXONOMY)
    overall = output['views']['successful_unique_quality']['overall']
    assert overall['end_to_end_pair'] == {
        'tp': 1, 'fp': 4, 'fn': 4, 'precision': .2, 'recall': .2, 'micro_f1': .2,
        'gold_pair_support': 5, 'predicted_pair_support': 5}
    assert overall['pair_fp_decomposition']['aspect_absent_from_gold'] == 2
    assert overall['pair_fp_decomposition']['aspect_exists_sentiment_incorrect_or_extra'] == 2
    assert overall['pair_fn_decomposition']['aspect_not_detected'] == 2
    assert overall['pair_fn_decomposition']['aspect_detected_sentiment_missing_or_incorrect'] == 2
    for view in output['views'].values():
        for scope in view.values():
            assert scope['pair_fp_decomposition']['sum_verified']
            assert scope['pair_fn_decomposition']['sum_verified']
            assert scope['matches_monitoring_end_to_end_counts']
    assert output['views']['successful_unique_quality']['heldout']['end_to_end_pair']['fp'] == 3
    assert output['views']['successful_unique_quality']['seen']['end_to_end_pair']['fp'] == 1


def test_conditional_sentiment_excludes_presence_errors_and_counts_aspect_instances():
    labels, records = example()
    output = diagnose_records(records, labels, TAXONOMY)
    value = output['views']['successful_unique_quality']['overall']['conditional_sentiment_on_correctly_detected_aspects']
    assert (value['tp'], value['fp'], value['fn']) == (1, 2, 2)
    assert value['micro_f1'] == pytest.approx(1 / 3)
    assert value['gold_pair_support'] == value['predicted_pair_support'] == 3
    assert value['correctly_detected_aspect_instances'] == 2
    assert value['reviews_with_correctly_detected_aspect'] == 2
    assert 'cannot replace' in value['limitation']


def test_delivery_failure_is_visible_in_separate_view_without_inflating_conditional_support():
    labels, records = example()
    output = diagnose_records(records, labels, TAXONOMY)
    success = output['views']['successful_unique_quality']['overall']
    delivered = output['views']['delivered_quality_failure_as_empty']['overall']
    assert output['coverage']['unique_successful_reviews'] == 5
    assert output['coverage']['unique_reviews_without_success'] == 1
    assert delivered['end_to_end_pair']['fn'] == success['end_to_end_pair']['fn'] + 1
    assert delivered['pair_fn_decomposition']['aspect_not_detected'] == 3
    assert delivered['conditional_sentiment_on_correctly_detected_aspects']['gold_pair_support'] == 3


def test_repeated_better_success_does_not_replace_first_success_or_add_samples():
    labels, records = example()
    _, improved = sample('wrong-sentiment', [('held', 'positive')], [('held', 'positive')], started=10)
    records.append(improved)
    output = diagnose_records(records, labels, TAXONOMY)
    assert output['coverage']['successful_attempts_omitted_as_repeats'] == 1
    assert output['coverage']['repeated_success_prediction_disagreements'] == 1
    assert output['views']['successful_unique_quality']['overall']['end_to_end_pair']['tp'] == 1
    serialized = json.dumps(output)
    assert 'wrong-sentiment' not in serialized and 'attempt-missed-4' not in serialized


def test_running_campaign_is_rejected_before_any_labels_or_results_are_read(tmp_path):
    campaign = tmp_path / 'running'
    campaign.mkdir()
    (campaign / 'state.json').write_text('{"status":"running"}', encoding='utf-8')
    with pytest.raises(ValueError, match='Only completed'):
        diagnose_campaigns({'primary': campaign}, tmp_path / 'missing-labels', tmp_path / 'missing-artifacts')


def test_zero_conditional_support_and_unknown_label_are_explicit():
    label, record = sample('only-presence-error', [('held', 'positive')], [('seen', 'negative')])
    output = diagnose_records([record], [label], TAXONOMY)
    value = output['views']['successful_unique_quality']['overall']['conditional_sentiment_on_correctly_detected_aspects']
    assert value['gold_pair_support'] == value['correctly_detected_aspect_instances'] == 0
    assert value['micro_f1'] == 0
    record['response']['predictions'][0]['sentiments'] = ['unsupported']
    with pytest.raises(ValueError, match='outside'):
        diagnose_records([record], [label], TAXONOMY)
