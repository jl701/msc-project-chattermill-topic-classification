"""CPU-only paired analysis of completed prefix-on/off quality campaigns.

Strictly match frozen model inputs and unique evaluation reviews. Export only
aggregates. A 100-review development gate is never extrapolated to unseen replay
rows. Sequential single-run timing ratios are descriptive, not causal estimates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from msc_project.serving.monitoring import prediction_pairs, summarise_load
from serving_quality_intervals import analyse as bootstrap_quality

SENTIMENTS = {'negative', 'neutral', 'positive'}
SERVER_FIELDS = ('pair_batch_size', 'max_batch_tokens', 'request_batch_size',
                 'batch_wait_ms', 'queue_size', 'timeout_s', 'logits_mode')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines()
            if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def successful_by_sample(records):
    first = {}
    for record in sorted(records, key=lambda r: (r.get('started_s') is None,
                                                r.get('started_s') or 0)):
        if record.get('success'):
            first.setdefault(str(record.get('sample_id', record.get('request_id'))), record['response'])
    return first


def change_counts(keys, off, on):
    """Direction is uncached/off to cached/on; absent deliveries become empty."""
    result = {'reviews_compared': len(keys), 'pair_label_changed_reviews': 0,
              'presence_changed_reviews': 0, 'sentiment_only_changed_reviews': 0,
              'pair_assignments_added': 0, 'pair_assignments_removed': 0,
              'aspect_assignments_added': 0, 'aspect_assignments_removed': 0}
    for key in keys:
        left, right = prediction_pairs(off.get(key, {})), prediction_pairs(on.get(key, {}))
        la, ra = {a for a, _ in left}, {a for a, _ in right}
        result['pair_label_changed_reviews'] += left != right
        result['presence_changed_reviews'] += la != ra
        result['sentiment_only_changed_reviews'] += left != right and la == ra
        result['pair_assignments_added'] += len(right - left)
        result['pair_assignments_removed'] += len(left - right)
        result['aspect_assignments_added'] += len(ra - la)
        result['aspect_assignments_removed'] += len(la - ra)
    result['pair_label_changed_fraction'] = result['pair_label_changed_reviews'] / len(keys) if keys else None
    return result


def finite_probability(value):
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError('Scores must be finite probabilities in [0, 1]')
    return number


def score_differences(keys, off, on, taxonomy):
    aspect_deltas, sentiment_deltas = [], []
    common_aspect_instances = 0
    for key in keys:
        responses = [off[key], on[key]]
        aspect_maps = []
        for response in responses:
            scores = response.get('aspect_scores', [])
            values = {item['aspect']: finite_probability(item['score']) for item in scores}
            if len(values) != len(scores) or set(values) != taxonomy:
                raise ValueError('Each successful response must contain all frozen aspect scores exactly once')
            aspect_maps.append(values)
        aspect_deltas.extend(abs(aspect_maps[1][a] - aspect_maps[0][a]) for a in taxonomy)
        predictions = [{row['aspect']: row for row in r['predictions']} for r in responses]
        for aspect in predictions[0].keys() & predictions[1].keys():
            common_aspect_instances += 1
            values = [p[aspect].get('sentiment_scores', {}) for p in predictions]
            if any(set(v) != SENTIMENTS for v in values):
                raise ValueError('Common selected aspects require all three sentiment scores')
            sentiment_deltas.extend(abs(finite_probability(values[1][s]) - finite_probability(values[0][s]))
                                    for s in SENTIMENTS)
    return {
        'aspect_scores_compared': len(aspect_deltas),
        'aspect_max_absolute_difference': max(aspect_deltas) if aspect_deltas else None,
        'aspect_nonidentical_score_count': sum(v != 0 for v in aspect_deltas),
        'common_selected_aspect_instances': common_aspect_instances,
        'sentiment_scores_compared': len(sentiment_deltas),
        'sentiment_max_absolute_difference': max(sentiment_deltas) if sentiment_deltas else None,
        'sentiment_nonidentical_score_count': sum(v != 0 for v in sentiment_deltas),
        'scope': 'All taxonomy aspect scores for common-success reviews; sentiment differences only for aspects selected by BOTH systems. No missing sentiment score is fabricated.',
    }


def ratio(numerator, denominator):
    return numerator / denominator if numerator is not None and denominator is not None and denominator > 0 else None


def safe_performance(records, wall):
    result = summarise_load(records, wall)
    # The actual driver emits categorical error codes, but do not export an
    # arbitrary manually edited log string as an aggregate dictionary key.
    failures = {}
    for key, count in result['failure_counts'].items():
        safe = key if re.fullmatch(r'(http_[0-9]{3}|timeout|connection_error|invalid_response_json|response_request_id_mismatch|response_missing_predictions|response_variant_mismatch|client_capacity_drop|client_[A-Za-z][A-Za-z0-9_]*|none)', key) else 'other_error'
        failures[safe] = failures.get(safe, 0) + count
    result['failure_counts'] = failures
    result['length_groups'] = {key: value for key, value in result['length_groups'].items()
                               if key in {'short', 'medium', 'long', 'unknown', *map(str, range(1, 11))}}
    result['http_status_counts'] = {key: value for key, value in result['http_status_counts'].items()
                                    if key == 'None' or re.fullmatch(r'[0-9]{3}', key)}
    return result


def compare_records(on_records, off_records, labels, artifacts, *, on_wall, off_wall,
                    replicates=2000, seed=20261109):
    """Compare complete equal input sets; expose failed-delivery coverage separately."""
    expected_ids = {str(row['request_id']) for row in labels}
    for records in [on_records, off_records]:
        keys = [str(r.get('sample_id', r.get('request_id'))) for r in records]
        if len(keys) != len(expected_ids) or len(set(keys)) != len(keys) or set(keys) != expected_ids:
            raise ValueError('Both quality runs must attempt the exact frozen evaluation set once per review')
    intervals = bootstrap_quality({'cache_on': on_records, 'cache_off': off_records}, labels, artifacts,
                                  replicates=replicates, seed=seed)
    on, off = successful_by_sample(on_records), successful_by_sample(off_records)
    common = set(on) & set(off)
    changes = change_counts(common, off, on)
    full_success = len(common) == len(expected_ids)
    performance = {'cache_on': safe_performance(on_records, on_wall),
                   'cache_off': safe_performance(off_records, off_wall)}
    on_perf, off_perf = performance['cache_on'], performance['cache_off']
    performance['ratios'] = {
        'on_over_off_p50_latency': ratio(on_perf['successful_latency']['p50_ms'], off_perf['successful_latency']['p50_ms']),
        'on_over_off_p95_latency': ratio(on_perf['successful_latency']['p95_ms'], off_perf['successful_latency']['p95_ms']),
        'on_over_off_successful_throughput': ratio(on_perf['successful_reviews_per_second'], off_perf['successful_reviews_per_second']),
        'interpretation': 'One sequential full-replay run per mode, not a randomized or repeated causal performance estimate. Latency uses successful attempts only; compare coverage/success rates and thermal/clock telemetry. Throughput includes client drain.',
    }
    return {
        'coverage': {'frozen_evaluation_reviews': len(expected_ids), 'both_successful_reviews': len(common),
                     'on_only_successful_reviews': len(set(on) - set(off)),
                     'off_only_successful_reviews': len(set(off) - set(on)),
                     'neither_successful_reviews': len(expected_ids - (set(on) | set(off))),
                     'complete_model_output_comparison': full_success},
        'change_direction': 'cache_off_to_cache_on',
        'model_output_changes_common_success': changes,
        'delivered_changes_failure_as_empty': change_counts(expected_ids, off, on),
        'all_evaluation_final_label_sets_equal': changes['pair_label_changed_reviews'] == 0 if full_success else None,
        'score_differences_common_success': score_differences(common, off, on, set(artifacts['taxonomy']['canonical_order'])),
        'quality_intervals': intervals,
        'quality_difference_direction': 'cache_off_minus_cache_on, as explicitly identified in paired_differences',
        'performance': performance,
    }


def select_run(campaign, variant, mode, artifacts, artifact_hash, expected_label_ids):
    campaign = Path(campaign)
    if read_json(campaign / 'state.json').get('status') != 'completed':
        raise ValueError('Both campaigns must be completed before cache-mode comparison')
    plan = read_json(campaign / 'plan.json')
    if plan.get('artifacts_sha256') != artifact_hash:
        raise ValueError('Campaign artifact identity differs from the frozen manifest')
    candidates = []
    for session in plan['sessions']:
        if session['server'].get('variant') != variant or session['server'].get('prefix_cache') != mode:
            continue
        for run in session['runs']:
            if run.get('quality_evaluation') is True:
                candidates.append((session, run))
    if len(candidates) != 1:
        raise ValueError(f'Expected one {variant}/prefix-{mode} quality run')
    session, registered = candidates[0]
    if (registered.get('mode') != 'closed' or registered.get('concurrency') != 1
            or registered.get('requests') != len(expected_label_ids) or registered.get('shuffle', False)):
        raise ValueError('Cache-mode references require an unshuffled complete C1 quality run')
    directory = campaign / 'runs' / (session['name'] + '__' + registered['name'])
    config, summary = read_json(directory / 'config.json'), read_json(directory / 'summary.json')
    expected_replay_hash = artifacts['replay']['splits']['evaluation']['files']['replay']['sha256']
    if config.get('replay_sha256') != expected_replay_hash:
        raise ValueError('Quality input content hash differs from the frozen evaluation replay')
    if config.get('expected_variant') != variant or config.get('mode') != 'closed' or config.get('concurrency') != 1:
        raise ValueError('Executed client configuration differs from the registered comparison')
    if summary.get('interrupted') or summary.get('attempts') != len(expected_label_ids):
        raise ValueError('Quality run is interrupted or has incomplete attempt coverage')
    records = read_jsonl(directory / 'requests.jsonl')
    if len(records) != summary['attempts']:
        raise ValueError('Persisted request count differs from the final summary')
    keys = [str(r.get('sample_id', r.get('request_id'))) for r in records]
    if len(set(keys)) != len(keys) or set(keys) != expected_label_ids:
        raise ValueError('Quality run repeats or omits frozen evaluation identities')
    for record in records:
        if record.get('success'):
            response = record.get('response', {})
            version = response.get('model_version', '')
            if (response.get('request_id') != record.get('request_id')
                    or response.get('variant') != variant or response.get('fold_id') != artifacts['fold_id']
                    or not version.endswith(f'-{variant}-prefix{mode}') or artifact_hash[:12] not in version):
                raise ValueError('Successful response has incompatible request/model/fold/cache identity')
    runtime, runtime_settings = [], []
    for path in (campaign / 'server_sessions').glob(session['name'] + '_*/server_metadata.json'):
        metadata = read_json(path)
        if metadata.get('arguments', {}).get('variant') != variant:
            continue
        engine = metadata.get('engine', {})
        if (engine.get('artifacts_sha256') != artifact_hash or engine.get('fold_id') != artifacts['fold_id']
                or engine.get('model_id') != artifacts['model']['model_id']
                or engine.get('model_revision') != artifacts['model']['revision']
                or engine.get('prefix_cache_enabled') is not (mode == 'on')):
            raise ValueError('Actual server model metadata does not match the comparison identity')
        if any(metadata.get('arguments', {}).get(field) != session['server'].get(field)
               for field in SERVER_FIELDS):
            raise ValueError('Actual server engineering settings differ from the registered plan')
        source = metadata.get('source_sha256', {})
        if not source.get('inference.py') or not source.get('prefix_cache.py'):
            raise ValueError('Server did not record inference and prefix source fingerprints')
        runtime.append(source)
        settings = {'device': metadata.get('device'), 'versions': metadata.get('versions'),
                    'quantization': engine.get('quantization')}
        if not settings['device'] or not settings['versions'] or not settings['quantization']:
            raise ValueError('Server lacks device, dependency or quantization provenance')
        runtime_settings.append(settings)
    if not runtime:
        raise ValueError('No matching actual server metadata found')
    implementation = {name: {r[name] for r in runtime} for name in ['inference.py', 'prefix_cache.py']}
    if any(len(values) != 1 for values in implementation.values()):
        raise ValueError('Multiple implementation versions occurred within a comparison session')
    if any(settings != runtime_settings[0] for settings in runtime_settings):
        raise ValueError('Multiple device/dependency/quantization settings occurred in one session')
    wall = summary.get('wall_seconds_including_drain')
    if not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall <= 0:
        raise ValueError('Quality run lacks a finite positive elapsed wall time')
    return {
        'records': records, 'wall': wall, 'server': session['server'],
        'implementation': {name: next(iter(values)) for name, values in implementation.items()},
        'runtime_settings': runtime_settings[0],
        'provenance': {'campaign_plan_sha256': sha(campaign / 'plan.json'),
                       'requests_sha256': sha(directory / 'requests.jsonl'),
                       'summary_sha256': sha(directory / 'summary.json'),
                       'client_config_sha256': sha(directory / 'config.json')},
    }


def analyse_campaigns(primary, comparison, labels_path, artifacts_path, *, replicates=2000, seed=20261109):
    artifacts, labels = read_json(artifacts_path), read_jsonl(labels_path)
    expected_labels = artifacts['replay']['splits']['evaluation']['files']['labels']['sha256']
    if sha(labels_path) != expected_labels:
        raise ValueError('Label file differs from the independently frozen evaluation labels')
    label_ids = {str(row['request_id']) for row in labels}
    if len(label_ids) != len(labels):
        raise ValueError('Ground-truth identities are duplicated')
    artifact_hash = sha(artifacts_path)
    output = {'schema_version': 'serving_cache_mode_comparison_v1',
              'fold_id': artifacts['fold_id'], 'artifacts_sha256': artifact_hash,
              'evaluation_labels_sha256': expected_labels,
              'scope': 'Complete frozen validation engineering replay, not a new blind test or historical twelve-fold average. Aggregate-only output.',
              'variants': {}}
    for variant in ['few_shot', 'hybrid']:
        on = select_run(primary, variant, 'on', artifacts, artifact_hash, label_ids)
        off = select_run(comparison, variant, 'off', artifacts, artifact_hash, label_ids)
        if any(on['server'].get(field) != off['server'].get(field) for field in SERVER_FIELDS):
            raise ValueError('Prefix-on/off engineering settings differ beyond cache mode')
        if on['implementation'] != off['implementation']:
            raise ValueError('Prefix-on/off inference implementations differ; cannot attribute comparison to cache mode')
        if on['runtime_settings'] != off['runtime_settings']:
            raise ValueError('Prefix-on/off device, dependency versions or quantization differ')
        result = compare_records(on['records'], off['records'], labels, artifacts,
                                 on_wall=on['wall'], off_wall=off['wall'], replicates=replicates, seed=seed)
        result['provenance'] = {'cache_on': on['provenance'], 'cache_off': off['provenance'],
                                'implementation_sha256': on['implementation'],
                                'runtime_settings': on['runtime_settings']}
        output['variants'][variant] = result
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary-campaign', type=Path, required=True)
    parser.add_argument('--comparison-campaign', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replicates', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261109)
    args = parser.parse_args()
    result = analyse_campaigns(args.primary_campaign, args.comparison_campaign, args.labels,
                               args.artifacts, replicates=args.replicates, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'variants': list(result['variants']), 'fold_id': result['fold_id'],
                      'bootstrap_replicates': args.replicates}, indent=2))


if __name__ == '__main__':
    main()
