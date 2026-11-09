"""Read-only quality diagnostics for completed, explicitly marked serving runs.

Rules are preregistered in serving_quality_diagnostics_20261109.json. No current
run is inspected for selection, no GPU is used and no model setting is changed.
Pair errors are partitioned exactly; conditional sentiment excludes presence
errors and must never be presented as end-to-end F1. Outputs are aggregate-only.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from msc_project.experiments.taxonomy_final_test import canonical_sha256
from msc_project.serving.monitoring import evaluate_quality, label_pairs, prediction_pairs

REGISTRATION = ROOT / 'configs/experiments/serving_quality_diagnostics_20261109.json'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines()
            if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prf(tp, fp, fn):
    return {'tp': tp, 'fp': fp, 'fn': fn,
            'precision': tp / (tp + fp) if tp + fp else 0.0,
            'recall': tp / (tp + fn) if tp + fn else 0.0,
            'micro_f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            'gold_pair_support': tp + fn, 'predicted_pair_support': tp + fp}


def scope_diagnostics(keys, predictions, gold, aspects):
    counts = Counter()
    for key in keys:
        predicted = {(a, s) for a, s in predictions.get(key, set()) if a in aspects}
        target = {(a, s) for a, s in gold[key] if a in aspects}
        pred_aspects, gold_aspects = {a for a, _ in predicted}, {a for a, _ in target}
        fp, fn = predicted - target, target - predicted
        counts['tp'] += len(predicted & target)
        counts['fp'] += len(fp)
        counts['fn'] += len(fn)
        counts['fp_absent_aspect'] += sum(a not in gold_aspects for a, _ in fp)
        counts['fp_sentiment_with_existing_aspect'] += sum(a in gold_aspects for a, _ in fp)
        counts['fn_missed_aspect'] += sum(a not in pred_aspects for a, _ in fn)
        counts['fn_sentiment_with_detected_aspect'] += sum(a in pred_aspects for a, _ in fn)
        common = pred_aspects & gold_aspects
        conditional_predicted = {(a, s) for a, s in predicted if a in common}
        conditional_gold = {(a, s) for a, s in target if a in common}
        counts['conditional_tp'] += len(conditional_predicted & conditional_gold)
        counts['conditional_fp'] += len(conditional_predicted - conditional_gold)
        counts['conditional_fn'] += len(conditional_gold - conditional_predicted)
        counts['correctly_detected_aspect_instances'] += len(common)
        counts['reviews_with_correctly_detected_aspect'] += bool(common)
        counts['gold_aspect_instances'] += len(gold_aspects)
        counts['predicted_aspect_instances'] += len(pred_aspects)
        counts['gold_positive_reviews'] += bool(target)
    if counts['fp_absent_aspect'] + counts['fp_sentiment_with_existing_aspect'] != counts['fp']:
        raise AssertionError('FP decomposition does not sum to end-to-end pair FP')
    if counts['fn_missed_aspect'] + counts['fn_sentiment_with_detected_aspect'] != counts['fn']:
        raise AssertionError('FN decomposition does not sum to end-to-end pair FN')
    if (counts['conditional_fp'] != counts['fp_sentiment_with_existing_aspect']
            or counts['conditional_fn'] != counts['fn_sentiment_with_detected_aspect']
            or counts['conditional_tp'] != counts['tp']):
        raise AssertionError('Conditional sentiment and error decomposition disagree')
    return {
        'unique_reviews': len(keys), 'gold_positive_reviews': counts['gold_positive_reviews'],
        'end_to_end_pair': prf(counts['tp'], counts['fp'], counts['fn']),
        'pair_fp_decomposition': {
            'aspect_absent_from_gold': counts['fp_absent_aspect'],
            'aspect_exists_sentiment_incorrect_or_extra': counts['fp_sentiment_with_existing_aspect'],
            'total': counts['fp'], 'sum_verified': True},
        'pair_fn_decomposition': {
            'aspect_not_detected': counts['fn_missed_aspect'],
            'aspect_detected_sentiment_missing_or_incorrect': counts['fn_sentiment_with_detected_aspect'],
            'total': counts['fn'], 'sum_verified': True},
        'conditional_sentiment_on_correctly_detected_aspects': {
            **prf(counts['conditional_tp'], counts['conditional_fp'], counts['conditional_fn']),
            'correctly_detected_aspect_instances': counts['correctly_detected_aspect_instances'],
            'reviews_with_correctly_detected_aspect': counts['reviews_with_correctly_detected_aspect'],
            'all_gold_aspect_instances': counts['gold_aspect_instances'],
            'all_predicted_aspect_instances': counts['predicted_aspect_instances'],
            'population': 'predicted aspects intersect gold aspects for each unique review',
            'limitation': 'Missed and false-positive aspects are excluded. Correctly detected aspect means presence TP, not necessarily correct sentiment. This conditional F1 cannot replace end-to-end pair F1.',
        },
    }


def diagnose_records(records, labels, taxonomy):
    universe, heldout = set(taxonomy['canonical_order']), set(taxonomy['heldout_aspects'])
    if not heldout or not heldout < universe:
        raise ValueError('Frozen held-out and seen taxonomy definitions are required')
    baseline = evaluate_quality(records, labels, heldout, universe)
    gold = {str(row['request_id']): label_pairs(row) for row in labels}
    predicted, attempted = {}, set()
    for record in sorted(records, key=lambda r: (r.get('started_s') is None,
                                                r.get('started_s') or 0)):
        key = str(record.get('sample_id', record.get('request_id')))
        attempted.add(key)
        if record.get('success') and key not in predicted:
            predicted[key] = prediction_pairs(record.get('response', {}))
    valid_sentiments = {'negative', 'neutral', 'positive'}
    if any(a not in universe or s not in valid_sentiments
           for pairs in list(gold.values()) + list(predicted.values()) for a, s in pairs):
        raise ValueError('Gold/predicted labels contain values outside the frozen taxonomy')
    result = {'variant': baseline['variant'], 'fold_id': baseline['fold_id'], 'f1_scale': '0-1',
              'coverage': {key: baseline[key] for key in [
                  'unique_attempted_reviews', 'unique_successful_reviews', 'unique_reviews_without_success',
                  'unattempted_labelled_reviews', 'successful_attempts_omitted_as_repeats',
                  'repeated_success_prediction_disagreements']}, 'views': {}}
    successes = sum(row.get('success') is True for row in records)
    result['coverage'].update(attempts=len(records), successful_attempts=successes,
                              attempt_success_rate=successes / len(records) if records else None)
    scopes = {'overall': universe, 'seen': universe - heldout, 'heldout': heldout}
    for view, keys in [('successful_unique_quality', set(predicted)),
                       ('delivered_quality_failure_as_empty', attempted)]:
        result['views'][view] = {}
        for scope, aspects in scopes.items():
            report = scope_diagnostics(keys, predicted, gold, aspects)
            expected = baseline[view][scope]['pair']
            if any(report['end_to_end_pair'][key] != expected[key] for key in ['tp', 'fp', 'fn']):
                raise AssertionError('Diagnostics disagree with existing monitoring counts')
            if report['end_to_end_pair']['micro_f1'] != expected['micro_f1']:
                raise AssertionError('Diagnostics disagree with existing monitoring micro-F1')
            report['matches_monitoring_end_to_end_counts'] = True
            result['views'][view][scope] = report
    result['view_semantics'] = {
        'successful_unique_quality': 'Unique samples with at least one successful response; excludes never-successful samples',
        'delivered_quality_failure_as_empty': 'All unique attempted samples; never-successful samples become empty predictions. A later success rescues earlier failed attempts, so also report attempt-level success rate.',
        'deduplication': 'First successful response in request-start order, identical to monitoring; repeats are not independent quality samples',
        'unattempted': 'Unattempted labelled reviews are not scored; coverage is explicit',
        'zero_denominator': 'Precision/recall/F1 use zero, matching monitoring; always read support alongside values',
    }
    return result


def diagnose_campaigns(campaigns, labels_path, artifacts_path):
    # Reject ongoing campaigns before reading any of their request/prediction files.
    for name, directory in campaigns.items():
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
            raise ValueError('Campaign names must be public identifiers, not paths')
        if read_json(Path(directory) / 'state.json').get('status') != 'completed':
            raise ValueError('Only completed campaigns may enter the preregistered diagnostic')
    artifacts, labels = read_json(artifacts_path), read_jsonl(labels_path)
    frozen = artifacts['replay']['splits']['evaluation']
    if sha(labels_path) != frozen['files']['labels']['sha256']:
        raise ValueError('Labels do not match the frozen evaluation label file')
    label_ids = [str(row['request_id']) for row in labels]
    if (len(label_ids) != frozen['count'] or len(set(label_ids)) != len(label_ids)
            or canonical_sha256(label_ids) != frozen['request_id_sha256']):
        raise ValueError('Labels do not match the exact frozen evaluation identities')
    artifact_hash = sha(artifacts_path)
    output = {'schema_version': 'serving_quality_diagnostics_v1',
              'registration_sha256': sha(REGISTRATION), 'artifacts_sha256': artifact_hash,
              'label_file_sha256': sha(labels_path), 'frozen_evaluation_reviews': len(labels),
              'scope': 'Descriptive bottleneck diagnosis on existing non-blind validation replay. No model, threshold, prompt or operating-point selection is made. Aggregate-only output.',
              'runs': []}
    for campaign_name, directory in campaigns.items():
        directory = Path(directory)
        plan = read_json(directory / 'plan.json')
        if plan.get('artifacts_sha256') != artifact_hash:
            raise ValueError('Campaign uses a different frozen artifact manifest')
        for session in plan['sessions']:
            for registered in session['runs']:
                if registered.get('quality_evaluation') is not True:
                    continue
                run_name = session['name'] + '__' + registered['name']
                if not re.fullmatch(r'[A-Za-z0-9_.-]+', run_name):
                    raise ValueError('Run name is not a safe public identifier')
                run = directory / 'runs' / run_name
                config, summary = read_json(run / 'config.json'), read_json(run / 'summary.json')
                if summary.get('interrupted'):
                    raise ValueError('Interrupted quality run cannot be reported as completed')
                if config.get('replay_sha256') != frozen['files']['replay']['sha256']:
                    raise ValueError('Quality run input differs from the frozen evaluation replay')
                records = read_jsonl(run / 'requests.jsonl')
                if summary.get('attempts') != len(records):
                    raise ValueError('Request records disagree with the final attempt count')
                variant = session['server']['variant']
                for record in records:
                    if record.get('success'):
                        response = record.get('response', {})
                        if (response.get('variant') != variant
                                or response.get('fold_id') != artifacts['fold_id']
                                or response.get('request_id') != record.get('request_id')):
                            raise ValueError('Successful response identity does not match its quality run')
                result = diagnose_records(records, labels, artifacts['taxonomy'])
                result.update(campaign=campaign_name, run=run_name,
                              registered_variant=variant, prefix_cache=session['server'].get('prefix_cache'),
                              complete_evaluation_attempt_coverage=result['coverage']['unique_attempted_reviews'] == len(labels),
                              provenance={'plan_sha256': sha(directory / 'plan.json'),
                                          'requests_sha256': sha(run / 'requests.jsonl'),
                                          'summary_sha256': sha(run / 'summary.json')})
                output['runs'].append(result)
    if not output['runs']:
        raise ValueError('No explicitly registered quality_evaluation runs were found')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', action='append', required=True, metavar='NAME=DIR')
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    campaigns = {}
    for value in args.campaign:
        if '=' not in value:
            parser.error('--campaign requires NAME=DIR')
        name, directory = value.split('=', 1)
        if name in campaigns:
            parser.error('Campaign names must be unique')
        campaigns[name] = Path(directory)
    result = diagnose_campaigns(campaigns, args.labels, args.artifacts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'quality_runs': len(result['runs']),
                      'frozen_evaluation_reviews': result['frozen_evaluation_reviews']}))


if __name__ == '__main__':
    main()
