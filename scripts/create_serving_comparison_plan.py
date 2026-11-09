"""Freeze prefix-off references and a common-rate comparison after the primary study.

CPU-only planning. The rules were registered before primary evaluation; only
development capacity aggregates determine the common rate. No labels, F1 values
or GPU results are used to select thresholds, prompts, weights or service settings.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('few_shot', 'qlora', 'hybrid')
CAPACITY_NAMES = ('closed_c1_r1', 'closed_c1_r2', 'closed_c1_r3')
REGISTRATION = 'configs/experiments/serving_comparison_extension_20261109.json'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(root, path):
    path = Path(path)
    result = path.resolve() if path.is_absolute() else (root / path).resolve()
    result.relative_to(root)
    return result


def create_plan(primary, output, *, root=ROOT):
    root = Path(root).resolve()
    primary, output = resolve(root, primary), resolve(root, output)
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing extension plan directory')
    registration_path = root / REGISTRATION
    registration = read_json(registration_path)
    if registration.get('study_id') != 'serving_comparison_extension_20261109':
        raise ValueError('Missing preregistered comparison-extension rules')
    state = read_json(primary / 'state.json')
    if state.get('status') != 'completed':
        raise ValueError('Primary campaign must be completed before freezing the extension plan')
    original = read_json(primary / 'plan.json')
    artifacts_path = resolve(root, original['artifacts'])
    artifact_hash = sha(artifacts_path)
    if original.get('artifacts_sha256') != artifact_hash:
        raise ValueError('Primary plan artifact identity differs from the current frozen manifest')
    artifacts = read_json(artifacts_path)
    if original.get('prefix_cache') != 'on':
        raise ValueError('This extension requires a prefix-on primary campaign')
    inputs = {}
    for split, count in [('development', 100), ('evaluation', 957)]:
        frozen = artifacts['replay']['splits'][split]
        if frozen['count'] != count:
            raise ValueError(f'Frozen {split} review count differs from the registration')
        inputs[split] = {}
        for kind in ['replay', 'metadata']:
            source = resolve(root, frozen['files'][kind]['path'])
            if sha(source) != frozen['files'][kind]['sha256']:
                raise ValueError(f'Frozen {split}/{kind} content changed')
            inputs[split][kind] = {'path': source.relative_to(root).as_posix(), 'sha256': sha(source)}
    sessions = {}
    for variant in VARIANTS:
        matches = [s for s in original['sessions'] if s['server'].get('variant') == variant]
        if len(matches) != 1:
            raise ValueError('Primary must contain exactly one session per comparison variant')
        sessions[variant] = matches[0]
        if matches[0]['server'].get('prefix_cache') != 'on':
            raise ValueError('Every primary comparison variant must use the prefix-on policy')
    # Only method identity differs in the common-policy comparison.
    shared = ['pair_batch_size', 'max_batch_tokens', 'request_batch_size', 'batch_wait_ms',
              'queue_size', 'timeout_s', 'logits_mode', 'prefix_cache']
    reference_server = sessions['few_shot']['server']
    if any(s['server'].get(k) != reference_server.get(k) for s in sessions.values() for k in shared):
        raise ValueError('Primary variants do not share the registered engineering configuration')
    capacities, references, quality_references = {}, {}, {}
    for variant, session in sessions.items():
        registered = {run['name']: run for run in session['runs']}
        refs = []
        for name in CAPACITY_NAMES:
            run = registered.get(name, {})
            if run.get('mode') != 'closed' or run.get('concurrency') != 1:
                raise ValueError('Missing registered closed-C1 capacity reference')
            directory = primary / 'runs' / (session['name'] + '__' + name)
            summary_path = directory / 'summary.json'
            summary = read_json(summary_path)
            value = summary.get('successful_reviews_per_second')
            if (summary.get('interrupted') or summary.get('attempts') != run.get('requests')
                    or not summary.get('attempts') or summary.get('success_rate', 0) < .99
                    or isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError('Capacity reference is partial, failed or nonpositive')
            refs.append({'run': session['name'] + '__' + name,
                         'summary_sha256': sha(summary_path), 'reviews_per_second': value,
                         'success_rate': summary['success_rate']})
        references[variant] = refs
        capacities[variant] = statistics.median(row['reviews_per_second'] for row in refs)
        quality = registered.get('quality_evaluation', {})
        if (quality.get('quality_evaluation') is not True or quality.get('requests') != 957
                or quality.get('concurrency') != 1 or quality.get('mode') != 'closed'
                or quality.get('shuffle', False)):
            raise ValueError('Primary quality reference must be the frozen unshuffled 957-review C1 run')
        directory = primary / 'runs' / (session['name'] + '__quality_evaluation')
        summary = read_json(directory / 'summary.json')
        configuration = read_json(directory / 'config.json')
        if summary.get('interrupted') or summary.get('attempts') != 957:
            raise ValueError('Primary quality reference did not attempt all 957 reviews')
        if configuration.get('replay_sha256') != inputs['evaluation']['replay']['sha256']:
            raise ValueError('Primary quality input content differs from the frozen evaluation replay')
        quality_references[variant] = {
            'run': session['name'] + '__quality_evaluation',
            'summary_sha256': sha(directory / 'summary.json'),
            'config_sha256': sha(directory / 'config.json'),
            'scope': 'Reference identity/completion only; no labels or F1 read to generate this plan',
        }
    common_rate = .5 * min(capacities.values())
    if not math.isfinite(common_rate) or common_rate <= 0:
        raise ValueError('Common absolute rate must be finite and positive')

    def replay_run(name, split='development', **settings):
        return {'name': name, 'replay': inputs[split]['replay']['path'],
                'metadata': inputs[split]['metadata']['path'], **settings}

    generated_sessions = []
    for variant in ['few_shot', 'hybrid']:
        server = deepcopy(sessions[variant]['server'])
        server['prefix_cache'] = 'off'
        runs = [replay_run('warmup', mode='closed', requests=3, concurrency=1,
                          quality_evaluation=False, tag='warmup_excluded_from_headline'),
                replay_run('quality_evaluation', 'evaluation', mode='closed', requests=957,
                           concurrency=1, quality_evaluation=True,
                           tag='paired_prefix_off_reference_no_quality_tuning')]
        generated_sessions.append({'name': variant + '_prefix_off', 'server': server, 'runs': runs})
    for variant in VARIANTS:
        server = deepcopy(sessions[variant]['server'])
        runs = [replay_run('warmup', mode='closed', requests=3, concurrency=1,
                          quality_evaluation=False, tag='warmup_excluded_from_headline')]
        for repeat in range(3):
            runs.append(replay_run(f'common_rate_r{repeat + 1}', mode='open', duration=90,
                                  rate=common_rate, max_inflight=64, shuffle=True,
                                  seed=20261109 + repeat, quality_evaluation=False,
                                  tag='matched_absolute_arrival_rate'))
        generated_sessions.append({'name': variant + '_common_rate', 'server': server, 'runs': runs})
    plan = {
        'study_id': 'serving_comparison_extension_20261109',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'registration': REGISTRATION, 'registration_sha256': sha(registration_path),
        'primary_plan_sha256': sha(primary / 'plan.json'),
        'artifacts': artifacts_path.relative_to(root).as_posix(), 'artifacts_sha256': artifact_hash,
        'fold_id': artifacts['fold_id'], 'inputs': inputs,
        'capacity_references': references, 'per_variant_median_capacity': capacities,
        'common_absolute_rate_reviews_per_second': common_rate,
        'common_rate_rule': '0.5 * min(each variant median of three completed closed-C1 development runs)',
        'quality_on_references': quality_references,
        'quality_comparison': 'Pair each prefix-off quality response to the matching primary prefix-on review ID. Report final-label set changes, presence/pair F1 and latency; never tune thresholds or other settings using this comparison.',
        'quality_scope': 'Historical validation engineering replay; not a new blind test. Partial delivery is reported with coverage and failure-as-empty view.',
        'timing_scope': 'Same full-review workload and copied settings; prefix-on/off execute sequentially, so report thermal/clock telemetry and order effects.',
        'selection_boundaries': {'labels_read': False, 'quality_used_to_choose_rate': False,
                                 'official_test_accessed': False, 'model_or_threshold_changes': False},
        'sessions': generated_sessions,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    manifest = {'status': 'frozen_plan_only_not_executed', 'plan_sha256': sha(output / 'plan.json'),
                'registration_sha256': sha(registration_path), 'factory_sha256': sha(Path(__file__)),
                'labels_read': False}
    (output / 'generation_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary-campaign', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    plan = create_plan(args.primary_campaign, args.output_dir)
    print(json.dumps({'status': 'plan_only_not_executed',
                      'sessions': len(plan['sessions']),
                      'common_absolute_rate_reviews_per_second': plan['common_absolute_rate_reviews_per_second']}, indent=2))


if __name__ == '__main__':
    main()
