"""Freeze explicit workload plans after the development correctness gates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX_VARIANTS = {"few_shot", "hybrid"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines()
            if line.strip()]


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def portable_reference(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def verify_prefix_gate(gate_path, artifacts_path, development_path, *,
                       pair_batch_size=4, max_batch_tokens=4096):
    """Fail closed on incomplete, stale, wrong-fold or differently configured gates.

    Admission checks the existing frozen replay and all persisted per-review
    comparisons. The initial gate format did not record source-code hashes at
    process start; admission-time source fingerprints are therefore explicitly
    not represented as proof of execution-time source identity.
    """
    gate_path, artifacts_path, development_path = map(Path,
        (gate_path, artifacts_path, development_path))
    gate, artifacts = read_json(gate_path), read_json(artifacts_path)
    if gate.get('status') != 'completed' or gate.get('label_gate_pass') is not True:
        raise ValueError('Prefix gate must be completed with label_gate_pass=true')
    if gate.get('official_test_accessed') is not False:
        raise ValueError('Prefix gate must explicitly exclude official test access')
    replay = read_jsonl(development_path)
    frozen = artifacts['replay']['splits']['development']
    if frozen['count'] != 100 or len(replay) != 100:
        raise ValueError('The prefix gate must cover the exact frozen 100-review development replay')
    if sha256(development_path) != frozen['files']['replay']['sha256']:
        raise ValueError('Development replay content differs from the frozen artifact manifest')
    ids = [row['request_id'] for row in replay]
    if len(set(ids)) != 100 or canonical_sha256(ids) != frozen['request_id_sha256']:
        raise ValueError('Development replay identities differ from the frozen manifest')
    if any(set(row) != {'request_id', 'text'} or not isinstance(row['text'], str) for row in replay):
        raise ValueError('Development replay must contain label-free request records')
    expected_lengths = {row['request_id']: len(row['text']) for row in replay}
    samples = gate.get('samples')
    if not isinstance(samples, list) or len(samples) != 100:
        raise ValueError('Prefix gate must enumerate exactly 100 distinct development reviews')
    if len({row.get('request_id') for row in samples}) != 100:
        raise ValueError('Prefix gate contains duplicate sample identities')
    if {row.get('request_id'): row.get('characters') for row in samples} != expected_lengths:
        raise ValueError('Prefix gate samples/lengths differ from the frozen development replay')
    engine, parameters = gate.get('engine', {}), gate.get('parameters', {})
    if engine.get('artifacts_sha256') != sha256(artifacts_path):
        raise ValueError('Prefix gate used a different artifact manifest')
    identity = {'fold_id': artifacts['fold_id'], 'model_id': artifacts['model']['model_id'],
                'model_revision': artifacts['model']['revision'], 'logits_mode': 'last_token',
                'pair_batch_size': pair_batch_size, 'max_batch_tokens': max_batch_tokens}
    if any(engine.get(key) != value for key, value in identity.items()):
        raise ValueError('Prefix gate model/fold/batching/logit configuration differs from the plan')
    if (parameters.get('samples') != 100 or parameters.get('pair_batch_size') != pair_batch_size
            or parameters.get('max_batch_tokens') != max_batch_tokens):
        raise ValueError('Prefix gate invocation parameters differ from the plan')
    for name, expected_path in [('replay', development_path), ('artifacts', artifacts_path)]:
        supplied = Path(parameters.get(name, ''))
        if not supplied.is_absolute():
            supplied = ROOT / supplied
        if supplied.resolve() != expected_path.resolve():
            raise ValueError(f'Prefix gate {name} source path differs from its admitted frozen input')
    invoked = str(parameters.get('variants', '')).split(',')
    if len(invoked) != 2 or set(invoked) != PREFIX_VARIANTS:
        raise ValueError('Prefix gate must run both few_shot and hybrid exactly once')
    variants = gate.get('variants', {})
    if set(variants) != PREFIX_VARIANTS:
        raise ValueError('Prefix gate lacks a completed result for both few_shot and hybrid')
    for name, value in variants.items():
        if (value.get('request_count') != 100 or value.get('label_gate_pass') is not True
                or value.get('label_changed_ids') != []):
            raise ValueError(f'Prefix gate has incomplete or changed-label results for {name}')
        for key in ['aspect_max_abs_difference', 'sentiment_max_abs_difference']:
            number = value.get(key)
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number < 0:
                raise ValueError('Prefix gate must retain finite, nonnegative probability differences')
    combined = gate.get('combined_batch_checks', [])
    if len(combined) != 2 or {row.get('variant') for row in combined} != PREFIX_VARIANTS:
        raise ValueError('Prefix gate requires one combined-batch check for each variant')
    if any(row.get('label_gate_pass') is not True or row.get('stored_prefix_immutable') is not True
           or row.get('request_count') != 3 or row.get('label_changed_requests') != 0 for row in combined):
        raise ValueError('Prefix combined-batch labels or cached tensor immutability failed')
    journal_path = gate_path.parent / 'samples.jsonl'
    journal = read_jsonl(journal_path)
    for variant in PREFIX_VARIANTS:
        for phase in ['cache_off', 'cache_on', 'comparison']:
            records = [row for row in journal if row.get('variant') == variant and row.get('phase') == phase]
            observed = []
            for row in records:
                if phase == 'comparison':
                    if row.get('label_gate_pass') is not True:
                        raise ValueError('Persisted prefix comparison includes a label failure')
                    observed.append(row['request_id'])
                else:
                    request_ids = row.get('request_ids', [])
                    outputs = row.get('outputs', [])
                    if len(request_ids) != 1 or [value.get('request_id') for value in outputs] != request_ids:
                        raise ValueError('Prefix journal output identities are incomplete')
                    observed.extend(request_ids)
            if len(observed) != 100 or set(observed) != set(ids):
                raise ValueError(f'Prefix journal has incomplete or duplicate {variant}/{phase} coverage')
    return {
        'status': 'passed', 'gate_summary_sha256': sha256(gate_path),
        'gate_journal_sha256': sha256(journal_path), 'artifacts_sha256': sha256(artifacts_path),
        'development_replay_sha256': sha256(development_path),
        'development_request_id_sha256': frozen['request_id_sha256'], 'unique_development_reviews': 100,
        'verified_variants': sorted(PREFIX_VARIANTS), 'model_and_runtime': identity,
        'probability_differences': {name: {key: value[key] for key in
            ['aspect_max_abs_difference', 'sentiment_max_abs_difference']} for name, value in variants.items()},
        'gate_meaning': 'Final labels matched on all 100 development reviews and combined checks. This does not assert numerical score identity or evaluation-set equivalence.',
        'source_binding_note': 'Existing gate recorded artifact identity but not process-start source hashes. The following are admission-time fingerprints only; rerun the gate if inference/prefix implementation changed after the gate started.',
        'source_sha256_at_admission': {name: sha256(ROOT / name) for name in
            ['scripts/check_serving_prefix_gpu.py', 'src/msc_project/serving/inference.py',
             'src/msc_project/serving/prefix_cache.py']},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prefix-cache', choices=['on', 'off'], default='off')
    p.add_argument('--prefix-gate', type=Path)
    p.add_argument('--artifacts', type=Path, default=ROOT / 'outputs/serving_20261109/artifacts.json')
    p.add_argument('--development-replay', type=Path, default=ROOT / 'outputs/serving_20261109/replay_development.jsonl')
    p.add_argument('--variants', default='few_shot,qlora,hybrid')
    p.add_argument('--pressure-requests', type=int, default=30)
    p.add_argument('--open-duration', type=float, default=90)
    p.add_argument('--repeats', type=int, default=3)
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite a registered campaign plan')
    gate_provenance = None
    if args.prefix_cache == 'on':
        if not args.prefix_gate:
            raise SystemExit('A completed 100-development-input prefix equivalence gate is required')
        try:
            gate_provenance = verify_prefix_gate(args.prefix_gate, args.artifacts, args.development_replay)
        except (ValueError, KeyError, TypeError, OSError) as error:
            raise SystemExit(f'Prefix admission rejected: {error}') from error
    artifacts = read_json(args.artifacts)
    replay_sources = artifacts['replay']['splits']
    sessions = []
    for variant in args.variants.split(','):
        if variant not in ['few_shot', 'qlora', 'hybrid']:
            raise SystemExit('Invalid variant')
        runs = []
        def add(name, *, split='development', **kw):
            files = replay_sources[split]['files']
            replay_path = args.development_replay if split == 'development' else files['replay']['path']
            if sha256(replay_path) != files['replay']['sha256'] or sha256(files['metadata']['path']) != files['metadata']['sha256']:
                raise SystemExit('Replay/metadata content differs from the frozen artifact manifest')
            runs.append({'name': name, 'replay': portable_reference(replay_path),
                         'metadata': portable_reference(files['metadata']['path']), **kw})
        add('warmup', mode='closed', requests=3, concurrency=1,
            tag='warmup_excluded_from_headline')
        add('quality_evaluation', split='evaluation', mode='closed', requests=957,
            concurrency=1, quality_evaluation=True, tag='fixed_validation_engineering_replay')
        for concurrency in [1, 2, 4, 8]:
            for repeat in range(args.repeats):
                add(f'closed_c{concurrency}_r{repeat+1}', mode='closed',
                    requests=args.pressure_requests, concurrency=concurrency,
                    shuffle=True, seed=20261109 + repeat, tag='closed_load')
        references = [f'closed_c1_r{repeat+1}' for repeat in range(args.repeats)]
        for fraction in [0.25, 0.5, 0.75, 1.0, 1.25]:
            for repeat in range(args.repeats):
                add(f'open_f{str(fraction).replace(".", "p")}_r{repeat+1}', mode='open',
                    duration=args.open_duration, rate_fraction=fraction,
                    capacity_reference=references, max_inflight=64,
                    shuffle=True, seed=20261109 + repeat, tag='independent_arrival_load')
        sessions.append({'name': variant, 'server': {
            'variant': variant, 'port': 8765, 'pair_batch_size': 4,
            'max_batch_tokens': 4096, 'request_batch_size': 1, 'batch_wait_ms': 0,
            'queue_size': 8, 'timeout_s': 120, 'logits_mode': 'last_token',
            'prefix_cache': args.prefix_cache,
        }, 'runs': runs})
    plan = {'study_id': 'serving_primary_20261109',
            'registered_utc': datetime.now(timezone.utc).isoformat(),
            'purpose': 'Common engineering configuration across three frozen model pipelines',
            'artifacts': portable_reference(args.artifacts),
            'artifacts_sha256': sha256(args.artifacts),
            'prefix_gate_provenance': gate_provenance,
            'operating_setting_reason': 'Common pair batch 4 and request batch 1 preserve a straightforward matched comparison. Development screens found little further throughput benefit from combining requests, with greater response time per combined batch.',
            'quality_scope': '957 historical validation reviews; engineering replay, not a fresh blind test. Same first L2 fold for all systems.',
            'prefix_cache': args.prefix_cache,
            'arrival_rate_rule': 'Median measured c1 development throughput times the registered fraction; each resolved rate is saved before its run.',
            'sessions': sessions}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
