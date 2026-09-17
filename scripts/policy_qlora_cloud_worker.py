"""Bounded pilot orchestration. No training/selection algorithm is implemented here.

Runs one registered fold, then exits for verified local backup and Pod Stop.
A wall-clock or telemetry guard stops the worker, not the RunPod billing service.
The controller must Stop the Pod after backup or an operational failure.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

STUDY = 'taxonomy_policy_qlora_completion_v1'
REPO = Path(__file__).resolve().parents[1]
PINNED = {'torch': '2.10.0+cu128', 'transformers': '4.57.6', 'accelerate': '1.14.0',
          'bitsandbytes': '0.49.2', 'peft': '0.19.1', 'numpy': '2.2.6',
          'pandas': '2.3.3', 'scikit-learn': '1.8.0', 'tokenizers': '0.22.2',
          'safetensors': '0.7.0'}

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)

def validate_transport(root):
    value = json.loads((root / 'transport.json').read_text())
    if value['study_id'] != STUDY or value['include_official_test'] is not False:
        raise ValueError('Incorrect release boundary')
    for record in value['files']:
        p = (root / record['path']).resolve()
        if not p.is_relative_to(root.resolve()) or not p.is_file() or sha(p) != record['sha256']:
            raise ValueError('Release file integrity failed: ' + record['path'])
    return value

def gpu():
    fields = 'utilization.gpu,memory.used,temperature.gpu,power.draw,pstate,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.hw_slowdown'
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=' + fields,
        '--format=csv,noheader,nounits'], text=True, timeout=20).strip()
    if len(result.splitlines()) != 1:
        raise ValueError('The pilot requires exactly one GPU')
    columns = [part.strip() for part in result.split(',')]
    return dict(zip(fields.split(','), columns))

def heartbeat_ok(root, grace=False):
    path = root / 'result/_sync/receiver_heartbeat.json'
    if not path.is_file():
        return False
    value = json.loads(path.read_text())
    stamp = dt.datetime.fromisoformat(value['at']).timestamp()
    return value.get('status') == 'pass' and -120 < time.time() - stamp < (900 if not grace else 180)

def stop_child(child):
    if child and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=15)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--scope', default='heldout-a01', choices=[f'heldout-a{i:02d}' for i in range(1,13)])
    parser.add_argument('--max-hours', type=float, default=6.0)
    args = parser.parse_args()
    if not 0 < args.max_hours <= 6:
        raise ValueError('Pilot may not exceed six worker hours')
    root = args.root.resolve()
    runtime = root / 'runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    if any((runtime / name).exists() for name in ['worker.lock', 'COMPLETE.json', 'FAILED.json']):
        raise RuntimeError('Existing terminal or running evidence: no automatic restart')
    lock = runtime / 'worker.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    started = time.time()
    child = None
    state = {'study_id': STUDY, 'planned_scopes': [args.scope], 'completed_scopes': [],
             'failure_count': 0, 'test_contract_count': 0, 'started_unix': started,
             'worker_pid': os.getpid(), 'max_worker_hours': args.max_hours,
             'pod_stop_performed': False}
    try:
        transport = validate_transport(root)
        versions = {name: metadata.version(name) for name in PINNED}
        if versions != PINNED:
            raise ValueError('Pinned scientific environment mismatch: ' + repr(versions))
        write(runtime / 'environment.json', {'versions': versions, 'python': sys.version,
            'transport_sha256': sha(root / 'transport.json'), 'gpu': gpu()})
        base = [sys.executable, '-u', str(REPO / 'scripts/run_taxonomy_policy_qlora_completion.py'),
            '--config', str(REPO / 'configs/experiments/taxonomy_policy_qlora_completion_v1.json'),
            '--data-dir', str(root / 'data'), '--output-root', str(root / 'result'),
            '--scope-id', args.scope, '--local-files-only', '--resume']
        # Prepare identity before the receiver may create its heartbeat directory.
        subprocess.run(base + ['--phase', 'prepare-scope'], check=True, timeout=180)
        state.update(status='waiting_for_local_receiver', updated_unix=time.time())
        write(runtime / 'state.json', state)
        deadline = time.monotonic() + 300
        while not heartbeat_ok(root, grace=True):
            if time.monotonic() > deadline:
                raise RuntimeError('No verified local receiver heartbeat within five minutes')
            time.sleep(5)
        phases = [('train-candidate', rate) for rate in ['2e-6', '5e-6', '1e-5']]
        phases += [('select-scope', None), ('score-selected', None)]
        slowdown_streak = 0
        with (runtime / 'worker.log').open('a', buffering=1) as log:
            for phase, rate in phases:
                command = base + ['--phase', phase]
                if rate:
                    command += ['--learning-rate', rate]
                state.update(status='running', current_phase=phase, current_learning_rate=rate,
                             current_scope=args.scope, updated_unix=time.time())
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                state['child_pid'] = child.pid
                write(runtime / 'state.json', state)
                while child.poll() is None:
                    reading = gpu()
                    bad = (float(reading['temperature.gpu']) >= 86 or
                        any(reading[key].lower() == 'active' for key in (
                            'clocks_event_reasons.hw_thermal_slowdown', 'clocks_event_reasons.hw_slowdown')))
                    slowdown_streak = slowdown_streak + 1 if bad else 0
                    if slowdown_streak >= 3:
                        raise RuntimeError('Repeated hardware/thermal slowdown')
                    if time.time() - started > args.max_hours * 3600:
                        raise RuntimeError('Pilot wall-clock ceiling reached')
                    if not heartbeat_ok(root):
                        raise RuntimeError('Local receiver heartbeat stale or missing')
                    if __import__('shutil').disk_usage(root).free < 10 * 1024**3:
                        raise RuntimeError('Disk safety margin below 10 GiB')
                    state.update(updated_unix=time.time(), gpu=reading)
                    write(runtime / 'state.json', state)
                    with (runtime / 'telemetry.jsonl').open('a') as telemetry:
                        telemetry.write(json.dumps({'at': time.time(), 'phase': phase, 'rate': rate, **reading}) + '\n')
                    time.sleep(30)
                if child.returncode != 0:
                    raise RuntimeError(f'{phase} {rate} exited {child.returncode}; no automatic retry')
                child = None
        state.update(status='READY_FOR_VERIFIED_LOCAL_BACKUP', completed_scopes=[args.scope],
                     updated_unix=time.time(), child_pid=None)
        write(runtime / 'state.json', state)
        write(runtime / 'COMPLETE.json', state)
        return 0
    except BaseException as error:
        stop_child(child)
        state.update(status='FAILED', failure_count=1, error=repr(error), updated_unix=time.time())
        write(runtime / 'state.json', state)
        write(runtime / 'FAILED.json', state)
        raise
    finally:
        lock.unlink(missing_ok=True)

if __name__ == '__main__':
    raise SystemExit(main())
