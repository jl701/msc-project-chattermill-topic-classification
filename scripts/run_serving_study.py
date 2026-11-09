"""Execute a preregistered serial HTTP campaign with local-only raw evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def endpoint(base, path, *, post=False):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(base + path, data=b'{}' if post else None,
                                     method='POST' if post else 'GET')
    with opener.open(request, timeout=5) as response:
        return json.loads(response.read())


def wait_idle(base, timeout=300):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        metrics = endpoint(base, '/metrics')
        if metrics.get('unfinished_tasks') == 0:
            return metrics
        time.sleep(0.25)
    raise RuntimeError('GPU worker did not drain after the client run')


def validate_completed_run(summary, resolved):
    """Never promote a gracefully interrupted or short run to completed evidence."""
    if summary.get('interrupted') is not False:
        raise RuntimeError('Run is interrupted or missing its completion declaration')
    if resolved.get('mode') == 'closed' and resolved.get('requests') is not None:
        if summary.get('attempts') != int(resolved['requests']):
            raise RuntimeError('Closed-loop run did not attempt its registered request count')
    if resolved.get('mode') == 'open':
        horizon = resolved.get('duration')
        if resolved.get('phase_plan'):
            phase_path = Path(resolved['phase_plan'])
            if not phase_path.is_absolute():
                phase_path = ROOT / phase_path
            phases = json.loads(phase_path.read_text(encoding='utf-8'))['phases']
            horizon = sum(float(phase['duration']) for phase in phases)
        if horizon is not None and summary.get('wall_seconds_including_drain', 0) < float(horizon):
            raise RuntimeError('Open-loop run did not cover its registered observation period')


def probe_api_contract(base):
    """Exercise rejected requests without invoking GPU inference."""
    cases = [
        ('malformed_json', b'{', 400),
        ('empty_text', b'{"request_id":"invalid-empty","text":""}', 400),
        ('labels_rejected', b'{"request_id":"invalid-labels","text":"synthetic","labels":[]}', 400),
        ('oversize_text', json.dumps({'request_id': 'invalid-long', 'text': 'x' * 20001}).encode(), 400),
        ('oversize_body', b'x' * 131073, 413),
    ]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    checks = []
    before = endpoint(base, '/metrics')
    for name, body, expected in cases:
        request = urllib.request.Request(base + '/predict', data=body, method='POST',
                                         headers={'Content-Type': 'application/json'})
        try:
            with opener.open(request, timeout=5) as response:
                status = response.status
        except urllib.error.HTTPError as exc:
            status = exc.code
        checks.append({'case': name, 'status': status, 'expected': expected,
                       'pass': status == expected})
    after = endpoint(base, '/metrics')
    if not all(check['pass'] for check in checks) or after['accepted'] != before['accepted']:
        raise RuntimeError('HTTP input-contract check failed')
    return {'checks': checks, 'gpu_requests_admitted': after['accepted'] - before['accepted']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    if plan.get('artifacts_sha256'):
        artifact_path = Path(plan['artifacts'])
        if not artifact_path.is_absolute():
            artifact_path = ROOT / artifact_path
        if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != plan['artifacts_sha256']:
            raise RuntimeError('Artifact manifest changed after campaign registration')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    plan_hash = hashlib.sha256(args.plan.read_bytes()).hexdigest()
    frozen_plan = args.output / 'plan.json'
    if frozen_plan.exists():
        if hashlib.sha256(frozen_plan.read_bytes()).hexdigest() != plan_hash:
            raise RuntimeError('Existing campaign has a different plan')
    else:
        frozen_plan.write_bytes(args.plan.read_bytes())
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1', PYTHONUTF8='1')
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    state = {'plan_sha256': plan_hash, 'status': 'running', 'completed': [],
             'started_utc': datetime.now(timezone.utc).isoformat()}
    state_path = args.output / 'state.json'

    def run_child(argv, log, description):
        record = {'utc': datetime.now(timezone.utc).isoformat(),
                  'description': description, 'argv': argv, 'log': str(log)}
        with (args.output / 'commands.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        with log.open('w', encoding='utf-8') as f:
            child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=f,
                                     stderr=subprocess.STDOUT, creationflags=flags)
            returncode = child.wait()
        if returncode:
            raise RuntimeError(f'{description} failed, exit {returncode}; inspect {log}')

    try:
        for session in plan['sessions']:
            name = session['name']
            server_cfg = session['server']
            variant = server_cfg['variant']
            port = int(server_cfg.get('port', 8765))
            base = f'http://127.0.0.1:{port}'
            try:
                endpoint(base, '/health')
            except (OSError, urllib.error.URLError):
                pass
            else:
                raise RuntimeError(f'Port {port} is already occupied; no existing service was stopped')
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            directory = args.output / 'server_sessions' / f'{name}_{timestamp}'
            directory.mkdir(parents=True, exist_ok=False)
            server_cmd = [sys.executable, '-u', 'scripts/serve_chattermill.py',
                          '--artifacts', plan['artifacts'], '--output-dir', str(directory)]
            for key, value in server_cfg.items():
                server_cmd.extend(['--' + key.replace('_', '-'), str(value)])
            write_json(directory / 'launch.json', {'argv': server_cmd, 'environment': {
                k: env[k] for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'TOKENIZERS_PARALLELISM']}})
            log = (directory / 'server.log').open('w', encoding='utf-8')
            process = subprocess.Popen(server_cmd, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, creationflags=flags)
            telemetry_log = (directory / 'telemetry.log').open('w', encoding='utf-8')
            telemetry = subprocess.Popen([
                sys.executable, '-u', 'scripts/collect_serving_telemetry.py',
                '--pid', str(process.pid), '--output', str(directory / 'telemetry.jsonl'),
                '--interval', '5'], cwd=ROOT, env=env, stdout=telemetry_log,
                stderr=subprocess.STDOUT, creationflags=flags)
            state.update(session=name, server_pid=process.pid, server_dir=str(directory))
            write_json(state_path, state)
            print(f'START server {name}, PID={process.pid}', flush=True)
            try:
                ready_start = time.monotonic()
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f'Server exited before ready: {directory / "server.log"}')
                    try:
                        if endpoint(base, '/health').get('ready'):
                            break
                    except (OSError, urllib.error.URLError):
                        pass
                    if time.monotonic() - ready_start > 180:
                        raise RuntimeError('Model server startup exceeded 180 seconds')
                    time.sleep(0.5)
                write_json(directory / 'startup_observation.json', {
                    'launch_to_http_ready_s': time.monotonic() - ready_start,
                    'model_load_s': json.loads((directory / 'server_metadata.json').read_text())['model_load_s']})
                write_json(directory / 'api_contract_checks.json', probe_api_contract(base))
                if args.resume:
                    warmup = next((run for run in session['runs'] if run['name'] == 'warmup'), None)
                    if warmup is None:
                        raise RuntimeError('A resumed model session requires a registered warmup')
                    restart_dir = directory / 'restart_warmup'
                    restart_dir.mkdir()
                    restart_cmd = [sys.executable, '-u', 'scripts/benchmark_serving.py',
                                   '--url', base, '--output', str(restart_dir),
                                   '--expected-variant', variant, '--replay', warmup['replay'],
                                   '--mode', 'closed', '--requests', str(warmup['requests']),
                                   '--concurrency', '1', '--timeout',
                                   str(float(server_cfg.get('timeout_s', 120)) + 10),
                                   '--tag', 'restart_warmup_excluded_from_headline']
                    run_child(restart_cmd, restart_dir / 'client.log', name + ' restart warmup')
                    wait_idle(base)
                    validate_completed_run(json.loads((restart_dir / 'summary.json').read_text()), warmup)
                for run in session['runs']:
                    run_name = name + '__' + run['name']
                    output = args.output / 'runs' / run_name
                    if output.exists() and (output / 'summary.json').exists() and args.resume:
                        previous_summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
                        validate_completed_run(previous_summary, run)
                        state['completed'].append(run_name)
                        print('RESUME skip ' + run_name, flush=True)
                        continue
                    output.mkdir(parents=True, exist_ok=True)
                    if (output / 'requests.jsonl').exists():
                        raise RuntimeError('Incomplete prior run retained; select a new run name: ' + run_name)
                    resolved = dict(run)
                    if 'rate_fraction' in resolved:
                        candidates = []
                        for prior_name in resolved['capacity_reference']:
                            prior = args.output / 'runs' / (name + '__' + prior_name) / 'summary.json'
                            summary = json.loads(prior.read_text())
                            if summary['success_rate'] < 0.99:
                                raise RuntimeError('Capacity reference has excessive failures')
                            candidates.append(summary['successful_reviews_per_second'])
                        resolved['rate'] = statistics.median(candidates) * resolved['rate_fraction']
                        resolved['reference_capacity_reviews_per_second'] = statistics.median(candidates)
                    write_json(output / 'registered_run.json', resolved)
                    cmd = [sys.executable, '-u', 'scripts/benchmark_serving.py',
                           '--url', base, '--output', str(output), '--expected-variant', variant,
                           '--timeout', str(float(server_cfg.get('timeout_s', 120)) + 10)]
                    for key in ['replay', 'metadata', 'mode', 'requests', 'duration', 'concurrency',
                                'rate', 'max_inflight', 'seed', 'metrics_interval', 'tag', 'phase_plan']:
                        if key in resolved:
                            cmd.extend(['--' + key.replace('_', '-'), str(resolved[key])])
                    if resolved.get('shuffle'):
                        cmd.append('--shuffle')
                    before = wait_idle(base)
                    write_json(output / 'server_metrics_before.json', before)
                    state.update(current_run=run_name, current_output=str(output),
                                 current_run_started_utc=datetime.now(timezone.utc).isoformat())
                    write_json(state_path, state)
                    print(f'START {run_name}', flush=True)
                    run_child(cmd, output / 'client.log', run_name)
                    after = wait_idle(base)
                    write_json(output / 'server_metrics_after.json', after)
                    if after['log_write_failures'] > before['log_write_failures']:
                        raise RuntimeError('Worker log write failure')
                    summary = json.loads((output / 'summary.json').read_text())
                    validate_completed_run(summary, resolved)
                    state['completed'].append(run_name)
                    write_json(state_path, state)
                    print(json.dumps({'DONE': run_name, 'attempts': summary['attempts'],
                        'successful': summary['successful'], 'reviews_per_s': summary['successful_reviews_per_second'],
                        'p50_ms': summary['successful_latency']['p50_ms'],
                        'p95_ms': summary['successful_latency']['p95_ms']}, ensure_ascii=False), flush=True)
            finally:
                if process.poll() is None:
                    try:
                        endpoint(base, '/shutdown', post=True)
                    except (OSError, urllib.error.URLError):
                        pass
                    try:
                        process.wait(timeout=150)
                    except subprocess.TimeoutExpired:
                        import psutil
                        owned = psutil.Process(process.pid)
                        for child in owned.children(recursive=True):
                            child.terminate()
                        process.terminate()
                        process.wait(timeout=30)
                log.close()
                try:
                    telemetry.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    telemetry.terminate()
                    telemetry.wait(timeout=15)
                telemetry_log.close()
            telemetry_path = directory / 'telemetry.jsonl'
            measurements = []
            if telemetry_path.exists():
                measurements = [json.loads(line) for line in telemetry_path.read_text(encoding='utf-8').splitlines()
                                if line.strip()]
                measurements = [row for row in measurements if row.get('event') == 'measurement']
            telemetry_check = {'exit_code': telemetry.returncode, 'measurements': len(measurements),
                               'gpu_measurements': sum(bool((row.get('gpu') or {}).get('name')) for row in measurements)}
            write_json(directory / 'telemetry_validation.json', telemetry_check)
            if telemetry.returncode != 0 or not telemetry_check['gpu_measurements']:
                raise RuntimeError('Session telemetry did not provide valid GPU measurements')
        state['status'] = 'completed'
    except Exception as exc:
        state.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(state_path, state)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
