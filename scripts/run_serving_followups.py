"""Run the already registered comparisons and resilience work after primary.

This is a one-off task controller, not a recurring automation. It starts no
GPU work until the primary controller has stopped its servers and declared
completion. Every child runs sequentially and all raw evidence stays local.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, default=Path('outputs/serving_20261109'))
    args = parser.parse_args()
    root = args.study_root.resolve()
    directory = root / 'followup_controller'
    directory.mkdir(parents=True, exist_ok=False)
    state = {'status': 'waiting_primary', 'completed': [],
             'started_utc': datetime.now(timezone.utc).isoformat()}
    def record(**updates):
        state.update(updates, updated_utc=datetime.now(timezone.utc).isoformat())
        (directory / 'state.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1', PYTHONUTF8='1')
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    def child(name, script, *options):
        argv = [sys.executable, '-u', '-X', 'utf8', 'scripts/' + script, *map(str, options)]
        record(status='running', current=name)
        with (directory / 'commands.jsonl').open('a', encoding='utf-8') as commands:
            commands.write(json.dumps({'name': name, 'argv': argv,
                'utc': datetime.now(timezone.utc).isoformat()}) + '\n')
        print('START ' + name, flush=True)
        with (directory / (name + '.log')).open('w', encoding='utf-8') as log:
            result = subprocess.run(argv, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, creationflags=flags)
        if result.returncode:
            raise RuntimeError(f'{name} failed with exit code {result.returncode}; inspect retained log')
        state['completed'].append(name)
        record()
        print('DONE ' + name, flush=True)
    try:
        record()
        primary = root / 'primary'
        while True:
            try:
                value = json.loads((primary / 'state.json').read_text(encoding='utf-8'))
            except json.JSONDecodeError:
                time.sleep(0.1)
                continue
            if value['status'] == 'completed':
                break
            if value['status'] != 'running':
                raise RuntimeError('Primary did not complete: ' + value['status'])
            time.sleep(10)
        child('comparison_plan', 'create_serving_comparison_plan.py',
              '--primary-campaign', primary, '--output-dir', root / 'comparison_plan')
        child('comparison', 'run_serving_study.py',
              '--plan', root / 'comparison_plan/plan.json', '--output', root / 'comparison')
        child('resilience_plan', 'create_serving_resilience_plan.py',
              '--primary-campaign', primary, '--output-dir', root / 'resilience_plan')
        child('resilience', 'run_serving_study.py',
              '--plan', root / 'resilience_plan/plan.json', '--output', root / 'resilience')
        for name in ['primary', 'comparison', 'resilience']:
            child('aggregate_' + name, 'summarise_serving_campaign.py',
                  '--campaign', root / name, '--labels', root / 'labels_evaluation.jsonl',
                  '--artifacts', root / 'artifacts.json', '--output-dir', root / ('aggregate_' + name))
        record(status='completed')
    except Exception as error:
        record(status='failed', error=type(error).__name__ + ': ' + str(error))
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
